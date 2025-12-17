"""
PULP EXPERIMENT - Solver de Otimização Inteira (Ground Truth)
==============================================================
Calcula explicações ÓTIMAS usando programação inteira com PuLP (CBC solver).
Este método serve como BASELINE MATEMÁTICO para avaliar qualidade de heurísticas.

Entrada:
- Modelo treinado (LogisticRegression)
- Thresholds t+ e t- (do PEAB)
- Instâncias de teste

Saída:
- json/pulp/{dataset}.json (formato consistente com anchor/minexp/peab)
- results/report/pulp/R_*.txt (relatórios por dataset)

Características:
✓ Garante solução ÓTIMA (cardinalidade mínima)
✓ Lento mas preciso (útil para análise acadêmica)
✓ Usado para calcular GAP das heurísticas
"""

import time
import pulp
import pandas as pd
import numpy as np
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any

# Importações do projeto
from peab import (
    carregar_hiperparametros,
    DATASET_CONFIG,
    RANDOM_STATE,
    _get_lr
)
from data.datasets import selecionar_dataset_e_classe
from utils.shared_training import get_shared_pipeline  # USAR MESMO PIPELINE QUE ANCHOR/MINEXP
from utils.progress_bar import ProgressBar
from utils.results_handler import _to_builtin, update_method_results

#==============================================================================
# CONSTANTES
#==============================================================================
OUTPUT_BASE_DIR = 'results/report/pulp'

#==============================================================================
# SOLVER DE OTIMIZAÇÃO INTEIRA (GROUND TRUTH)
#==============================================================================
def calcular_explicacao_otima_pulp(
    modelo: Pipeline, 
    instance_df: pd.DataFrame, 
    X_train: pd.DataFrame, 
    t_plus: float, 
    t_minus: float,
    max_abs: float
) -> Tuple[List[str], int, str]:
    """
    Calcula a explicação ÓTIMA (cardinalidade mínima) usando PuLP.
    
    IMPORTANTE: t_plus e t_minus são thresholds NORMALIZADOS (no espaço [-1, +1]).
    O score RAW precisa ser normalizado antes da comparação!
    
    Retorna:
        (features_selecionadas, cardinalidade, tipo_predicao)
    """
    logreg = _get_lr(modelo)
    scaler = modelo.named_steps['scaler']
    
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    feature_names = instance_df.columns.tolist()
    
    vals_scaled = scaler.transform(instance_df)[0]
    
    # [CORREÇÃO CRÍTICA] Usar os mesmos limites que o PEAB!
    # PEAB usa intervalo COMPLETO [0, 1] pós-MinMaxScaler
    # NÃO usar min/max OBSERVADOS no treino (muito conservador!)
    # X_train_scaled = scaler.transform(X_train)
    # min_scaled = X_train_scaled.min(axis=0)  # ERRADO: muito conservador
    # max_scaled = X_train_scaled.max(axis=0)  # ERRADO: muito conservador
    
    # Usar intervalo COMPLETO como o PEAB:
    min_scaled = np.zeros(len(coefs))  # MinMaxScaler: mínimo teórico = 0
    max_scaled = np.ones(len(coefs))   # MinMaxScaler: máximo teórico = 1
    
    # CRÍTICO: PuLP precisa usar os MESMOS thresholds que o PEAB
    # Os thresholds vêm NORMALIZADOS do get_shared_pipeline()
    # Então o score RAW precisa ser normalizado antes da comparação!
    score_raw = modelo.decision_function(instance_df)[0]
    score_norm = score_raw / max_abs if max_abs > 0 else score_raw
    
    # Determina tipo de predição (thresholds JÁ estão normalizados)
    # IMPORTANTE: Comparar score NORMALIZADO com thresholds normalizados
    if score_norm >= t_plus: 
        tipo_predicao = "POSITIVA"
        estado = 1
    elif score_norm <= t_minus: 
        tipo_predicao = "NEGATIVA"
        estado = 0
    else: 
        tipo_predicao = "REJEITADA"
        estado = 2

    # Formulação do problema de otimização
    prob = pulp.LpProblem("ExplicacaoMinima", pulp.LpMinimize)
    z = [pulp.LpVariable(f"z_{i}", cat='Binary') for i in range(len(coefs))]
    
    # Objetivo: minimizar número de features
    prob += pulp.lpSum(z)
    
    # Cálculo dos piores cenários
    base_worst_min = intercept
    base_worst_max = intercept
    termos_min = []
    termos_max = []
    
    for i, w in enumerate(coefs):
        v_worst_min = min_scaled[i] if w > 0 else max_scaled[i]
        v_worst_max = max_scaled[i] if w > 0 else min_scaled[i]
        
        contrib_worst_min = v_worst_min * w
        contrib_worst_max = v_worst_max * w
        contrib_real = vals_scaled[i] * w
        
        base_worst_min += contrib_worst_min
        base_worst_max += contrib_worst_max
        
        termos_min.append(z[i] * (contrib_real - contrib_worst_min))
        termos_max.append(z[i] * (contrib_real - contrib_worst_max))

    # [INVESTIGAÇÃO] Scores do PuLP vs Thresholds
    # Os scores (base_worst_min/max) estão na escala RAW do modelo.
    # Os thresholds (t_plus/t_minus) estão na escala NORMALIZADA.
    # CORREÇÃO: Normalizar expressões de restrição por max_abs para
    # comparar corretamente com os thresholds normalizados.

    # [CORREÇÃO] Adicionar EPSILON como o PEAB faz, para evitar conservadorismo excessivo
    EPSILON = 1e-6
    
    # Preparar termos normalizados para as restrições
    # Observação: pulp.lpSum(termos_*) retorna uma expressão linear;
    # dividir por max_abs mantém linearidade e corrige escala.
    denom = max_abs if max_abs > 0 else 1.0
    expr_min_norm = (base_worst_min + pulp.lpSum(termos_min)) / denom
    expr_max_norm = (base_worst_max + pulp.lpSum(termos_max)) / denom

    # Restrições baseadas no tipo de predição (com tolerância EPSILON)
    if estado == 1:  # POSITIVA
        # Dar uma pequena margem de tolerância (como o PEAB)
        prob += expr_min_norm >= t_plus - EPSILON
    elif estado == 0:  # NEGATIVA
        prob += expr_max_norm <= t_minus + EPSILON
    else:  # REJEITADA
        # Para rejeitadas, dar margem nas DUAS direções
        prob += expr_max_norm <= t_plus + EPSILON
        prob += expr_min_norm >= t_minus - EPSILON

    # Resolve o problema com configurações menos conservadoras
    # Ajustar tolerâncias do solver CBC para evitar conservadorismo excessivo
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        gapRel=0.0,      # Gap de otimalidade relativo (0 = buscar ótimo exato)
        gapAbs=0.0,      # Gap de otimalidade absoluto
        options=[
            'primalT 1e-7',    # Tolerância primal (menor = menos conservador)
            'dualT 1e-7',      # Tolerância dual
            'integerT 1e-6',   # Tolerância para variáveis inteiras
        ]
    )
    prob.solve(solver)
    
    # Extrai solução
    if pulp.LpStatus[prob.status] == 'Optimal':
        features_selecionadas = [
            feature_names[i] for i in range(len(z)) 
            if pulp.value(z[i]) > 0.5
        ]
        cardinalidade = len(features_selecionadas)
    else:
        # Fallback: todas as features
        features_selecionadas = feature_names
        cardinalidade = len(feature_names)
    
    return features_selecionadas, cardinalidade, tipo_predicao

#==============================================================================
# EXECUÇÃO DO EXPERIMENTO
#==============================================================================
def executar_experimento_pulp_para_dataset(dataset_name):
    """
    Executa o experimento PuLP para um dataset específico.
    
    Args:
        dataset_name: Nome do dataset a ser processado
    """
    print(f"\n🎯 Dataset selecionado: {dataset_name}")
    print("⚠️  AVISO: PuLP é lento mas garante soluções ÓTIMAS.\n")
    
    # Normalizar nome do dataset para MNIST (adicionar par de dígitos)
    dataset_json_key = dataset_name
    cfg = DATASET_CONFIG.get(dataset_name, {})
    if dataset_name == 'mnist':
        digit_pair = cfg.get('digit_pair')
        if digit_pair and len(digit_pair) == 2:
            dataset_json_key = f"mnist_{digit_pair[0]}_vs_{digit_pair[1]}"
            print(f"   Salvando como: {dataset_json_key}\n")
    
    # =========================================================================
    # USAR GET_SHARED_PIPELINE PARA GARANTIR 100% DE CONSISTÊNCIA COM PEAB
    # =========================================================================
    print("🔧 Usando get_shared_pipeline() para garantir consistência total com PEAB/Anchor/MinExp...")
    modelo, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)
    
    nomes_classes = meta['nomes_classes']
    rejection_cost = meta['rejection_cost']
    test_size = meta['test_size']
    model_params = meta['model_params']
    
    print(f"✅ Pipeline compartilhado carregado!")
    print(f"   • Modelo: {type(modelo).__name__}")
    print(f"   • Features: {len(meta['feature_names'])}")
    print(f"   • Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   • Thresholds: t+ = {t_plus:.4f}, t- = {t_minus:.4f}")
    print(f"   • Rejection cost: {rejection_cost}")
    if meta.get('subsample_size'):
        print(f"   • Subsample: {meta['subsample_size']*100:.1f}%")
    if meta.get('top_k_features'):
        print(f"   • Top-K features: {meta['top_k_features']}")
    print()
    
    # Calcular métricas do modelo
    y_pred = modelo.predict(X_test)
    decision_scores = modelo.decision_function(X_test)
    
    # Normalizar scores usando mesmo max_abs do treino
    max_abs = model_params['norm_params']['max_abs']
    decision_scores_norm = decision_scores / max_abs if max_abs > 0 else decision_scores
    
    # Classificar com reject option
    accepted_mask = (decision_scores_norm >= t_plus) | (decision_scores_norm <= t_minus)
    rejected_mask = ~accepted_mask
    
    # Métricas
    acc_sem_rej = float(np.mean(y_pred == y_test))
    acc_com_rej = float(np.mean(y_pred[accepted_mask] == y_test.iloc[accepted_mask])) if np.any(accepted_mask) else 0.0
    taxa_rej = float(np.mean(rejected_mask))
    error_rate = 1.0 - acc_com_rej if np.any(accepted_mask) else 1.0
    risco = float(error_rate + rejection_cost * taxa_rej)
    
    metricas = {
        'acuracia_sem_rejeicao': acc_sem_rej,
        'acuracia_com_rejeicao': acc_com_rej,
        'taxa_rejeicao': taxa_rej,
        'risco_empirico': risco
    }
    
    # Processa instâncias de teste
    explicacoes = []
    stats_por_tipo = {
        'POSITIVA': {'instancias': 0, 'tamanho_total': 0, 'tempo_total': 0.0},
        'NEGATIVA': {'instancias': 0, 'tamanho_total': 0, 'tempo_total': 0.0},
        'REJEITADA': {'instancias': 0, 'tamanho_total': 0, 'tempo_total': 0.0}
    }
    
    print(f"🔬 Processando {len(X_test)} instâncias de teste...")
    with ProgressBar(total=len(X_test), description=f"PuLP - {dataset_name}") as pbar:
        for i in range(len(X_test)):
            instancia = X_test.iloc[[i]]
            classe_real = nomes_classes[y_test.iloc[i]]
            
            # IMPORTANTE: Usar índice original do DataFrame (não sequencial i)
            # Isso garante que PuLP e PEAB usem o mesmo ID de instância
            instance_id = str(X_test.index[i])
            
            start_time = time.perf_counter()
            features_otimas, tamanho, tipo_pred = calcular_explicacao_otima_pulp(
                modelo, instancia, X_train, t_plus, t_minus, max_abs
            )
            tempo_gasto = time.perf_counter() - start_time
            
            # Atualiza estatísticas
            stats_por_tipo[tipo_pred]['instancias'] += 1
            stats_por_tipo[tipo_pred]['tamanho_total'] += tamanho
            stats_por_tipo[tipo_pred]['tempo_total'] += tempo_gasto
            
            # Armazena explicação
            explicacoes.append({
                'indice': instance_id,  # Usar ID original (compatível com PEAB)
                'classe_real': classe_real,
                'tipo_predicao': tipo_pred,
                'features_selecionadas': features_otimas,
                'tamanho': int(tamanho),
                'tempo_segundos': float(tempo_gasto)
            })
            
            pbar.update()
    
    # Calcula estatísticas gerais
    total_instancias = len(explicacoes)
    tempo_total = sum(e['tempo_segundos'] for e in explicacoes)
    tamanho_medio = np.mean([e['tamanho'] for e in explicacoes])
    
    # Prepara estrutura de resultados (formato consistente)
    results_data = {
        'dataset': dataset_name,
        'metodo': 'pulp',
        'num_instancias': total_instancias,
        'params': meta['params_modelo'],  # Usar parâmetros do pipeline compartilhado
        't_plus': float(t_plus),
        't_minus': float(t_minus),
        'rejection_cost': float(rejection_cost),
        'metricas_modelo': metricas,
        'estatisticas_gerais': {
            'tamanho_medio': float(tamanho_medio),
            'tempo_total_segundos': float(tempo_total),
            'tempo_medio_segundos': float(tempo_total / total_instancias)
        },
        'estatisticas_por_tipo': {}
    }
    
    # Estatísticas por tipo de predição
    for tipo, stats in stats_por_tipo.items():
        if stats['instancias'] > 0:
            results_data['estatisticas_por_tipo'][tipo.lower()] = {
                'instancias': stats['instancias'],
                'tamanho_medio': float(stats['tamanho_total'] / stats['instancias']),
                'tempo_medio': float(stats['tempo_total'] / stats['instancias'])
            }
    
    results_data['explicacoes'] = explicacoes
    
    # Salva em JSON (estrutura: json/pulp/{dataset_json_key}.json)
    update_method_results('pulp', dataset_json_key, results_data)
    
    print(f"\n✅ JSON salvo: json/pulp/{dataset_json_key}.json")
    
    # Gera relatório em TXT
    gerar_relatorio_pulp(results_data, dataset_json_key)
    
    # Resumo final
    print("\n" + "="*80)
    print("📊 RESUMO DO EXPERIMENTO")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Instâncias processadas: {total_instancias}")
    print(f"Tamanho médio: {tamanho_medio:.2f} features")
    print(f"Tempo total: {tempo_total:.2f}s")
    print(f"Tempo médio/instância: {tempo_total/total_instancias:.4f}s")
    print("\nDistribuição por tipo:")
    for tipo, stats in stats_por_tipo.items():
        if stats['instancias'] > 0:
            pct = (stats['instancias'] / total_instancias) * 100
            tam_medio = stats['tamanho_total'] / stats['instancias']
            print(f"  {tipo:10s}: {stats['instancias']:4d} ({pct:5.1f}%) - Tam. médio: {tam_medio:.2f}")
    print("="*80 + "\n")

#==============================================================================
# GERAÇÃO DE RELATÓRIO
#==============================================================================
def gerar_relatorio_pulp(results_data: Dict, dataset_name: str):
    """
    Gera relatório detalhado em formato TXT.
    """
    output_dir = os.path.join(OUTPUT_BASE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"pulp_{dataset_name}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO PULP - {dataset_name.upper()}\n")
        f.write("Solver de Otimização Inteira (Ground Truth)\n")
        f.write("="*80 + "\n\n")
        
        # Configuração
        f.write("-" * 80 + "\n")
        f.write("1. CONFIGURAÇÃO DO EXPERIMENTO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Instâncias processadas: {results_data['num_instancias']}\n")
        f.write(f"Thresholds: t+ = {results_data['t_plus']:.4f}, t- = {results_data['t_minus']:.4f}\n")
        f.write(f"Zona de rejeição: {results_data['t_plus'] - results_data['t_minus']:.4f}\n")
        f.write(f"Rejection cost: {results_data['rejection_cost']}\n")
        f.write(f"Hiperparâmetros:\n")
        f.write(json.dumps(results_data['params'], indent=2))
        f.write("\n\n")
        
        # Métricas do modelo
        f.write("-" * 80 + "\n")
        f.write("2. MÉTRICAS DO MODELO\n")
        f.write("-" * 80 + "\n")
        metricas = results_data['metricas_modelo']
        f.write(f"Acurácia (sem rejeição): {metricas.get('acuracia_sem_rejeicao', 0.0)*100:.2f}%\n")
        f.write(f"Acurácia (com rejeição): {metricas.get('acuracia_com_rejeicao', 0.0)*100:.2f}%\n")
        f.write(f"Taxa de rejeição: {metricas.get('taxa_rejeicao', 0.0)*100:.2f}%\n")
        f.write(f"Risco empírico: {metricas.get('risco_empirico', 0.0):.4f}\n")
        f.write("\n")
        
        # Estatísticas gerais
        f.write("-" * 80 + "\n")
        f.write("3. ESTATÍSTICAS DAS EXPLICAÇÕES (ÓTIMAS)\n")
        f.write("-" * 80 + "\n")
        stats = results_data['estatisticas_gerais']
        f.write(f"Tamanho médio: {stats['tamanho_medio']:.2f} features\n")
        f.write(f"Tempo total: {stats['tempo_total_segundos']:.2f}s\n")
        f.write(f"Tempo médio/instância: {stats['tempo_medio_segundos']:.4f}s\n")
        f.write("\n")
        
        # Estatísticas por tipo
        f.write("-" * 80 + "\n")
        f.write("4. DETALHAMENTO POR TIPO DE PREDIÇÃO\n")
        f.write("-" * 80 + "\n")
        for tipo, stats_tipo in results_data['estatisticas_por_tipo'].items():
            pct = (stats_tipo['instancias'] / results_data['num_instancias']) * 100
            f.write(f"\n{tipo.upper()}:\n")
            f.write(f"  - Instâncias: {stats_tipo['instancias']} ({pct:.1f}%)\n")
            f.write(f"  - Tamanho médio: {stats_tipo['tamanho_medio']:.2f} features\n")
            f.write(f"  - Tempo médio: {stats_tipo['tempo_medio']:.4f}s\n")
        f.write("\n")
        
        # TOP 10 maiores e menores explicações
        f.write("-" * 80 + "\n")
        f.write("5. TOP 10 MAIORES EXPLICAÇÕES\n")
        f.write("-" * 80 + "\n")
        explicacoes_ordenadas = sorted(
            results_data['explicacoes'], 
            key=lambda x: x['tamanho'], 
            reverse=True
        )
        for i, exp in enumerate(explicacoes_ordenadas[:10], 1):
            f.write(f"{i:2d}. Instância {exp['indice']:>8s} ({exp['tipo_predicao']:10s}): "
                   f"{exp['tamanho']:3d} features - {exp['tempo_segundos']:.4f}s\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("6. TOP 10 MENORES EXPLICAÇÕES\n")
        f.write("-" * 80 + "\n")
        explicacoes_menores = sorted(
            results_data['explicacoes'], 
            key=lambda x: x['tamanho']
        )
        for i, exp in enumerate(explicacoes_menores[:10], 1):
            f.write(f"{i:2d}. Instância {exp['indice']:>8s} ({exp['tipo_predicao']:10s}): "
                   f"{exp['tamanho']:3d} features - {exp['tempo_segundos']:.4f}s\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Relatório salvo: {output_file}")

#==============================================================================
# PONTO DE ENTRADA
#==============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("   PULP EXPERIMENT - Solver de Otimização Inteira (Ground Truth)")
    print("="*80 + "\n")
    print("⚠️  AVISO: PuLP é lento mas garante soluções ÓTIMAS.\n")
    
    # Seleção do dataset
    resultado = selecionar_dataset_e_classe()
    
    # Verifica se foram selecionados múltiplos datasets
    if resultado[0] == '__MULTIPLE__':
        datasets_lista = resultado[4]
        
        print(f"\n📋 Executando {len(datasets_lista)} datasets em sequência...")
        print("="*80 + "\n")
        
        for i, dataset_name in enumerate(datasets_lista, 1):
            print(f"\n{'='*80}")
            print(f"   [{i}/{len(datasets_lista)}] Executando: {dataset_name.upper()}")
            print(f"{'='*80}\n")
            
            try:
                executar_experimento_pulp_para_dataset(dataset_name)
                print(f"\n✅ [{i}/{len(datasets_lista)}] {dataset_name} concluído com sucesso!")
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Execução interrompida pelo usuário.")
                print(f"📊 Datasets concluídos: {i-1}/{len(datasets_lista)}")
                break
                
            except Exception as e:
                print(f"\n❌ Erro ao processar {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n⏭️  Continuando para o próximo dataset...\n")
                continue
        
        print(f"\n{'='*80}")
        print(f"   ✅ Execução em lote concluída!")
        print(f"{'='*80}\n")
    
    else:
        # Execução de dataset único
        dataset_name = resultado[0]
        
        if not dataset_name:
            print("❌ Nenhum dataset selecionado. Encerrando.")
        else:
            executar_experimento_pulp_para_dataset(dataset_name)
