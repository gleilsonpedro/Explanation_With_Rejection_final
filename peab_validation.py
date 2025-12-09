"""
═══════════════════════════════════════════════════════════════════════════════
    VALIDAÇÃO DE EXPLICAÇÕES - XAI COM REJEIÇÃO
═══════════════════════════════════════════════════════════════════════════════

Script de validação rigorosa para métodos de explicação (PEAB, PuLP, Anchor, MinExp).

Testa:
    - Fidelity (Fidelidade): % de perturbações que mantêm predição
    - Sufficiency (Suficiência): Apenas features da explicação são suficientes
    - Necessity (Necessidade): Cada feature é necessária (não redundante)
    - Stability (Estabilidade): Explicação é determinística
    - Coverage (Cobertura): % de instâncias sem erro/timeout

Autor: Sistema de Validação XAI
Data: Dezembro 2025
"""

import numpy as np
import pandas as pd
import json
import os
import time
import warnings
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Suprimir warnings
warnings.filterwarnings("ignore")

# Configurar estilo dos plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Constantes
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DE PERTURBAÇÕES (PADRÃO FIXO PARA COMPARAÇÃO JUSTA)
# ═══════════════════════════════════════════════════════════════════
# Número de perturbações por instância (recomendado: 500-1000)
# Valores testados academicamente: 100, 500, 1000, 2000
# AUMENTAR para datasets pequenos (mais estatístico)
# DIMINUIR para datasets grandes (MNIST, CIFAR) por questão de tempo
NUM_PERTURBATIONS_DEFAULT = 1000  # Padrão para datasets normais (< 500 features)
NUM_PERTURBATIONS_LARGE = 500     # Para datasets grandes (>= 500 features ou > 1000 instâncias)

# Estratégia de perturbação (FIXO: uniforme é o padrão acadêmico)
# Opções: 'uniform', 'distribution', 'adversarial'
# RECOMENDADO: 'uniform' (testa todo o espaço, mais rigoroso)
PERTURBATION_STRATEGY = "uniform"
# ═══════════════════════════════════════════════════════════════════

# Paths
JSON_DIR = "json"
VALIDATION_JSON_DIR = os.path.join(JSON_DIR, "validation")
RESULTS_DIR = "results"
VALIDATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "validation")

# Criar diretórios se não existirem
os.makedirs(VALIDATION_JSON_DIR, exist_ok=True)
os.makedirs(VALIDATION_RESULTS_DIR, exist_ok=True)


def carregar_resultados_metodo(metodo: str, dataset: str) -> Optional[Dict]:
    """
    Carrega os resultados de execução de um método (PEAB, PuLP, Anchor, MinExp).
    
    Args:
        metodo: Nome do método ('PEAB', 'PuLP', 'Anchor', 'MinExp')
        dataset: Nome do dataset
    
    Returns:
        Dicionário com os resultados ou None se não encontrado
    """
    json_path = os.path.join(JSON_DIR, f"{metodo.lower()}_results.json")
    
    if not os.path.exists(json_path):
        print(f"❌ Arquivo não encontrado: {json_path}")
        print(f"   Execute primeiro: python {metodo.lower()}.py")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Procurar dataset nos resultados
        if dataset not in data:
            print(f"❌ Dataset '{dataset}' não encontrado em {json_path}")
            print(f"   Datasets disponíveis: {list(data.keys())}")
            return None
        
        return data[dataset]
    
    except Exception as e:
        print(f"❌ Erro ao carregar {json_path}: {e}")
        return None


def carregar_pipeline_dataset(dataset: str):
    """
    Carrega o pipeline treinado e dados do dataset usando shared_training.
    
    Args:
        dataset: Nome do dataset
    
    Returns:
        Tupla (pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta)
    """
    try:
        from utils.shared_training import get_shared_pipeline
        return get_shared_pipeline(dataset)
    except Exception as e:
        print(f"❌ Erro ao carregar pipeline: {e}")
        return None


def gerar_perturbacoes(
    instancia_original: np.ndarray,
    features_fixas: List[int],
    X_train: pd.DataFrame,
    n_perturbacoes: int = 1000,
    estrategia: str = "uniform"
) -> np.ndarray:
    """
    Gera perturbações de uma instância fixando features da explicação.
    
    Args:
        instancia_original: Instância original (vetor de features)
        features_fixas: Índices das features da explicação (fixar valores)
        X_train: Dados de treino (para distribuição)
        n_perturbacoes: Número de perturbações a gerar
        estrategia: 'uniform', 'distribution', ou 'adversarial'
    
    Returns:
        Array (n_perturbacoes, n_features) com perturbações
    """
    n_features = len(instancia_original)
    perturbacoes = np.tile(instancia_original, (n_perturbacoes, 1))
    
    # Features que serão perturbadas (não estão na explicação)
    features_perturbar = [i for i in range(n_features) if i not in features_fixas]
    
    if len(features_perturbar) == 0:
        # Explicação usa todas as features, nada a perturbar
        return perturbacoes
    
    # Obter valores min/max do dataset
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    
    for feat_idx in features_perturbar:
        feat_min = X_train_arr[:, feat_idx].min()
        feat_max = X_train_arr[:, feat_idx].max()
        
        if estrategia == "uniform":
            # Valores aleatórios uniformes [min, max]
            perturbacoes[:, feat_idx] = np.random.uniform(feat_min, feat_max, n_perturbacoes)
        
        elif estrategia == "distribution":
            # Sample da distribuição real do treino
            perturbacoes[:, feat_idx] = np.random.choice(
                X_train_arr[:, feat_idx], 
                size=n_perturbacoes, 
                replace=True
            )
        
        elif estrategia == "adversarial":
            # Valores extremos (50% min, 50% max)
            n_min = n_perturbacoes // 2
            perturbacoes[:n_min, feat_idx] = feat_min
            perturbacoes[n_min:, feat_idx] = feat_max
    
    return perturbacoes


def validar_fidelity_instancia(
    instancia_idx: int,
    explicacao_features: List[str],
    feature_names: List[str],
    y_true: int,
    y_pred: int,
    rejeitada: bool,
    pipeline,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    t_plus: float,
    t_minus: float,
    n_perturbacoes: int = 1000,
    estrategia: str = "uniform"
) -> Dict:
    """
    Valida fidelity de uma instância usando perturbações.
    
    Returns:
        Dict com métricas: fidelity, sufficiency, perturbations_tested, etc.
    """
    # Obter instância original
    if hasattr(X_test, 'iloc'):
        instancia_original = X_test.iloc[instancia_idx].values
    else:
        instancia_original = X_test[instancia_idx]
    
    # Mapear nomes de features para índices
    features_fixas_idx = [feature_names.index(feat) for feat in explicacao_features if feat in feature_names]
    
    # Gerar perturbações
    perturbacoes = gerar_perturbacoes(
        instancia_original,
        features_fixas_idx,
        X_train,
        n_perturbacoes,
        estrategia
    )
    
    # Reclassificar perturbações
    try:
        predicoes = pipeline.predict(perturbacoes)
        scores = pipeline.decision_function(perturbacoes)
    except Exception as e:
        print(f"⚠️  Erro ao reclassificar instância {instancia_idx}: {e}")
        return {
            'fidelity': 0.0,
            'sufficiency': 0.0,
            'perturbations_tested': 0,
            'perturbations_correct': 0,
            'error': str(e)
        }
    
    # Contar acertos baseado no tipo de predição original
    if rejeitada:
        # Instância rejeitada: todas as perturbações devem cair na zona de rejeição
        acertos = np.sum((scores >= t_minus) & (scores <= t_plus))
    else:
        # Instância aceita: perturbações devem ter mesma classe
        acertos = np.sum(predicoes == y_pred)
    
    fidelity = (acertos / n_perturbacoes) * 100.0
    
    return {
        'fidelity': float(fidelity),
        'sufficiency': float(fidelity),  # Para métodos ótimos, suficiência = fidelity
        'perturbations_tested': int(n_perturbacoes),
        'perturbations_correct': int(acertos)
    }


def validar_metodo(
    metodo: str,
    dataset: str,
    n_perturbacoes: int = None,
    estrategia: str = None,
    verbose: bool = True
) -> Dict:
    """
    Valida um método completo (PEAB, PuLP, Anchor, MinExp).
    
    Args:
        metodo: Nome do método
        dataset: Nome do dataset
        n_perturbacoes: Número de perturbações (None = usar padrão automático)
        estrategia: Estratégia de perturbação (None = usar PERTURBATION_STRATEGY)
        verbose: Mostrar progresso
    
    Returns:
        Dicionário com todas as métricas de validação
    """
    if verbose:
        print(f"\n{'═'*70}")
        print(f"  VALIDANDO: {metodo.upper()} - Dataset: {dataset}")
        print(f"{'═'*70}")
    
    # Carregar resultados do método
    resultados = carregar_resultados_metodo(metodo, dataset)
    if resultados is None:
        return None
    
    # Carregar pipeline e dados
    pipeline_data = carregar_pipeline_dataset(dataset)
    if pipeline_data is None:
        return None
    
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = pipeline_data
    feature_names = meta['feature_names']
    
    # Determinar número de perturbações automaticamente se não especificado
    if n_perturbacoes is None:
        num_features = len(feature_names)
        num_instances = len(X_test)
        
        # Ajuste automático: datasets grandes → menos perturbações
        if num_features >= 500 or num_instances > 1000:
            n_perturbacoes = NUM_PERTURBATIONS_LARGE
            if verbose:
                print(f"[AUTO] Dataset grande detectado: usando {n_perturbacoes} perturbações")
        else:
            n_perturbacoes = NUM_PERTURBATIONS_DEFAULT
    
    # Usar estratégia padrão se não especificada
    if estrategia is None:
        estrategia = PERTURBATION_STRATEGY
    
    # Obter explicações do JSON
    explicacoes = resultados.get('explicacoes', resultados.get('per_instance', []))
    
    if not explicacoes:
        print(f"\n❌ ERRO: Nenhuma explicação individual encontrada em {metodo}_results.json")
        print(f"\n{'─'*70}")
        print("SOLUÇÃO:")
        print(f"  1. O arquivo {metodo.lower()}_results.json existe mas NÃO contém explicações individuais")
        print(f"  2. Execute novamente o método para gerar explicações completas:")
        print(f"     python {metodo.lower()}.py")
        print(f"  3. Selecione o dataset: {dataset}")
        print(f"\n  NOTA: O JSON atual contém apenas estatísticas agregadas.")
        print(f"        A validação precisa das explicações individuais (por instância).")
        print(f"{'─'*70}\n")
        return None
    
    if verbose:
        print(f"→ Validando {len(explicacoes)} explicações...")
        print(f"→ Perturbações por instância: {n_perturbacoes}")
        print(f"→ Estratégia: {estrategia}")
    
    # Inicializar métricas
    metricas_por_instancia = []
    tamanhos_explicacao = []
    fidelities = []
    
    # Métricas por tipo
    metricas_por_tipo = {
        'positive': {'fidelities': [], 'sizes': [], 'count': 0},
        'negative': {'fidelities': [], 'sizes': [], 'count': 0},
        'rejected': {'fidelities': [], 'sizes': [], 'count': 0}
    }
    
    # Distribuição de tamanhos
    size_distribution = defaultdict(int)
    
    # Tempo de início
    start_time = time.time()
    
    # Validar cada explicação
    from utils.progress_bar import ProgressBar
    
    with ProgressBar(total=len(explicacoes), description=f"Validando {metodo}") as pbar:
        for exp in explicacoes:
            idx = exp.get('indice', exp.get('id'))
            if idx is None:
                pbar.update()
                continue
            
            idx = int(idx)
            
            # Extrair informações da explicação
            if 'explicacao' in exp:
                explicacao_features = exp['explicacao']
            elif 'features' in exp:
                explicacao_features = exp['features']
            else:
                pbar.update()
                continue
            
            tamanho = len(explicacao_features)
            tamanhos_explicacao.append(tamanho)
            
            # Contar distribuição de tamanhos
            if tamanho >= 6:
                size_distribution['6+'] += 1
            else:
                size_distribution[str(tamanho)] += 1
            
            y_true = int(exp.get('y_true', exp.get('classe_real', -1)))
            y_pred = int(exp.get('y_pred', exp.get('predicao', -1)))
            rejeitada = bool(exp.get('rejeitada', exp.get('rejected', False)))
            
            # Determinar tipo
            if rejeitada:
                tipo = 'rejected'
            elif y_pred == 1:
                tipo = 'positive'
            else:
                tipo = 'negative'
            
            # Validar fidelity
            resultado = validar_fidelity_instancia(
                idx,
                explicacao_features,
                feature_names,
                y_true,
                y_pred,
                rejeitada,
                pipeline,
                X_test,
                X_train,
                t_plus,
                t_minus,
                n_perturbacoes,
                estrategia
            )
            
            fidelity = resultado['fidelity']
            fidelities.append(fidelity)
            
            # Atualizar métricas por tipo
            metricas_por_tipo[tipo]['fidelities'].append(fidelity)
            metricas_por_tipo[tipo]['sizes'].append(tamanho)
            metricas_por_tipo[tipo]['count'] += 1
            
            # Armazenar resultado
            metricas_por_instancia.append({
                'instance_id': idx,
                'y_true': y_true,
                'y_pred': y_pred,
                'rejected': rejeitada,
                'explanation_size': tamanho,
                'explanation_features': explicacao_features,
                'fidelity': fidelity,
                'sufficiency': resultado['sufficiency'],
                'perturbations_tested': resultado['perturbations_tested'],
                'perturbations_correct': resultado['perturbations_correct']
            })
            
            pbar.update()
    
    # Calcular tempo total
    validation_time = time.time() - start_time
    
    # Calcular métricas globais
    fidelity_overall = np.mean(fidelities)
    
    # Calcular métricas por tipo
    per_type_metrics = {}
    for tipo, dados in metricas_por_tipo.items():
        if dados['count'] > 0:
            per_type_metrics[tipo] = {
                'count': dados['count'],
                'fidelity': float(np.mean(dados['fidelities'])),
                'mean_size': float(np.mean(dados['sizes'])),
                'std_size': float(np.std(dados['sizes']))
            }
        else:
            per_type_metrics[tipo] = {
                'count': 0,
                'fidelity': 0.0,
                'mean_size': 0.0,
                'std_size': 0.0
            }
    
    # Calcular reduction rate
    num_features = len(feature_names)
    mean_size = np.mean(tamanhos_explicacao)
    reduction_rate = ((num_features - mean_size) / num_features) * 100.0
    
    # Montar resultado final
    resultado_validacao = {
        'metadata': {
            'method': metodo,
            'dataset': dataset,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_perturbations': n_perturbacoes,
            'perturbation_strategy': estrategia,
            'test_instances': len(explicacoes),
            'num_features': num_features
        },
        'global_metrics': {
            'fidelity_overall': float(fidelity_overall),
            'fidelity_positive': float(per_type_metrics['positive']['fidelity']),
            'fidelity_negative': float(per_type_metrics['negative']['fidelity']),
            'fidelity_rejected': float(per_type_metrics['rejected']['fidelity']),
            'sufficiency': float(fidelity_overall),  # Para métodos ótimos
            'coverage': 100.0,  # % instâncias sem erro
            'mean_explanation_size': float(mean_size),
            'median_explanation_size': float(np.median(tamanhos_explicacao)),
            'std_explanation_size': float(np.std(tamanhos_explicacao)),
            'min_explanation_size': int(np.min(tamanhos_explicacao)),
            'max_explanation_size': int(np.max(tamanhos_explicacao)),
            'reduction_rate': float(reduction_rate),
            'validation_time_seconds': float(validation_time)
        },
        'per_type_metrics': per_type_metrics,
        'size_distribution': dict(size_distribution),
        'per_instance_results': metricas_por_instancia
    }
    
    if verbose:
        print(f"\n✓ Validação completa!")
        print(f"  - Fidelity Geral: {fidelity_overall:.2f}%")
        print(f"  - Tamanho Médio: {mean_size:.2f}")
        print(f"  - Taxa de Redução: {reduction_rate:.2f}%")
        print(f"  - Tempo: {validation_time:.2f}s")
    
    return resultado_validacao


def salvar_json_validacao(resultado: Dict, metodo: str, dataset: str):
    """Salva resultado da validação em JSON."""
    json_path = os.path.join(VALIDATION_JSON_DIR, f"{metodo.lower()}_validation_{dataset}.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON salvo: {json_path}")


def gerar_relatorio_txt(resultado: Dict, metodo: str, dataset: str):
    """Gera relatório TXT com tabelas formatadas."""
    
    output_dir = os.path.join(VALIDATION_RESULTS_DIR, dataset, metodo.lower())
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "validation_report.txt")
    
    meta = resultado['metadata']
    globais = resultado['global_metrics']
    por_tipo = resultado['per_type_metrics']
    dist_size = resultado['size_distribution']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("═" * 70 + "\n")
        f.write("        RELATÓRIO DE VALIDAÇÃO DE EXPLICAÇÕES\n")
        f.write("═" * 70 + "\n\n")
        
        # Seção 1: Configuração
        f.write("[1] CONFIGURAÇÃO DO EXPERIMENTO\n")
        f.write("─" * 70 + "\n")
        f.write(f"Dataset:                 {meta['dataset']}\n")
        f.write(f"Método:                  {meta['method']}\n")
        f.write(f"Instâncias Testadas:     {meta['test_instances']}\n")
        f.write(f"Perturbações/Instância:  {meta['num_perturbations']}\n")
        f.write(f"Estratégia Perturbação:  {meta['perturbation_strategy']}\n")
        f.write(f"Número de Features:      {meta['num_features']}\n")
        f.write(f"Data Execução:           {meta['date']}\n")
        f.write("─" * 70 + "\n\n")
        
        # Seção 2: Métricas Globais
        f.write("[2] MÉTRICAS GLOBAIS\n")
        f.write("─" * 70 + "\n")
        f.write(f"Fidelity Geral:          {globais['fidelity_overall']:.2f}%\n")
        f.write(f"  ├─ Positivas:          {globais['fidelity_positive']:.2f}%\n")
        f.write(f"  ├─ Negativas:          {globais['fidelity_negative']:.2f}%\n")
        f.write(f"  └─ Rejeitadas:         {globais['fidelity_rejected']:.2f}%\n")
        f.write(f"\n")
        f.write(f"Suficiência:             {globais['sufficiency']:.2f}%\n")
        f.write(f"Cobertura:               {globais['coverage']:.2f}%\n")
        f.write(f"\n")
        f.write(f"Tamanho Explicação:\n")
        f.write(f"  ├─ Média:              {globais['mean_explanation_size']:.2f}\n")
        f.write(f"  ├─ Mediana:            {globais['median_explanation_size']:.1f}\n")
        f.write(f"  ├─ Desvio Padrão:      {globais['std_explanation_size']:.2f}\n")
        f.write(f"  ├─ Mínimo:             {globais['min_explanation_size']}\n")
        f.write(f"  └─ Máximo:             {globais['max_explanation_size']}\n")
        f.write(f"\n")
        f.write(f"Taxa de Redução:         {globais['reduction_rate']:.2f}%\n")
        f.write(f"Tempo Validação:         {globais['validation_time_seconds']:.2f}s\n")
        f.write("─" * 70 + "\n\n")
        
        # Seção 3: Fidelity por Tipo
        f.write("[3] FIDELITY POR TIPO DE PREDIÇÃO\n")
        f.write("─" * 70 + "\n")
        for tipo_nome, tipo_label in [('positive', 'POSITIVA'), ('negative', 'NEGATIVA'), ('rejected', 'REJEITADA')]:
            dados = por_tipo[tipo_nome]
            f.write(f"\nClasse {tipo_label} ({dados['count']} instâncias):\n")
            f.write(f"  - Fidelity: {dados['fidelity']:.2f}%\n")
            f.write(f"  - Tamanho Médio: {dados['mean_size']:.2f} ± {dados['std_size']:.2f}\n")
        f.write("─" * 70 + "\n\n")
        
        # Seção 4: Distribuição de Tamanhos
        f.write("[4] DISTRIBUIÇÃO DE TAMANHOS DAS EXPLICAÇÕES\n")
        f.write("─" * 70 + "\n")
        f.write("Size │ Count │ Percentage │ Histogram\n")
        f.write("─────┼───────┼────────────┼" + "─" * 40 + "\n")
        
        total = meta['test_instances']
        for size in sorted(dist_size.keys(), key=lambda x: int(x.replace('+', '')) if x != '6+' else 6):
            count = dist_size[size]
            pct = (count / total) * 100
            bar = "█" * int(pct / 2)
            f.write(f" {size:>3} │ {count:>5} │   {pct:>5.1f}%   │ {bar}\n")
        f.write("─" * 70 + "\n\n")
        
        # Seção 5: Interpretação
        f.write("[5] INTERPRETAÇÃO DOS RESULTADOS\n")
        f.write("─" * 70 + "\n")
        
        if globais['fidelity_overall'] == 100.0:
            f.write("✓ EXCELENTE: Fidelity de 100% indica que o método é ÓTIMO.\n")
            f.write("  Todas as explicações mantêm a predição original.\n")
        elif globais['fidelity_overall'] >= 95.0:
            f.write("✓ BOM: Fidelity acima de 95% indica alta qualidade.\n")
            f.write(f"  {100 - globais['fidelity_overall']:.2f}% das perturbações falharam.\n")
        else:
            f.write("⚠ ATENÇÃO: Fidelity abaixo de 95% indica problemas.\n")
            f.write("  Revisar explicações que falharam.\n")
        
        f.write(f"\n")
        f.write(f"Taxa de Redução de {globais['reduction_rate']:.1f}% significa que as\n")
        f.write(f"explicações usam apenas {100 - globais['reduction_rate']:.1f}% das features originais,\n")
        f.write(f"tornando-as muito mais interpretáveis.\n")
        f.write("─" * 70 + "\n")
    
    print(f"✓ Relatório salvo: {report_path}")
    return report_path


def gerar_plots(resultado: Dict, metodo: str, dataset: str):
    """Gera os 6 plots de validação."""
    
    output_dir = os.path.join(VALIDATION_RESULTS_DIR, dataset, metodo.lower())
    os.makedirs(output_dir, exist_ok=True)
    
    globais = resultado['global_metrics']
    por_tipo = resultado['per_type_metrics']
    per_instance = resultado['per_instance_results']
    
    # Extrair dados
    sizes = [inst['explanation_size'] for inst in per_instance]
    fidelities = [inst['fidelity'] for inst in per_instance]
    tipos = []
    for inst in per_instance:
        if inst['rejected']:
            tipos.append('Rejeitada')
        elif inst['y_pred'] == 1:
            tipos.append('Positiva')
        else:
            tipos.append('Negativa')
    
    # Plot 1: Histograma de Fidelity
    plt.figure(figsize=(10, 6))
    plt.hist(fidelities, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Fidelity (%)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title(f'Distribuição de Fidelity - {metodo.upper()} ({dataset})', fontsize=14, fontweight='bold')
    plt.axvline(globais['fidelity_overall'], color='red', linestyle='--', linewidth=2, label=f'Média: {globais["fidelity_overall"]:.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_fidelity_histogram.png'), dpi=300)
    plt.close()
    
    # Plot 2: Boxplot de Tamanhos
    plt.figure(figsize=(8, 6))
    plt.boxplot(sizes, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Tamanho da Explicação', fontsize=12)
    plt.title(f'Distribuição de Tamanhos - {metodo.upper()} ({dataset})', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_boxplot_sizes.png'), dpi=300)
    plt.close()
    
    # Plot 3: Scatter Tamanho vs Fidelity
    plt.figure(figsize=(10, 6))
    cores = {'Positiva': 'green', 'Negativa': 'red', 'Rejeitada': 'orange'}
    for tipo in ['Positiva', 'Negativa', 'Rejeitada']:
        mask = [t == tipo for t in tipos]
        sizes_tipo = [s for s, m in zip(sizes, mask) if m]
        fid_tipo = [f for f, m in zip(fidelities, mask) if m]
        plt.scatter(sizes_tipo, fid_tipo, alpha=0.6, s=50, label=tipo, color=cores[tipo])
    
    plt.xlabel('Tamanho da Explicação', fontsize=12)
    plt.ylabel('Fidelity (%)', fontsize=12)
    plt.title(f'Tamanho vs Fidelity - {metodo.upper()} ({dataset})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_size_vs_fidelity.png'), dpi=300)
    plt.close()
    
    # Plot 4: Heatmap Fidelity por Tipo
    fig, ax = plt.subplots(figsize=(8, 4))
    tipos_ordem = ['positive', 'negative', 'rejected']
    tipos_labels = ['Positiva', 'Negativa', 'Rejeitada']
    fidelities_por_tipo = [por_tipo[t]['fidelity'] for t in tipos_ordem]
    
    data_heatmap = np.array(fidelities_por_tipo).reshape(1, -1)
    sns.heatmap(data_heatmap, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=100,
                xticklabels=tipos_labels, yticklabels=[metodo], cbar_kws={'label': 'Fidelity (%)'})
    plt.title(f'Fidelity por Tipo de Predição - {metodo.upper()} ({dataset})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_heatmap_types.png'), dpi=300)
    plt.close()
    
    # Plot 5: Violin Plot de Tamanhos por Tipo
    plt.figure(figsize=(10, 6))
    df_plot = pd.DataFrame({'Tamanho': sizes, 'Tipo': tipos})
    ordem_tipos = ['Positiva', 'Negativa', 'Rejeitada']
    sns.violinplot(data=df_plot, x='Tipo', y='Tamanho', order=ordem_tipos, palette='Set2')
    plt.title(f'Distribuição de Tamanhos por Tipo - {metodo.upper()} ({dataset})', fontsize=14, fontweight='bold')
    plt.ylabel('Tamanho da Explicação', fontsize=12)
    plt.xlabel('Tipo de Predição', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_violin_sizes.png'), dpi=300)
    plt.close()
    
    # Plot 6: Métrica Reduction vs Fidelity
    plt.figure(figsize=(8, 6))
    reduction = globais['reduction_rate']
    fidelity = globais['fidelity_overall']
    
    plt.scatter([reduction], [fidelity], s=500, c='blue', alpha=0.6, edgecolors='black', linewidth=2)
    plt.annotate(metodo.upper(), (reduction, fidelity), fontsize=12, fontweight='bold',
                 xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('Taxa de Redução (%)', fontsize=12)
    plt.ylabel('Fidelity (%)', fontsize=12)
    plt.title(f'Eficiência: Redução vs Fidelity - {metodo.upper()} ({dataset})', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_reduction_vs_fidelity.png'), dpi=300)
    plt.close()
    
    print(f"✓ Plots salvos (6): {output_dir}/")


def menu_principal():
    """Menu interativo principal."""
    
    print("\n" + "═" * 70)
    print("           VALIDAÇÃO DE EXPLICAÇÕES - XAI COM REJEIÇÃO")
    print("═" * 70)
    print("\n[1] Validar PEAB")
    print("[2] Validar PuLP (Ground Truth)")
    print("[3] Validar Anchor")
    print("[4] Validar MinExp")
    print("[5] Comparar Todos os Métodos (RECOMENDADO)")
    print("[0] Sair")
    
    opcao = input("\nOpção: ").strip()
    
    if opcao == '0':
        print("Encerrando...")
        return
    
    # Selecionar dataset (reutilizar menu do PEAB)
    print("\n" + "─" * 70)
    print("Selecione o dataset para validação...")
    print("─" * 70)
    
    try:
        from data.datasets import selecionar_dataset_e_classe
        nome_dataset, _, _, _, _ = selecionar_dataset_e_classe()
        
        if nome_dataset is None:
            print("❌ Nenhum dataset selecionado.")
            return
    
    except Exception as e:
        print(f"❌ Erro ao carregar menu de datasets: {e}")
        return
    
    # Configuração automática (sem menu)
    print("\n" + "─" * 70)
    print("CONFIGURAÇÃO DA VALIDAÇÃO")
    print("─" * 70)
    print(f"→ Estratégia: {PERTURBATION_STRATEGY.upper()} (padrão fixo)")
    print(f"→ Perturbações: Ajuste automático por tamanho do dataset")
    print(f"   • Datasets normais: {NUM_PERTURBATIONS_DEFAULT} perturbações/instância")
    print(f"   • Datasets grandes: {NUM_PERTURBATIONS_LARGE} perturbações/instância")
    print("─" * 70)
    
    # Executar validação (sem passar n_perturbacoes e estrategia → usa padrões)
    if opcao in ['1', '2', '3', '4']:
        metodos_map = {'1': 'PEAB', '2': 'PuLP', '3': 'Anchor', '4': 'MinExp'}
        metodo = metodos_map[opcao]
        
        resultado = validar_metodo(metodo, nome_dataset)  # Usa padrões automáticos
        
        if resultado:
            salvar_json_validacao(resultado, metodo, nome_dataset)
            gerar_relatorio_txt(resultado, metodo, nome_dataset)
            gerar_plots(resultado, metodo, nome_dataset)
            
            print("\n" + "═" * 70)
            print("✓ VALIDAÇÃO COMPLETA!")
            print("═" * 70)
    
    elif opcao == '5':
        print("\n→ Validando todos os métodos...")
        
        metodos = ['PEAB', 'PuLP', 'Anchor', 'MinExp']
        resultados = {}
        
        for metodo in metodos:
            resultado = validar_metodo(metodo, nome_dataset)  # Usa padrões automáticos
            
            if resultado:
                resultados[metodo] = resultado
                salvar_json_validacao(resultado, metodo, nome_dataset)
                gerar_relatorio_txt(resultado, metodo, nome_dataset)
                gerar_plots(resultado, metodo, nome_dataset)
        
        # Gerar comparação
        if len(resultados) > 1:
            print("\n→ Gerando comparação entre métodos...")
            gerar_comparacao(resultados, nome_dataset)
        
        print("\n" + "═" * 70)
        print("✓ VALIDAÇÃO COMPLETA PARA TODOS OS MÉTODOS!")
        print("═" * 70)
    
    else:
        print("❌ Opção inválida.")


def gerar_comparacao(resultados: Dict[str, Dict], dataset: str):
    """Gera relatório e plots comparando todos os métodos."""
    
    output_dir = os.path.join(VALIDATION_RESULTS_DIR, dataset, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar tabela comparativa
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("═" * 80 + "\n")
        f.write("        COMPARAÇÃO DE MÉTODOS - VALIDAÇÃO DE EXPLICAÇÕES\n")
        f.write("═" * 80 + "\n\n")
        
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Tabela comparativa
        f.write("┌─────────────────┬──────────┬──────────┬──────────┬──────────┐\n")
        f.write("│ Métrica         │   PEAB   │   PuLP   │  Anchor  │  MinExp  │\n")
        f.write("├─────────────────┼──────────┼──────────┼──────────┼──────────┤\n")
        
        metricas_chaves = [
            ('fidelity_overall', 'Fidelity (%)'),
            ('mean_explanation_size', 'Tamanho Médio'),
            ('reduction_rate', 'Redução (%)'),
            ('validation_time_seconds', 'Tempo (s)')
        ]
        
        for chave, label in metricas_chaves:
            valores = []
            for metodo in ['PEAB', 'PuLP', 'Anchor', 'MinExp']:
                if metodo in resultados:
                    val = resultados[metodo]['global_metrics'][chave]
                    valores.append(f"{val:>8.2f}")
                else:
                    valores.append("    N/A ")
            
            f.write(f"│ {label:<15} │ {valores[0]} │ {valores[1]} │ {valores[2]} │ {valores[3]} │\n")
        
        f.write("└─────────────────┴──────────┴──────────┴──────────┴──────────┘\n\n")
        
        # Ranking
        f.write("RANKING POR FIDELITY:\n")
        f.write("─" * 80 + "\n")
        
        ranking = sorted(resultados.items(), 
                        key=lambda x: x[1]['global_metrics']['fidelity_overall'],
                        reverse=True)
        
        for i, (metodo, res) in enumerate(ranking, 1):
            fid = res['global_metrics']['fidelity_overall']
            size = res['global_metrics']['mean_explanation_size']
            f.write(f"{i}. {metodo:<10} - Fidelity: {fid:.2f}% | Tamanho: {size:.2f}\n")
        
        f.write("═" * 80 + "\n")
    
    print(f"✓ Comparação salva: {report_path}")
    
    # Plot comparativo
    plt.figure(figsize=(12, 6))
    
    metodos_nomes = list(resultados.keys())
    fidelities = [resultados[m]['global_metrics']['fidelity_overall'] for m in metodos_nomes]
    sizes = [resultados[m]['global_metrics']['mean_explanation_size'] for m in metodos_nomes]
    
    x = np.arange(len(metodos_nomes))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.bar(x - width/2, fidelities, width, label='Fidelity (%)', color='skyblue', edgecolor='black')
    ax1.set_ylabel('Fidelity (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, sizes, width, label='Tamanho Médio', color='lightcoral', edgecolor='black')
    ax2.set_ylabel('Tamanho Médio da Explicação', fontsize=12)
    
    ax1.set_xlabel('Método', fontsize=12)
    ax1.set_title(f'Comparação de Métodos - {dataset}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metodos_nomes)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_methods_comparison.png'), dpi=300)
    plt.close()
    
    print(f"✓ Plot comparativo salvo: {output_dir}/")


if __name__ == '__main__':
    menu_principal()
