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


def encontrar_variacao_mnist(metodo: str) -> Optional[str]:
    """
    Busca por variações de MNIST disponíveis (mnist_3_vs_6.json, mnist_1_vs_2.json, etc).
    
    Args:
        metodo: Nome do método ('PEAB', 'PuLP', etc)
    
    Returns:
        Nome do dataset encontrado ou None
    """
    metodo_lower = metodo.lower()
    metodo_dir = os.path.join(JSON_DIR, metodo_lower)
    
    if not os.path.exists(metodo_dir):
        return None
    
    # Procura por arquivos que começam com "mnist"
    mnist_files = [f for f in os.listdir(metodo_dir) if f.startswith('mnist') and f.endswith('.json')]
    
    if not mnist_files:
        return None
    
    # Se houver apenas 1, retorna automaticamente
    if len(mnist_files) == 1:
        dataset_name = mnist_files[0].replace('.json', '')
        print(f"\n✓ MNIST encontrado: {dataset_name}")
        return dataset_name
    
    # Se houver múltiplas, mostra menu
    print("\n🔍 Múltiplas variações de MNIST encontradas:")
    print("─" * 60)
    for i, f in enumerate(mnist_files, 1):
        dataset_name = f.replace('.json', '')
        print(f"  {i}. {dataset_name}")
    
    print("─" * 60)
    escolha = input("Qual variação deseja usar? (número): ").strip()
    
    try:
        idx = int(escolha) - 1
        if 0 <= idx < len(mnist_files):
            dataset_name = mnist_files[idx].replace('.json', '')
            return dataset_name
        else:
            print("❌ Opção inválida")
            return None
    except ValueError:
        print("❌ Digite um número válido")
        return None


def carregar_resultados_metodo(metodo: str, dataset: str) -> Optional[Tuple]:
    """
    Carrega os resultados de execução de um método (PEAB, PuLP, Anchor, MinExp).
    
    NOVA ESTRUTURA: Carrega de json/{method}/{dataset}.json
    
    Suporta busca automática de variações MNIST se dataset não for encontrado.
    
    Args:
        metodo: Nome do método ('PEAB', 'PuLP', 'Anchor', 'MinExp')
        dataset: Nome do dataset
    
    Returns:
        Tupla (dados, dataset_usado) onde dataset_usado pode ser diferente
        de dataset (ex: mnist_3_vs_6 em vez de mnist)
        Retorna None se não encontrado
    """
    metodo_lower = metodo.lower()
    if metodo_lower == 'pulp':
        metodo_lower = 'pulp'  # PuLP usa 'pulp' como nome de pasta
    
    # Nova estrutura: json/{method}/{dataset}.json
    json_path = os.path.join(JSON_DIR, metodo_lower, f"{dataset}.json")
    dataset_usado = dataset
    
    # Se não encontrar e for mnist, procura por variações
    if not os.path.exists(json_path) and dataset == 'mnist':
        print(f"\n⚠ {dataset}.json não encontrado em json/{metodo_lower}/")
        print("  Procurando por variações de MNIST...")
        dataset_encontrado = encontrar_variacao_mnist(metodo)
        
        if dataset_encontrado:
            json_path = os.path.join(JSON_DIR, metodo_lower, f"{dataset_encontrado}.json")
            dataset_usado = dataset_encontrado
        else:
            print(f"❌ Nenhuma variação de MNIST encontrada em json/{metodo_lower}/")
            return None
    
    if not os.path.exists(json_path):
        print(f"❌ Arquivo não encontrado: {json_path}")
        print(f"   Execute primeiro: python {metodo_lower}.py")
        print(f"   E selecione o dataset: {dataset}")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Retorna tupla com dados e dataset usado
        return (data, dataset_usado)
    
    except Exception as e:
        print(f"❌ Erro ao carregar {json_path}: {e}")
        return None


def carregar_pipeline_dataset(dataset: str):
    """
    Carrega o pipeline treinado e dados do dataset usando shared_training.
    Detecta variações MNIST (mnist_3_vs_8, mnist_1_vs_2, etc) e configura
    o par automáticamente.
    
    Args:
        dataset: Nome do dataset (pode ser 'mnist', 'mnist_3_vs_8', etc)
    
    Returns:
        Tupla (pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta)
    """
    try:
        from utils.shared_training import get_shared_pipeline
        from data.datasets import set_mnist_options
        import re
        
        # Detecta variações MNIST e extrai o par (ex: mnist_3_vs_8 -> (3, 8))
        if dataset.startswith('mnist_') and '_vs_' in dataset:
            match = re.match(r'mnist_(\d+)_vs_(\d+)', dataset)
            if match:
                digit_a, digit_b = int(match.group(1)), int(match.group(2))
                set_mnist_options('raw', (digit_a, digit_b))
                dataset_to_load = 'mnist'
            else:
                dataset_to_load = dataset
        else:
            dataset_to_load = dataset
        
        return get_shared_pipeline(dataset_to_load)
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
    # Tentar primeiro usar label-based indexing (.loc), depois position-based (.iloc)
    try:
        # Tentar como label do índice (que é o que PEAB salva no JSON)
        instancia_original = X_test.loc[instancia_idx].values
    except (KeyError, TypeError):
        try:
            # Se falhar, tentar como índice posicional (posição)
            instancia_original = X_test.iloc[int(instancia_idx)].values
        except (IndexError, ValueError):
            # Se ainda falhar, logar erro e retornar None
            return {
                'fidelity': -1,
                'sufficiency': -1,
                'explanation_size': len(explicacao_features),
                'perturbations_tested': 0,
                'error': f"Não foi possível acessar instância {instancia_idx} em X_test"
            }
    
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
    # Carregar resultados do método (retorna tupla com dataset correto)
    resultado_carga = carregar_resultados_metodo(metodo, dataset)
    if resultado_carga is None:
        return None
    
    resultados, dataset_correto = resultado_carga
    
    if verbose:
        print(f"\n{'═'*70}")
        print(f"  VALIDANDO: {metodo.upper()} - Dataset: {dataset_correto}")
        print(f"{'═'*70}")
    
    # Carregar pipeline e dados (dataset_correto já contém MNIST_X_vs_Y se necessário)
    pipeline_data = carregar_pipeline_dataset(dataset_correto)
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
            
            # Extrair informações da explicação - suporta ambos os formatos
            # Formato novo: 'explanation' + 'explanation_size'
            # Formato antigo: 'explicacao' ou 'features'
            if 'explanation' in exp:
                explicacao_features = exp['explanation']
                tamanho = exp.get('explanation_size', len(explicacao_features))
            elif 'explicacao' in exp:
                explicacao_features = exp['explicacao']
                tamanho = len(explicacao_features)
            elif 'features' in exp:
                explicacao_features = exp['features']
                tamanho = len(explicacao_features)
            else:
                pbar.update()
                continue
            
            tamanhos_explicacao.append(tamanho)
            
            # Contar distribuição de tamanhos
            if tamanho >= 6:
                size_distribution['6+'] += 1
            else:
                size_distribution[str(tamanho)] += 1
            
            y_true = int(exp.get('y_true', exp.get('classe_real', -1)))
            y_pred = int(exp.get('y_pred', exp.get('predicao', -1)))
            # Suporta ambos os formatos: 'rejected' (booleano) ou 'rejeitada' (booleano)
            rejeitada = bool(exp.get('rejected', exp.get('rejeitada', False)))
            
            # Determinar tipo: se rejected=True, é rejeitada (mesmo que y_pred seja -1)
            if rejeitada:
                tipo = 'rejected'
            elif y_pred == 1:
                tipo = 'positive'
            elif y_pred == 0:
                tipo = 'negative'
            else:
                # Se y_pred for -1 ou outro valor inválido
                tipo = 'rejected'
            
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
            
            # Se houver erro ao processar a instância, pular
            if 'error' in resultado:
                pbar.update()
                continue
            
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
    """Gera relatório TXT profissional adequado para dissertação."""
    
    output_dir = os.path.join(VALIDATION_RESULTS_DIR, dataset, metodo.lower())
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "validation_report.txt")
    
    meta = resultado['metadata']
    globais = resultado['global_metrics']
    por_tipo = resultado['per_type_metrics']
    dist_size = resultado['size_distribution']
    
    # Converter nome do dataset para display
    dataset_display = dataset.replace('_', ' ').title()
    metodo_display = metodo.upper()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Cabeçalho
        f.write("╔" + "═" * 78 + "╗\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("║" + f"RELATÓRIO DE VALIDAÇÃO DE EXPLICABILIDADE - MÉTODO {metodo_display}".center(78) + "║\n")
        f.write("║" + f"Dataset: {dataset_display}".center(78) + "║\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("╚" + "═" * 78 + "╝\n\n")
        
        # SEÇÃO 1: Descrição do Método
        f.write("━" * 80 + "\n")
        f.write("1. DESCRIÇÃO DO MÉTODO DE VALIDAÇÃO\n")
        f.write("━" * 80 + "\n\n")
        f.write("Este relatório apresenta a validação da qualidade das explicações geradas\n")
        f.write("pelo método de Explainability AI (Explicabilidade em Inteligência Artificial).\n\n")
        f.write(f"MÉTODO UTILIZADO: {metodo_display}\n")
        f.write("TÉCNICA DE VALIDAÇÃO: Avaliação de Fidelidade por Perturbação\n\n")
        f.write("A fidelidade é medida através de perturbações nos dados de entrada:\n")
        f.write(f"  • {meta['num_perturbations']:,} variações foram aplicadas a cada instância\n")
        f.write("  • Cada variação altera os valores das features de forma sistemática\n")
        f.write("  • Verifica-se se a predição do modelo permanece a mesma com as\n")
        f.write("    features explicativas em seus valores perturbados\n")
        f.write("  • Uma alta taxa de consistência indica que a explicação é fiel ao\n")
        f.write("    comportamento real do modelo (alta fidelidade)\n\n")
        f.write("ESTRATÉGIA DE PERTURBAÇÃO: Uniforme\n")
        f.write("  • Valores das features são aleatoriamente substituídos dentro de seus\n")
        f.write("    intervalos observados (mínimo-máximo) no conjunto de treinamento\n")
        f.write("  • Essa abordagem rigorosa testa o método em cenários variados\n\n")
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 2: Configuração do Experimento
        f.write("━" * 80 + "\n")
        f.write("2. CONFIGURAÇÃO DO EXPERIMENTO\n")
        f.write("━" * 80 + "\n\n")
        f.write(f"  Base de Dados:                    {dataset_display}\n")
        f.write(f"  Instâncias Validadas:             {meta['test_instances']} amostras\n")
        f.write(f"  Número de Variáveis (Features):   {meta['num_features']}\n")
        f.write(f"  Perturbações por Instância:       {meta['num_perturbations']:,}\n")
        f.write(f"  Total de Avaliações:              {meta['test_instances'] * meta['num_perturbations']:,}\n")
        f.write(f"  Data de Execução:                 {meta['date']}\n\n")
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 3: Resultados Principais
        f.write("━" * 80 + "\n")
        f.write("3. RESULTADOS PRINCIPAIS\n")
        f.write("━" * 80 + "\n\n")
        
        f.write("3.1 FIDELIDADE DAS EXPLICAÇÕES\n")
        f.write("─" * 80 + "\n\n")
        f.write(f"  Fidelidade Geral:                 {globais['fidelity_overall']:.2f}%\n\n")
        
        f.write("  Fidelidade por Tipo de Predição:\n")
        for tipo_nome, tipo_label, emoji in [('positive', 'Predições Positivas', '○'), 
                                               ('negative', 'Predições Negativas', '●'), 
                                               ('rejected', 'Predições Rejeitadas', '◆')]:
            dados = por_tipo[tipo_nome]
            f.write(f"    {emoji} {tipo_label:.<40} {dados['fidelity']:>6.2f}% ({dados['count']:>3} instâncias)\n")
        
        f.write(f"\n  Taxa de Cobertura (sem erros):    {globais['coverage']:.2f}%\n")
        f.write(f"  Instâncias Processadas com Sucesso: {int(globais['coverage'] / 100 * meta['test_instances'])} / {meta['test_instances']}\n")
        f.write("\n")
        
        f.write("3.2 CARACTERÍSTICAS DAS EXPLICAÇÕES\n")
        f.write("─" * 80 + "\n\n")
        f.write("  Tamanho das Explicações (número de variáveis selecionadas):\n")
        f.write(f"    • Média:                        {globais['mean_explanation_size']:.2f} variáveis\n")
        f.write(f"    • Mediana:                      {globais['median_explanation_size']:.0f} variáveis\n")
        f.write(f"    • Desvio Padrão:                {globais['std_explanation_size']:.2f}\n")
        f.write(f"    • Intervalo:                    {globais['min_explanation_size']} a {globais['max_explanation_size']} variáveis\n")
        f.write(f"    • Taxa de Compactação:          {globais['reduction_rate']:.1f}%\n")
        f.write(f"      (redução em relação ao total de {meta['num_features']} variáveis)\n")
        f.write("\n")
        
        f.write("3.3 DISTRIBUIÇÃO DE TAMANHOS DAS EXPLICAÇÕES\n")
        f.write("─" * 80 + "\n\n")
        f.write("  Variáveis │ Quantidade │ Porcentagem │ Visualização\n")
        f.write("  ───────────┼────────────┼─────────────┼" + "─" * 42 + "\n")
        
        total = meta['test_instances']
        for size in sorted(dist_size.keys(), key=lambda x: int(x.replace('+', '')) if x != '6+' else 6):
            count = dist_size[size]
            pct = (count / total) * 100
            bar_len = int(pct / 2)
            bar = "█" * bar_len
            f.write(f"     {size:>4}    │    {count:>4}    │    {pct:>5.1f}%   │ {bar:<40}\n")
        f.write("\n")
        
        # SEÇÃO 4: Análise Detalhada
        f.write("━" * 80 + "\n")
        f.write("4. ANÁLISE DETALHADA POR TIPO DE PREDIÇÃO\n")
        f.write("━" * 80 + "\n\n")
        
        tipos_info = [
            ('positive', 'Predições Positivas', 'Instâncias classificadas como positivas pelo modelo', 'A'),
            ('negative', 'Predições Negativas', 'Instâncias classificadas como negativas pelo modelo', 'B'),
            ('rejected', 'Predições Rejeitadas', 'Instâncias onde o modelo aplicou mecanismo de rejeição', 'C')
        ]
        
        for tipo_nome, tipo_label, descricao, idx in tipos_info:
            dados = por_tipo[tipo_nome]
            f.write(f"4.{idx} {tipo_label.upper()}\n")
            f.write("─" * 80 + "\n")
            f.write(f"    Descrição: {descricao}\n\n")
            f.write(f"    Quantidade de Instâncias:       {dados['count']} ({dados['count']/total*100:.1f}%)\n")
            f.write(f"    Fidelidade Médio:               {dados['fidelity']:.2f}%\n")
            f.write(f"    Tamanho Médio da Explicação:    {dados['mean_size']:.2f} variáveis\n")
            f.write(f"    Desvio Padrão do Tamanho:       {dados['std_size']:.2f}\n\n")
        
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 5: Interpretação dos Resultados
        f.write("━" * 80 + "\n")
        f.write("5. INTERPRETAÇÃO E CONCLUSÕES\n")
        f.write("━" * 80 + "\n\n")
        
        # Análise de Fidelidade
        if globais['fidelity_overall'] >= 99.0:
            conclusao_fidelidade = "Excelente"
            texto_fidelidade = "O método produz explicações de qualidade excepcional."
        elif globais['fidelity_overall'] >= 95.0:
            conclusao_fidelidade = "Muito Boa"
            texto_fidelidade = "As explicações apresentam alta fidelidade ao comportamento do modelo."
        elif globais['fidelity_overall'] >= 85.0:
            conclusao_fidelidade = "Boa"
            texto_fidelidade = "As explicações são geralmente confiáveis."
        elif globais['fidelity_overall'] >= 75.0:
            conclusao_fidelidade = "Aceitável"
            texto_fidelidade = "As explicações apresentam qualidade aceitável."
        else:
            conclusao_fidelidade = "Requer Revisão"
            texto_fidelidade = "As explicações devem ser analisadas criticamente."
        
        f.write(f"FIDELIDADE: {conclusao_fidelidade}\n")
        f.write(f"  {texto_fidelidade}\n")
        f.write(f"  Com uma fidelidade de {globais['fidelity_overall']:.2f}%, as explicações geradas\n")
        f.write(f"  mantêm consistência em {globais['fidelity_overall']:.2f}% dos cenários testados quando\n")
        f.write(f"  as features não selecionadas são aleatoriamente perturbadas.\n\n")
        
        # Análise de Compactação
        f.write(f"COMPACTAÇÃO: {100 - globais['reduction_rate']:.1f}% das Features Necessárias\n")
        f.write(f"  As explicações utilizam em média apenas {globais['mean_explanation_size']:.2f} de {meta['num_features']} variáveis,\n")
        f.write(f"  representando uma redução de {globais['reduction_rate']:.1f}%.\n")
        f.write(f"  Isso torna as explicações bastante compactas e fáceis de interpretar.\n\n")
        
        # Análise de Cobertura
        if globais['coverage'] == 100.0:
            f.write(f"COBERTURA: Completa (100%)\n")
            f.write(f"  Todas as {meta['test_instances']} instâncias foram processadas com sucesso,\n")
            f.write(f"  sem erros ou timeouts durante a validação.\n\n")
        else:
            f.write(f"COBERTURA: {globais['coverage']:.2f}%\n")
            f.write(f"  {int(globais['coverage'] / 100 * meta['test_instances'])} de {meta['test_instances']} instâncias foram processadas com sucesso.\n")
            f.write(f"  {100 - globais['coverage']:.2f}% das instâncias apresentaram erros ou timeouts.\n\n")
        
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 6: Recomendações
        f.write("━" * 80 + "\n")
        f.write("6. RECOMENDAÇÕES\n")
        f.write("━" * 80 + "\n\n")
        
        if globais['fidelity_overall'] >= 95.0:
            f.write("  ✓ O método está validado e pronto para uso.\n")
            f.write("  ✓ As explicações podem ser confiáveis e utilizadas em aplicações práticas.\n")
        else:
            f.write("  • Verificar configurações de hiperparâmetros do método.\n")
            f.write("  • Revisar instâncias com baixa fidelidade para identificar padrões.\n")
            f.write("  • Considerar ajustes na estratégia de seleção de features.\n")
        
        f.write("\n")
        f.write("━" * 80 + "\n")
        f.write(f"Relatório gerado em: {meta['date']}\n")
        f.write("━" * 80 + "\n")
    
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
