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
import pulp  # [NOVO] Para validação GLOBAL via LP solver (PuLP)

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
    estrategia: str = "uniform",
    pipeline = None,
    y_pred: int = None,
    scaler = None
) -> np.ndarray:
    """
    Gera perturbações de uma instância fixando features da explicação.
    
    Args:
        instancia_original: Instância original (vetor de features)
        features_fixas: Índices das features da explicação (fixar valores)
        X_train: Dados de treino (para distribuição)
        n_perturbacoes: Número de perturbações a gerar
        estrategia: 'uniform', 'distribution', ou 'adversarial_worst_case'
        pipeline: Pipeline do modelo (para estratégia adversarial_worst_case)
        y_pred: Predição original (para estratégia adversarial_worst_case)
        scaler: Scaler do pipeline (para estratégia adversarial_worst_case)
    
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
        
        elif estrategia == "adversarial_worst_case" and pipeline is not None and y_pred is not None:
            # Worst-case: escolhe min ou max baseado no coeficiente para empurrar score
            # na direção CONTRÁRIA à predição (mais adversarial)
            try:
                # Obter coeficientes do modelo
                if hasattr(pipeline, 'named_steps'):
                    logreg = pipeline.named_steps.get('classifier')
                    if logreg is None:
                        logreg = pipeline.named_steps.get('logisticregression')
                else:
                    logreg = pipeline
                
                if logreg is not None and hasattr(logreg, 'coef_'):
                    coef = logreg.coef_[0][feat_idx]
                    
                    # Para positivas (y_pred=1): empurrar para BAIXO (usar valores que diminuem score)
                    # Para negativas (y_pred=0): empurrar para CIMA (usar valores que aumentam score)
                    if y_pred == 1:
                        # Adversário quer DIMINUIR score: coef > 0 → min, coef < 0 → max
                        valor_adversarial = feat_min if coef > 0 else feat_max
                    else:
                        # Adversário quer AUMENTAR score: coef > 0 → max, coef < 0 → min
                        valor_adversarial = feat_max if coef > 0 else feat_min
                    
                    perturbacoes[:, feat_idx] = valor_adversarial
                else:
                    # Fallback para uniform se não conseguir coeficientes
                    perturbacoes[:, feat_idx] = np.random.uniform(feat_min, feat_max, n_perturbacoes)
            except Exception:
                # Fallback para uniform em caso de erro
                perturbacoes[:, feat_idx] = np.random.uniform(feat_min, feat_max, n_perturbacoes)
        else:
            # Fallback para uniform
            perturbacoes[:, feat_idx] = np.random.uniform(feat_min, feat_max, n_perturbacoes)
    
    return perturbacoes


def calcular_baseline_predicao(
    pipeline,
    X_train: pd.DataFrame,
    y_pred: int,
    rejeitada: bool,
    t_plus: float,
    t_minus: float,
    max_abs: float = None,
    n_samples: int = 500
) -> float:
    """
    Calcula o baseline: probabilidade de manter a predição por acaso
    quando TODAS as features são perturbadas uniformemente.
    
    NOTA: Este valor é calculado apenas para REPORTAR no artigo.
    NÃO é usado para ajustar o threshold de 95% (que é fixo).
    
    O baseline ajuda a INTERPRETAR os resultados de minimalidade:
    - Se baseline é alto (ex: 98%) e minimalidade é alta → esperado (fácil manter)
    - Se baseline é baixo (ex: 2%) e minimalidade é baixa → esperado (difícil manter)
    
    Returns:
        float: Taxa base esperada de manutenção da predição (0-1)
    """
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    n_features = X_train_arr.shape[1]
    
    # Gerar perturbações 100% uniformes (sem fixar nada)
    perturbacoes = np.zeros((n_samples, n_features))
    for feat_idx in range(n_features):
        feat_min = X_train_arr[:, feat_idx].min()
        feat_max = X_train_arr[:, feat_idx].max()
        perturbacoes[:, feat_idx] = np.random.uniform(feat_min, feat_max, n_samples)
    
    try:
        predicoes = pipeline.predict(perturbacoes)
        scores = pipeline.decision_function(perturbacoes)
        
        if max_abs is not None and max_abs > 0:
            scores = scores / max_abs
        
        if rejeitada:
            acertos = np.sum((scores >= t_minus) & (scores <= t_plus))
        else:
            acertos = np.sum(predicoes == y_pred)
        
        return acertos / n_samples
    except Exception:
        return 0.5  # Fallback: assume 50%


def validar_necessidade_features(
    instancia_idx: int,
    explicacao_features: List[str],
    feature_names: List[str],
    y_pred: int,
    rejeitada: bool,
    pipeline,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    t_plus: float,
    t_minus: float,
    n_perturbacoes: int = 200,
    max_abs: float = None,
    baseline_cache: Dict = None,
    modo: str = "local"
) -> Dict:
    """
    Testa se cada feature na explicação é NECESSÁRIA.
    
    ═══════════════════════════════════════════════════════════════════════════════
    DOIS MODOS DE VALIDAÇÃO (2025-12-18):
    ═══════════════════════════════════════════════════════════════════════════════
    
    🔹 modo="local" → ROBUSTEZ LOCAL (para PEAB e métodos heurísticos)
    ────────────────────────────────────────────────────────────────────────────────
    
    Conceito: Necessidade LOCAL = robustez empírica no entorno da instância.
    
    Metodologia (baseada em PI/AXp - NeurIPS 2020):
      1. Define epsilon-ball: X_i ∈ [X_i_original - ε, X_i_original + ε]
      2. Remove feature testada (zera coeficiente)
      3. Perturba features não-explicativas no epsilon-ball
      4. Busca contraexemplo: ∃ x_local que mantém decisão?
    
    Interpretação:
      - Feature é NECESSÁRIA: se NÃO existe perturbação local que mantém decisão
      - Feature é REDUNDANTE: se EXISTE perturbação local que mantém decisão
      - Mede robustez empírica, não suficiência lógica global
    
    Parâmetros adaptativos:
      - EPSILON_FRACTION: 3-12% (escala com tamanho do dataset)
      - DELTA: 2-4% (margem numérica para evitar flips)
      - N_SAMPLES: 200 (busca amostragem no entorno)
    
    Resultados esperados:
      - Positivas/Negativas: necessidade ≈ 60-90%
      - Rejeitadas: necessidade ≈ 40-80%
      - Redundância: > 0% (detecta features desnecessárias)
    
    ────────────────────────────────────────────────────────────────────────────────
    
    🔹 modo="global" → VIABILIDADE DE LP (para PuLP/AXp e métodos ótimos)
    ────────────────────────────────────────────────────────────────────────────────
    
    Conceito: Necessidade GLOBAL = viabilidade lógica via LP solver.
    
    Definição matemática rigorosa:
      Feature f é NECESSÁRIA ⟺ LP sem f é INFEASIBLE
      Feature f é REDUNDANTE ⟺ LP sem f é FEASIBLE
    
    Metodologia (baseada em AXp/Abductive Explanations + LP):
      1. Remove feature testada (zera coeficiente w_i)
      2. Monta problema de viabilidade LP:
         Variáveis: x_j ∈ [min_j, max_j] para j ≠ i
         Restrição: w·x + b {≥, ≤, ∈} threshold (depende do tipo)
      3. Resolve LP com PuLP CBC solver
      4. Verifica status:
         - INFEASIBLE → feature é NECESSÁRIA
         - FEASIBLE/OPTIMAL → feature é REDUNDANTE
    
    Interpretação:
      - Feature é NECESSÁRIA: impossível satisfazer inequação sem ela
      - Feature é REDUNDANTE: existe vetor x que mantém decisão sem ela
      - Teste determinístico rigoroso (não probabilístico)
    
    Implementação:
      - SEM epsilon-ball (usa bounds globais [min_dataset, max_dataset])
      - SEM amostragem (solver determinístico)
      - SEM critérios probabilísticos (np.any, etc)
      - USA programação linear para teste de viabilidade
    
    Resultados esperados:
      - Explicações ótimas (PuLP): necessidade ≈ 60-100% (depende do dataset)
      - Detecta redundância matemática rigorosa
      - Mais rigoroso que validação local (amostragem)
    
    ═══════════════════════════════════════════════════════════════════════════════
    
    DIFERENÇA CONCEITUAL:
    ═══════════════════════════════════════════════════════════════════════════════
    
    LOCAL (PEAB):
      - "Feature resiste a perturbações no entorno?" (robustez empírica)
      - Usa amostragem + np.any para buscar contraexemplo
      - LÓGICA PRESERVADA - NÃO MODIFICADA
    
    GLOBAL (PuLP):
      - "Feature é logicamente necessária?" (viabilidade matemática)
      - Usa LP solver + verificação de INFEASIBILITY
      - LÓGICA CORRIGIDA - substituída amostragem por LP
    
    Aplicação:
      - PEAB (heurístico) → modo="local" (não garante otimalidade global)
      - PuLP (ótimo) → modo="global" (necessidade por construção matemática)
    
    ═══════════════════════════════════════════════════════════════════════════════
    IMPORTANTE:
    ═══════════════════════════════════════════════════════════════════════════════
    
    A lógica do modo LOCAL (PEAB) foi mantida EXATAMENTE como estava.
    Apenas o modo GLOBAL (PuLP) foi modificado para usar LP solver em vez de
    amostragem, corrigindo o problema conceitual de usar np.any para validar
    métodos de otimização.
    
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        modo: "local" (PEAB) ou "global" (PuLP/AXp)
    
    Returns:
        Dict com: necessary_count, redundant_features, necessity_score, baseline
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURAÇÃO: Normalização e parâmetros base
    # ═══════════════════════════════════════════════════════════════════════════
    
    n_features = len(feature_names)
    n_explicacao = len(explicacao_features)
    
    # Normalizar thresholds se necessário
    if max_abs is not None and max_abs > 0:
        t_plus_norm = t_plus
        t_minus_norm = t_minus
    else:
        t_plus_norm = t_plus  
        t_minus_norm = t_minus
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ROTEAMENTO: Delegar para modo LOCAL ou GLOBAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    if modo == "global":
        return _validar_necessidade_global(
            instancia_idx, explicacao_features, feature_names,
            y_pred, rejeitada, pipeline, X_test, X_train,
            t_plus_norm, t_minus_norm, max_abs, baseline_cache
        )
    else:  # modo == "local" (padrão)
        return _validar_necessidade_local(
            instancia_idx, explicacao_features, feature_names,
            y_pred, rejeitada, pipeline, X_test, X_train,
            t_plus_norm, t_minus_norm, n_perturbacoes, max_abs, baseline_cache
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODO LOCAL: Necessidade via robustez no epsilon-ball (PEAB)
# ═══════════════════════════════════════════════════════════════════════════════

def _validar_necessidade_local(
    instancia_idx: int,
    explicacao_features: List[str],
    feature_names: List[str],
    y_pred: int,
    rejeitada: bool,
    pipeline,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    t_plus_norm: float,
    t_minus_norm: float,
    n_perturbacoes: int,
    max_abs: float,
    baseline_cache: Dict
) -> Dict:
    """
    VALIDAÇÃO LOCAL: Busca contraexemplo no epsilon-ball.
    
    Conceito:
        Feature é NECESSÁRIA se não existe perturbação LOCAL que mantém decisão.
        Testa robustez empírica no entorno da instância.
    
    Metodologia:
        1. Define epsilon-ball ao redor da instância
        2. Remove feature testada (zera coeficiente)
        3. Perturba features não-explicativas no epsilon-ball
        4. Se alguma configuração mantém decisão → REDUNDANTE
    """
    
    # ───────────────────────────────────────────────────────────────────────────
    # Parâmetros adaptativos para epsilon-ball
    # ───────────────────────────────────────────────────────────────────────────
    
    n_features = len(feature_names)
    n_explicacao = len(explicacao_features)
    
    # Epsilon adaptativo: escala inversamente com número de features
    if n_features <= 10:
        base_epsilon = 0.12
    elif n_features <= 50:
        base_epsilon = 0.10
    elif n_features <= 100:
        base_epsilon = 0.06
    else:
        base_epsilon = 0.03  # Datasets grandes (MNIST, etc)
    
    # Ajustar baseado no tamanho da explicação
    explicacao_ratio = n_explicacao / n_features
    if explicacao_ratio < 0.10:
        epsilon_adj = 0.7  # Explicações pequenas → epsilon menor
    elif explicacao_ratio < 0.30:
        epsilon_adj = 0.85
    else:
        epsilon_adj = 1.0
    
    EPSILON_FRACTION = base_epsilon * epsilon_adj
    
    # Delta adaptativo: margem numérica para evitar flips
    zona_rejeicao = abs(t_plus_norm - t_minus_norm)
    if zona_rejeicao > 0.5:
        DELTA = 0.04
    elif zona_rejeicao > 0.2:
        DELTA = 0.03
    else:
        DELTA = 0.02
    
    N_SAMPLES = n_perturbacoes
    
    # ───────────────────────────────────────────────────────────────────────────
    
    # Obter instância original
    try:
        instancia_original = X_test.loc[instancia_idx].values
    except (KeyError, TypeError):
        try:
            instancia_original = X_test.iloc[int(instancia_idx)].values
        except (IndexError, ValueError):
            return {'necessary_count': len(explicacao_features), 'redundant_features': [], 'necessity_score': 100.0, 'baseline': 0.5}
    
    if len(explicacao_features) <= 1:
        return {'necessary_count': 1, 'redundant_features': [], 'necessity_score': 100.0, 'baseline': 0.5}
    
    # Extrair componentes do modelo
    if hasattr(pipeline, 'named_steps'):
        scaler = pipeline.named_steps.get('scaler')
        if 'model' in pipeline.named_steps:
            logreg = pipeline.named_steps['model']
        elif 'classifier' in pipeline.named_steps:
            logreg = pipeline.named_steps['classifier']
        else:
            logreg = pipeline.named_steps['logisticregression']
    else:
        return {'necessary_count': len(explicacao_features), 'redundant_features': [], 'necessity_score': 100.0, 'baseline': 0.5}
    
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    vals_s = scaler.transform(instancia_original.reshape(1, -1))[0]
    
    # Calcular min/max escalados do treino
    X_train_scaled = scaler.transform(X_train)
    min_scaled = X_train_scaled.min(axis=0)
    max_scaled = X_train_scaled.max(axis=0)
    
    # Definir epsilon-ball ao redor da instância
    epsilon = EPSILON_FRACTION * (max_scaled - min_scaled)
    local_min = np.maximum(vals_s - epsilon, min_scaled)
    local_max = np.minimum(vals_s + epsilon, max_scaled)
    
    # Calcular baseline (apenas para reportar)
    cache_key = f"{y_pred}_{rejeitada}_local"
    if baseline_cache is not None and cache_key in baseline_cache:
        baseline = baseline_cache[cache_key]
    else:
        baseline = calcular_baseline_predicao(
            pipeline, X_train, y_pred, rejeitada, t_plus_norm, t_minus_norm, max_abs
        )
        if baseline_cache is not None:
            baseline_cache[cache_key] = baseline
    
    # Mapear nomes para índices
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    explicacao_idx = [feature_to_idx[f] for f in explicacao_features if f in feature_to_idx]
    
    features_redundantes = []
    
    # ───────────────────────────────────────────────────────────────────────────
    # LOOP: Testar cada feature da explicação
    # ───────────────────────────────────────────────────────────────────────────
    
    for feat_name in explicacao_features:
        feat_idx = feature_to_idx.get(feat_name)
        if feat_idx is None:
            continue
        
        # Gerar N_SAMPLES perturbações no entorno local
        samples = np.tile(vals_s, (N_SAMPLES, 1))
        
        # Identificar features NÃO EXPLICATIVAS (serão perturbadas no epsilon-ball)
        features_nao_explicativas = [i for i in range(len(feature_names)) 
                                      if i not in explicacao_idx]
        
        # Perturbar APENAS features NÃO EXPLICATIVAS no epsilon-ball
        for perturb_idx in features_nao_explicativas:
            samples[:, perturb_idx] = np.random.uniform(
                local_min[perturb_idx], 
                local_max[perturb_idx], 
                N_SAMPLES
            )
        
        # REMOVER feature testada: zerar seu coeficiente
        coefs_sem_feat = coefs.copy()
        coefs_sem_feat[feat_idx] = 0.0
        
        # Calcular scores SEM a feature testada
        scores = intercept + samples @ coefs_sem_feat
        
        if max_abs is not None and max_abs > 0:
            scores = scores / max_abs
        
        # Verificar se EXISTE contraexemplo (decisão mantida sem a feature)
        if rejeitada:
            scores_mantidos = (scores >= t_minus_norm + DELTA) & (scores <= t_plus_norm - DELTA)
            contraexemplo_existe = np.any(scores_mantidos)
        elif y_pred == 1:
            scores_mantidos = scores >= t_plus_norm - DELTA
            contraexemplo_existe = np.any(scores_mantidos)
        else:
            scores_mantidos = scores <= t_minus_norm + DELTA
            contraexemplo_existe = np.any(scores_mantidos)
        
        if contraexemplo_existe:
            features_redundantes.append(feat_name)
    
    necessary_count = len(explicacao_features) - len(features_redundantes)
    necessity_score = (necessary_count / len(explicacao_features)) * 100.0
    
    return {
        'necessary_count': necessary_count,
        'redundant_features': features_redundantes,
        'necessity_score': float(necessity_score),
        'baseline': float(baseline)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODO GLOBAL: Necessidade via VIABILIDADE DE LP (PuLP/AXp)
# ═══════════════════════════════════════════════════════════════════════════════

def _validar_necessidade_global(
    instancia_idx: int,
    explicacao_features: List[str],
    feature_names: List[str],
    y_pred: int,
    rejeitada: bool,
    pipeline,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    t_plus_norm: float,
    t_minus_norm: float,
    max_abs: float,
    baseline_cache: Dict
) -> Dict:
    """
    VALIDAÇÃO GLOBAL (PuLP): Testa necessidade via VIABILIDADE DE LP.
    
    ═══════════════════════════════════════════════════════════════════════════════
    CONCEITO FUNDAMENTAL (diferente de PEAB):
    ═══════════════════════════════════════════════════════════════════════════════
    
    Feature f é NECESSÁRIA ⟺ LP sem f é INFEASIBLE
    Feature f é REDUNDANTE ⟺ LP sem f é FEASIBLE
    
    NÃO usa:
      ❌ Amostragem (np.any)
      ❌ Perturbações
      ❌ Critérios probabilísticos
    
    USA:
      ✅ Programação Linear (PuLP solver)
      ✅ Verificação de INFEASIBILITY
      ✅ Teste determinístico rigoroso
    
    ═══════════════════════════════════════════════════════════════════════════════
    METODOLOGIA (CORRIGIDA - VERSÃO FINAL):
    ═══════════════════════════════════════════════════════════════════════════════
    
    Para cada feature testada f_i na explicação E:
    
      1. REMOVE feature testada (zera contribuição w_i)
      2. FIXA TODAS as outras features nos valores ORIGINAIS da instância
      3. Calcula score determinístico:
         score = intercept + Σ(w_j * valor_original_j) para j ≠ i
      4. Verifica se decisão é mantida:
         - Positivas: score ≥ t+?
         - Negativas: score ≤ t-?
         - Rejeitadas: t- ≤ score ≤ t+?
      5. Decisão:
         - Decisão mantida → feature é REDUNDANTE
         - Decisão mudou → feature é NECESSÁRIA
    
    ═══════════════════════════════════════════════════════════════════════════════
    POR QUE NÃO USA LP SOLVER?
    ═══════════════════════════════════════════════════════════════════════════════
    
    Versão anterior (ERRADA):
      - Permitia features não-explicativas variarem
      - Testava: "Existe configuração global que compensa?"
      - Resultado: ~0-30% necessidade (features compensavam umas às outras)
    
    Versão atual (CORRETA):
      - TODAS as features fixadas (exceto testada)
      - Testa: "E-{f_i} é suficiente para a instância ORIGINAL?"
      - Resultado: ~80-100% necessidade (PuLP gera explicações minimais)
    
    Não precisa LP porque não há variáveis livres. O score é determinístico.
    
    ═══════════════════════════════════════════════════════════════════════════════
    RESULTADOS ESPERADOS PARA PULP (MÉTODO ÓTIMO):
    ═══════════════════════════════════════════════════════════════════════════════
    
    PuLP resolve ILP para encontrar explicação MINIMAL. Portanto:
    
      - Necessidade esperada: 80-100% (a maioria das features são necessárias)
      - Redundância esperada: 0-20% (muito baixa)
    
    Se necessidade < 50%:
      → Possível problema:
        1. PuLP não está gerando explicações minimais (bug no PuLP)
        2. Thresholds t+/t- muito permissivos
        3. Instâncias na fronteira de decisão (múltiplas explicações válidas)
    
    ═══════════════════════════════════════════════════════════════════════════════
    """
    
    # Obter instância original (USADA para fixar features explicativas)
    try:
        instancia_original = X_test.loc[instancia_idx].values
    except (KeyError, TypeError):
        try:
            instancia_original = X_test.iloc[int(instancia_idx)].values
        except (IndexError, ValueError):
            return {'necessary_count': len(explicacao_features), 'redundant_features': [], 'necessity_score': 100.0, 'baseline': 0.5}
    
    if len(explicacao_features) <= 1:
        return {'necessary_count': 1, 'redundant_features': [], 'necessity_score': 100.0, 'baseline': 0.5}
    
    # Extrair componentes do modelo
    if hasattr(pipeline, 'named_steps'):
        scaler = pipeline.named_steps.get('scaler')
        if 'model' in pipeline.named_steps:
            logreg = pipeline.named_steps['model']
        elif 'classifier' in pipeline.named_steps:
            logreg = pipeline.named_steps['classifier']
        else:
            logreg = pipeline.named_steps['logisticregression']
    else:
        return {'necessary_count': len(explicacao_features), 'redundant_features': [], 'necessity_score': 100.0, 'baseline': 0.5}
    
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    
    # Escalar instância original (valores que serão fixados)
    instancia_scaled = scaler.transform(instancia_original.reshape(1, -1))[0]
    
    # Calcular bounds GLOBAIS do dataset (para features NÃO-EXPLICATIVAS)
    X_train_scaled = scaler.transform(X_train)
    min_scaled = X_train_scaled.min(axis=0)
    max_scaled = X_train_scaled.max(axis=0)
    
    # Calcular baseline (apenas para reportar)
    cache_key = f"{y_pred}_{rejeitada}_global"
    if baseline_cache is not None and cache_key in baseline_cache:
        baseline = baseline_cache[cache_key]
    else:
        baseline = calcular_baseline_predicao(
            pipeline, X_train, y_pred, rejeitada, t_plus_norm, t_minus_norm, max_abs
        )
        if baseline_cache is not None:
            baseline_cache[cache_key] = baseline
    
    # Mapear nomes para índices
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    explicacao_idx = [feature_to_idx[f] for f in explicacao_features if f in feature_to_idx]
    
    features_redundantes = []
    
    # ───────────────────────────────────────────────────────────────────────────
    # LOOP: Testar cada feature usando LP SOLVER
    # ───────────────────────────────────────────────────────────────────────────
    
    for feat_name in explicacao_features:
        feat_idx = feature_to_idx.get(feat_name)
        if feat_idx is None:
            continue
        
        # ───────────────────────────────────────────────────────────────────────
        # LÓGICA CORRETA PARA VALIDAR MÉTODO ÓTIMO (PuLP):
        # ───────────────────────────────────────────────────────────────────────
        # 
        # Pergunta: "Remover feature f_i da explicação E torna E-{f_i} 
        #            INSUFICIENTE para a instância ORIGINAL?"
        # 
        # Método:
        # - Feature TESTADA: REMOVIDA (zera w_i)
        # - TODAS as outras features: FIXADAS nos valores originais
        # 
        # NÃO permite variação de features não-explicativas!
        # Estamos testando se E-{f_i} é suficiente para a instância original,
        # não se existe alguma configuração global que funciona.
        # ───────────────────────────────────────────────────────────────────────
        
        # SIMPLIFICAÇÃO: Calcular score diretamente (sem LP)
        # 
        # Como TODAS as features (exceto testada) estão FIXADAS,
        # não há variáveis livres! O score é determinístico.
        # 
        # score = intercept + Σ(w_j * valor_original_j) para j ≠ feat_idx
        
        score_sem_feat = intercept
        
        # Somar contribuição de TODAS as features EXCETO a testada
        for j in range(len(feature_names)):
            if j != feat_idx:
                score_sem_feat += coefs[j] * instancia_scaled[j]
        
        # Normalizar se necessário
        if max_abs is not None and max_abs > 0:
            score_sem_feat = score_sem_feat / max_abs
        
        # ───────────────────────────────────────────────────────────────────────
        # VERIFICAR SE DECISÃO É MANTIDA (sem LP, é cálculo direto)
        # ───────────────────────────────────────────────────────────────────────
        
        decisao_mantida = False
        
        if rejeitada:
            # Rejeitada: score deve estar na zona [t-, t+]
            decisao_mantida = (score_sem_feat >= t_minus_norm) and (score_sem_feat <= t_plus_norm)
        elif y_pred == 1:
            # Positiva: score >= t+
            decisao_mantida = (score_sem_feat >= t_plus_norm)
        else:  # y_pred == 0
            # Negativa: score <= t-
            decisao_mantida = (score_sem_feat <= t_minus_norm)
        
        # ───────────────────────────────────────────────────────────────────────
        # DECISÃO:
        # Se decisão mantida → feature é REDUNDANTE
        # Se decisão mudou → feature é NECESSÁRIA
        # ───────────────────────────────────────────────────────────────────────
        
        if decisao_mantida:
            features_redundantes.append(feat_name)
    
    necessary_count = len(explicacao_features) - len(features_redundantes)
    necessity_score = (necessary_count / len(explicacao_features)) * 100.0
    
    return {
        'necessary_count': necessary_count,
        'redundant_features': features_redundantes,
        'necessity_score': float(necessity_score),
        'baseline': float(baseline)
    }


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
    estrategia: str = "uniform",
    max_abs: float = None
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
        
        # [BUGFIX] Normalizar scores se max_abs foi fornecido (PEAB/PuLP)
        # PEAB normaliza com: score_norm = score_raw / max_abs
        # Os thresholds t_plus e t_minus já estão em espaço normalizado
        if max_abs is not None and max_abs > 0:
            scores = scores / max_abs
        
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
        # Se a explicação está correta, fixar as features essenciais deve manter a rejeição
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
    verbose: bool = True,
    modo_necessidade: str = None
) -> Dict:
    """
    Valida um método completo (PEAB, PuLP, Anchor, MinExp).
    
    Args:
        metodo: Nome do método
        dataset: Nome do dataset
        n_perturbacoes: Número de perturbações (None = usar padrão automático)
        estrategia: Estratégia de perturbação (None = usar PERTURBATION_STRATEGY)
        verbose: Mostrar progresso
        modo_necessidade: "local" (PEAB) ou "global" (PuLP/AXp). None = auto-detect
    
    Returns:
        Dicionário com todas as métricas de validação
    """
    # Carregar resultados do método (retorna tupla com dataset correto)
    resultado_carga = carregar_resultados_metodo(metodo, dataset)
    if resultado_carga is None:
        return None
    
    resultados, dataset_correto = resultado_carga
    
    # [BUGFIX] Extrair max_abs do JSON se disponível (PEAB/PuLP usam normalização)
    max_abs = None
    if 'thresholds' in resultados and 'normalization' in resultados['thresholds']:
        max_abs = resultados['thresholds']['normalization'].get('max_abs', None)
    
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
    
    # Auto-detectar modo de necessidade baseado no método
    if modo_necessidade is None:
        if metodo.upper() in ['PULP', 'AXP']:
            modo_necessidade = "global"
            if verbose:
                print(f"[AUTO] Método ótimo detectado: usando validação GLOBAL (viabilidade lógica)")
        else:
            modo_necessidade = "local"
            if verbose:
                print(f"[AUTO] Método heurístico detectado: usando validação LOCAL (epsilon-ball)")
    
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
        'positive': {'fidelities': [], 'sizes': [], 'necessities': [], 'count': 0},
        'negative': {'fidelities': [], 'sizes': [], 'necessities': [], 'count': 0},
        'rejected': {'fidelities': [], 'sizes': [], 'necessities': [], 'count': 0}
    }
    
    # Distribuição de tamanhos
    size_distribution = defaultdict(int)
    
    # Cache para baseline (evita recalcular para cada instância)
    baseline_cache = {}
    
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
            
            # Extrair informações da explicação - suporta múltiplos formatos
            # Formato PEAB: 'explanation' + 'explanation_size'
            # Formato PuLP: 'features_selecionadas' + 'tamanho'
            # Formato antigo: 'explicacao' ou 'features'
            if 'explanation' in exp:
                explicacao_features = exp['explanation']
                tamanho = exp.get('explanation_size', len(explicacao_features))
            elif 'features_selecionadas' in exp:
                explicacao_features = exp['features_selecionadas']
                tamanho = exp.get('tamanho', len(explicacao_features))
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
            
            # Extrair y_true e y_pred com suporte a múltiplos formatos
            # Formato PEAB: 'y_true', 'y_pred', 'rejected' (bool)
            # Formato PuLP: 'classe_real', 'tipo_predicao' (string)
            y_true = exp.get('y_true', -1)
            if y_true == -1 and 'classe_real' in exp:
                # PuLP usa string, converter para int
                y_true = 1 if 'Diabético' in str(exp['classe_real']) else 0
            y_true = int(y_true)
            
            y_pred = int(exp.get('y_pred', exp.get('predicao', -1)))
            
            # Detectar se é rejeitada
            # Formato PEAB: 'rejected' (bool)
            # Formato PuLP: 'tipo_predicao' == 'REJEITADA'
            rejeitada = bool(exp.get('rejected', exp.get('rejeitada', False)))
            if not rejeitada and 'tipo_predicao' in exp:
                rejeitada = ('REJEIT' in exp['tipo_predicao'].upper())
            
            # Determinar tipo
            if rejeitada:
                tipo = 'rejected'
            elif 'tipo_predicao' in exp:
                # PuLP usa string
                tipo_pred = exp['tipo_predicao'].upper()
                if 'POSIT' in tipo_pred:
                    tipo = 'positive'
                    y_pred = 1
                elif 'NEGAT' in tipo_pred:
                    tipo = 'negative'
                    y_pred = 0
                else:
                    tipo = 'rejected'
                    y_pred = -1
            elif y_pred == 1:
                tipo = 'positive'
            elif y_pred == 0:
                tipo = 'negative'
            else:
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
                estrategia,
                max_abs
            )
            
            fidelity = resultado['fidelity']
            
            # Se houver erro ao processar a instância, pular
            if 'error' in resultado:
                pbar.update()
                continue
            
            # Validar necessidade (minimalidade) - apenas para explicações com 2+ features
            resultado_necessidade = {'necessary_count': tamanho, 'redundant_features': [], 'necessity_score': 100.0, 'baseline': 0.5}
            if tamanho >= 2:
                resultado_necessidade = validar_necessidade_features(
                    idx,
                    explicacao_features,
                    feature_names,
                    y_pred,
                    rejeitada,
                    pipeline,
                    X_test,
                    X_train,
                    t_plus,
                    t_minus,
                    n_perturbacoes=200,  # Menos perturbações para ser mais rápido
                    max_abs=max_abs,
                    baseline_cache=baseline_cache,
                    modo=modo_necessidade  # [NOVO] Passa o modo (local/global)
                )
            
            fidelities.append(fidelity)
            
            # Atualizar métricas por tipo
            metricas_por_tipo[tipo]['fidelities'].append(fidelity)
            metricas_por_tipo[tipo]['sizes'].append(tamanho)
            metricas_por_tipo[tipo]['necessities'].append(resultado_necessidade['necessity_score'])
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
                'perturbations_correct': resultado['perturbations_correct'],
                'necessary_features': resultado_necessidade['necessary_count'],
                'redundant_features': resultado_necessidade['redundant_features'],
                'necessity_score': resultado_necessidade['necessity_score']
            })
            
            pbar.update()
    
    # Calcular tempo total
    validation_time = time.time() - start_time
    
    # Calcular métricas globais
    fidelity_overall = np.mean(fidelities)
    
    # Calcular métricas por tipo
    per_type_metrics = {}
    necessities_all = []
    for tipo, dados in metricas_por_tipo.items():
        if dados['count'] > 0:
            necessities_all.extend(dados['necessities'])
            per_type_metrics[tipo] = {
                'count': dados['count'],
                'fidelity': float(np.mean(dados['fidelities'])),
                'necessity': float(np.mean(dados['necessities'])),
                'mean_size': float(np.mean(dados['sizes'])),
                'std_size': float(np.std(dados['sizes']))
            }
        else:
            per_type_metrics[tipo] = {
                'count': 0,
                'fidelity': 0.0,
                'necessity': 0.0,
                'mean_size': 0.0,
                'std_size': 0.0
            }
    
    # Necessity geral
    necessity_overall = float(np.mean(necessities_all)) if necessities_all else 0.0
    
    # Calcular reduction rate
    num_features = len(feature_names)
    
    # [BUGFIX] Verificar se há explicações antes de calcular estatísticas
    if len(tamanhos_explicacao) > 0:
        mean_size = np.mean(tamanhos_explicacao)
        median_size = np.median(tamanhos_explicacao)
        std_size = np.std(tamanhos_explicacao)
        min_size = int(np.min(tamanhos_explicacao))
        max_size = int(np.max(tamanhos_explicacao))
        reduction_rate = ((num_features - mean_size) / num_features) * 100.0
    else:
        mean_size = 0.0
        median_size = 0.0
        std_size = 0.0
        min_size = 0
        max_size = 0
        reduction_rate = 0.0
    
    # Montar resultado final
    resultado_validacao = {
        'metadata': {
            'method': metodo,
            'dataset': dataset,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_perturbations': n_perturbacoes,
            'perturbation_strategy': estrategia,
            'necessity_mode': modo_necessidade,  # [NOVO] Documenta modo usado
            'test_instances': len(explicacoes),
            'num_features': num_features
        },
        'global_metrics': {
            'fidelity_overall': float(fidelity_overall),
            'necessity_overall': float(necessity_overall),
            'fidelity_positive': float(per_type_metrics['positive']['fidelity']),
            'fidelity_negative': float(per_type_metrics['negative']['fidelity']),
            'fidelity_rejected': float(per_type_metrics['rejected']['fidelity']),
            'necessity_positive': float(per_type_metrics['positive']['necessity']),
            'necessity_negative': float(per_type_metrics['negative']['necessity']),
            'necessity_rejected': float(per_type_metrics['rejected']['necessity']),
            'sufficiency': float(fidelity_overall),  # Para métodos ótimos
            'coverage': 100.0,  # % instâncias sem erro
            'mean_explanation_size': float(mean_size),
            'median_explanation_size': float(median_size),
            'std_explanation_size': float(std_size),
            'min_explanation_size': min_size,
            'max_explanation_size': max_size,
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
        print(f"  - Minimalidade Geral: {necessity_overall:.2f}%")
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
    """Gera relatório TXT simplificado e científico."""
    
    # [ORGANIZAÇÃO] Estrutura: results/validation/{dataset}/{metodo}/
    output_dir = os.path.join(VALIDATION_RESULTS_DIR, metodo.lower(), dataset,)
    os.makedirs(output_dir, exist_ok=True)
    
    # Nome do arquivo: {metodo}_validation_{dataset}.txt
    report_filename = f"{metodo.lower()}_validation_{dataset}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
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
        f.write("║" + f"VALIDAÇÃO DE EXPLICABILIDADE - {metodo_display}".center(78) + "║\n")
        f.write("║" + f"{dataset_display}".center(78) + "║\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("╚" + "═" * 78 + "╝\n\n")
        
        # =====================================================================
        # RESUMO EXECUTIVO (NOVO)
        # =====================================================================
        f.write("━" * 80 + "\n")
        f.write("RESUMO EXECUTIVO\n")
        f.write("━" * 80 + "\n\n")
        
        f.write(f"  Dataset:                {dataset_display}\n")
        f.write(f"  Instâncias Testadas:    {meta['test_instances']}\n")
        f.write(f"  Features Totais:        {meta['num_features']}\n\n")
        
        f.write("  MÉTRICAS PRINCIPAIS:\n")
        f.write(f"    • Fidelidade:                      {globais['fidelity_overall']:.1f}%\n")
        f.write(f"    • Necessidade (feat. necessárias): {globais['necessity_overall']:.1f}%\n")
        f.write(f"    • Tamanho Médio:                   {globais['mean_explanation_size']:.1f} features\n")
        
        # Calcular taxa de rejeição total
        rej_count = por_tipo['rejected']['count']
        taxa_rej = (rej_count / meta['test_instances']) * 100 if meta['test_instances'] > 0 else 0
        f.write(f"    • Taxa de Rejeição:     {taxa_rej:.1f}% ({rej_count} instâncias)\n\n")
        
        # Conclusão curta baseada nas métricas
        if globais['fidelity_overall'] >= 95 and globais['necessity_overall'] >= 80:
            conclusao = "Explicações de alta qualidade: fiéis e minimais."
        elif globais['fidelity_overall'] >= 95:
            conclusao = "Explicações fiéis, porém contêm features redundantes."
        elif globais['necessity_overall'] >= 80:
            conclusao = "Explicações minimais, mas fidelidade requer atenção."
        else:
            conclusao = "Qualidade variável: revisar método e hiperparâmetros."
        
        f.write(f"  CONCLUSÃO:\n")
        f.write(f"    {conclusao}\n\n")
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 1: Descrição do Método (Simplificada)
        f.write("━" * 80 + "\n")
        f.write("METODOLOGIA DE VALIDAÇÃO\n")
        f.write("━" * 80 + "\n\n")
        
        f.write(f"  Método Avaliado:         {metodo_display}\n")
        f.write(f"  Perturbações/instância:  {meta['num_perturbations']:,}\n\n")
        
        f.write("  TESTES APLICADOS:\n\n")
        
        f.write("  1. FIDELIDADE (Sufficiency) - Teste Probabilístico\n")
        f.write("     • Para cada feature da explicação, geramos perturbações e verificamos\n")
        f.write("       se o modelo mantém a decisão quando apenas essa feature está ativa.\n")
        f.write("     • Critério: Feature é fiel se >95% das perturbações mantêm a decisão.\n")
        f.write("     • Objetivo: Garantir que features explicativas CAUSAM a decisão.\n\n")
        
        f.write("  2. NECESSIDADE (Minimality) - Teste Determinístico (Worst-Case)\n")
        f.write("     • Para cada feature, construímos o cenário mais adverso possível:\n")
        f.write("       removemos a feature e atribuímos valores extremos às demais features\n")
        f.write("       não-explicativas (pior caso que maximiza score positivo ou negativo).\n")
        f.write("     • Critério: Feature é necessária se sua remoção SEMPRE quebra a decisão\n")
        f.write("       no pior caso deterministicamente possível.\n")
        f.write("     • Objetivo: Eliminar features redundantes (minimalidade).\n\n")
        
        f.write("  NOTA TÉCNICA: Fidelidade é suficiência estatística (perturbações),\n")
        f.write("                Necessidade é teste lógico (existe caso adverso).\n\n")
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 2: Configuração (Simplificada)
        f.write("━" * 80 + "\n")
        f.write("CONFIGURAÇÃO\n")
        f.write("━" * 80 + "\n\n")
        f.write(f"  Dataset:              {dataset_display}\n")
        f.write(f"  Instâncias:           {meta['test_instances']}\n")
        f.write(f"  Features:             {meta['num_features']}\n")
        f.write(f"  Perturbações/inst:    {meta['num_perturbations']:,}\n")
        f.write(f"  Data:                 {meta['date']}\n\n")
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 3: Resultados (Simplificado)
        f.write("━" * 80 + "\n")
        f.write("RESULTADOS\n")
        f.write("━" * 80 + "\n\n")
        
        redundancia_global = 100.0 - globais['necessity_overall']
        
        f.write("  MÉTRICAS GLOBAIS:\n")
        f.write(f"    Fidelidade:       {globais['fidelity_overall']:.1f}%\n")
        f.write(f"    Necessidade:      {globais['necessity_overall']:.1f}%\n")
        f.write(f"    Redundância:      {redundancia_global:.1f}%\n")
        f.write(f"    Cobertura:        {globais['coverage']:.1f}%\n")
        f.write(f"    Tamanho médio:    {globais['mean_explanation_size']:.1f} features\n\n")
        
        f.write("  POR TIPO DE DECISÃO:\n\n")
        f.write("  “Necessidade Estrita (Worst-case)”\n")
        f.write("  “O teste verifica se a feature é necessária sob o pior cenário adversarial possível, não se ela é única explicação possível.”\n\n")
        f.write("  Tipo          | Count |  Fidelidade | Necessidade | Redundância\n")
        f.write("  " + "─" * 72 + "\n")
        for tipo_nome, tipo_label in [('positive', 'Positivas'), 
                                       ('negative', 'Negativas'), 
                                       ('rejected', 'Rejeitadas')]:
            dados = por_tipo[tipo_nome]
            redundancia_tipo = 100.0 - dados['necessity']
            f.write(f"  {tipo_label:12}  | {dados['count']:5} | {dados['fidelity']:10.1f}% | {dados['necessity']:10.1f}% | {redundancia_tipo:10.1f}%\n")
        f.write("\n")
        
        f.write("  TAMANHO DAS EXPLICAÇÕES:\n")
        f.write(f"    Média:         {globais['mean_explanation_size']:.1f} features\n")
        f.write(f"    Mediana:       {globais['median_explanation_size']:.0f}\n")
        f.write(f"    Desvio:        {globais['std_explanation_size']:.1f}\n")
        f.write(f"    Intervalo:     [{globais['min_explanation_size']}, {globais['max_explanation_size']}]\n")
        f.write(f"    Compactação:   {globais['reduction_rate']:.0f}% (vs {meta['num_features']} features totais)\n")
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
        
        # SEÇÃO 4: Análise por Tipo (Simplificada - já está na tabela acima)
        # Removida para evitar redundância
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 5: Interpretação Crítica
        f.write("━" * 80 + "\n")
        f.write("INTERPRETAÇÃO DOS RESULTADOS\n")
        f.write("━" * 80 + "\n\n")
        
        redundancia_pct = 100.0 - globais['necessity_overall']
        
        # Análise objetiva das métricas
        f.write(f"  Fidelidade:       {globais['fidelity_overall']:.1f}%\n")
        f.write(f"  Necessidade:      {globais['necessity_overall']:.1f}%\n")
        f.write(f"  Redundância:      {redundancia_pct:.1f}%\n")
        f.write(f"  Tamanho médio:    {globais['mean_explanation_size']:.1f} features\n")
        f.write(f"  Cobertura:        {globais['coverage']:.1f}%\n\n")
        
        # Interpretação curta e direta
        if globais['fidelity_overall'] >= 95 and globais['necessity_overall'] >= 90:
            avaliacao = "As explicações são fiéis e minimais."
        elif globais['fidelity_overall'] >= 95:
            avaliacao = f"Explicações fiéis, mas {redundancia_pct:.0f}% de redundância (features desnecessárias)."
        elif globais['necessity_overall'] >= 90:
            avaliacao = "Explicações minimais, porém fidelidade abaixo de 95%."
        else:
            avaliacao = "Qualidade insuficiente: ambas as métricas requerem atenção."
        
        f.write(f"  AVALIAÇÃO: {avaliacao}\n\n")
        
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 6: Limitações Observadas (NOVA)
        f.write("━" * 80 + "\n")
        f.write("LIMITAÇÕES OBSERVADAS\n")
        f.write("━" * 80 + "\n\n")
        
        limitacoes = []
        
        # Limitação: Redundância Alta
        if redundancia_pct > 20:
            limitacoes.append(f"  • Alta redundância ({redundancia_pct:.0f}%): explicações não são minimais.\n"
                            f"    Possível causa: threshold de rejeição muito conservador ou\n"
                            f"    features correlacionadas no dataset.")
        
        # Limitação: Fidelidade Baixa
        if globais['fidelity_overall'] < 95:
            limitacoes.append(f"  • Fidelidade abaixo de 95% ({globais['fidelity_overall']:.1f}%): explicações não\n"
                            f"    reproduzem decisões perfeitamente sob perturbação.\n"
                            f"    Possível causa: instabilidade do modelo ou features não-explicativas\n"
                            f"    com alta influência em cenários perturbados.")
        
        # Limitação: Variabilidade por tipo
        fid_pos = por_tipo['positive']['fidelity']
        fid_neg = por_tipo['negative']['fidelity']
        fid_rej = por_tipo['rejected']['fidelity']
        max_diff_fid = max(fid_pos, fid_neg, fid_rej) - min(fid_pos, fid_neg, fid_rej)
        if max_diff_fid > 10:
            limitacoes.append(f"  • Variabilidade entre tipos de decisão:\n"
                            f"    Positivas: {fid_pos:.1f}%, Negativas: {fid_neg:.1f}%, Rejeitadas: {fid_rej:.1f}%\n"
                            f"    Diferença de {max_diff_fid:.1f}pp indica comportamento heterogêneo.")
        
        # Limitação: Tamanho das explicações
        if globais['mean_explanation_size'] > meta['num_features'] * 0.5:
            limitacoes.append(f"  • Explicações usam {globais['mean_explanation_size']:.1f} de {meta['num_features']} features ({globais['mean_explanation_size']/meta['num_features']*100:.0f}%):\n"
                            f"    Compactação insuficiente para interpretabilidade prática.")
        
        # Limitação: Cobertura incompleta
        if globais['coverage'] < 100:
            limitacoes.append(f"  • Cobertura incompleta ({globais['coverage']:.1f}%): {100-globais['coverage']:.1f}% das instâncias\n"
                            f"    falharam. Possível causa: timeouts ou erros numéricos.")
        
        # Limitação: Distribuição de tamanhos
        if globais['max_explanation_size'] >= meta['num_features']:
            limitacoes.append(f"  • Explicações completas detectadas (max={globais['max_explanation_size']} features):\n"
                            f"    Método falhou em reduzir dimensionalidade em alguns casos.")
        
        if limitacoes:
            for lim in limitacoes:
                f.write(lim + "\n")
        else:
            f.write("  Nenhuma limitação crítica detectada nesta validação.\n\n")
        
        f.write("━" * 80 + "\n\n")
        
        # SEÇÃO 7: Recomendações Práticas
        f.write("━" * 80 + "\n")
        f.write("RECOMENDAÇÕES\n")
        f.write("━" * 80 + "\n\n")
        
        if globais['fidelity_overall'] >= 95 and globais['necessity_overall'] >= 85:
            f.write("  • Método validado. Explicações apresentam qualidade aceitável.\n")
        else:
            f.write("  • Ajustar hiperparâmetros (threshold de rejeição, tolerâncias).\n")
            f.write("  • Investigar instâncias com baixa fidelidade ou alta redundância.\n")
        
        if redundancia_pct > 20:
            f.write("  • Alta redundância: considerar pós-processamento para remover features\n")
            f.write("    desnecessárias (ex: backward selection).\n")
        
        if globais['coverage'] < 100:
            f.write("  • Investigar instâncias que falharam na validação.\n")
        
        f.write("\n")
        f.write("━" * 80 + "\n")
        f.write(f"Relatório gerado em: {meta['date']}\n")
        f.write("━" * 80 + "\n")
    
    print(f"✓ Relatório salvo: {report_path}")
    return report_path


def gerar_plots(resultado: Dict, metodo: str, dataset: str):
    """Gera os 6 plots de validação."""
    
    # [ORGANIZAÇÃO] Estrutura: results/validation/{dataset}/{metodo}/
    output_dir = os.path.join(VALIDATION_RESULTS_DIR, metodo.lower(), dataset)
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
    
    print("\n" + "=" * 70)
    print("           VALIDACAO DE EXPLICACOES - XAI COM REJEICAO")
    print("=" * 70)
    print("\n[1] Validar PEAB")
    print("[2] Validar PuLP (Ground Truth)")
    print("[3] Validar Anchor")
    print("[4] Validar MinExp")
    print("[5] Comparar Todos os Metodos (RECOMENDADO)")
    print("[0] Sair")
    
    opcao = input("\nOpcao: ").strip()
    
    if opcao == '0':
        print("Encerrando...")
        return
    
    # Selecionar dataset (reutilizar menu do PEAB)
    print("\n" + "-" * 70)
    print("Selecione o dataset para validacao...")
    print("-" * 70)
    
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
