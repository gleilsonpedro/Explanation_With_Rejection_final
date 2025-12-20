#este é o peab vetorizado_ funciona mais rapido- esta em teste
import os
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Tuple, Dict, Any, Set

# Imports do seu projeto
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
from utils.results_handler import update_method_results
from utils.progress_bar import ProgressBar

#==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
#==============================================================================
RANDOM_STATE: int = 42

# Configurações específicas de MNIST
MNIST_CONFIG = {
    'feature_mode': 'raw',           
    'digit_pair': (3, 8),            
    'top_k_features': None,          
    'test_size': 0.3,                
    'rejection_cost': 0.10,          
    'subsample_size': 1.0
}

DATASET_CONFIG = {
    "mnist":                MNIST_CONFIG,
    "breast_cancer":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "pima_indians_diabetes":{'test_size': 0.3, 'rejection_cost': 0.24},
    "vertebral_column":     {'test_size': 0.3, 'rejection_cost': 0.24},
    "sonar":                {'test_size': 0.3, 'rejection_cost': 0.24},
    "spambase":             {'test_size': 0.3, 'rejection_cost': 0.24},
    "banknote":             {'test_size': 0.3, 'rejection_cost': 0.24},
    "heart_disease":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "creditcard":           {'subsample_size': 0.5, 'test_size': 0.3, 'rejection_cost': 0.05},
    "covertype":            {'subsample_size': 0.3, 'test_size': 0.3, 'rejection_cost': 0.10},
    "gas_sensor":           {'test_size': 0.3, 'rejection_cost': 0.24},
    "newsgroups":           {'subsample_size': 0.1, 'test_size': 0.3, 'rejection_cost': 0.24},
    "rcv1":                 {'subsample_size': 0.05, 'test_size': 0.3, 'rejection_cost': 0.24},
}
OUTPUT_BASE_DIR: str = 'results/report/peab'
HIPERPARAMETROS_FILE: str = 'json/hiperparametros.json'
DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2', 'C': 0.01, 'solver': 'liblinear', 'max_iter': 1000
}

#==============================================================================
# UTILITÁRIOS MATEMÁTICOS OTIMIZADOS
#==============================================================================

def _get_lr(modelo: Pipeline):
    if 'model' in modelo.named_steps: return modelo.named_steps['model']
    if 'modelo' in modelo.named_steps: return modelo.named_steps['modelo']
    raise KeyError("Nenhum passo de regressão logística encontrado.")

def carregar_hiperparametros(caminho_arquivo: str = HIPERPARAMETROS_FILE) -> dict:
    try:
        with open(caminho_arquivo, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def precompute_linear_contributions(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    PRÉ-CÁLCULO VETORIAL (O SEGREDO DA VELOCIDADE):
    Em vez de rodar o modelo mil vezes, calculamos:
    1. O score base se TODAS as features fossem para o pior caso possível (min e max).
    2. O vetor de ganho/perda (delta) que cada feature traz ao ser fixada.
    
    Retorna:
      - intercept: viés do modelo
      - base_min_score: score mínimo teórico (todas features jogando contra para baixo)
      - base_max_score: score máximo teórico (todas features jogando contra para cima)
      - gains_from_min: vetor de quanto cada feature, se fixada, levanta o score do mínimo
      - losses_from_max: vetor de quanto cada feature, se fixada, abaixa o score do máximo
    """
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    
    # 1. Obter valores da instância no espaço escalado
    instance_df_ordered = instance_df[X_train.columns]
    if hasattr(scaler, 'feature_range'):
        vals_s = scaler.transform(instance_df_ordered)[0]
    else:
        vals_s = instance_df_ordered.values[0]

    # 2. Definir limites teóricos (MinMaxScaler [0,1])
    # Se coef > 0, o menor valor (0) minimiza e maior (1) maximiza.
    # Se coef < 0, o maior valor (1) minimiza e menor (0) maximiza.
    X_min_contribution = np.where(coefs > 0, 0.0, 1.0)
    X_max_contribution = np.where(coefs > 0, 1.0, 0.0)

    # 3. Calcular Scores Base (Piores Cenários Globais)
    # base_min: score se o adversário conseguir jogar TUDO para baixo
    # base_max: score se o adversário conseguir jogar TUDO para cima
    base_min_score = intercept + np.dot(coefs, X_min_contribution)
    base_max_score = intercept + np.dot(coefs, X_max_contribution)

    # 4. Calcular Vetores de Delta (Contribuição ao fixar)
    # Se eu fixo a feature 'i', eu ganho a diferença entre o valor real e o pior valor.
    # gain_from_min[i] = w_i * (x_i - x_worst_min) -> Sempre >= 0
    # loss_from_max[i] = w_i * (x_worst_max - x_i) -> Sempre >= 0 (na verdade reduz o max)
    
    current_contribution = vals_s * coefs
    min_contribution = X_min_contribution * coefs
    max_contribution = X_max_contribution * coefs
    
    gains_from_min = current_contribution - min_contribution
    losses_from_max = max_contribution - current_contribution

    return intercept, base_min_score, base_max_score, gains_from_min, losses_from_max

def check_validity_fast(fixed_indices: Set[int], 
                        base_min: float, base_max: float, 
                        gains_min: np.ndarray, losses_max: np.ndarray,
                        t_plus: float, t_minus: float, 
                        mode: str) -> bool:
    """
    VERIFICAÇÃO OTIMIZADA (O(K) onde K é tamanho da explicação):
    Apenas soma os deltas das features fixadas e compara com thresholds.
    Sem alocação de memória, sem dot product.
    
    mode: 'positive' (score >= t+), 'negative' (score <= t-), 'rejected' (score entre t- e t+)
    """
    # Converter set para lista para indexação rápida do numpy
    if not fixed_indices:
        current_gain = 0.0
        current_loss = 0.0
    else:
        idx = list(fixed_indices)
        current_gain = np.sum(gains_min[idx])
        current_loss = np.sum(losses_max[idx])
    
    # O Pior Score Mínimo (Adversário empurra pra baixo, mas features fixas seguram)
    worst_case_min_score = base_min + current_gain
    
    # O Pior Score Máximo (Adversário empurra pra cima, mas features fixas seguram)
    worst_case_max_score = base_max - current_loss
    
    EPSILON = 1e-5

    if mode == 'positive':
        # Deve garantir que mesmo no pior caso (min), ficamos acima de t+
        return worst_case_min_score >= t_plus - EPSILON

    elif mode == 'negative':
        # Deve garantir que mesmo no pior caso (max), ficamos abaixo de t-
        return worst_case_max_score <= t_minus + EPSILON

    elif mode == 'rejected':
        # Deve garantir que não sai por baixo (min >= t-) E não sai por cima (max <= t+)
        return (worst_case_min_score >= t_minus - EPSILON) and \
               (worst_case_max_score <= t_plus + EPSILON)
    
    return False

#==============================================================================
# FASES DO ALGORITMO (AGORA COM LÓGICA VETORIAL)
#==============================================================================

def fase_1_reforco_fast(feature_names: pd.Index, 
                        base_min: float, base_max: float, 
                        gains_min: np.ndarray, losses_max: np.ndarray,
                        t_plus: float, t_minus: float, 
                        mode: str, 
                        sorting_metric: np.ndarray) -> Tuple[List[str], int, Set[int]]:
    """
    Fase 1 Otimizada: Constrói explicação robusta adicionando features (Guloso).
    """
    expl_indices = set()
    expl_str = []
    adicoes = 0
    
    # Ordenação heurística (mantendo a lógica do PEAB original)
    indices_ordenados = np.argsort(-sorting_metric) # Descrescente
    
    # Loop guloso
    for idx in indices_ordenados:
        # 1. Verifica se já satisfaz (Check Instantâneo)
        if check_validity_fast(expl_indices, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode):
            break
            
        # 2. Adiciona a melhor feature
        if idx not in expl_indices:
            expl_indices.add(idx)
            # Para o log/output (valor não importa aqui pois usamos índices na validação)
            expl_str.append(feature_names[idx]) 
            adicoes += 1
            
    # Verifica validade final (caso acabe as features)
    # Se ainda não válido, é porque a instância é impossível de explicar (raro) ou limites
    return expl_str, adicoes, expl_indices

def fase_2_minimizacao_fast(feature_names: pd.Index, 
                            expl_indices_inicial: Set[int],
                            base_min: float, base_max: float, 
                            gains_min: np.ndarray, losses_max: np.ndarray,
                            t_plus: float, t_minus: float, 
                            mode: str, 
                            sorting_metric: np.ndarray,
                            instance_df: pd.DataFrame) -> Tuple[List[str], int]:
    """
    Fase 2 Otimizada: Remove redundâncias (Backward Elimination).
    """
    expl_indices = expl_indices_inicial.copy()
    remocoes = 0
    
    # Identifica features atuais para tentar remover
    # Ordena para remover primeiro as de MENOR impacto (delta menor) - Heurística comum em pruning
    # Ou maior impacto? O PEAB original usava reverse=False (menor impacto primeiro)
    features_presentes = list(expl_indices)
    features_presentes.sort(key=lambda i: sorting_metric[i], reverse=False)
    
    for idx in features_presentes:
        if len(expl_indices) <= 1: break # Evitar conjunto vazio se desejado
        
        # Tenta remover
        expl_indices.remove(idx)
        
        # Validação Instantânea
        is_valid = check_validity_fast(expl_indices, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode)
        
        if is_valid:
            remocoes += 1
        else:
            # Se ficou inválido, bota de volta (é essencial)
            expl_indices.add(idx)
            
    # Reconstrói strings finais
    expl_final_str = []
    for idx in expl_indices:
        val = instance_df.iloc[0, idx] # Pega valor original só para formatação
        expl_final_str.append(f"{feature_names[idx]} = {val:.4f}")
        
    return expl_final_str, remocoes

def gerar_explicacao_instancia_fast(instancia_df: pd.DataFrame, modelo: Pipeline, X_train: pd.DataFrame, 
                                    t_plus: float, t_minus: float) -> Tuple[List[str], List[str], int, int]:
    
    # 1. Setup do ambiente de validação rápida (O(N))
    # Calcula todos os vetores necessários UMA VEZ
    intercept, base_min, base_max, gains_min, losses_max = precompute_linear_contributions(modelo, instancia_df, X_train)
    
    # Score real para decidir o caso
    score_raw = modelo.decision_function(instancia_df)[0]
    
    adicoes_total = 0
    remocoes_total = 0
    expl_final_str = []
    
    feature_names = X_train.columns

    # Lógica de decisão
    if score_raw >= t_plus:
        # CASO POSITIVO
        mode = 'positive'
        # Heurística: features que mais ajudam a subir (gains_min) são as mais importantes para manter
        sorting_metric = gains_min 
        
        _, ad, indices_robustos = fase_1_reforco_fast(feature_names, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, sorting_metric)
        expl_final_str, rm = fase_2_minimizacao_fast(feature_names, indices_robustos, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, sorting_metric, instancia_df)
        adicoes_total = ad
        remocoes_total = rm

    elif score_raw <= t_minus:
        # CASO NEGATIVO
        mode = 'negative'
        # Heurística: features que mais ajudam a descer (losses_max) são as importantes
        sorting_metric = losses_max
        
        _, ad, indices_robustos = fase_1_reforco_fast(feature_names, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, sorting_metric)
        expl_final_str, rm = fase_2_minimizacao_fast(feature_names, indices_robustos, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, sorting_metric, instancia_df)
        adicoes_total = ad
        remocoes_total = rm
        
    else:
        # CASO REJEITADA (O mais complexo)
        mode = 'rejected'
        # Precisamos satisfazer DOIS lados. O PEAB original tenta duas estratégias e pega a melhor.
        
        # Estratégia A: Priorizar features que evitam descer (seguram no min)
        _, ad1, ind1 = fase_1_reforco_fast(feature_names, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, gains_min)
        expl1, rm1 = fase_2_minimizacao_fast(feature_names, ind1, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, gains_min, instancia_df)
        
        # Estratégia B: Priorizar features que evitam subir (seguram no max)
        _, ad2, ind2 = fase_1_reforco_fast(feature_names, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, losses_max)
        expl2, rm2 = fase_2_minimizacao_fast(feature_names, ind2, base_min, base_max, gains_min, losses_max, t_plus, t_minus, mode, losses_max, instancia_df)
        
        # Escolhe a menor explicação
        if len(expl1) <= len(expl2):
            expl_final_str = expl1
            adicoes_total = ad1
            remocoes_total = rm1
        else:
            expl_final_str = expl2
            adicoes_total = ad2
            remocoes_total = rm2

    return [f.split(' = ')[0] for f in expl_final_str], [], adicoes_total, remocoes_total

#==============================================================================
# FUNÇÕES DE SUPORTE (TREINO, CONFIGURAÇÃO, RELATÓRIO)
#==============================================================================

def configurar_experimento(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], float, float]:
    if dataset_name == 'mnist':
        from data import datasets as ds_module
        cfg = DATASET_CONFIG.get(dataset_name, {})
        ds_module.set_mnist_options(cfg.get('feature_mode', 'raw'), cfg.get('digit_pair', None))
    
    X, y, nomes_classes = carregar_dataset(dataset_name)
    cfg = DATASET_CONFIG.get(dataset_name, {'test_size': 0.3, 'rejection_cost': 0.24})

    if 'subsample_size' in cfg and cfg['subsample_size']:
        frac = cfg['subsample_size']
        if frac < 1.0:
            idx = np.arange(len(y))
            sample_idx, _ = train_test_split(idx, test_size=(1 - frac), random_state=RANDOM_STATE, stratify=y)
            X = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]
            y = y.iloc[sample_idx] if isinstance(y, pd.Series) else y[sample_idx]

    return X, y, nomes_classes, cfg['rejection_cost'], cfg['test_size']

def treinar_e_avaliar_modelo(X_train: pd.DataFrame, y_train: pd.Series, rejection_cost: float, logreg_params: Dict[str, Any], dataset_name: str = "UNKNOWN", val_size: float = 0.2) -> Tuple[Pipeline, float, float, Dict[str, Any]]:
    # Split Treino/Validação
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=RANDOM_STATE, stratify=y_train)
    
    # Treino
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params))])
    pipeline.fit(X_train_sub, y_train_sub)

    # Grid Search Thresholds (Validacao)
    decision_scores_raw = pipeline.decision_function(X_val)
    min_raw, max_raw = float(decision_scores_raw.min()), float(decision_scores_raw.max())
    
    scores_neg = decision_scores_raw[decision_scores_raw < 0]
    scores_pos = decision_scores_raw[decision_scores_raw > 0]
    
    t_minus_grid = np.linspace(scores_neg.min(), -0.01, 50) if len(scores_neg) > 0 else np.linspace(-1.0, -0.01, 50)
    t_plus_grid = np.linspace(0.01, scores_pos.max(), 50) if len(scores_pos) > 0 else np.linspace(0.01, 1.0, 50)
    
    best_risk, best_t_plus, best_t_minus = float('inf'), 0.1, -0.1
    for tm in t_minus_grid:
        for tp in t_plus_grid:
            if not (tm < 0 < tp): continue
            y_pred = np.full(y_val.shape, -1)
            acc_mask = (decision_scores_raw >= tp) | (decision_scores_raw <= tm)
            y_pred[decision_scores_raw >= tp] = 1
            y_pred[decision_scores_raw <= tm] = 0
            err = np.mean(y_pred[acc_mask] != y_val[acc_mask]) if np.any(acc_mask) else 0.0
            rej = 1.0 - np.mean(acc_mask)
            risk = err + rejection_cost * rej
            if risk < best_risk: best_risk, best_t_plus, best_t_minus = risk, tp, tm
            
    # Retreino Final
    pipeline.fit(X_train, y_train)
    coefs = pipeline.named_steps['model'].coef_[0]
    
    # Calcular max_abs para normalização (compatibilidade com PuLP)
    decision_scores_train = pipeline.decision_function(X_train)
    max_abs = float(np.abs(decision_scores_train).max()) if len(decision_scores_train) > 0 else 1.0
    
    model_params = {
        'coefs': {name: float(w) for name, w in zip(list(X_train.columns), coefs)},
        'intercepto': float(pipeline.named_steps['model'].intercept_[0]),
        'scaler_params': {'min': pipeline.named_steps['scaler'].min_, 'scale': pipeline.named_steps['scaler'].scale_},
        'norm_params': {'max_abs': max_abs}  # Adicionar para compatibilidade com PuLP
    }
    return pipeline, float(best_t_plus), float(best_t_minus), model_params

def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n[INFO] Executando PEAB (OTIMIZADO) para: {dataset_name.upper()}")
    todos_hiperparametros = carregar_hiperparametros()
    X_full, y_full, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)

    params = DEFAULT_LOGREG_PARAMS.copy()
    if dataset_name in todos_hiperparametros and 'params' in todos_hiperparametros[dataset_name]:
        params.update(todos_hiperparametros[dataset_name]['params'])

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=RANDOM_STATE, stratify=y_full)
    
    # Feature Selection (Opcional, configurado em DATASET_CONFIG)
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0 and top_k < X_train.shape[1]:
        print(f"[FEAT_SEL] Selecionando Top-{top_k} features...")
        selector = SelectKBest(score_func=f_classif, k=top_k).fit(X_train, y_train)
        cols = X_train.columns[selector.get_support()]
        X_train, X_test = X_train[cols], X_test[cols]

    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params, dataset_name)
    
    print(f"THRESHOLDS: T+={t_plus:.4f}, T-={t_minus:.4f}")

    # Loop de Explicação OTIMIZADO
    print(f"[INFO] Explicando {len(X_test)} instâncias...")
    start_total = time.perf_counter()
    resultados = []
    
    # Pré-cálculo das predições (Vectorized no sklearn)
    scores = modelo.decision_function(X_test)
    preds = np.full(len(X_test), 2) # 2 = Rejected
    preds[scores >= t_plus] = 1
    preds[scores <= t_minus] = 0
    
    with ProgressBar(total=len(X_test)) as pbar:
        for i in range(len(X_test)):
            start_inst = time.perf_counter()
            inst = X_test.iloc[[i]]
            
            # CHAMADA DA VERSÃO FAST
            expl, _, _, _ = gerar_explicacao_instancia_fast(inst, modelo, X_train, t_plus, t_minus)
            
            duracao = time.perf_counter() - start_inst
            p_code = preds[i]
            
            # CORREÇÃO: Usar índice original do DataFrame, não sequencial
            original_idx = str(X_test.index[i])
            
            resultados.append({
                'id': original_idx,  # ✅ CORRIGIDO: Usa índice original
                'pred_code': int(p_code),
                'explicacao': sorted(expl), 
                'tamanho_explicacao': len(expl),
                'tempo': duracao,  # Adicionar tempo individual
                'log_detalhado': []
            })
            pbar.update()

    total_time = time.perf_counter() - start_total
    print(f"[INFO] Tempo Total: {total_time:.2f}s | Média: {total_time/len(X_test):.4f}s/inst")

    # =========================================================================
    # SALVAR RESULTADOS COM update_method_results (NECESSÁRIO PARA COMPARAÇÃO)
    # =========================================================================
    mask_rej = (preds == 2)
    y_pred_final = modelo.predict(X_test)
    
    # Preparar dados no formato esperado por update_method_results
    per_instance_data = []
    for res in resultados:
        idx = res['id']
        idx_int = int(X_test.index.get_loc(int(idx)))  # Posição no array
        
        per_instance_data.append({
            'id': idx,
            'y_true': int(y_test.iloc[idx_int]),
            'y_pred': int(y_pred_final[idx_int]),
            'rejected': bool(preds[idx_int] == 2),
            'decision_score': float(scores[idx_int]),
            'explanation': res['explicacao'],
            'explanation_size': res['tamanho_explicacao'],
            'computation_time': res.get('tempo', 0.0)
        })
    
    # Calcular tempos por tipo
    tempos_por_tipo = {'positive': [], 'negative': [], 'rejected': []}
    for i, res in enumerate(resultados):
        if preds[i] == 1:
            tempos_por_tipo['positive'].append(res.get('tempo', 0.0))
        elif preds[i] == 0:
            tempos_por_tipo['negative'].append(res.get('tempo', 0.0))
        else:
            tempos_por_tipo['rejected'].append(res.get('tempo', 0.0))
    
    # Preparar estrutura de resultados completa
    results_data = {
        'config': {
            'dataset_name': dataset_name,
            'test_size': test_size,
            'random_state': RANDOM_STATE,
            'rejection_cost': rejection_cost,
            'subsample_size': cfg.get('subsample_size', None)
        },
        'thresholds': {
            't_plus': float(t_plus),
            't_minus': float(t_minus),
            'rejection_zone_width': float(t_plus - t_minus)
        },
        'performance': {
            'accuracy_without_rejection': float(np.mean(y_pred_final == y_test) * 100),
            'accuracy_with_rejection': float(np.mean(preds[~mask_rej] == y_test.iloc[~mask_rej]) * 100) if np.any(~mask_rej) else 100.0,
            'rejection_rate': float(np.mean(mask_rej) * 100),
            'num_test_instances': len(X_test),
            'num_rejected': int(np.sum(mask_rej)),
            'num_accepted': int(np.sum(~mask_rej))
        },
        'explanation_stats': {
            'positive': {
                'count': int(np.sum(preds == 1)),
                'mean_length': float(np.mean([r['tamanho_explicacao'] for r in resultados if preds[resultados.index(r)] == 1])) if np.sum(preds == 1) > 0 else 0.0,
            },
            'negative': {
                'count': int(np.sum(preds == 0)),
                'mean_length': float(np.mean([r['tamanho_explicacao'] for r in resultados if preds[resultados.index(r)] == 0])) if np.sum(preds == 0) > 0 else 0.0,
            },
            'rejected': {
                'count': int(np.sum(preds == 2)),
                'mean_length': float(np.mean([r['tamanho_explicacao'] for r in resultados if preds[resultados.index(r)] == 2])) if np.sum(preds == 2) > 0 else 0.0,
            }
        },
        'computation_time': {
            'total': float(total_time),
            'mean_per_instance': float(total_time / len(X_test)),
            'positive': float(np.mean(tempos_por_tipo['positive'])) if tempos_por_tipo['positive'] else 0.0,
            'negative': float(np.mean(tempos_por_tipo['negative'])) if tempos_por_tipo['negative'] else 0.0,
            'rejected': float(np.mean(tempos_por_tipo['rejected'])) if tempos_por_tipo['rejected'] else 0.0,
        },
        'model': {
            'params': params,
            'num_features': len(X_train.columns)
        },
        'per_instance': per_instance_data
    }
    
    # Normalizar nome do dataset para MNIST
    dataset_json_key = dataset_name
    if dataset_name == 'mnist':
        digit_pair = cfg.get('digit_pair')
        if digit_pair and len(digit_pair) == 2:
            dataset_json_key = f"mnist_{digit_pair[0]}_vs_{digit_pair[1]}"
    
    # Chamar update_method_results com assinatura correta
    update_method_results(
        method='peab',
        dataset=dataset_json_key,
        results=results_data
    )
    
    print(f"[INFO] Resultados salvos em json/peab/{dataset_json_key}.json")
    
    # Salvar TXT (Básico) também
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    with open(f"{OUTPUT_BASE_DIR}/peab_{dataset_name}.txt", 'w') as f:
        f.write(f"RELATÓRIO PEAB FAST - {dataset_name}\n")
        f.write(f"Tempo Total: {total_time:.4f}s\n")
        f.write(f"Tempo Médio: {total_time/len(X_test):.6f}s\n")
        f.write(f"Acurácia com rejeição: {float(np.mean(preds[~mask_rej] == y_test.iloc[~mask_rej])*100) if np.any(~mask_rej) else 100.0:.2f}%\n")
        f.write(f"Taxa Rejeição: {float(np.mean(mask_rej)*100):.2f}%\n")

if __name__ == '__main__':
    resultado = selecionar_dataset_e_classe()
    if resultado[0] == '__MULTIPLE__':
        for dataset in resultado[4]:
            try: executar_experimento_para_dataset(dataset)
            except Exception as e: print(f"Erro em {dataset}: {e}")
    elif resultado[0]:
        executar_experimento_para_dataset(resultado[0])