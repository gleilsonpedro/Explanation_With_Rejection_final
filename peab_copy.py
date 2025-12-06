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
    'rejection_cost': 0.24,          
    'subsample_size': 1.0 
}

DATASET_CONFIG = {
    "mnist":                MNIST_CONFIG,
    "wine":                 {'test_size': 0.3, 'rejection_cost': 0.24},
    "breast_cancer":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "pima_indians_diabetes":{'test_size': 0.3, 'rejection_cost': 0.24},
    "vertebral_column":     {'test_size': 0.3, 'rejection_cost': 0.24},
    "sonar":                {'test_size': 0.3, 'rejection_cost': 0.24},
    "spambase":             {'test_size': 0.3, 'rejection_cost': 0.24},
    "banknote_auth":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "heart_disease":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine_quality":         {'test_size': 0.3, 'rejection_cost': 0.24},
    "creditcard":           {'subsample_size': 0.5, 'test_size': 0.3, 'rejection_cost': 0.24},
    "newsgroups":           {'test_size': 0.3, 'rejection_cost': 0.24, 'top_k_features': 2000}
}
OUTPUT_BASE_DIR: str = 'results/report/peab'
HIPERPARAMETROS_FILE: str = 'json/hiperparametros.json'
DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2', 'C': 0.01, 'solver': 'liblinear', 'max_iter': 1000
}

#==============================================================================
# CONTROLES DE LOG
#==============================================================================
TECHNICAL_LOGS: bool = True
MAX_LOG_FEATURES: int = 200
MAX_LOG_STEPS: int = 60

SYMBOL_LEGEND = [
    "LEGENDA DOS SÍMBOLOS",
    "   δ = Contribuição individual (w_i × x_i)",
    "   Σδ = Soma acumulada (intercepto + Σδ_i)",
    "   ● = Feature mantida (essencial)",
    "   ○ = Feature removida (não essencial)",
    "   ↑ = Feature aumenta score (favorável à classe 1)",
    "   ↓ = Feature diminui score (favorável à classe 0)",
    "   s'= Valor da feature no pior cenário(wrost case)"
]

LOG_TEMPLATES = {
    'processamento_header': "********** PROCESSAMENTO POR INSTÂNCIA  **********\n",
    'classificada_analise': "├── Análise: Score está {posicao}. Buscando o menor conjunto que garante a classificação.",
    'classificada_min_inicio': "├── Iniciando processo de minimização com {num_features} features.",
    'classificada_ordem': "├── Tentativas de desafixação (ordem de maior impacto |δ|): {lista}",
    'classificada_step_sucesso': "├─ ○ {feat} (δ: {delta:+.3f}): s' = {score:.3f} ({cond}) → SUCESSO. DESAFIXADA.",
    'classificada_step_falha': "├─ ● {feat} (δ: {delta:+.3f}): s' = {score:.3f} ({cond}) → FALHA. ESSENCIAL.",
    'rejeitada_analise': "├── Zona de Rejeição: [{t_minus:.4f}, {t_plus:.4f}]",
    'rejeitada_prova_header': "├── Prova de Minimalidade (partindo de conjunto robusto após heurística; verificação bidirecional):",
    'rejeitada_feat_header_sucesso': "├─ ○ {feat} (δ: {delta:+.3f}):",
    'rejeitada_feat_header_falha': "├─ ● {feat} (δ: {delta:+.3f}):",
    'rejeitada_subteste_neg': "│   ├─ Teste vs Lado Negativo: s' = {score:.3f} ({cmp}) {ok}",
    'rejeitada_subteste_pos': "│   └─ Teste vs Lado Positivo: s' = {score:.3f} ({cmp}) {ok}",
    'rejeitada_feat_footer_sucesso': "│   └─> SUCESSO. Feature DESAFIXADA.",
    'rejeitada_feat_footer_falha': "│   └─> FALHA. Feature ESSENCIAL (precisa ser fixada).",
}

DISABLE_REFORCO_CLASSIFICADAS: bool = True
MIN_REJECTION_WIDTH: float = 0.05

def _get_lr(modelo: Pipeline):
    if 'model' in modelo.named_steps: return modelo.named_steps['model']
    if 'modelo' in modelo.named_steps: return modelo.named_steps['modelo']
    raise KeyError("Nenhum passo de regressão logística encontrado.")

def carregar_hiperparametros(caminho_arquivo: str = HIPERPARAMETROS_FILE) -> dict:
    try:
        with open(caminho_arquivo, 'r') as f:
            params = json.load(f)
        print(f"\n[INFO] Arquivo de hiperparâmetros '{caminho_arquivo}' carregado com sucesso.")
        return params
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"\n[AVISO] Arquivo '{caminho_arquivo}' não encontrado ou inválido. Usando padrão.")
        return {}

#==============================================================================
#  LÓGICA FORMAL DE EXPLICAÇÃO (OTIMIZADA E CORRIGIDA)
#==============================================================================

def calculate_deltas(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, premis_class: int) -> np.ndarray:
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    
    instance_df_ordered = instance_df[X_train.columns]
    scaled_instance_vals = scaler.transform(instance_df_ordered)[0]
    
    # [CORREÇÃO B] Usar o feature_range do scaler para definir min/max teóricos
    f_min, f_max = scaler.feature_range
    X_train_scaled_min = np.full_like(coefs, f_min)
    X_train_scaled_max = np.full_like(coefs, f_max)
    
    deltas = np.zeros_like(coefs)
    
    if premis_class == 1:
        pior_valor = np.where(coefs > 0, X_train_scaled_min, X_train_scaled_max)
    else:
        pior_valor = np.where(coefs > 0, X_train_scaled_max, X_train_scaled_min)
        
    deltas = (scaled_instance_vals - pior_valor) * coefs
    return deltas

def one_explanation_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, premis_class: int) -> List[str]:
    score = modelo.decision_function(instance_df)[0]
    explicacao = []
    
    deltas = calculate_deltas(modelo, instance_df, X_train, premis_class)
    indices_ordenados = np.argsort(-np.abs(deltas))
    
    score_base = score - np.sum(deltas)
    soma_deltas_cumulativa = score_base 
    target_score = t_plus if premis_class == 1 else t_minus
    EPSILON = 1e-6 

    for i in indices_ordenados:
        feature_nome = X_train.columns[i]
        valor_original_feature = instance_df.iloc[0, X_train.columns.get_loc(feature_nome)]
        
        if abs(deltas[i]) > 1e-9:
             soma_deltas_cumulativa += deltas[i]
             explicacao.append(f"{feature_nome} = {valor_original_feature:.4f}")
        
        if premis_class == 1:
            if soma_deltas_cumulativa >= (target_score - EPSILON) and explicacao:
                break
        else:
            if soma_deltas_cumulativa <= (target_score + EPSILON) and explicacao:
                break
                
    if not explicacao and len(X_train.columns) > 0:
         logreg = _get_lr(modelo)
         idx_max = np.argmax(np.abs(logreg.coef_[0]))
         feat_nome = X_train.columns[idx_max]
         valor_feat = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
         explicacao.append(f"{feat_nome} = {valor_feat:.4f}")

    return explicacao

def perturbar_e_validar_otimizado(vals_s: np.ndarray, coefs: np.ndarray, score_original: float, 
                                  indices_explicacao: Set[int], 
                                  intercept: float,
                                  modelo: Pipeline, 
                                  t_plus: float, t_minus: float, direcao_override: int, 
                                  pred_class_orig: int, is_rejected: bool) -> Tuple[bool, float]:
    
    if not indices_explicacao:
        return False, 0.0

    # Obtém limites reais do scaler do modelo para consistência
    if 'scaler' in modelo.named_steps:
        MIN_VAL, MAX_VAL = modelo.named_steps['scaler'].feature_range
    else:
        MIN_VAL, MAX_VAL = 0.0, 1.0

    perturbar_para_diminuir = (direcao_override == 1)
    
    if perturbar_para_diminuir:
        # Baixar score: se w>0 -> MIN, se w<0 -> MAX
        X_teste = np.where(coefs > 0, MIN_VAL, MAX_VAL)
    else:
        # Subir score: se w>0 -> MAX, se w<0 -> MIN
        X_teste = np.where(coefs > 0, MAX_VAL, MIN_VAL)
    
    # Restaura valores originais nas features fixas
    if indices_explicacao:
        idx_fixos = list(indices_explicacao)
        X_teste[idx_fixos] = vals_s[idx_fixos]
    
    # Cálculo Instantâneo do Score
    score_pert = intercept + np.dot(X_teste, coefs)
    
    EPSILON = 1e-6
    
    if is_rejected:
        if perturbar_para_diminuir:
            return (score_pert >= t_minus - EPSILON), score_pert
        else:
            return (score_pert <= t_plus + EPSILON), score_pert
    else:
        if pred_class_orig == 1:
            return (score_pert >= t_plus - EPSILON), score_pert
        else:
            return (score_pert <= t_minus + EPSILON), score_pert

def fase_1_reforco(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], 
                   X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, 
                   premisa_ordenacao: int, benchmark_mode: bool = False) -> Tuple[List[str], int]:
    
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    vals_s = scaler.transform(instance_df)[0]
    score_orig = modelo.decision_function(instance_df)[0]
    
    pred_class_orig = int(modelo.predict(instance_df)[0])
    col_to_idx = {c: i for i, c in enumerate(X_train.columns)}
    
    expl_robusta_indices = {col_to_idx[f.split(' = ')[0]] for f in expl_inicial}
    expl_robusta_str = list(expl_inicial)
    
    adicoes = 0
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao)
    indices_ordenados = np.argsort(-np.abs(deltas_para_ordenar))
    num_features_total = X_train.shape[1]
    
    while True:
        if is_rejected:
            valido1, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, expl_robusta_indices, intercept, modelo, t_plus, t_minus, 0, pred_class_orig, True)
            valido2, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, expl_robusta_indices, intercept, modelo, t_plus, t_minus, 1, pred_class_orig, True)
            is_valid = valido1 and valido2
        else:
            direcao = 1 if pred_class_orig == 1 else 0
            is_valid, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, expl_robusta_indices, intercept, modelo, t_plus, t_minus, direcao, pred_class_orig, False)
            
        if is_valid: break
        if len(expl_robusta_indices) == num_features_total: break

        adicionou_feature = False
        for idx in indices_ordenados:
            if idx not in expl_robusta_indices:
                expl_robusta_indices.add(idx)
                feat_nome = X_train.columns[idx]
                val = instance_df.iloc[0, idx]
                expl_robusta_str.append(f"{feat_nome} = {val:.4f}")
                adicoes += 1
                adicionou_feature = True
                break
        if not adicionou_feature: break
    
    return expl_robusta_str, adicoes

def fase_2_minimizacao(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], 
                       X_train: pd.DataFrame, t_plus: float, t_minus: float, 
                       is_rejected: bool, premisa_ordenacao: int, 
                       log_passos: List[Dict], benchmark_mode: bool = False) -> Tuple[List[str], int]:
    
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    vals_s = scaler.transform(instance_df)[0]
    score_orig = modelo.decision_function(instance_df)[0]
    
    pred_class_orig = int(modelo.predict(instance_df)[0])
    col_to_idx = {c: i for i, c in enumerate(X_train.columns)}

    expl_minima_str = list(expl_robusta)
    indices_atuais = {col_to_idx[f.split(' = ')[0]] for f in expl_robusta}
    
    remocoes = 0
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao)
    
    features_para_remover = sorted(
        [f.split(' = ')[0] for f in expl_minima_str],
        key=lambda nome: abs(deltas_para_ordenar[col_to_idx[nome]]),
        reverse=True
    )

    for feat_nome in features_para_remover:
        if len(indices_atuais) <= 1: break
        idx_alvo = col_to_idx[feat_nome]
        indices_atuais.remove(idx_alvo)
        
        remocao_bem_sucedida = False
        score_p1, score_p2 = 0.0, 0.0
        
        if is_rejected:
            valido1, score_p1 = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, indices_atuais, intercept, modelo, t_plus, t_minus, 1, pred_class_orig, True)
            valido2, score_p2 = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, indices_atuais, intercept, modelo, t_plus, t_minus, 0, pred_class_orig, True)
            if valido1 and valido2:
                remocao_bem_sucedida = True
            
            if not benchmark_mode and log_passos is not None:
                log_passos.append({
                    'feat_nome': feat_nome, 'sucesso': remocao_bem_sucedida,
                    'score_neg': score_p1, 'score_pos': score_p2
                })
        else:
            direcao = 1 if pred_class_orig == 1 else 0
            valido, score_p1 = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, indices_atuais, intercept, modelo, t_plus, t_minus, direcao, pred_class_orig, False)
            if valido:
                remocao_bem_sucedida = True
            
            if not benchmark_mode and log_passos is not None:
                log_passos.append({'feat_nome': feat_nome, 'sucesso': remocao_bem_sucedida, 'score_perturbado': score_p1})

        if remocao_bem_sucedida:
            remocoes += 1
            expl_minima_str = [f for f in expl_minima_str if not f.startswith(feat_nome)]
        else:
            indices_atuais.add(idx_alvo)

    return expl_minima_str, remocoes

#==============================================================================
# GERAÇÃO DE LOG E ORQUESTRAÇÃO
#==============================================================================

def gerar_explicacao_instancia(instancia_df: pd.DataFrame, modelo: Pipeline, X_train: pd.DataFrame, t_plus: float, t_minus: float, benchmark_mode: bool = False) -> Tuple[List[str], List[str], int, int]:
    is_rejected = t_minus <= modelo.decision_function(instancia_df)[0] <= t_plus
    log_formatado: List[str] = []
    emit_tech_logs = (not benchmark_mode) and TECHNICAL_LOGS and (X_train.shape[1] <= MAX_LOG_FEATURES)

    if is_rejected:
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['rejeitada_analise'].format(t_minus=t_minus, t_plus=t_plus))

        # Caminho 1 (Tentando provar Classe 1)
        expl_inicial_p1 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 1)
        expl_robusta_p1, adicoes1 = fase_1_reforco(modelo, instancia_df, expl_inicial_p1, X_train, t_plus, t_minus, True, 1, benchmark_mode)
        passos_p1: List[Dict[str, Any]] = []
        expl_final_p1, remocoes1 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p1, X_train, t_plus, t_minus, True, 1, passos_p1, benchmark_mode)

        # Caminho 2 (Tentando provar Classe 0)
        expl_inicial_p2 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 0)
        expl_robusta_p2, adicoes2 = fase_1_reforco(modelo, instancia_df, expl_inicial_p2, X_train, t_plus, t_minus, True, 0, benchmark_mode)
        passos_p2: List[Dict[str, Any]] = []
        expl_final_p2, remocoes2 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p2, X_train, t_plus, t_minus, True, 0, passos_p2, benchmark_mode)

        # Seleção do melhor resultado da minimização
        if len(expl_final_p1) <= len(expl_final_p2):
            expl_final = expl_final_p1
            adicoes, remocoes = adicoes1, remocoes1
            passos_escolhidos = passos_p1
        else:
            expl_final = expl_final_p2
            adicoes, remocoes = adicoes2, remocoes2
            passos_escolhidos = passos_p2

        # =====================================================================
        # [NOVA TRAVA INTELIGENTE]
        # Se a explicação final for vazia (0 features), não devolvemos tudo.
        # Em vez disso, pegamos a "Explicação Formal Inicial" mais curta.
        # Isso garante que mostramos pelo menos as features que empurram o score
        # para a borda, sem precisar mostrar a imagem inteira.
        # =====================================================================
        if len(expl_final) == 0:
            # Recalcula a explicação base (one_explanation) para os dois lados
            base_p1 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 1)
            base_p0 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 0)
            
            # Escolhe a menor não-vazia
            if len(base_p1) > 0 and (len(base_p1) <= len(base_p0) or len(base_p0) == 0):
                expl_final = base_p1
            elif len(base_p0) > 0:
                expl_final = base_p0
            else:
                # Se até a base for vazia (raríssimo), aí sim fallback total
                expl_final = [f"{c} = {instancia_df.iloc[0, X_train.columns.get_loc(c)]:.4f}" for c in X_train.columns]
            
            # Zera contadores de processo pois foi um fallback
            remocoes = 0 
        # =====================================================================

        if emit_tech_logs:
             for passo in passos_escolhidos[:MAX_LOG_STEPS]:
                key = 'rejeitada_feat_header_sucesso' if passo.get('sucesso') else 'rejeitada_feat_header_falha'
                log_formatado.append(LOG_TEMPLATES[key].format(feat=passo['feat_nome'], delta=0.0))

    else:
        # Lógica das Classificadas (Mantida igual)
        pred_class = int(modelo.predict(instancia_df)[0])
        if emit_tech_logs:
            posicao = 'acima de t+' if pred_class == 1 else 'abaixo de t-'
            log_formatado.append(LOG_TEMPLATES['classificada_analise'].format(posicao=posicao))

        expl_inicial = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, pred_class)
        if DISABLE_REFORCO_CLASSIFICADAS:
            expl_robusta = expl_inicial
            adicoes = 0
        else:
            expl_robusta, adicoes = fase_1_reforco(modelo, instancia_df, expl_inicial, X_train, t_plus, t_minus, False, pred_class, benchmark_mode)

        passos: List[Dict[str, Any]] = []
        expl_final, remocoes = fase_2_minimizacao(modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, False, pred_class, passos, benchmark_mode)

    return [f.split(' = ')[0] for f in expl_final], log_formatado, adicoes, remocoes

#==============================================================================
# FUNÇÕES DE SUPORTE ESSENCIAIS (REINSERIDAS)
#==============================================================================

def configurar_experimento(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], float, float]:
    """Carrega dados e configs. Se MNIST, aplica configs especiais."""
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

def aplicar_selecao_top_k_features(X_train: pd.DataFrame, X_test: pd.DataFrame, pipeline: Pipeline, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    logreg = _get_lr(pipeline)
    coefs = logreg.coef_[0]
    importances = [(name, abs(coefs[i])) for i, name in enumerate(X_train.columns)]
    indices_top = sorted(range(len(importances)), key=lambda i: importances[i][1], reverse=True)[:top_k]
    selected_features = [X_train.columns[i] for i in indices_top]
    return X_train[selected_features], X_test[selected_features], selected_features

def treinar_e_avaliar_modelo(X_train: pd.DataFrame, y_train: pd.Series, rejection_cost: float, logreg_params: Dict[str, Any]) -> Tuple[Pipeline, float, float, Dict[str, Any]]:
    """
    [CORREÇÃO A] Recebe X_train e y_train JÁ DIVIDIDOS para garantir consistência.
    """
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params)),
    ])
    pipeline.fit(X_train, y_train)

    decision_scores = pipeline.decision_function(X_train)
    qs = np.linspace(0, 1, 100)
    search_space = np.unique(np.quantile(decision_scores, qs))
    best_risk, best_t_plus, best_t_minus = float('inf'), 0.0, 0.0
    
    for i in range(len(search_space)):
        for j in range(i, len(search_space)):
            t_minus, t_plus = float(search_space[i]), float(search_space[j])
            if MIN_REJECTION_WIDTH > 0.0 and (t_plus - t_minus) < MIN_REJECTION_WIDTH: continue
            
            y_pred = np.full(y_train.shape, -1)
            accepted = (decision_scores >= t_plus) | (decision_scores <= t_minus)
            y_pred[decision_scores >= t_plus] = 1
            y_pred[decision_scores <= t_minus] = 0
            
            error_rate = np.mean(y_pred[accepted] != y_train[accepted]) if np.any(accepted) else 0.0
            rejection_rate = 1.0 - np.mean(accepted)
            risk = float(error_rate + rejection_cost * rejection_rate)
            
            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, t_plus, t_minus

    coefs = pipeline.named_steps['model'].coef_[0]
    model_params = {
        'coefs': {name: float(w) for name, w in zip(list(X_train.columns), coefs)},
        'intercepto': float(pipeline.named_steps['model'].intercept_[0]),
        'scaler_params': {'min': pipeline.named_steps['scaler'].min_, 'scale': pipeline.named_steps['scaler'].scale_},
        **logreg_params
    }
    return pipeline, float(best_t_plus), float(best_t_minus), model_params

def coletar_metricas(resultados_instancias, y_test, y_pred_test_final, rejected_mask,
                     tempo_total, model_params, modelo: Pipeline, X_test: pd.DataFrame, feature_names: List[str],
                     times_pos, times_neg, times_rej, adicoes_pos, adicoes_neg, adicoes_rej, remocoes_pos, remocoes_neg, remocoes_rej):
    stats_pos_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 1]
    stats_neg_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 0]
    stats_rej_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 2]

    acc_sem_rej = float(np.mean(modelo.predict(X_test) == y_test) * 100)
    acc_com_rej = float(np.mean(y_pred_test_final[~rejected_mask] == y_test.iloc[~rejected_mask]) * 100) if np.any(~rejected_mask) else 100.0
    taxa_rej = float(np.mean(rejected_mask) * 100)

    avg_time_pos = float(np.mean(times_pos)) if times_pos else 0.0
    avg_time_neg = float(np.mean(times_neg)) if times_neg else 0.0
    avg_time_rej = float(np.mean(times_rej)) if times_rej else 0.0

    def get_proc_stats(adicoes, remocoes):
        inst_com_adicao = sum(1 for x in adicoes if x > 0)
        media_adicoes = float(np.mean([x for x in adicoes if x > 0])) if inst_com_adicao > 0 else 0.0
        inst_com_remocao = sum(1 for x in remocoes if x > 0)
        media_remocoes = float(np.mean([x for x in remocoes if x > 0])) if inst_com_remocao > 0 else 0.0
        return {'inst_com_adicao': int(inst_com_adicao), 'media_adicoes': media_adicoes, 
                'inst_com_remocao': int(inst_com_remocao), 'media_remocoes': media_remocoes,
                'perc_adicao': float(inst_com_adicao/len(adicoes)*100) if adicoes else 0.0,
                'perc_remocao': float(inst_com_remocao/len(remocoes)*100) if remocoes else 0.0}

    return {
        'acuracia_sem_rejeicao': acc_sem_rej, 'acuracia_com_rejeicao': acc_com_rej, 'taxa_rejeicao': taxa_rej,
        'stats_explicacao_positiva': {'instancias': len(stats_pos_list), 'media': float(np.mean(stats_pos_list)) if stats_pos_list else 0.0, 'std_dev': float(np.std(stats_pos_list)) if stats_pos_list else 0.0, 'min': int(np.min(stats_pos_list)) if stats_pos_list else 0, 'max': int(np.max(stats_pos_list)) if stats_pos_list else 0},
        'stats_explicacao_negativa': {'instancias': len(stats_neg_list), 'media': float(np.mean(stats_neg_list)) if stats_neg_list else 0.0, 'std_dev': float(np.std(stats_neg_list)) if stats_neg_list else 0.0, 'min': int(np.min(stats_neg_list)) if stats_neg_list else 0, 'max': int(np.max(stats_neg_list)) if stats_neg_list else 0},
        'stats_explicacao_rejeitada': {'instancias': len(stats_rej_list), 'media': float(np.mean(stats_rej_list)) if stats_rej_list else 0.0, 'std_dev': float(np.std(stats_rej_list)) if stats_rej_list else 0.0, 'min': int(np.min(stats_rej_list)) if stats_rej_list else 0, 'max': int(np.max(stats_rej_list)) if stats_rej_list else 0},
        'tempo_total': float(tempo_total), 'tempo_medio_instancia': float(tempo_total / len(y_test) if len(y_test) > 0 else 0.0),
        'tempo_medio_positivas': avg_time_pos, 'tempo_medio_negativas': avg_time_neg, 'tempo_medio_rejeitadas': avg_time_rej,
        'features_frequentes': Counter([feat for r in resultados_instancias for feat in r['explicacao']]).most_common(),
        'pesos_modelo': sorted(((name, float(model_params['coefs'][name])) for name in feature_names), key=lambda item: abs(item[1]), reverse=True),
        'intercepto': float(model_params['intercepto']),
        'processo_stats_pos': get_proc_stats(adicoes_pos, remocoes_pos),
        'processo_stats_neg': get_proc_stats(adicoes_neg, remocoes_neg),
        'processo_stats_rej': get_proc_stats(adicoes_rej, remocoes_rej)
    }

def gerar_relatorio_texto(dataset_name, test_size, wr, modelo, t_plus, t_minus, num_test, metricas, resultados_instancias):
    output_path = os.path.join(OUTPUT_BASE_DIR, f"peab_{dataset_name}.txt")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE ANÁLISE - MÉTODO PEAB\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Acurácia (sem rejeição): {metricas['acuracia_sem_rejeicao']:.2f}%\n")
        f.write(f"Thresholds: t+={t_plus:.4f}, t-={t_minus:.4f}\n\n")
        # Completar com template original se desejar

def montar_dataset_cache(*args, **kwargs):
    pass

def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n[INFO] Executando PEAB para: {dataset_name.upper()}")
    todos_hiperparametros = carregar_hiperparametros()
    X_full, y_full, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)

    params = DEFAULT_LOGREG_PARAMS.copy()
    if dataset_name in todos_hiperparametros and 'params' in todos_hiperparametros[dataset_name]:
        params.update(todos_hiperparametros[dataset_name]['params'])

    # [CORREÇÃO] Split Único e Consistente
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=RANDOM_STATE, stratify=y_full)

    # Redução Top-K
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0 and top_k < X_train.shape[1]:
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params)
        logreg_tmp = _get_lr(modelo_temp)
        idx_top = np.argsort(np.abs(logreg_tmp.coef_[0]))[::-1][:top_k]
        feats = X_train.columns[idx_top]
        X_train, X_test = X_train[feats], X_test[feats]

    # Treino Final com dados consistentes
    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params)
    
    # Previsões
    decision_scores = modelo.decision_function(X_test)
    y_pred = np.full(y_test.shape, -1, dtype=int)
    y_pred[decision_scores >= t_plus] = 1
    y_pred[decision_scores <= t_minus] = 0
    mask_rej = (y_pred == -1)
    y_pred_final = y_pred.copy()
    y_pred_final[mask_rej] = 2

    print(f"[INFO] Explicando {len(X_test)} instâncias...")
    start_total = time.perf_counter()
    resultados = []
    
    # Listas auxiliares para estatísticas
    t_pos, t_neg, t_rej = [], [], []
    ad_pos, ad_neg, ad_rej = [], [], []
    rm_pos, rm_neg, rm_rej = [], [], []

    with ProgressBar(total=len(X_test)) as pbar:
        for i in range(len(X_test)):
            start_inst = time.perf_counter()
            inst = X_test.iloc[[i]]
            expl, logs, ad, rm = gerar_explicacao_instancia(inst, modelo, X_train, t_plus, t_minus, benchmark_mode=False)
            duracao = time.perf_counter() - start_inst
            
            p_code = y_pred_final[i]
            resultados.append({
                'id': i, 'classe_real': nomes_classes[y_test.iloc[i]],
                'pred_code': int(p_code),
                'predicao': 'REJEITADA' if p_code == 2 else nomes_classes[p_code],
                'explicacao': sorted(expl), 'tamanho_explicacao': len(expl),
                'log_detalhado': logs
            })
            
            if p_code == 2:
                t_rej.append(duracao); ad_rej.append(ad); rm_rej.append(rm)
            elif p_code == 1:
                t_pos.append(duracao); ad_pos.append(ad); rm_pos.append(rm)
            else:
                t_neg.append(duracao); ad_neg.append(ad); rm_neg.append(rm)
            pbar.update()

    total_time = time.perf_counter() - start_total
    
    metricas = coletar_metricas(resultados, y_test, y_pred_final, mask_rej, total_time, model_params, modelo, X_test, X_train.columns, t_pos, t_neg, t_rej, ad_pos, ad_neg, ad_rej, rm_pos, rm_neg, rm_rej)
    
    gerar_relatorio_texto(dataset_name, test_size, rejection_cost, modelo, t_plus, t_minus, len(X_test), metricas, resultados)
    print(f"[SUCESSO] Relatório gerado em {OUTPUT_BASE_DIR}")

if __name__ == '__main__':
    ds, _, _, _, _ = selecionar_dataset_e_classe()
    if ds:
        executar_experimento_para_dataset(ds)