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
    'subsample_size': 0.3  # REDUZIDO para MinExp conseguir processar (5% = ~210 instâncias)
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
MIN_REJECTION_WIDTH: float = 0.0

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

#==============================================================================
#  LÓGICA FORMAL (OTIMIZADA E CONSISTENTE)
#==============================================================================

def calculate_deltas(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, premis_class: int) -> np.ndarray:
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo) #acesssa o pipeline e pega o modelo
    coefs = logreg.coef_[0] # pega os coeficientes do modelo
    
    instance_df_ordered = instance_df[X_train.columns] 
    
    # Transforma a instância para o espaço escalado
    if hasattr(scaler, 'feature_range'):
        f_min, f_max = scaler.feature_range
        scaled_instance_vals = scaler.transform(instance_df_ordered)[0]
    else:
        # Fallback caso não tenha scaler
        # Assume MinMaxScaler com range [0, 1]
        f_min, f_max = 0.0, 1.0
        scaled_instance_vals = instance_df_ordered.values[0]
    # Cria vetores para os valores mínimos e máximos no espaço escalado
    X_train_scaled_min = np.full_like(coefs, f_min)
    X_train_scaled_max = np.full_like(coefs, f_max)
    
    # Define o pior valor para cada feature com base na classe alvo
    # Classe 1: adversário empurra para BAIXO (quer fazer score < t_plus)
    # Classe 0: adversário empurra para CIMA (quer fazer score > t_minus)
    if premis_class == 1:
        # Pior caso: valores que DIMINUEM o score
        pior_valor = np.where(coefs > 0, X_train_scaled_min, X_train_scaled_max)
    else:
        # Pior caso: valores que AUMENTAM o score (invertido!)
        pior_valor = np.where(coefs > 0, X_train_scaled_max, X_train_scaled_min)
    
    # Calcula os deltas (contribuição de cada feature)
    deltas = (scaled_instance_vals - pior_valor) * coefs
    return deltas

def one_explanation_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, premis_class: int) -> List[str]:
    score = modelo.decision_function(instance_df)[0] # pega o score da instancia
    explicacao = [] # lsita de explicação vazia que sera preenchida
    
    deltas = calculate_deltas(modelo, instance_df, X_train, premis_class) # calcula os deltas 
    indices_ordenados = np.argsort(-np.abs(deltas)) # indices ordenados por impacto absoluto decrescente (negativo para ordem do maior p menor)
    
    # score_base = score no pior caso (todas features no pior valor)
    score_base = score - np.sum(deltas)
    soma_deltas_cumulativa = score_base 
    target_score = t_plus if premis_class == 1 else t_minus
    EPSILON = 1e-6 # margem de tolerancia para comparações de ponto flutuante
    
    for i in indices_ordenados:
        feature_nome = X_train.columns[i]
        valor_original_feature = instance_df.iloc[0, X_train.columns.get_loc(feature_nome)]
        
        if abs(deltas[i]) > 0:
             soma_deltas_cumulativa += deltas[i]
             explicacao.append(f"{feature_nome} = {valor_original_feature:.4f}")
        
        # Verifica se a condição de score alvo foi atingida
        # Classe 1: deltas positivos, acumula até SUBIR e atingir t_plus
        # Classe 0: deltas negativos, acumula até DESCER e atingir t_minus
        if premis_class == 1:
            if soma_deltas_cumulativa >= target_score and explicacao:
                break
        else:
            if soma_deltas_cumulativa <= target_score and explicacao:
                break
    
    # se sair do loop sem explicação, adiciona a feature de maior coeficiente            
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
                                  t_plus: float, t_minus: float, norm_params: Dict[str, float], direcao_override: int, 
                                  pred_class_orig: int, is_rejected: bool) -> Tuple[bool, float]:
    """
    Valida mantendo features fixas em seus valores originais e empurrando
    o restante para o pior caso na direção especificada. Suporta rejeição.
    Permite conjunto vazio (intercepto pode ser suficiente).
    
    direcao_override: 0 = empurrar para CIMA (tentar sair por cima)
                      1 = empurrar para BAIXO (tentar sair por baixo)
                      
    [CORREÇÃO CRÍTICA] Para rejeitadas, testa AMBAS as direções simultaneamente
    garantindo que a explicação mantém a instância na zona de rejeição sob
    qualquer perturbação adversária (para cima OU para baixo).
    """
    # Limites por-feature no espaço escalado do MinMaxScaler
    if 'scaler' in modelo.named_steps:
        scaler = modelo.named_steps['scaler']
        # Para MinMaxScaler, extremos no espaço escalado são 0.0 e 1.0 para todas as features
        MIN_VEC = np.zeros_like(coefs) # vetor de mínimos
        MAX_VEC = np.ones_like(coefs) # vetor de máximos
    else:
        MIN_VEC = np.zeros_like(coefs) 
        MAX_VEC = np.ones_like(coefs)

    EPSILON = 1e-3
    
    #[CORREÇÃO CRÍTICA] Para rejeitadas: testar AMBAS as direções
    # USAR SCORES BRUTOS (SEM NORMALIZAÇÃO)
    if is_rejected:
        # Perturbação para BAIXO (tentar cair abaixo de t_minus)
        X_teste_baixo = np.where(coefs > 0, MIN_VEC, MAX_VEC)
        if indices_explicacao:
            idx_fixos = list(indices_explicacao)
            X_teste_baixo[idx_fixos] = vals_s[idx_fixos]
        score_baixo = intercept + np.dot(X_teste_baixo, coefs)
        
        # Perturbação para CIMA (tentar subir acima de t_plus)
        X_teste_cima = np.where(coefs > 0, MAX_VEC, MIN_VEC)
        if indices_explicacao:
            X_teste_cima[idx_fixos] = vals_s[idx_fixos]
        score_cima = intercept + np.dot(X_teste_cima, coefs)
        
        # Explicação é suficiente se AMBAS as perturbações ficam na zona de rejeição
        # USAR SCORES BRUTOS diretamente com thresholds brutos
        valido_baixo = (score_baixo >= t_minus - EPSILON)
        valido_cima = (score_cima <= t_plus + EPSILON)
        
        # Retorna média dos scores como referência (para debug)
        score_medio = (score_baixo + score_cima) / 2.0
        return (valido_baixo and valido_cima), score_medio
    
    else:
        # Classificadas: testar apenas uma direção (comportamento original)
        # USAR SCORES BRUTOS (SEM NORMALIZAÇÃO)
        empurrar_para_baixo = (direcao_override == 1)
        X_teste = np.where(coefs > 0, MIN_VEC, MAX_VEC) if empurrar_para_baixo else np.where(coefs > 0, MAX_VEC, MIN_VEC)
        
        if indices_explicacao:
            idx_fixos = list(indices_explicacao)
            X_teste[idx_fixos] = vals_s[idx_fixos]

        score_pert = intercept + np.dot(X_teste, coefs)
        
        # Classificadas: manter no lado correto do limiar correspondente
        # USAR SCORES BRUTOS diretamente com thresholds brutos
        if pred_class_orig == 1:
            return (score_pert >= t_plus - EPSILON), score_pert
        else:
            return (score_pert <= t_minus + EPSILON), score_pert

def fase_1_reforco(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], 
                   X_train: pd.DataFrame, t_plus: float, t_minus: float, norm_params: Dict[str, float], is_rejected: bool, 
                   premisa_ordenacao: int, benchmark_mode: bool = False) -> Tuple[List[str], int]:
    
    # norm_params mantido para compatibilidade (max_abs=1.0)
    # Pré-cálculos
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    # Garantir mesma ordem de colunas do treino antes de transformar
    instance_df_ordered = instance_df[X_train.columns]
    vals_s = scaler.transform(instance_df_ordered)[0]
    score_orig = modelo.decision_function(instance_df)[0]

    # Classe predita original
    pred_class_orig = int(modelo.predict(instance_df)[0])
    col_to_idx = {c: i for i, c in enumerate(X_train.columns)} # mapeia nome da coluna para índice
    
    expl_robusta_indices = {col_to_idx[f.split(' = ')[0]] for f in expl_inicial} # índices das features na explicação inicial
    expl_robusta_str = list(expl_inicial) 
    
    adicoes = 0
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao) # calcula os deltas
    indices_ordenados = np.argsort(-np.abs(deltas_para_ordenar)) # ordena os índices por impacto absoluto decrescente
    num_features_total = X_train.shape[1]
    
    while True:
        if is_rejected: 
            valido1, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, expl_robusta_indices, intercept, modelo, t_plus, t_minus, norm_params, 1, pred_class_orig, True) # testa empurrar para baixo
            valido2, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, expl_robusta_indices, intercept, modelo, t_plus, t_minus, norm_params, 0, pred_class_orig, True) # testa empurrar para cima
            is_valid = valido1 and valido2 # ambos os lados devem ser válidos
        else:
            direcao = 1 if pred_class_orig == 1 else 0
            is_valid, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, expl_robusta_indices, intercept, modelo, t_plus, t_minus, norm_params, direcao, pred_class_orig, False)
            
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
                       X_train: pd.DataFrame, t_plus: float, t_minus: float, norm_params: Dict[str, float],
                       is_rejected: bool, premisa_ordenacao: int, 
                       log_passos: List[Dict], benchmark_mode: bool = False) -> Tuple[List[str], int]:
    
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    # Garantir mesma ordem de colunas do treino antes de transformar
    instance_df_ordered = instance_df[X_train.columns]
    vals_s = scaler.transform(instance_df_ordered)[0]
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
        reverse=False # se True remove as do maior delta se False as de menro delta(menor impacto)
    )

    for feat_nome in features_para_remover:
        if len(indices_atuais) <= 1: break
        idx_alvo = col_to_idx[feat_nome]
        indices_atuais.remove(idx_alvo)
        
        remocao_bem_sucedida = False
        
        if is_rejected:
            valido1, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, indices_atuais, intercept, modelo, t_plus, t_minus, norm_params, 1, pred_class_orig, True)
            valido2, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, indices_atuais, intercept, modelo, t_plus, t_minus, norm_params, 0, pred_class_orig, True)
            if valido1 and valido2: remocao_bem_sucedida = True
            
            if not benchmark_mode and log_passos is not None:
                log_passos.append({'feat_nome': feat_nome, 'sucesso': remocao_bem_sucedida})
        else:
            direcao = 1 if pred_class_orig == 1 else 0
            valido, _ = perturbar_e_validar_otimizado(vals_s, coefs, score_orig, indices_atuais, intercept, modelo, t_plus, t_minus, norm_params, direcao, pred_class_orig, False)
            if valido: remocao_bem_sucedida = True
            
            if not benchmark_mode and log_passos is not None:
                log_passos.append({'feat_nome': feat_nome, 'sucesso': remocao_bem_sucedida})

        if remocao_bem_sucedida:
            remocoes += 1
            # Remove feature da explicação usando comparação exata do nome
            expl_minima_str = [f for f in expl_minima_str if f.split(' = ')[0] != feat_nome]
        else:
            indices_atuais.add(idx_alvo)
    
    return expl_minima_str, remocoes

def gerar_explicacao_instancia(instancia_df: pd.DataFrame, modelo: Pipeline, X_train: pd.DataFrame, t_plus: float, t_minus: float, norm_params: Dict[str, float], benchmark_mode: bool = False) -> Tuple[List[str], List[str], int, int]:
    # USAR SCORES BRUTOS (SEM NORMALIZAÇÃO)
    score_raw = modelo.decision_function(instancia_df)[0]
    
    is_rejected = t_minus <= score_raw <= t_plus
    log_formatado: List[str] = []
    emit_tech_logs = (not benchmark_mode) and TECHNICAL_LOGS and (X_train.shape[1] <= MAX_LOG_FEATURES)

    if is_rejected:
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['rejeitada_analise'].format(t_minus=t_minus, t_plus=t_plus))

        # Para rejeitadas, começar com conjunto vazio e usar fase 1 para adicionar features
        expl_inicial_vazia = []
        expl_robusta_p1, adicoes1 = fase_1_reforco(modelo, instancia_df, expl_inicial_vazia, X_train, t_plus, t_minus, norm_params, True, 1, benchmark_mode)
        passos_p1: List[Dict[str, Any]] = []
        expl_final_p1, remocoes1 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p1, X_train, t_plus, t_minus, norm_params, True, 1, passos_p1, benchmark_mode)

        expl_robusta_p2, adicoes2 = fase_1_reforco(modelo, instancia_df, expl_inicial_vazia, X_train, t_plus, t_minus, norm_params, True, 0, benchmark_mode)
        passos_p2: List[Dict[str, Any]] = []
        expl_final_p2, remocoes2 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p2, X_train, t_plus, t_minus, norm_params, True, 0, passos_p2, benchmark_mode)

        if len(expl_final_p1) <= len(expl_final_p2):
            expl_final, adicoes, remocoes = expl_final_p1, adicoes1, remocoes1
            passos_escolhidos = passos_p1
        else:
            expl_final, adicoes, remocoes = expl_final_p2, adicoes2, remocoes2
            passos_escolhidos = passos_p2

        if emit_tech_logs:
             for passo in passos_escolhidos[:MAX_LOG_STEPS]:
                key = 'rejeitada_feat_header_sucesso' if passo.get('sucesso') else 'rejeitada_feat_header_falha'
                log_formatado.append(LOG_TEMPLATES[key].format(feat=passo['feat_nome'], delta=0.0))

    else:
        pred_class = int(modelo.predict(instancia_df)[0])
        if emit_tech_logs:
            posicao = 'acima de t+' if pred_class == 1 else 'abaixo de t-'
            log_formatado.append(LOG_TEMPLATES['classificada_analise'].format(posicao=posicao))

        expl_inicial = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, pred_class)
        if DISABLE_REFORCO_CLASSIFICADAS:
            expl_robusta = expl_inicial
            adicoes = 0
        else:
            expl_robusta, adicoes = fase_1_reforco(modelo, instancia_df, expl_inicial, X_train, t_plus, t_minus, norm_params, False, pred_class, benchmark_mode)

        passos: List[Dict[str, Any]] = []
        expl_final, remocoes = fase_2_minimizacao(modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, norm_params, False, pred_class, passos, benchmark_mode)

    return [f.split(' = ')[0] for f in expl_final], log_formatado, adicoes, remocoes

#==============================================================================
# FUNÇÕES DE SUPORTE (INCLUÍDAS PARA CORRIGIR O ERRO DE IMPORT)
#==============================================================================

def configurar_experimento(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], float, float]:
    """Carrega dataset e configurações."""
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

def treinar_e_avaliar_modelo(X_train: pd.DataFrame, y_train: pd.Series, rejection_cost: float, logreg_params: Dict[str, Any], dataset_name: str = "UNKNOWN", val_size: float = 0.2) -> Tuple[Pipeline, float, float, Dict[str, Any]]:
    """
    [CORREÇÃO CRÍTICA] Otimiza thresholds em conjunto de VALIDAÇÃO separado.
    
    Fundamentação:
    - Fumera & Roli (2002): "threshold optimization should be performed on independent validation set"
    - Otimizar thresholds no mesmo conjunto usado para treinar causa overfitting
    - Split treino/validação garante generalização dos thresholds
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        rejection_cost: Custo de rejeição
        logreg_params: Hiperparâmetros do LogisticRegression
        dataset_name: Nome do dataset (para logs)
        val_size: Fração do treino usada para validação (padrão: 0.2)
    
    Returns:
        pipeline: Modelo treinado com TODO o conjunto de treino
        t_plus: Threshold positivo otimizado em validação
        t_minus: Threshold negativo otimizado em validação
        model_params: Parâmetros do modelo
    """
    
    #==============================================================================
    # PASSO 1: SPLIT TREINO/VALIDAÇÃO (evitar overfitting dos thresholds)
    #==============================================================================
    print(f"\n[VALIDAÇÃO] Dividindo treino em: {int((1-val_size)*100)}% treino + {int(val_size*100)}% validação")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_size, 
        random_state=RANDOM_STATE, 
        stratify=y_train
    )
    print(f"[VALIDAÇÃO] Treino: {len(X_train_sub)} instâncias, Validação: {len(X_val)} instâncias")
    
    #==============================================================================
    # PASSO 2: TREINAR MODELO NO SUBCONJUNTO DE TREINO
    #==============================================================================
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params)),
    ])
    pipeline.fit(X_train_sub, y_train_sub)

    #==============================================================================
    # PASSO 3: OTIMIZAR THRESHOLDS NO CONJUNTO DE VALIDAÇÃO (NÃO NO TREINO!)
    #==============================================================================
    # Usar scores BRUTOS (sem normalização!)
    decision_scores_val = pipeline.decision_function(X_val)
    decision_scores_raw = decision_scores_val

    min_raw = float(decision_scores_raw.min())
    max_raw = float(decision_scores_raw.max())

    print(f"\n[GRID_ADAPTATIVO] Dataset: {dataset_name}")
    print(f"[GRID_ADAPTATIVO] Scores brutos (VALIDAÇÃO): [{min_raw:.6f}, {max_raw:.6f}]")

    # ESTRATÉGIA: Garantir que 0 esteja entre t- e t+
    num_points = 50

    # SEPARA scores negativos e positivos reais
    scores_neg = decision_scores_raw[decision_scores_raw < 0]
    scores_pos = decision_scores_raw[decision_scores_raw > 0]

    # CRÍTICO: Garantir que t- seja negativo e t+ positivo
    if len(scores_neg) == 0:
        # Não há scores negativos → criar grid negativo artificial
        t_minus_grid = np.linspace(min_raw - 1.0, -0.01, num_points)
    else:
        t_minus_grid = np.linspace(scores_neg.min(), max(scores_neg.max() * 0.9, -0.01), num_points)

    if len(scores_pos) == 0:
        # Não há scores positivos → criar grid positivo artificial
        t_plus_grid = np.linspace(0.01, max_raw + 1.0, num_points)
    else:
        t_plus_grid = np.linspace(min(scores_pos.min() * 1.1, 0.01), scores_pos.max(), num_points)

    # GARANTIR t- < 0 < t+ (filtra valores incorretos)
    t_minus_grid = t_minus_grid[t_minus_grid < 0]
    t_plus_grid = t_plus_grid[t_plus_grid > 0]

    # Fallback se ainda vazio
    if len(t_minus_grid) == 0:
        t_minus_grid = np.linspace(-2.0, -0.01, num_points)
    if len(t_plus_grid) == 0:
        t_plus_grid = np.linspace(0.01, 2.0, num_points)

    print(f"[GRID_ADAPTATIVO] Grid negativo: {len(t_minus_grid)} pontos em [{t_minus_grid.min():.6f}, {t_minus_grid.max():.6f}]")
    print(f"[GRID_ADAPTATIVO] Grid positivo: {len(t_plus_grid)} pontos em [{t_plus_grid.min():.6f}, {t_plus_grid.max():.6f}]")
    print(f"[GRID_ADAPTATIVO] Total combinações: {len(t_minus_grid) * len(t_plus_grid)}")

    best_risk, best_t_plus, best_t_minus = float('inf'), 0.1, -0.1

    # Grid search em scores BRUTOS do conjunto de VALIDAÇÃO
    for t_minus in t_minus_grid:
        for t_plus in t_plus_grid:
            # GARANTIR RESTRIÇÃO TEÓRICA: t- < 0 < t+
            if not (t_minus < 0 < t_plus):
                continue
                
            if MIN_REJECTION_WIDTH > 0.0 and (t_plus - t_minus) < MIN_REJECTION_WIDTH:
                continue
            
            # Classificar com reject option (usando scores BRUTOS de VALIDAÇÃO)
            y_pred = np.full(y_val.shape, -1)
            accepted = (decision_scores_raw >= t_plus) | (decision_scores_raw <= t_minus)
            y_pred[decision_scores_raw >= t_plus] = 1
            y_pred[decision_scores_raw <= t_minus] = 0
            
            error_rate = np.mean(y_pred[accepted] != y_val[accepted]) if np.any(accepted) else 0.0
            rejection_rate = 1.0 - np.mean(accepted)
            risk = float(error_rate + rejection_cost * rejection_rate)
            
            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, t_plus, t_minus

    print(f"[GRID_ADAPTATIVO] Thresholds ótimos (validação): T+={best_t_plus:.6f}, T-={best_t_minus:.6f}, risk={best_risk:.6f}")
    
    #==============================================================================
    # PASSO 4: RETREINAR COM TODO O CONJUNTO DE TREINO (usando thresholds ótimos)
    #==============================================================================
    print(f"[VALIDAÇÃO] Retreinando com TODO o treino ({len(X_train)} instâncias) usando thresholds ótimos\n")
    pipeline.fit(X_train, y_train)

    # REMOVER cálculo de max_abs e normalização
    norm_params_temp = {'max_abs': 1.0}  # Placeholder para compatibilidade
    coefs = pipeline.named_steps['model'].coef_[0]
    model_params = {
        'coefs': {name: float(w) for name, w in zip(list(X_train.columns), coefs)},
        'intercepto': float(pipeline.named_steps['model'].intercept_[0]),
        'scaler_params': {'min': pipeline.named_steps['scaler'].min_, 'scale': pipeline.named_steps['scaler'].scale_},
        'norm_params': norm_params_temp,
        **logreg_params
    }
    # Retorna thresholds em scores BRUTOS (RAW)
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

def montar_dataset_cache(dataset_name: str,
                         X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series,
                         nomes_classes: List[str],
                         t_plus: float,
                         t_minus: float,
                         WR_REJECTION_COST: float,
                         test_size_atual: float,
                         model_params: Dict[str, Any],
                         metricas_dict: Dict[str, Any],
                         y_pred_test: np.ndarray,
                         decision_scores_test: np.ndarray,
                         rejected_mask: np.ndarray,
                         resultados_instancias: List[Dict[str, Any]]):
    
    scaler_params = {
        'min': [float(v) for v in model_params['scaler_params']['min']],
        'scale': [float(v) for v in model_params['scaler_params']['scale']]
    }
    feature_names = list(X_train.columns)
    coefs_ordered = [float(model_params['coefs'][col]) for col in feature_names]
    intercepto = float(model_params['intercepto'])
    X_test_dict = {str(col): [float(x) for x in X_test[col].tolist()] for col in X_test.columns}
    y_test_list = [int(v) for v in y_test.tolist()]

    per_instance = []
    for i, rid in enumerate(X_test.index):
        per_instance.append({
            'id': str(rid),
            'y_true': int(y_test.iloc[i]),
            'y_pred': int(y_pred_test[i]) if int(y_pred_test[i]) in (0, 1) else -1,
            'rejected': bool(rejected_mask[i]),
            'decision_score': float(decision_scores_test[i]),
            'explanation': list(resultados_instancias[i]['explicacao']),
            'explanation_size': int(resultados_instancias[i]['tamanho_explicacao'])
        })

    mnist_meta = {}
    if dataset_name == 'mnist':
        try:
            from data.datasets import MNIST_FEATURE_MODE, MNIST_SELECTED_PAIR
            mnist_meta = {
                'mnist_feature_mode': MNIST_FEATURE_MODE,
                'mnist_digit_pair': list(MNIST_SELECTED_PAIR) if MNIST_SELECTED_PAIR is not None else None
            }
        except Exception:
            mnist_meta = {}

    dataset_cache = {
        'config': {
            'dataset_name': dataset_name,
            'test_size': float(test_size_atual),
            'random_state': RANDOM_STATE,
            'rejection_cost': float(WR_REJECTION_COST),
            'subsample_size': float(DATASET_CONFIG.get(dataset_name, {}).get('subsample_size', 0.0)) if DATASET_CONFIG.get(dataset_name, {}).get('subsample_size') else None,
            **mnist_meta
        },
        'thresholds': {
            't_plus': float(t_plus),  # Threshold no espaço de scores brutos (RAW)
            't_minus': float(t_minus),  # Threshold no espaço de scores brutos (RAW)
            'rejection_zone_width': float(t_plus - t_minus),
            'note': 'Thresholds are in RAW score space (no normalization applied). Guarantees t- < 0 < t+ to preserve probabilistic interpretation.'
        },
        'performance': {
            'accuracy_without_rejection': float(metricas_dict['acuracia_sem_rejeicao']),
            'accuracy_with_rejection': float(metricas_dict['acuracia_com_rejeicao']),
            'rejection_rate': float(metricas_dict['taxa_rejeicao']),
            'num_test_instances': len(y_test),
            'num_rejected': int(np.sum(rejected_mask)),
            'num_accepted': int(len(y_test) - np.sum(rejected_mask))
        },
        'explanation_stats': {
            'positive': {
                'count': int(metricas_dict['stats_explicacao_positiva']['instancias']),
                'mean_length': float(metricas_dict['stats_explicacao_positiva']['media']),
                'std_length': float(metricas_dict['stats_explicacao_positiva']['std_dev']),
                'min_length': int(metricas_dict['stats_explicacao_positiva'].get('min', 0)),
                'max_length': int(metricas_dict['stats_explicacao_positiva'].get('max', 0))
            },
            'negative': {
                'count': int(metricas_dict['stats_explicacao_negativa']['instancias']),
                'mean_length': float(metricas_dict['stats_explicacao_negativa']['media']),
                'std_length': float(metricas_dict['stats_explicacao_negativa']['std_dev']),
                'min_length': int(metricas_dict['stats_explicacao_negativa'].get('min', 0)),
                'max_length': int(metricas_dict['stats_explicacao_negativa'].get('max', 0))
            },
            'rejected': {
                'count': int(metricas_dict['stats_explicacao_rejeitada']['instancias']),
                'mean_length': float(metricas_dict['stats_explicacao_rejeitada']['media']),
                'std_length': float(metricas_dict['stats_explicacao_rejeitada']['std_dev']),
                'min_length': int(metricas_dict['stats_explicacao_rejeitada'].get('min', 0)),
                'max_length': int(metricas_dict['stats_explicacao_rejeitada'].get('max', 0))
            }
        },
        'computation_time': {
            'total': float(metricas_dict.get('tempo_total', 0.0)),
            'mean_per_instance': float(metricas_dict.get('tempo_medio_instancia', 0.0)),
            'positive': float(metricas_dict.get('tempo_medio_positivas', 0.0)),
            'negative': float(metricas_dict.get('tempo_medio_negativas', 0.0)),
            'rejected': float(metricas_dict.get('tempo_medio_rejeitadas', 0.0))
        },
        'top_features': [
            {"feature": feat, "count": int(count)}
            for feat, count in metricas_dict.get('features_frequentes', [])[:20]  # Top 20
        ],
        'model': {
            'type': 'LogisticRegression',
            'num_features': len(feature_names),
            'class_names': list(nomes_classes),
            'params': {k: v for k, v in model_params.items() if k not in ['coefs', 'intercepto', 'scaler_params']},
            'coefs': coefs_ordered,
            'intercept': intercepto,
            'scaler_params': scaler_params
        },
        'per_instance': per_instance  # Adiciona detalhes de cada instância
    }
    return dataset_cache

def gerar_relatorio_texto(dataset_name, test_size, wr, modelo, t_plus, t_minus, num_test, num_features, metricas, resultados_instancias, model_params):
    """Gera relatório TXT completo (tempo não contabilizado no experimento)."""
    output_path = os.path.join(OUTPUT_BASE_DIR, f"peab_{dataset_name}.txt")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("          RELATÓRIO DE ANÁLISE - MÉTODO PEAB (EXPLAINABLE AI)\n")
        f.write("="*80 + "\n\n")
        
        # SEÇÃO 1: CONFIGURAÇÃO DO EXPERIMENTO
        f.write("-"*80 + "\n")
        f.write("1. CONFIGURAÇÃO DO EXPERIMENTO\n")
        f.write("-"*80 + "\n")
        f.write(f"  Dataset: {dataset_name}\n")
        f.write(f"  Instâncias de teste: {num_test}\n")
        f.write(f"  Features por instância: {num_features}\n")
        f.write(f"  Test size: {test_size:.2%}\n")
        f.write(f"  Custo de rejeição (WR): {wr:.4f}\n\n")
        
        # SEÇÃO 2: HIPERPARÂMETROS DO MODELO
        f.write("-"*80 + "\n")
        f.write("2. HIPERPARÂMETROS DO MODELO (Regressão Logística)\n")
        f.write("-"*80 + "\n")
        for k, v in model_params.items():
            if k not in ['coefs', 'intercepto', 'scaler_params']:
                f.write(f"  {k}: {v}\n")
        f.write(f"  Intercepto: {model_params.get('intercepto', 0.0):.6f}\n\n")
        
        # SEÇÃO 3: THRESHOLDS DE REJEIÇÃO
        f.write("-"*80 + "\n")
        f.write("3. THRESHOLDS DE REJEIÇÃO\n")
        f.write("-"*80 + "\n")
        f.write(f"  t+ (limiar superior): {t_plus:.6f}\n")
        f.write(f"  t- (limiar inferior): {t_minus:.6f}\n")
        f.write(f"  Largura da zona de rejeição: {t_plus - t_minus:.6f}\n\n")
        
        # SEÇÃO 4: DESEMPENHO DO MODELO
        f.write("-"*80 + "\n")
        f.write("4. DESEMPENHO DO MODELO\n")
        f.write("-"*80 + "\n")
        f.write(f"  Acurácia sem rejeição: {metricas['acuracia_sem_rejeicao']:.2f}%\n")
        f.write(f"  Acurácia com rejeição: {metricas['acuracia_com_rejeicao']:.2f}%\n")
        f.write(f"  Taxa de rejeição: {metricas['taxa_rejeicao']:.2f}%\n\n")
        
        # SEÇÃO 5: ESTATÍSTICAS DAS EXPLICAÇÕES
        f.write("-"*80 + "\n")
        f.write("5. ESTATÍSTICAS DAS EXPLICAÇÕES\n")
        f.write("-"*80 + "\n")
        for tipo_label, tipo_key in [('POSITIVAS', 'stats_explicacao_positiva'), 
                                      ('NEGATIVAS', 'stats_explicacao_negativa'), 
                                      ('REJEITADAS', 'stats_explicacao_rejeitada')]:
            stats = metricas[tipo_key]
            f.write(f"  {tipo_label}:\n")
            f.write(f"    Quantidade: {stats['instancias']}\n")
            f.write(f"    Tamanho médio: {stats['media']:.2f} features\n")
            f.write(f"    Desvio padrão: {stats['std_dev']:.2f}\n")
            f.write(f"    Mínimo: {stats['min']} features\n")
            f.write(f"    Máximo: {stats['max']} features\n\n")
        
        # SEÇÃO 6: TEMPOS DE EXECUÇÃO
        f.write("-"*80 + "\n")
        f.write("6. TEMPOS DE EXECUÇÃO (apenas geração de explicações)\n")
        f.write("-"*80 + "\n")
        f.write(f"  Tempo total: {metricas['tempo_total']:.4f}s\n")
        f.write(f"  Tempo médio por instância: {metricas['tempo_medio_instancia']:.6f}s\n")
        f.write(f"  Tempo médio POSITIVAS: {metricas['tempo_medio_positivas']:.6f}s\n")
        f.write(f"  Tempo médio NEGATIVAS: {metricas['tempo_medio_negativas']:.6f}s\n")
        f.write(f"  Tempo médio REJEITADAS: {metricas['tempo_medio_rejeitadas']:.6f}s\n\n")
        
        # SEÇÃO 7: TOP 10 FEATURES MAIS FREQUENTES
        f.write("-"*80 + "\n")
        f.write("7. TOP 10 FEATURES MAIS FREQUENTES NAS EXPLICAÇÕES\n")
        f.write("-"*80 + "\n")
        top_feats = metricas['features_frequentes'][:10]
        for feat, count in top_feats:
            freq_pct = (count / num_test * 100)
            f.write(f"  {feat}: {count} ocorrências ({freq_pct:.1f}%)\n")
        f.write("\n")
        
        # SEÇÃO 8: LOGS DETALHADOS (opcional, apenas para datasets pequenos)
        if num_test <= 50 and TECHNICAL_LOGS:
            f.write("-"*80 + "\n")
            f.write("8. LOGS DETALHADOS POR INSTÂNCIA\n")
            f.write("-"*80 + "\n\n")
            for r in resultados_instancias:
                if 'log_detalhado' in r:
                    for log_line in r['log_detalhado']:
                        f.write(f"{log_line}\n")
                    f.write(f"\n   --> RESULTADO FINAL (Instância #{r['id']}):\n")
                    f.write(f"       - EXPLICAÇÃO: {sorted(r['explicacao'])}\n\n")

def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n[INFO] Executando PEAB para: {dataset_name.upper()}")
    todos_hiperparametros = carregar_hiperparametros()
    X_full, y_full, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)

    params = DEFAULT_LOGREG_PARAMS.copy()
    if dataset_name in todos_hiperparametros and 'params' in todos_hiperparametros[dataset_name]:
        params.update(todos_hiperparametros[dataset_name]['params'])

    # [CORREÇÃO] Split Único e Consistente
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=RANDOM_STATE, stratify=y_full)

    # [CORREÇÃO CRÍTICA] Seleção de features sem data leakage
    # ANTES: Usava modelo treinado com rejection → data leakage!
    # DEPOIS: Usa teste estatístico (f_classif) independente do modelo final
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0 and top_k < X_train.shape[1]:
        print(f"\n[FEATURE_SELECTION] Selecionando top-{top_k} features (método: f_classif ANOVA)")
        print(f"[FEATURE_SELECTION] Features antes: {X_train.shape[1]}")
        
        # Seleção usando APENAS dados de treino (sem vazar informação)
        selector = SelectKBest(score_func=f_classif, k=top_k)
        selector.fit(X_train, y_train)  # APENAS TREINO!
        
        selected_indices = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices]
        
        # Aplicar seleção em treino E teste
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        print(f"[FEATURE_SELECTION] Features após: {X_train.shape[1]}")
        print(f"[FEATURE_SELECTION] Features selecionadas: {list(selected_features)[:10]}..." if len(selected_features) > 10 else f"[FEATURE_SELECTION] Features selecionadas: {list(selected_features)}")
        print()

    # Treino Final com dados consistentes
    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params, dataset_name)

    print(f"\n{'='*40}")
    print(f"THRESHOLDS CALCULADOS (scores brutos):")
    print(f" T+ (Plus):     {t_plus:.4f}")
    print(f" T- (Minus):    {t_minus:.4f}")
    print(f" Rejection Zone: {t_plus - t_minus:.4f}")
    print(f"{'='*40}\n")

    # USAR SCORES BRUTOS (SEM NORMALIZAÇÃO)
    # Previsões
    decision_scores = modelo.decision_function(X_test)
    norm_params = model_params.get('norm_params', {'max_abs': 1.0})
    
    y_pred = np.full(y_test.shape, -1, dtype=int)
    y_pred[decision_scores >= t_plus] = 1
    y_pred[decision_scores <= t_minus] = 0
    mask_rej = (y_pred == -1)
    y_pred_final = y_pred.copy()
    y_pred_final[mask_rej] = 2

    # Loop de Explicação
    print(f"[INFO] Explicando {len(X_test)} instâncias...")
    start_total = time.perf_counter()
    resultados = []
    
    t_pos, t_neg, t_rej = [], [], []
    ad_pos, ad_neg, ad_rej = [], [], []
    rm_pos, rm_neg, rm_rej = [], [], []
    
    with ProgressBar(total=len(X_test)) as pbar:
        for i in range(len(X_test)):
            start_inst = time.perf_counter()
            inst = X_test.iloc[[i]]
            expl, logs, ad, rm = gerar_explicacao_instancia(inst, modelo, X_train, t_plus, t_minus, norm_params, benchmark_mode=False)
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

    total_time_experimento = time.perf_counter() - start_total
    print(f"[INFO] Tempo de experimento (explicações): {total_time_experimento:.2f}s")
    
    # ========== GERAÇÃO DE RELATÓRIOS (TEMPO NÃO CONTABILIZADO) ==========
    print("[INFO] Gerando relatórios (JSON + TXT)...")
    start_relatorios = time.perf_counter()
    
    metricas = coletar_metricas(resultados, y_test, y_pred_final, mask_rej, total_time_experimento, model_params, modelo, X_test, X_train.columns, t_pos, t_neg, t_rej, ad_pos, ad_neg, ad_rej, rm_pos, rm_neg, rm_rej)
    
    # Monta e salva JSON
    dataset_json_key = dataset_name
    if dataset_name == 'mnist':
        cfg_mnist = DATASET_CONFIG.get('mnist', {})
        digit_pair = cfg_mnist.get('digit_pair')
        if digit_pair and len(digit_pair) == 2:
            dataset_json_key = f"mnist_{digit_pair[0]}_vs_{digit_pair[1]}"
    
    dataset_cache = montar_dataset_cache(
        dataset_name, X_train, X_test, y_train, y_test, nomes_classes,
        t_plus, t_minus, rejection_cost, test_size, model_params, metricas,
        y_pred_final, decision_scores, mask_rej, resultados
    )
    update_method_results('peab', dataset_json_key, dataset_cache)
    
    # Gera relatório TXT completo
    gerar_relatorio_texto(dataset_name, test_size, rejection_cost, modelo, t_plus, t_minus, len(X_test), X_test.shape[1], metricas, resultados, model_params)
    
    tempo_relatorios = time.perf_counter() - start_relatorios
    print(f"[INFO] Relatórios gerados em {tempo_relatorios:.2f}s (não contabilizado no experimento)")
    print(f"\n{'='*80}")
    print(f"[SUCESSO] Arquivos salvos:")
    print(f"  📊 JSON: json/peab_results.json")
    print(f"  📄 TXT:  {OUTPUT_BASE_DIR}/peab_{dataset_name}.txt")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    resultado = selecionar_dataset_e_classe()
    
    # Verificar se é seleção múltipla
    if resultado[0] == '__MULTIPLE__':
        datasets_lista = resultado[4]  # Lista de datasets está na 5ª posição
        
        print(f"\n{'='*80}")
        print(f"  EXECUÇÃO EM SEQUÊNCIA: {len(datasets_lista)} DATASETS")
        print(f"{'='*80}\n")
        
        for i, dataset_name in enumerate(datasets_lista, 1):
            print(f"\n{'─'*80}")
            print(f"  [{i}/{len(datasets_lista)}] Executando: {dataset_name.upper()}")
            print(f"{'─'*80}\n")
            
            try:
                executar_experimento_para_dataset(dataset_name)
                print(f"\n✅ [{i}/{len(datasets_lista)}] {dataset_name} concluído com sucesso!")
            
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Execução interrompida pelo usuário.")
                print(f"   Datasets concluídos: {i-1}/{len(datasets_lista)}")
                break
            
            except Exception as e:
                print(f"\n❌ [{i}/{len(datasets_lista)}] Erro em {dataset_name}: {e}")
                print(f"   Continuando com próximo dataset...\n")
                continue
        
        print(f"\n{'='*80}")
        print(f"  EXECUÇÃO EM SEQUÊNCIA FINALIZADA")
        print(f"  Datasets processados: {i}/{len(datasets_lista)}")
        print(f"{'='*80}\n")
    
    # Seleção única tradicional
    elif resultado[0]:
        executar_experimento_para_dataset(resultado[0])