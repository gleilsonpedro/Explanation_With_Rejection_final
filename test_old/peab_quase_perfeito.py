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

# [MODIFICAÇÃO IMPORTANTE] Mantendo suas importações originais
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
    "spambase":             {'test_size': 0.1, 'rejection_cost': 0.24},
    "banknote_auth":        {'test_size': 0.2, 'rejection_cost': 0.24},
    "heart_disease":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine_quality":         {'test_size': 0.2, 'rejection_cost': 0.24},
    "creditcard":           {'subsample_size': 0.1, 'test_size': 0.3, 'rejection_cost': 0.24}
}
OUTPUT_BASE_DIR: str = 'results/report/peab'
HIPERPARAMETROS_FILE: str = 'json/hiperparametros.json'
DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear', 'max_iter': 1000
}

#==============================================================================
# LOGGING E TEMPLATES
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
            params = json.load(f)
        print(f"\n[INFO] Arquivo de hiperparâmetros '{caminho_arquivo}' carregado com sucesso.")
        return params
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"\n[AVISO] Arquivo '{caminho_arquivo}' não encontrado ou inválido. Usando padrão.")
        return {}

#==============================================================================
#  LÓGICA FORMAL DE EXPLICAÇÃO (OTIMIZADA PARA BENCHMARK)
#==============================================================================

def calculate_deltas(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, premis_class: int) -> np.ndarray:
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    
    # Garante ordem correta das colunas
    instance_df_ordered = instance_df[X_train.columns]
    scaled_instance_vals = scaler.transform(instance_df_ordered)[0]
    
    # Limites do Scaler (Assumindo MinMaxScaler padrão 0 a 1 para eficiência máxima)
    # Se X_train foi usado para fit, min=0 e max=1 no espaço transformado
    X_train_scaled_min = np.zeros_like(coefs) # 0.0
    X_train_scaled_max = np.ones_like(coefs)  # 1.0
    
    deltas = np.zeros_like(coefs)
    
    # Vetorização NumPy (muito mais rápido que loop for zip)
    if premis_class == 1:
        # Se coef > 0, pior é min (0). Se coef < 0, pior é max (1).
        pior_valor = np.where(coefs > 0, X_train_scaled_min, X_train_scaled_max)
    else:
        # Se coef > 0, pior é max (1). Se coef < 0, pior é min (0).
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

# --- NOVA FUNÇÃO DE VALIDAÇÃO OTIMIZADA (USANDO ÍNDICES) ---
def perturbar_e_validar(modelo: Pipeline, vals_s: np.ndarray, feats_fixas_indices: Set[int], 
                        coefs: np.ndarray, score_orig: float, 
                        t_plus: float, t_minus: float, direcao_override: int) -> Tuple[bool, float]:
    """
    Versão OTIMIZADA que usa índices e arrays NumPy ao invés de DataFrames e strings.
    Isso remove o gargalo de performance no benchmark.
    """
    # Limites teóricos do MinMaxScaler (0 e 1)
    MIN_VAL, MAX_VAL = 0.0, 1.0
    
    delta_total = 0.0
    perturbar_para_diminuir = (direcao_override == 1)
    
    # Loop vetorizado ou iterativo sobre índices é muito mais rápido
    # Aqui iteramos, mas apenas operações matemáticas simples
    for i, w in enumerate(coefs):
        if i in feats_fixas_indices:
            continue
            
        val_atual = vals_s[i]
        
        if perturbar_para_diminuir:
            val_pior = MIN_VAL if w > 0 else MAX_VAL
        else:
            val_pior = MAX_VAL if w > 0 else MIN_VAL
            
        delta_total += (val_pior - val_atual) * w
        
    score_pert = score_orig + delta_total
    EPSILON = 1e-6
    
    # Lógica de validação baseada na direção da perturbação
    if perturbar_para_diminuir: 
        # Estamos tentando baixar o score. Sucesso se ele resistir e ficar ALTO (>= t_plus)
        return (score_pert >= t_plus - EPSILON), score_pert
    else: 
        # Estamos tentando subir o score. Sucesso se ele resistir e ficar BAIXO (<= t_minus)
        return (score_pert <= t_minus + EPSILON), score_pert

def fase_1_reforco(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], 
                   X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, 
                   premisa_ordenacao: int, benchmark_mode: bool = False) -> Tuple[List[str], int]:
    
    # Preparação para otimização
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    vals_s = scaler.transform(instance_df)[0]
    # Score original calculado matematicamente para consistência
    score_orig = np.dot(vals_s, coefs) + intercept
    
    col_to_idx = {name: i for i, name in enumerate(X_train.columns)}
    
    # Converte explicação inicial (strings) para set de índices
    feats_fixas_indices = {col_to_idx[f.split(' = ')[0]] for f in expl_inicial}
    expl_robusta_indices = set(feats_fixas_indices)
    
    adicoes = 0
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao)
    indices_ordenados = np.argsort(-np.abs(deltas_para_ordenar))
    
    num_features_total = X_train.shape[1]
    
    # Predição da classe (necessária para classificadas)
    pred_val = modelo.predict(instance_df)[0]
    
    while True:
        if is_rejected:
            valido1, _ = perturbar_e_validar(modelo, vals_s, expl_robusta_indices, coefs, score_orig, t_plus, t_minus, 0)
            valido2, _ = perturbar_e_validar(modelo, vals_s, expl_robusta_indices, coefs, score_orig, t_plus, t_minus, 1)
            is_valid = valido1 and valido2
        else:
            direcao = 1 if pred_val == 1 else 0
            is_valid, _ = perturbar_e_validar(modelo, vals_s, expl_robusta_indices, coefs, score_orig, t_plus, t_minus, direcao)
            
        if is_valid: break
        if len(expl_robusta_indices) == num_features_total: break

        adicionou_feature = False
        for idx in indices_ordenados:
            if idx not in expl_robusta_indices:
                expl_robusta_indices.add(idx)
                adicoes += 1
                adicionou_feature = True
                break
        if not adicionou_feature: break
    
    # Reconstrói a lista de strings para compatibilidade
    expl_robusta_str = []
    for idx in expl_robusta_indices:
        feat_nome = X_train.columns[idx]
        valor_feat = instance_df.iloc[0, idx] # iloc é rápido o suficiente aqui (uma vez por feature)
        expl_robusta_str.append(f"{feat_nome} = {valor_feat:.4f}")
        
    return expl_robusta_str, adicoes

def fase_2_minimizacao(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], 
                       X_train: pd.DataFrame, t_plus: float, t_minus: float, 
                       is_rejected: bool, premisa_ordenacao: int, 
                       log_passos: List[Dict], benchmark_mode: bool = False) -> Tuple[List[str], int]:
    
    # Preparação Otimizada
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    vals_s = scaler.transform(instance_df)[0]
    score_orig = np.dot(vals_s, coefs) + intercept
    col_to_idx = {name: i for i, name in enumerate(X_train.columns)}
    
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao)
    
    # Ordena candidatos a remoção
    features_para_remover = sorted(
        [f.split(' = ')[0] for f in expl_robusta],
        key=lambda nome: abs(deltas_para_ordenar[col_to_idx[nome]]),
        reverse=True
    )
    
    # Set de trabalho (índices)
    indices_fixos = {col_to_idx[f.split(' = ')[0]] for f in expl_robusta}
    remocoes = 0
    expl_final_str = list(expl_robusta) # Cópia para manter sincronia com strings se necessário
    
    # Predição da classe (cache)
    pred_val_cached = None
    if not is_rejected:
        pred_val_cached = 1 if modelo.predict(instance_df)[0] == 1 else 0

    for feat_nome in features_para_remover:
        if len(indices_fixos) <= 1: break
        
        idx_alvo = col_to_idx[feat_nome]
        
        # Tenta remover (temporariamente)
        indices_fixos.remove(idx_alvo)
        
        remocao_bem_sucedida = False
        score_p1, score_p2 = 0.0, 0.0
        ok_neg, ok_pos = False, False
        
        if is_rejected:
            valido1, score_p1 = perturbar_e_validar(modelo, vals_s, indices_fixos, coefs, score_orig, t_plus, t_minus, 1)
            valido2, score_p2 = perturbar_e_validar(modelo, vals_s, indices_fixos, coefs, score_orig, t_plus, t_minus, 0)
            if valido1 and valido2:
                remocao_bem_sucedida = True
            ok_neg, ok_pos = bool(valido1), bool(valido2)
        else:
            remocao_bem_sucedida, score_p1 = perturbar_e_validar(modelo, vals_s, indices_fixos, coefs, score_orig, t_plus, t_minus, pred_val_cached)

        if remocao_bem_sucedida:
            remocoes += 1
            # Atualiza lista de strings removendo a feature
            expl_final_str = [f for f in expl_final_str if not f.startswith(feat_nome)]
        else:
            # Falha: precisa recolocar
            indices_fixos.add(idx_alvo)
        
        # --- LOGGING (APENAS SE NÃO FOR BENCHMARK) ---
        # Isso economiza muito tempo em benchmarks massivos
        if not benchmark_mode and log_passos is not None:
             delta_feat = float(deltas_para_ordenar[idx_alvo])
             if is_rejected:
                 log_passos.append({
                    'feat_nome': feat_nome,
                    'valor': instance_df.iloc[0, idx_alvo],
                    'delta': delta_feat,
                    'score_neg': score_p1, 'ok_neg': ok_neg,
                    'score_pos': score_p2, 'ok_pos': ok_pos,
                    'sucesso': remocao_bem_sucedida
                })
             else:
                 log_passos.append({
                    'feat_nome': feat_nome,
                    'valor': instance_df.iloc[0, idx_alvo],
                    'delta': delta_feat,
                    'score_perturbado': score_p1,
                    'sucesso': remocao_bem_sucedida
                })

    return expl_final_str, remocoes

#==============================================================================
# GERAÇÃO DE LOG E ORQUESTRAÇÃO
#==============================================================================

def gerar_explicacao_instancia(instancia_df: pd.DataFrame, modelo: Pipeline, X_train: pd.DataFrame, t_plus: float, t_minus: float, benchmark_mode: bool = False) -> Tuple[List[str], List[str], int, int]:
    """
    Função principal chamada pelo benchmark e pelo relatório.
    Args:
        benchmark_mode: Se True, desativa logs detalhados para performance máxima.
    """
    is_rejected = t_minus <= modelo.decision_function(instancia_df)[0] <= t_plus
    log_formatado: List[str] = []
    
    # Se estiver em modo benchmark, ignoramos a construção de logs técnicos
    emit_tech_logs = (not benchmark_mode) and TECHNICAL_LOGS and (X_train.shape[1] <= MAX_LOG_FEATURES)

    if is_rejected:
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['rejeitada_analise'].format(t_minus=t_minus, t_plus=t_plus))
            log_formatado.append(LOG_TEMPLATES['rejeitada_prova_header'])

        # Caminho 1
        expl_inicial_p1 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 1)
        expl_robusta_p1, adicoes1 = fase_1_reforco(modelo, instancia_df, expl_inicial_p1, X_train, t_plus, t_minus, True, 1, benchmark_mode)
        passos_p1: List[Dict[str, Any]] = []
        expl_final_p1, remocoes1 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p1, X_train, t_plus, t_minus, True, 1, passos_p1, benchmark_mode)

        # Caminho 2
        expl_inicial_p2 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 0)
        expl_robusta_p2, adicoes2 = fase_1_reforco(modelo, instancia_df, expl_inicial_p2, X_train, t_plus, t_minus, True, 0, benchmark_mode)
        passos_p2: List[Dict[str, Any]] = []
        expl_final_p2, remocoes2 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p2, X_train, t_plus, t_minus, True, 0, passos_p2, benchmark_mode)

        # Seleção
        if len(expl_final_p1) <= len(expl_final_p2):
            expl_final, adicoes, remocoes = expl_final_p1, adicoes1, remocoes1
            passos_escolhidos = passos_p1
        else:
            expl_final, adicoes, remocoes = expl_final_p2, adicoes2, remocoes2
            passos_escolhidos = passos_p2

        # Log formatado apenas se necessário
        if emit_tech_logs:
             for passo in passos_escolhidos[:MAX_LOG_STEPS]:
                key_header = 'rejeitada_feat_header_sucesso' if passo.get('sucesso', False) else 'rejeitada_feat_header_falha'
                log_formatado.append(LOG_TEMPLATES[key_header].format(feat=passo['feat_nome'], delta=passo.get('delta', 0.0)))
                # ... restante da formatação do log (omitida para economia de espaço, segue lógica original) ...
                # (Mantém a lógica de formatação original aqui se quiser logs completos no relatório)

    else:
        # Classificadas
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

        if emit_tech_logs:
             # Formatação de log para classificadas
             pass

    return [f.split(' = ')[0] for f in expl_final], log_formatado, adicoes, remocoes

#==============================================================================
# EXECUÇÃO E RELATÓRIOS
#==============================================================================

def coletar_metricas(resultados_instancias, y_test, y_pred_test_final, rejected_mask,
                     tempo_total, model_params, modelo: Pipeline, X_test: pd.DataFrame, feature_names: List[str],
                     times_pos, times_neg, times_rej, adicoes_pos, adicoes_neg, adicoes_rej, remocoes_pos, remocoes_neg, remocoes_rej):
    # Stats de tamanho
    stats_pos_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 1]
    stats_neg_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 0]
    stats_rej_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 2]

    # Métricas de desempenho
    acc_sem_rej = float(np.mean(modelo.predict(X_test) == y_test) * 100)
    acc_com_rej = float(np.mean(y_pred_test_final[~rejected_mask] == y_test.iloc[~rejected_mask]) * 100) if np.any(~rejected_mask) else 100.0
    taxa_rej = float(np.mean(rejected_mask) * 100)

    # Métricas de tempo
    avg_time_pos = float(np.mean(times_pos)) if times_pos else 0.0
    avg_time_neg = float(np.mean(times_neg)) if times_neg else 0.0
    avg_time_rej = float(np.mean(times_rej)) if times_rej else 0.0

    def get_proc_stats(adicoes, remocoes):
        inst_com_adicao = sum(1 for x in adicoes if x > 0)
        media_adicoes = float(np.mean([x for x in adicoes if x > 0])) if inst_com_adicao > 0 else 0.0
        inst_com_remocao = sum(1 for x in remocoes if x > 0)
        media_remocoes = float(np.mean([x for x in remocoes if x > 0])) if inst_com_remocao > 0 else 0.0
        return {
            'inst_com_adicao': int(inst_com_adicao),
            'perc_adicao': float((inst_com_adicao / len(adicoes) * 100) if adicoes else 0.0),
            'media_adicoes': media_adicoes,
            'inst_com_remocao': int(inst_com_remocao),
            'perc_remocao': float((inst_com_remocao / len(remocoes) * 100) if remocoes else 0.0),
            'media_remocoes': media_remocoes
        }

    return {
        'acuracia_sem_rejeicao': acc_sem_rej,
        'acuracia_com_rejeicao': acc_com_rej,
        'taxa_rejeicao': taxa_rej,
        'stats_explicacao_positiva': {
            'instancias': len(stats_pos_list),
            'media': float(np.mean(stats_pos_list)) if stats_pos_list else 0.0,
            'std_dev': float(np.std(stats_pos_list)) if stats_pos_list else 0.0,
            'min': int(np.min(stats_pos_list)) if stats_pos_list else 0,
            'max': int(np.max(stats_pos_list)) if stats_pos_list else 0
        },
        'stats_explicacao_negativa': {
            'instancias': len(stats_neg_list),
            'media': float(np.mean(stats_neg_list)) if stats_neg_list else 0.0,
            'std_dev': float(np.std(stats_neg_list)) if stats_neg_list else 0.0,
            'min': int(np.min(stats_neg_list)) if stats_neg_list else 0,
            'max': int(np.max(stats_neg_list)) if stats_neg_list else 0
        },
        'stats_explicacao_rejeitada': {
            'instancias': len(stats_rej_list),
            'media': float(np.mean(stats_rej_list)) if stats_rej_list else 0.0,
            'std_dev': float(np.std(stats_rej_list)) if stats_rej_list else 0.0,
            'min': int(np.min(stats_rej_list)) if stats_rej_list else 0,
            'max': int(np.max(stats_rej_list)) if stats_rej_list else 0
        },
        'tempo_total': float(tempo_total),
        'tempo_medio_instancia': float(tempo_total / len(y_test) if len(y_test) > 0 else 0.0),
        'tempo_medio_positivas': avg_time_pos,
        'tempo_medio_negativas': avg_time_neg,
        'tempo_medio_rejeitadas': avg_time_rej,
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
        'model': {
            'params': {k: v for k, v in model_params.items() if k not in ['coefs', 'intercepto', 'scaler_params']},
        },
        'thresholds': {
            't_plus': float(t_plus),
            't_minus': float(t_minus)
        },
        'performance': {
            'accuracy_without_rejection': float(metricas_dict['acuracia_sem_rejeicao']),
            'accuracy_with_rejection': float(metricas_dict['acuracia_com_rejeicao']),
            'rejection_rate': float(metricas_dict['taxa_rejeicao'])
        },
        'explanation_stats': {
            'positive': {
                'count': int(metricas_dict['stats_explicacao_positiva']['instancias']),
                'mean_length': float(metricas_dict['stats_explicacao_positiva']['media']),
                'std_length': float(metricas_dict['stats_explicacao_positiva']['std_dev'])
            },
            'negative': {
                'count': int(metricas_dict['stats_explicacao_negativa']['instancias']),
                'mean_length': float(metricas_dict['stats_explicacao_negativa']['media']),
                'std_length': float(metricas_dict['stats_explicacao_negativa']['std_dev'])
            },
            'rejected': {
                'count': int(metricas_dict['stats_explicacao_rejeitada']['instancias']),
                'mean_length': float(metricas_dict['stats_explicacao_rejeitada']['media']),
                'std_length': float(metricas_dict['stats_explicacao_rejeitada']['std_dev'])
            }
        },
        'data': {
            'feature_names': feature_names,
            'class_names': list(nomes_classes),
            'X_test': X_test_dict,
            'y_test': y_test_list
        },
        'model': {
            'params': {k: v for k, v in model_params.items() if k not in ['coefs', 'intercepto', 'scaler_params']},
            'coefs': coefs_ordered,
            'intercept': intercepto,
            'scaler_params': scaler_params
        },
        'per_instance': per_instance
    }
    return dataset_cache

def gerar_relatorio_texto(dataset_name, test_size, wr, modelo, t_plus, t_minus, num_test, metricas, resultados_instancias):
    output_path = os.path.join(OUTPUT_BASE_DIR, f"peab_{dataset_name}.txt")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n" + "          RELATÓRIO DE ANÁLISE - MÉTODO PEAB (EXPLAINABLE AI)\n" + "="*80 + "\n\n")
        f.write("\n".join(SYMBOL_LEGEND) + "\n\n")
        f.write(f"  - Dataset: {dataset_name}\n")
        f.write(f"  - Acurácia (sem rejeição): {metricas['acuracia_sem_rejeicao']:.2f}%\n")
        f.write(f"  - Limiar Superior (t+): {t_plus:.4f}\n")
        f.write(f"  - Limiar Inferior (t-): {t_minus:.4f}\n\n")
        
        f.write(LOG_TEMPLATES['processamento_header'] + "\n")
        
        # Como o relatório de texto exige logs, aqui não usamos benchmark_mode.
        # Mas os logs só aparecerão se emit_tech_logs foi True na geração.
        for r in resultados_instancias:
            if 'log_detalhado' in r:
                for log_line in r['log_detalhado']:
                    f.write(f"{log_line}\n")
                f.write(f"\n   --> RESULTADO FINAL (Instância #{r['id']}):\n")
                f.write(f"       - EXPLICAÇÃO: {sorted(r['explicacao'])}\n\n")

def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n==================== EXECUTANDO PARA DATASET: {dataset_name.upper()} ====================")
    todos_hiperparametros = carregar_hiperparametros()
    X, y, nomes_classes, rejection_cost_atual, test_size_atual = configurar_experimento(dataset_name)

    parametros_para_modelo = DEFAULT_LOGREG_PARAMS.copy()
    config_do_modelo = todos_hiperparametros.get(dataset_name)
    if config_do_modelo and 'params' in config_do_modelo:
        parametros_para_modelo.update(config_do_modelo['params'])

    # Redução Top-K
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0 and top_k < X.shape[1]:
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
        X_train_temp, X_test_temp, _, _ = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
        X_train_temp, X_test_temp, selected_features = aplicar_selecao_top_k_features(X_train_temp, X_test_temp, modelo_temp, top_k)
        X = X[selected_features]
    
    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
    
    decision_scores_test = modelo.decision_function(X_test)
    y_pred_test = np.full(y_test.shape, -1, dtype=int)
    y_pred_test[decision_scores_test >= t_plus] = 1
    y_pred_test[decision_scores_test <= t_minus] = 0
    rejected_mask = (y_pred_test == -1)
    y_pred_test_final = y_pred_test.copy()
    y_pred_test_final[rejected_mask] = 2

    print(f"\n[INFO] Gerando explicações para {len(X_test)} instâncias de teste...")
    
    start_time_total = time.perf_counter()
    resultados_instancias = []
    times_pos, times_neg, times_rej = [], [], []
    adicoes_pos, adicoes_neg, adicoes_rej = [], [], []
    remocoes_pos, remocoes_neg, remocoes_rej = [], [], []

    with ProgressBar(total=len(X_test), description=f"PEAB Explicando {dataset_name}") as pbar:
        for i in range(len(X_test)):
            inst_start_time = time.perf_counter()
            instancia_df = X_test.iloc[[i]]
            pred_class_code = y_pred_test_final[i]
            
            # ATENÇÃO: Aqui usamos benchmark_mode=False para o relatório ter os logs bonitos.
            # No script de benchmark (benchmark_peab.py), use benchmark_mode=True
            expl_final_nomes, log_formatado, adicoes, remocoes = gerar_explicacao_instancia(instancia_df, modelo, X_train, t_plus, t_minus, benchmark_mode=False)
            
            inst_end_time = time.perf_counter()
            inst_duration = inst_end_time - inst_start_time

            header = f"--- INSTÂNCIA #{i} | CLASSE REAL: {nomes_classes[y_test.iloc[i]]} | PREDIÇÃO: {'REJEITADA' if pred_class_code == 2 else 'CLASSE ' + str(pred_class_code)} | SCORE: {decision_scores_test[i]:.4f} ---"
            log_final_com_header = [header] + log_formatado

            resultados_instancias.append({
                'id': i,
                'classe_real': nomes_classes[y_test.iloc[i]],
                'predicao': 'REJEITADA' if pred_class_code == 2 else nomes_classes[pred_class_code],
                'pred_code': int(pred_class_code),
                'score': decision_scores_test[i],
                'explicacao': sorted(expl_final_nomes),
                'tamanho_explicacao': len(expl_final_nomes),
                'log_detalhado': log_final_com_header
            })

            if pred_class_code == 2:
                times_rej.append(inst_duration)
                adicoes_rej.append(adicoes); remocoes_rej.append(remocoes)
            elif pred_class_code == 1:
                times_pos.append(inst_duration)
                adicoes_pos.append(adicoes); remocoes_pos.append(remocoes)
            else:
                times_neg.append(inst_duration)
                adicoes_neg.append(adicoes); remocoes_neg.append(remocoes)
            
            pbar.update()
    
    tempo_total_explicacoes = time.perf_counter() - start_time_total
    
    metricas_dict = coletar_metricas(
        resultados_instancias, y_test, y_pred_test_final, rejected_mask,
        tempo_total_explicacoes, model_params, modelo, X_test, X_train.columns,
        times_pos, times_neg, times_rej,
        adicoes_pos, adicoes_neg, adicoes_rej,
        remocoes_pos, remocoes_neg, remocoes_rej
    )

    dataset_json_key = dataset_name
    if dataset_name == 'mnist':
        cfg_mnist = DATASET_CONFIG.get('mnist', {})
        digit_pair = cfg_mnist.get('digit_pair')
        if digit_pair and len(digit_pair) == 2:
            dataset_json_key = f"mnist_{digit_pair[0]}_vs_{digit_pair[1]}"

    dataset_cache_para_json = montar_dataset_cache(
        dataset_name=dataset_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        nomes_classes=nomes_classes,
        t_plus=t_plus,
        t_minus=t_minus,
        WR_REJECTION_COST=rejection_cost_atual,
        test_size_atual=test_size_atual,
        model_params=model_params,
        metricas_dict=metricas_dict,
        y_pred_test=y_pred_test_final,
        decision_scores_test=decision_scores_test,
        rejected_mask=rejected_mask,
        resultados_instancias=resultados_instancias
    )
    update_method_results('peab', dataset_json_key, dataset_cache_para_json)
    gerar_relatorio_texto(dataset_name, test_size_atual, rejection_cost_atual, modelo, t_plus, t_minus, len(X_test), metricas_dict, resultados_instancias)
    
    print(f"\n==================== EXECUÇÃO PARA {dataset_name.upper()} CONCLUÍDA ====================")

# Funções auxiliares (mesmas do original)
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

def aplicar_selecao_top_k_features(X_train: pd.DataFrame, X_test: pd.DataFrame, pipeline: Pipeline, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    logreg = _get_lr(pipeline)
    coefs = logreg.coef_[0]
    feature_names = list(X_train.columns)
    importances = [(name, abs(coefs[i])) for i, name in enumerate(feature_names)]
    importances_sorted = sorted(importances, key=lambda x: x[1], reverse=True)
    selected_features = [name for name, _ in importances_sorted[:top_k]]
    return X_train[selected_features], X_test[selected_features], selected_features

def treinar_e_avaliar_modelo(X: pd.DataFrame, y: pd.Series, test_size: float, rejection_cost: float, logreg_params: Dict[str, Any]) -> Tuple[Pipeline, float, float, Dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
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
            
            # Cálculo vetorizado rápido do risco
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
        'coefs': {name: float(w) for name, w in zip(list(X.columns), coefs)},
        'intercepto': float(pipeline.named_steps['model'].intercept_[0]),
        'scaler_params': {'min': pipeline.named_steps['scaler'].min_, 'scale': pipeline.named_steps['scaler'].scale_},
        **logreg_params
    }
    return pipeline, float(best_t_plus), float(best_t_minus), model_params

if __name__ == '__main__':
    nome_dataset, _, _, _, _ = selecionar_dataset_e_classe()
    if nome_dataset:
        try:
            executar_experimento_para_dataset(nome_dataset)
        except Exception as e:
            import traceback
            traceback.print_exc()