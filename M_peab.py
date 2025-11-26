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
from typing import List, Tuple, Dict, Any

# [MODIFICAÇÃO IMPORTANTE] Mantendo suas importações originais
# Certifique-se que esses arquivos existem no seu projeto
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
from utils.results_handler import update_method_results

#==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
#==============================================================================
RANDOM_STATE: int = 42

# Configurações específicas de MNIST
MNIST_CONFIG = {
    'feature_mode': 'raw',           
    'digit_pair': (9, 4),            
    'top_k_features': None,          
    'test_size': 0.3,                
    'rejection_cost': 0.24,          
    'subsample_size': 0.05           
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

# [MODIFICAÇÃO] Nomes de arquivos alterados para evitar sobrescrita
OUTPUT_BASE_DIR: str = 'results/report/peab' 
HIPERPARAMETROS_FILE: str = 'json/hiperparametros.json'
JSON_RESULTS_FILE: str = 'json/M_comparative_results.json' # Nome personalizado para o JSON

DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2', 'C': 0.01, 'solver': 'liblinear', 'max_iter': 1000
}

#==============================================================================
# FUNÇÃO AUXILIAR PARA SALVAR JSON CUSTOMIZADO
#==============================================================================
def save_custom_json_results(method: str, dataset: str, results: Dict[str, Any], filepath: str) -> None:
    """Salva resultados em um arquivo JSON específico."""
    
    def _to_builtin_local(obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _to_builtin_local(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_builtin_local(v) for v in obj]
        return obj

    data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[AVISO] Erro ao ler {filepath}: {e}. Criando novo.")
    
    if method not in data:
        data[method] = {}
    
    data[method][dataset] = results
    
    serializable = _to_builtin_local(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Resultados salvos em {filepath}")

#==============================================================================
# CONTROLES E TEMPLATES DE LOG TÉCNICO
#==============================================================================

TECHNICAL_LOGS: bool = True
MAX_LOG_FEATURES: int = 200  
MAX_LOG_STEPS: int = 60      

SYMBOL_LEGEND = [
    "LEGENDA DOS SÍMBOLOS",
    "   δ  = Contribuição individual (w_i × x_i) [Usado para Classes 0 e 1]",
    "   |w|= Peso Absoluto / Risco [Usado para Classe Rejeitada]",
    "   Σδ = Soma acumulada (intercepto + Σδ_i)",
    "   ●  = Feature mantida (essencial)",
    "   ○  = Feature removida (não essencial)",
    "   s' = Valor da feature no pior cenário (worst case)"
]

LOG_TEMPLATES = {
    'processamento_header': "********** PROCESSAMENTO POR INSTÂNCIA  **********\n",
    'classificada_analise': "├── Análise: Score está {posicao}. Buscando o menor conjunto via DELTA.",
    'classificada_min_inicio': "├── Iniciando minimização com {num_features} features.",
    'classificada_ordem': "├── Tentativas de desafixação (ordem de maior impacto |δ|): {lista}",
    'classificada_step_sucesso': "├─ ○ {feat} (δ: {delta:+.3f}): s' = {score:.3f} ({cond}) → SUCESSO. DESAFIXADA.",
    'classificada_step_falha': "├─ ● {feat} (δ: {delta:+.3f}): s' = {score:.3f} ({cond}) → FALHA. ESSENCIAL.",

    'rejeitada_analise': "├── Zona de Rejeição: [{t_minus:.4f}, {t_plus:.4f}]",
    'rejeitada_prova_header': "├── Prova de Minimalidade (Estratégia: Redução de Risco/Variância):",
    # Templates adaptados para exibir Risco (|w|) em vez de Delta na rejeição
    'rejeitada_feat_header_sucesso': "├─ ○ {feat} (Risco |w|: {delta:.3f}):",
    'rejeitada_feat_header_falha': "├─ ● {feat} (Risco |w|: {delta:.3f}):",
    'rejeitada_subteste_neg': "│   ├─ Teste vs Lado Negativo: s' = {score:.3f} ({cmp}) {ok}",
    'rejeitada_subteste_pos': "│   └─ Teste vs Lado Positivo: s' = {score:.3f} ({cmp}) {ok}",
    'rejeitada_feat_footer_sucesso': "│   └─> SUCESSO. Feature DESAFIXADA.",
    'rejeitada_feat_footer_falha': "│   └─> FALHA. Feature ESSENCIAL (precisa ser fixada).",
}

DISABLE_REFORCO_CLASSIFICADAS: bool = True
MIN_REJECTION_WIDTH: float = 0.0

def _get_lr(modelo: Pipeline):
    """Retorna a etapa de Regressão Logística independente do nome do passo."""
    if 'model' in modelo.named_steps:
        return modelo.named_steps['model']
    if 'modelo' in modelo.named_steps:
        return modelo.named_steps['modelo']
    raise KeyError("Nenhum passo de regressão logística encontrado no Pipeline")

# [NOVA FUNÇÃO AUXILIAR]
def _get_abs_weights(modelo: Pipeline) -> np.ndarray:
    """Retorna os pesos absolutos (Risco) de cada feature."""
    logreg = _get_lr(modelo)
    return np.abs(logreg.coef_[0])

#==============================================================================
#  LÓGICA FORMAL DE EXPLICAÇÃO (CORE ATUALIZADO)
#==============================================================================
def carregar_hiperparametros(caminho_arquivo: str = HIPERPARAMETROS_FILE) -> dict:
    try:
        with open(caminho_arquivo, 'r') as f:
            params = json.load(f)
        print(f"\n[INFO] Arquivo de hiperparâmetros '{caminho_arquivo}' carregado com sucesso.")
        return params
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"\n[AVISO] Arquivo '{caminho_arquivo}' não encontrado ou corrompido. Usando padrão.")
        return {}

def calculate_deltas(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, premis_class: int) -> np.ndarray:
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    instance_df_ordered = instance_df[X_train.columns]
    scaled_instance_vals = scaler.transform(instance_df_ordered)[0]
    X_train_scaled = scaler.transform(X_train) 
    X_train_scaled_min = X_train_scaled.min(axis=0) 
    X_train_scaled_max = X_train_scaled.max(axis=0)
    deltas = np.zeros_like(coefs)
    for i, (coef, scaled_val) in enumerate(zip(coefs, scaled_instance_vals)):
        if premis_class == 1:
            pior_valor_escalonado = X_train_scaled_min[i] if coef > 0 else X_train_scaled_max[i]
        else:
            pior_valor_escalonado = X_train_scaled_max[i] if coef > 0 else X_train_scaled_min[i]
        deltas[i] = (scaled_val - pior_valor_escalonado) * coef
    return deltas

# [FUNÇÃO REESCRITA COM NOVA LÓGICA DE RISCO]
def one_explanation_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, premis_class: int) -> List[str]:
    score = modelo.decision_function(instance_df)[0]
    
    # Verifica se a instância está DE FATO na zona de rejeição
    is_in_rejection_zone = (t_minus <= score <= t_plus)
    
    # Se premis_class for rejeição (no código original chamamos com 0 ou 1, 
    # mas aqui detectamos automaticamente pelo score para aplicar a estratégia correta)
    use_rejection_strategy = is_in_rejection_zone

    explicacao = []
    
    if use_rejection_strategy:
        # --- ESTRATÉGIA INOVADORA PARA REJEIÇÃO (Baseada em Peso/Risco) ---
        # 1. Pega os pesos absolutos (Potencial de Variação)
        risks = _get_abs_weights(modelo)
        
        # 2. Ordena features do MAIOR peso para o MENOR peso (os mais perigosos primeiro)
        indices_ordenados = np.argsort(-risks)
        
        # 3. Calcula quanto o score pode "balançar" se tudo estiver solto (Pior cenário global)
        current_possible_swing = np.sum(risks)
        
        # 4. Distâncias para as bordas da zona de rejeição
        dist_to_top = t_plus - score
        dist_to_bottom = score - t_minus
        
        for i in indices_ordenados:
            # Se o balanço restante (das features ainda livres) for menor que a distância 
            # para ambas as bordas, estamos seguros.
            if (current_possible_swing < dist_to_top) and (current_possible_swing < dist_to_bottom):
                break

            # Se ainda não é seguro, trava a feature atual
            feature_nome = X_train.columns[i]
            valor_original_feature = instance_df.iloc[0, X_train.columns.get_loc(feature_nome)]
            explicacao.append(f"{feature_nome} = {valor_original_feature:.4f}")
            
            # Ao fixar, removemos o risco dela da equação
            current_possible_swing -= risks[i]

    else:
        # --- ESTRATÉGIA ORIGINAL PARA CLASSIFICAÇÃO (Baseada em Delta) ---
        deltas = calculate_deltas(modelo, instance_df, X_train, premis_class)
        indices_ordenados = np.argsort(-np.abs(deltas))
        
        score_base = score - np.sum(deltas)
        soma_deltas_cumulativa = score_base 
        target_score = t_plus if premis_class == 1 else t_minus
        
        for i in indices_ordenados:
            feature_nome = X_train.columns[i]
            valor_original_feature = instance_df.iloc[0, X_train.columns.get_loc(feature_nome)]
            
            if abs(deltas[i]) > 1e-9:
                 soma_deltas_cumulativa += deltas[i]
                 explicacao.append(f"{feature_nome} = {valor_original_feature:.4f}")
            
            if (premis_class == 1 and soma_deltas_cumulativa > target_score and explicacao) or \
               (premis_class == 0 and soma_deltas_cumulativa < target_score and explicacao):
                break
                
    # Fallback de segurança (garante pelo menos 1 feature)
    if not explicacao and len(X_train.columns) > 0:
         if use_rejection_strategy:
             # Para rejeição, fallback é a feature de maior peso
             metrica = _get_abs_weights(modelo)
         else:
             # Para classificação, fallback é a feature de maior delta
             metrica = np.abs(calculate_deltas(modelo, instance_df, X_train, premis_class))
             
         idx_max = np.argmax(metrica)
         feat_nome = X_train.columns[idx_max]
         valor_feat = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
         explicacao.append(f"{feat_nome} = {valor_feat:.4f}")

    return explicacao

def perturbar_e_validar(modelo: Pipeline, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, direcao_override: int) -> Tuple[bool, float]:
    if not explicacao:
        return False, 0.0
    inst_pert = instance_df.copy()
    features_explicacao = {f.split(' = ')[0] for f in explicacao}
    
    perturbar_para_diminuir_score = (direcao_override == 1)
    modelo_interno = _get_lr(modelo)
    X_train_min = X_train.min(axis=0) 
    X_train_max = X_train.max(axis=0)
    
    for feat_idx, feat_nome in enumerate(X_train.columns):
        if feat_nome in features_explicacao:
            continue 
        coef = modelo_interno.coef_[0][feat_idx] 
        valor_pert = (X_train_min[feat_nome] if coef > 0 else X_train_max[feat_nome]) if perturbar_para_diminuir_score else (X_train_max[feat_nome] if coef > 0 else X_train_min[feat_nome])
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
        
    score_pert = modelo.decision_function(inst_pert)[0]
    pert_rejeitada = t_minus <= score_pert <= t_plus
   
    score_original = modelo.decision_function(instance_df)[0]
    is_original_rejected = t_minus <= score_original <= t_plus
    
    if is_original_rejected:
        return pert_rejeitada, score_pert
    else:
        pred_original_class = int(modelo.predict(instance_df)[0])
        if pred_original_class == 1:
            return (score_pert >= t_plus), score_pert
        else:
            return (score_pert <= t_minus), score_pert

# [FUNÇÃO REESCRITA COM NOVA LÓGICA DE ORDENAÇÃO]
def fase_1_reforco(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int) -> Tuple[List[str], int]:
    expl_robusta = list(expl_inicial)
    adicoes = 0
    
    # Seleção da métrica de ordenação para tentativa de adição
    if is_rejected:
        # Rejeição: Ordena por PESO/RISCO (do maior para o menor)
        metricas_ordenacao = _get_abs_weights(modelo)
    else:
        # Classificação: Ordena por DELTA (do maior para o menor)
        metricas_ordenacao = np.abs(calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao))
    
    indices_ordenados = np.argsort(-metricas_ordenacao)

    while True:
        if is_rejected:
            valido1, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, 0)
            valido2, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, 1)
            if valido1 and valido2: break 
        else:
            direcao = 1 if modelo.predict(instance_df)[0] == 1 else 0
            valido, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, direcao) 
            if valido: break 
        
        if len(expl_robusta) == X_train.shape[1]: break
        
        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta}
        adicionou_feature = False 
        
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set:
                valor_feat = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
                expl_robusta.append(f"{feat_nome} = {valor_feat:.4f}")
                adicoes += 1
                adicionou_feature = True 
                break
        if not adicionou_feature: break
            
    return expl_robusta, adicoes 

# [FUNÇÃO REESCRITA COM NOVA LÓGICA DE ORDENAÇÃO]
def fase_2_minimizacao(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int, log_passos: List[Dict]) -> Tuple[List[str], int]:
    expl_minima = list(expl_robusta)
    remocoes = 0
    
    # Seleção da métrica para log e ordenação de remoção
    if is_rejected:
        # Rejeição: Usa PESOS
        metricas = _get_abs_weights(modelo)
    else:
        # Classificação: Usa DELTAS
        metricas = np.abs(calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao))
    
    # Ordena as features presentes na explicação para tentativa de remoção.
    # Tentamos remover da MAIOR métrica para a MENOR (reverse=True).
    # Isso é uma heurística gulosa reversa: tenta tirar as mais importantes primeiro para ver se é possível.
    features_para_remover = sorted(
        [f.split(' = ')[0] for f in expl_minima],
        key=lambda nome: metricas[X_train.columns.get_loc(nome)],
        reverse=True
    )
    
    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1: break
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        
        remocao_bem_sucedida = False
        score_pert_final = None
        
        # Valor para log (Delta ou Peso)
        idx_feat = X_train.columns.get_loc(feat_nome)
        metric_val = float(metricas[idx_feat])

        if is_rejected:
            valido1, score_p1 = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, 1)
            valido2, score_p2 = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, 0)
            ok_neg = bool(valido1)
            ok_pos = bool(valido2)
            if valido1 and valido2:
                remocao_bem_sucedida = True
            
            log_passos.append({
                'feat_nome': feat_nome,
                'valor': instance_df.iloc[0, idx_feat],
                'delta': metric_val, # Aqui 'delta' no log representará o Peso/Risco para rejeitadas
                'score_neg': score_p1,
                'ok_neg': ok_neg,
                'score_pos': score_p2,
                'ok_pos': ok_pos,
                'sucesso': remocao_bem_sucedida
            })
        else:
            direcao = 1 if modelo.predict(instance_df)[0] == 1 else 0
            remocao_bem_sucedida, score_pert_final = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, direcao)
            
            log_passos.append({
                'feat_nome': feat_nome,
                'valor': instance_df.iloc[0, idx_feat],
                'delta': metric_val,
                'score_perturbado': score_pert_final,
                'sucesso': remocao_bem_sucedida
            })

        if remocao_bem_sucedida:
            expl_minima = expl_temp
            remocoes += 1
            
    return expl_minima, remocoes

#==============================================================================
# FUNÇÕES DE GERAÇÃO E FORMATAÇÃO DE LOG
#==============================================================================

def gerar_explicacao_instancia(instancia_df: pd.DataFrame, modelo: Pipeline, X_train: pd.DataFrame, t_plus: float, t_minus: float) -> Tuple[List[str], List[str], int, int]:
    is_rejected = t_minus <= modelo.decision_function(instancia_df)[0] <= t_plus
    log_formatado: List[str] = []
    emit_tech_logs = TECHNICAL_LOGS and (X_train.shape[1] <= MAX_LOG_FEATURES)

    if is_rejected:
        # --- FLUXO DE REJEIÇÃO OTIMIZADO ---
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['rejeitada_analise'].format(t_minus=t_minus, t_plus=t_plus))
            log_formatado.append(LOG_TEMPLATES['rejeitada_prova_header'])

        # Agora chamamos ONE explanation apenas uma vez, pois a lógica interna
        # já detecta que é rejeição e usa a estratégia de Risco (não depende de premissa 0 ou 1)
        # Passamos premis_class=1 apenas por compatibilidade de assinatura, mas será ignorado na lógica de risco.
        expl_inicial = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 1)
        
        # Fase 1: Reforço usando ordenação por Peso
        expl_robusta, adicoes = fase_1_reforco(modelo, instancia_df, expl_inicial, X_train, t_plus, t_minus, True, 1)
        
        # Fase 2: Minimização usando ordenação por Peso
        passos: List[Dict[str, Any]] = []
        expl_final, remocoes = fase_2_minimizacao(modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, True, 1, passos)

        if emit_tech_logs:
            feats_iniciais = sorted([f.split(' = ')[0] for f in expl_robusta])
            log_formatado.append(f"├── Conjunto inicial (Heurística Risco): {len(feats_iniciais)} features {feats_iniciais}")
            
            for passo in passos[:MAX_LOG_STEPS]:
                key_header = 'rejeitada_feat_header_sucesso' if passo.get('sucesso', False) else 'rejeitada_feat_header_falha'
                log_formatado.append(LOG_TEMPLATES[key_header].format(feat=passo['feat_nome'], delta=passo.get('delta', 0.0)))
                
                cmp_neg = f"> t- ({t_minus:.4f})" if passo.get('score_neg', 0.0) > t_minus else f"< t- ({t_minus:.4f})"
                ok_neg = "OK." if passo.get('ok_neg', False) else "FALHA."
                log_formatado.append(LOG_TEMPLATES['rejeitada_subteste_neg'].format(score=passo.get('score_neg', 0.0), cmp=cmp_neg, ok=ok_neg))
                
                cmp_pos = f"< t+ ({t_plus:.4f})" if passo.get('score_pos', 0.0) < t_plus else f"\u2265 t+ ({t_plus:.4f})"
                ok_pos = "OK." if passo.get('ok_pos', False) else "FALHA."
                log_formatado.append(LOG_TEMPLATES['rejeitada_subteste_pos'].format(score=passo.get('score_pos', 0.0), cmp=cmp_pos, ok=ok_pos))
                
                footer_key = 'rejeitada_feat_footer_sucesso' if passo.get('sucesso', False) else 'rejeitada_feat_footer_falha'
                log_formatado.append(LOG_TEMPLATES[footer_key])
                
            if len(passos) > MAX_LOG_STEPS:
                log_formatado.append(f"│   ... {len(passos) - MAX_LOG_STEPS} passos omitidos por limite de log ...")

    else:
        # --- FLUXO DE CLASSIFICAÇÃO (MANTIDO) ---
        pred_class = int(modelo.predict(instancia_df)[0])
        posicao = 'acima de t+' if pred_class == 1 else 'abaixo de t-'
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['classificada_analise'].format(posicao=posicao))

        expl_inicial = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, pred_class)
        
        if DISABLE_REFORCO_CLASSIFICADAS:
            expl_robusta = expl_inicial
            adicoes = 0
            if emit_tech_logs:
                log_formatado.append(f"├── Fase de reforço DESATIVADA. Usando explicação inicial com {len(expl_robusta)} features.")
        else:
            expl_robusta, adicoes = fase_1_reforco(modelo, instancia_df, expl_inicial, X_train, t_plus, t_minus, False, pred_class)

        # Para log, calculamos deltas
        deltas = calculate_deltas(modelo, instancia_df, X_train, premis_class=pred_class)
        feats_em_robusta = [f.split(' = ')[0] for f in expl_robusta]
        ordem = sorted(
            feats_em_robusta,
            key=lambda nome: abs(deltas[X_train.columns.get_loc(nome)]),
            reverse=True
        )
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['classificada_min_inicio'].format(num_features=len(expl_robusta)))
            log_formatado.append(LOG_TEMPLATES['classificada_ordem'].format(lista=str(ordem)))

        passos: List[Dict[str, Any]] = []
        expl_final, remocoes = fase_2_minimizacao(modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, False, pred_class, passos)

        if emit_tech_logs:
            limiar = t_plus if pred_class == 1 else t_minus
            for p in passos[:MAX_LOG_STEPS]:
                cond = ("> t+ (" + f"{limiar:.4f})") if pred_class == 1 else ("< t- (" + f"{limiar:.4f})")
                key = 'classificada_step_sucesso' if p.get('sucesso', False) else 'classificada_step_falha'
                score_show = p.get('score_perturbado', np.nan)
                log_formatado.append(LOG_TEMPLATES[key].format(feat=p['feat_nome'], delta=p.get('delta', 0.0), score=score_show, cond=cond))
            if len(passos) > MAX_LOG_STEPS:
                log_formatado.append(f"├─ ... {len(passos) - MAX_LOG_STEPS} passos omitidos por limite de log ...")

    return [f.split(' = ')[0] for f in expl_final], log_formatado, adicoes, remocoes

#==============================================================================
# EXECUÇÃO PRINCIPAL
#==============================================================================
def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n==================== EXECUTANDO (MODO HÍBRIDO) PARA: {dataset_name.upper()} ====================")
    
    # 1. Carregar Configurações
    todos_hiperparametros = carregar_hiperparametros()
    X, y, nomes_classes, rejection_cost_atual, test_size_atual = configurar_experimento(dataset_name)

    parametros_para_modelo = DEFAULT_LOGREG_PARAMS.copy()
    config_do_modelo = todos_hiperparametros.get(dataset_name)
    if config_do_modelo and 'params' in config_do_modelo:
        valid_keys = LogisticRegression().get_params().keys()
        parametros_carregados = {k: v for k, v in config_do_modelo['params'].items() if k in valid_keys}
        parametros_para_modelo.update(parametros_carregados)
        print(f"[INFO] Usando hiperparâmetros otimizados: {parametros_para_modelo}")
    else:
        print(f"[AVISO] Usando modelo padrão.")

    # 2. Redução de Features (Top-K)
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    
    if top_k and top_k > 0 and top_k < X.shape[1]:
        print(f"\n[INFO] Treinando modelo temporário para seleção Top-{top_k} features...")
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
        X_train_temp, X_test_temp, _, _ = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
        X_train_temp, X_test_temp, selected_features = aplicar_selecao_top_k_features(X_train_temp, X_test_temp, modelo_temp, top_k)
        X = X[selected_features]
        print(f"[INFO] Dataset reduzido para {top_k} features.")
    
    # 3. Treino
    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
    
    print(f"[INFO] Modelo treinado. Acurácia Teste: {modelo.score(X_test, y_test):.2%}")
    print(f"[INFO] T+: {t_plus:.4f}, T-: {t_minus:.4f} (WR: {rejection_cost_atual:.2f})")
    
    # 4. Predições
    decision_scores_test = modelo.decision_function(X_test)
    y_pred_test = np.full(y_test.shape, -1, dtype=int)
    y_pred_test[decision_scores_test >= t_plus] = 1
    y_pred_test[decision_scores_test <= t_minus] = 0
    rejected_mask = (y_pred_test == -1)
    y_pred_test_final = y_pred_test.copy()
    y_pred_test_final[rejected_mask] = 2

    # 5. Explicações
    print(f"\n[INFO] Gerando explicações...")
    from utils.progress_bar import ProgressBar
    
    start_time_total = time.perf_counter()
    resultados_instancias = []
    times_pos, times_neg, times_rej = [], [], []
    adicoes_pos, adicoes_neg, adicoes_rej = [], [], []
    remocoes_pos, remocoes_neg, remocoes_rej = [], [], []

    with ProgressBar(total=len(X_test), description=f"PEAB Híbrido {dataset_name}") as pbar:
        for i in range(len(X_test)):
            inst_start_time = time.perf_counter()
            
            instancia_df = X_test.iloc[[i]]
            pred_class_code = y_pred_test_final[i]
            
            expl_final_nomes, log_formatado, adicoes, remocoes = gerar_explicacao_instancia(instancia_df, modelo, X_train, t_plus, t_minus)
            
            inst_end_time = time.perf_counter()
            inst_duration = inst_end_time - inst_start_time

            header = f"--- INSTÂNCIA #{i} | REAL: {nomes_classes[y_test.iloc[i]]} | PRED: {'REJEITADA' if pred_class_code == 2 else 'CLASSE ' + str(pred_class_code)} | SCORE: {decision_scores_test[i]:.4f} ---"
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
                adicoes_rej.append(adicoes)
                remocoes_rej.append(remocoes)
            elif pred_class_code == 1:
                times_pos.append(inst_duration)
                adicoes_pos.append(adicoes)
                remocoes_pos.append(remocoes)
            else:
                times_neg.append(inst_duration)
                adicoes_neg.append(adicoes)
                remocoes_neg.append(remocoes)
            
            pbar.update()
    
    tempo_total = time.perf_counter() - start_time_total
    print(f"\n[INFO] Concluído em {tempo_total:.2f}s.")

    # 6. Métricas e Salvamento
    metricas_dict = coletar_metricas(
        resultados_instancias, y_test, y_pred_test_final, rejected_mask,
        tempo_total, model_params, modelo, X_test, X_train.columns,
        times_pos, times_neg, times_rej,
        adicoes_pos, adicoes_neg, adicoes_rej,
        remocoes_pos, remocoes_neg, remocoes_rej
    )

    dataset_json_key = dataset_name
    if dataset_name == 'mnist':
        cfg_mnist = DATASET_CONFIG.get('mnist', {})
        digit_pair = cfg_mnist.get('digit_pair')
        if digit_pair:
            dataset_json_key = f"mnist_{digit_pair[0]}_vs_{digit_pair[1]}"

    dataset_cache = montar_dataset_cache(
        dataset_name, X_train, X_test, y_train, y_test, nomes_classes,
        t_plus, t_minus, rejection_cost_atual, test_size_atual,
        model_params, metricas_dict, y_pred_test_final, decision_scores_test,
        rejected_mask, resultados_instancias
    )
    
    # [MODIFICAÇÃO] Usa a função local para salvar no JSON customizado
    save_custom_json_results('peab_hibrido', dataset_json_key, dataset_cache, filepath=JSON_RESULTS_FILE)

    # [MODIFICAÇÃO] Prefixar 'M_' no relatório
    gerar_relatorio_texto(f"M_peab_{dataset_name}", test_size_atual, rejection_cost_atual, modelo, t_plus, t_minus, len(X_test), metricas_dict, resultados_instancias)
    
    print(f"\n==================== FIM ====================")

# [MANTENDO FUNÇÕES AUXILIARES IGUAIS, APENAS COPIANDO PARA INTEGRIDADE]
def coletar_metricas(resultados_instancias, y_test, y_pred_test_final, rejected_mask,
                     tempo_total, model_params, modelo, X_test, feature_names,
                     times_pos, times_neg, times_rej, adicoes_pos, adicoes_neg, adicoes_rej, remocoes_pos, remocoes_neg, remocoes_rej):
    stats_pos_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 1]
    stats_neg_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 0]
    stats_rej_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 2]

    acc_sem_rej = float(np.mean(modelo.predict(X_test) == y_test) * 100)
    acc_com_rej = float(np.mean(y_pred_test_final[~rejected_mask] == y_test.iloc[~rejected_mask]) * 100) if np.any(~rejected_mask) else 100.0
    taxa_rej = float(np.mean(rejected_mask) * 100)

    def get_stats(lst):
        return {
            'instancias': len(lst),
            'media': float(np.mean(lst)) if lst else 0.0,
            'std_dev': float(np.std(lst)) if lst else 0.0,
            'min': int(np.min(lst)) if lst else 0,
            'max': int(np.max(lst)) if lst else 0
        }

    def get_proc_stats(adicoes, remocoes):
        inst_com_adicao = sum(1 for x in adicoes if x > 0)
        return {
            'inst_com_adicao': int(inst_com_adicao),
            'perc_adicao': float((inst_com_adicao / len(adicoes) * 100) if adicoes else 0.0),
            'media_adicoes': float(np.mean([x for x in adicoes if x > 0])) if inst_com_adicao > 0 else 0.0,
            'inst_com_remocao': int(sum(1 for x in remocoes if x > 0)),
            'perc_remocao': float((sum(1 for x in remocoes if x > 0) / len(remocoes) * 100) if remocoes else 0.0),
            'media_remocoes': float(np.mean([x for x in remocoes if x > 0])) if sum(1 for x in remocoes if x > 0) > 0 else 0.0
        }

    return {
        'acuracia_sem_rejeicao': acc_sem_rej,
        'acuracia_com_rejeicao': acc_com_rej,
        'taxa_rejeicao': taxa_rej,
        'stats_explicacao_positiva': get_stats(stats_pos_list),
        'stats_explicacao_negativa': get_stats(stats_neg_list),
        'stats_explicacao_rejeitada': get_stats(stats_rej_list),
        'tempo_total': float(tempo_total),
        'tempo_medio_instancia': float(tempo_total / len(y_test) if len(y_test) > 0 else 0.0),
        'tempo_medio_positivas': float(np.mean(times_pos)) if times_pos else 0.0,
        'tempo_medio_negativas': float(np.mean(times_neg)) if times_neg else 0.0,
        'tempo_medio_rejeitadas': float(np.mean(times_rej)) if times_rej else 0.0,
        'features_frequentes': Counter([feat for r in resultados_instancias for feat in r['explicacao']]).most_common(),
        'pesos_modelo': sorted(((name, float(model_params['coefs'][name])) for name in feature_names), key=lambda item: abs(item[1]), reverse=True),
        'intercepto': float(model_params['intercepto']),
        'processo_stats_pos': get_proc_stats(adicoes_pos, remocoes_pos),
        'processo_stats_neg': get_proc_stats(adicoes_neg, remocoes_neg),
        'processo_stats_rej': get_proc_stats(adicoes_rej, remocoes_rej)
    }

def configurar_experimento(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], float, float]:
    if dataset_name == 'mnist':
        from data import datasets as ds_module
        cfg = DATASET_CONFIG.get(dataset_name, {})
        ds_module.set_mnist_options(cfg.get('feature_mode', 'raw'), cfg.get('digit_pair', (9, 4)))
    
    X, y, nomes_classes = carregar_dataset(dataset_name)
    cfg = DATASET_CONFIG.get(dataset_name, {'test_size': 0.3, 'rejection_cost': 0.24})

    if 'subsample_size' in cfg and cfg['subsample_size']:
        idx = np.arange(len(y))
        sample_idx, _ = train_test_split(idx, test_size=(1 - cfg['subsample_size']), random_state=RANDOM_STATE, stratify=y)
        X = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]
        y = y.iloc[sample_idx] if isinstance(y, pd.Series) else y[sample_idx]

    return X, y, nomes_classes, cfg['rejection_cost'], cfg['test_size']

def aplicar_selecao_top_k_features(X_train, X_test, pipeline, top_k):
    logreg = _get_lr(pipeline)
    coefs = logreg.coef_[0]
    importances = sorted([(name, abs(coefs[i])) for i, name in enumerate(X_train.columns)], key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in importances[:top_k]]
    return X_train[selected], X_test[selected], selected

def treinar_e_avaliar_modelo(X, y, test_size, rejection_cost, logreg_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params))])
    pipeline.fit(X_train, y_train)

    decision_scores = pipeline.decision_function(X_train)
    qs = np.linspace(0, 1, 100)
    search_space = np.unique(np.quantile(decision_scores, qs))
    best_risk, best_tp, best_tm = float('inf'), 0.0, 0.0
    
    for i in range(len(search_space)):
        for j in range(i, len(search_space)):
            tm, tp = float(search_space[i]), float(search_space[j])
            if MIN_REJECTION_WIDTH > 0 and (tp - tm) < MIN_REJECTION_WIDTH: continue
            
            y_pred = np.full(y_train.shape, -1)
            accepted = (decision_scores >= tp) | (decision_scores <= tm)
            y_pred[decision_scores >= tp] = 1
            y_pred[decision_scores <= tm] = 0
            
            error_rate = np.mean(y_pred[accepted] != y_train[accepted]) if np.any(accepted) else 0.0
            rej_rate = 1.0 - np.mean(accepted)
            risk = error_rate + rejection_cost * rej_rate
            
            if risk < best_risk:
                best_risk, best_tp, best_tm = risk, tp, tm

    model_params = {
        'coefs': {name: float(w) for name, w in zip(X.columns, pipeline.named_steps['model'].coef_[0])},
        'intercepto': float(pipeline.named_steps['model'].intercept_[0]),
        'scaler_params': {'min': list(pipeline.named_steps['scaler'].min_), 'scale': list(pipeline.named_steps['scaler'].scale_)},
        **logreg_params
    }
    return pipeline, best_tp, best_tm, model_params

def gerar_relatorio_texto(filename_base, test_size, wr, modelo, t_plus, t_minus, num_test, metricas, resultados_instancias):
    output_path = os.path.join(OUTPUT_BASE_DIR, f"{filename_base}.txt")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO COMPARATIVO - MÉTODO MODIFICADO (HÍBRIDO: DELTA + RISCO)\n")
        f.write(f"Dataset: {filename_base} | WR: {wr} | Test Size: {test_size}\n")
        f.write("\n".join(SYMBOL_LEGEND) + "\n\n")
        
        # Resumo Estatístico Focado
        f.write("[ ESTATÍSTICAS DE TAMANHO DE EXPLICAÇÃO ]\n")
        for cls, key in [("POSITIVA", 'stats_explicacao_positiva'), ("NEGATIVA", 'stats_explicacao_negativa'), ("REJEITADA", 'stats_explicacao_rejeitada')]:
            stats = metricas[key]
            f.write(f"  - {cls}: {stats['media']:.2f} ± {stats['std_dev']:.2f} (Min: {stats['min']}, Max: {stats['max']})\n")
        
        f.write("\n" + LOG_TEMPLATES['processamento_header'] + "\n")
        for r in resultados_instancias:
            # Salva apenas se for rejeitada para análise focada, ou todas se preferir
            if r['predicao'] == 'REJEITADA':
                f.write("-" * 80 + "\n")
                for line in r['log_detalhado']: f.write(f"{line}\n")
                f.write(f"\n   --> FINAL: {sorted(r['explicacao'])}\n")

def montar_dataset_cache(dataset_name, X_train, X_test, y_train, y_test, nomes_classes, t_plus, t_minus, wr, test_size, model_params, metricas, y_pred, scores, rej_mask, resultados):
    # Simplificado para caber; mesma estrutura do original
    return {
        'config': {'dataset_name': dataset_name, 'rejection_cost': wr},
        'performance': {'rejection_rate': metricas['taxa_rejeicao']},
        'explanation_stats': metricas['stats_explicacao_rejeitada'], # Focando no que importa
        'per_instance': [
            {'id': str(idx), 'y_pred': int(y_pred[i]), 'explanation_size': len(resultados[i]['explicacao'])}
            for i, idx in enumerate(X_test.index)
        ]
    }

if __name__ == '__main__':
    nome_dataset, _, _, _, _ = selecionar_dataset_e_classe()
    if nome_dataset:
        executar_experimento_para_dataset(nome_dataset)