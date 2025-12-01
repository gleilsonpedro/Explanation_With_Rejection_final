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

# Importações do seu projeto (mantém compatibilidade)
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
# Removi update_method_results pois faremos manualmente para garantir o arquivo correto
from utils.progress_bar import ProgressBar

#==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
#==============================================================================
RANDOM_STATE: int = 42

MNIST_CONFIG = {
    'feature_mode': 'raw',           
    'digit_pair': (3, 8),            
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

OUTPUT_BASE_DIR: str = 'results/report/peab' 
HIPERPARAMETROS_FILE: str = 'json/hiperparametros.json'
JSON_RESULTS_FILE: str = 'json/2comparative_results.json' # JSON DIFERENTE (v2)

DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2', 'C': 0.01, 'solver': 'liblinear', 'max_iter': 1000
}

TECHNICAL_LOGS: bool = True
MAX_LOG_FEATURES: int = 200  
MAX_LOG_STEPS: int = 60      

SYMBOL_LEGEND = [
    "LEGENDA DOS SÍMBOLOS (PEAB v2 - Híbrido)",
    "   δ  = Contribuição (Delta) [Usado em Classificadas]",
    "   |w|= Peso Absoluto (Risco) [Usado em Rejeitadas]",
    "   ε  = Epsilon (Tolerância 1e-6)",
    "   ●  = Feature mantida (essencial)",
    "   ○  = Feature removida (não essencial)"
]

LOG_TEMPLATES = {
    'processamento_header': "********** PROCESSAMENTO POR INSTÂNCIA (v2) **********\n",
    'classificada_analise': "├── Análise (Classificada): Buscando menor conjunto via DELTA com Epsilon.",
    'classificada_min_inicio': "├── Iniciando minimização com {num_features} features.",
    'classificada_ordem': "├── Tentativas (ordem |δ|): {lista}",
    'classificada_step_sucesso': "├─ ○ {feat} (δ: {delta:+.3f}): s'={score:.3f} ({cond}) → SUCESSO.",
    'classificada_step_falha': "├─ ● {feat} (δ: {delta:+.3f}): s'={score:.3f} ({cond}) → FALHA.",

    'rejeitada_analise': "├── Análise (Rejeitada): Zona [{t_minus:.4f}, {t_plus:.4f}]. Estratégia: RISCO (|w|).",
    'rejeitada_prova_header': "├── Prova de Estabilidade (Redução de Variância):",
    'rejeitada_feat_header_sucesso': "├─ ○ {feat} (|w|: {delta:.3f}):",
    'rejeitada_feat_header_falha': "├─ ● {feat} (|w|: {delta:.3f}):",
    'rejeitada_subteste_neg': "│   ├─ Teste T-: s'={score:.3f} ({cmp}) {ok}",
    'rejeitada_subteste_pos': "│   └─ Teste T+: s'={score:.3f} ({cmp}) {ok}",
    'rejeitada_feat_footer_sucesso': "│   └─> SUCESSO.",
    'rejeitada_feat_footer_falha': "│   └─> FALHA (Essencial).",
}

DISABLE_REFORCO_CLASSIFICADAS: bool = True
MIN_REJECTION_WIDTH: float = 0.0

#==============================================================================
# FUNÇÕES AUXILIARES
#==============================================================================

def _get_lr(modelo: Pipeline):
    if 'model' in modelo.named_steps: return modelo.named_steps['model']
    if 'modelo' in modelo.named_steps: return modelo.named_steps['modelo']
    raise KeyError("Pipeline sem passo 'model' ou 'modelo'")

def _get_abs_weights(modelo: Pipeline) -> np.ndarray:
    """Retorna os pesos absolutos (Risco) de cada feature."""
    logreg = _get_lr(modelo)
    return np.abs(logreg.coef_[0])

def carregar_hiperparametros(caminho_arquivo: str = HIPERPARAMETROS_FILE) -> dict:
    try:
        with open(caminho_arquivo, 'r') as f: return json.load(f)
    except: return {}

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

# [NOVO] Função local para salvar o JSON no arquivo correto (v2)
def salvar_resultados_v2(dataset_name, metricas):
    data = {}
    
    # Tenta carregar se já existir
    if os.path.exists(JSON_RESULTS_FILE):
        try:
            with open(JSON_RESULTS_FILE, 'r') as f:
                data = json.load(f)
        except: pass
    
    if 'peab_v2' not in data:
        data['peab_v2'] = {}
        
    data['peab_v2'][dataset_name] = metricas
    
    # Garante que a pasta existe
    os.makedirs(os.path.dirname(JSON_RESULTS_FILE), exist_ok=True)
    
    with open(JSON_RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

#==============================================================================
#  CORE LÓGICO V2 (HÍBRIDO + EPSILON)
#==============================================================================

def one_explanation_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, premis_class: int) -> List[str]:
    score = modelo.decision_function(instance_df)[0]
    explicacao = []
    
    is_rejection_zone = (t_minus <= score <= t_plus)
    
    if is_rejection_zone:
        # --- ESTRATÉGIA DE RISCO (REJEIÇÃO) ---
        logreg = _get_lr(modelo)
        risks = np.abs(logreg.coef_[0])
        indices_ordenados = np.argsort(-risks)
        
        current_possible_swing = np.sum(risks)
        dist_to_top = t_plus - score
        dist_to_bottom = score - t_minus
        
        for i in indices_ordenados:
            if (current_possible_swing < dist_to_top) and (current_possible_swing < dist_to_bottom):
                break
            feat_nome = X_train.columns[i]
            val = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
            explicacao.append(f"{feat_nome} = {val:.4f}")
            current_possible_swing -= risks[i]

    else:
        # --- ESTRATÉGIA DE DELTA + EPSILON (CLASSIFICAÇÃO) ---
        deltas = calculate_deltas(modelo, instance_df, X_train, premis_class)
        indices_ordenados = np.argsort(-np.abs(deltas))
        
        score_base = score - np.sum(deltas)
        soma_cumulativa = score_base 
        target = t_plus if premis_class == 1 else t_minus
        
        EPSILON = 1e-6 

        for i in indices_ordenados:
            feature_nome = X_train.columns[i]
            val = instance_df.iloc[0, X_train.columns.get_loc(feature_nome)]
            
            if abs(deltas[i]) > 1e-9:
                 soma_cumulativa += deltas[i]
                 explicacao.append(f"{feature_nome} = {val:.4f}")
            
            if premis_class == 1:
                if soma_cumulativa >= (target - EPSILON) and explicacao:
                    break
            else: 
                if soma_cumulativa <= (target + EPSILON) and explicacao:
                    break
                    
    if not explicacao and len(X_train.columns) > 0:
         metrica = _get_abs_weights(modelo) if is_rejection_zone else np.abs(calculate_deltas(modelo, instance_df, X_train, premis_class))
         idx_max = np.argmax(metrica)
         feat_nome = X_train.columns[idx_max]
         val = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
         explicacao.append(f"{feat_nome} = {val:.4f}")

    return explicacao

def perturbar_e_validar(modelo: Pipeline, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, direcao_override: int) -> Tuple[bool, float]:
    if not explicacao: return False, 0.0
    inst_pert = instance_df.copy()
    features_explicacao = {f.split(' = ')[0] for f in explicacao}
    
    perturbar_para_diminuir = (direcao_override == 1)
    modelo_interno = _get_lr(modelo)
    X_min = X_train.min(axis=0) 
    X_max = X_train.max(axis=0)
    
    for feat_idx, feat_nome in enumerate(X_train.columns):
        if feat_nome in features_explicacao: continue 
        coef = modelo_interno.coef_[0][feat_idx] 
        valor_pert = (X_min[feat_nome] if coef > 0 else X_max[feat_nome]) if perturbar_para_diminuir else (X_max[feat_nome] if coef > 0 else X_min[feat_nome])
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
        
    score_pert = modelo.decision_function(inst_pert)[0]
    score_orig = modelo.decision_function(instance_df)[0]
    is_rejected = t_minus <= score_orig <= t_plus
    
    if is_rejected:
        return (t_minus <= score_pert <= t_plus), score_pert
    else:
        pred_class = int(modelo.predict(instance_df)[0])
        if pred_class == 1:
            return (score_pert >= t_plus), score_pert
        else:
            return (score_pert <= t_minus), score_pert

def fase_1_reforco(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int) -> Tuple[List[str], int]:
    expl_robusta = list(expl_inicial)
    adicoes = 0
    
    if is_rejected:
        metricas = _get_abs_weights(modelo) 
    else:
        metricas = np.abs(calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao))
    
    indices_ordenados = np.argsort(-metricas)

    while True:
        if is_rejected:
            val1, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, 0)
            val2, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, 1)
            if val1 and val2: break
        else:
            direcao = 1 if modelo.predict(instance_df)[0] == 1 else 0
            valido, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, direcao)
            if valido: break
        
        if len(expl_robusta) == X_train.shape[1]: break
        
        feats_set = {f.split(' = ')[0] for f in expl_robusta}
        adicionou = False
        
        for idx in indices_ordenados:
            feat = X_train.columns[idx]
            if feat not in feats_set:
                val = instance_df.iloc[0, X_train.columns.get_loc(feat)]
                expl_robusta.append(f"{feat} = {val:.4f}")
                adicoes += 1
                adicionou = True
                break
        if not adicionou: break
    return expl_robusta, adicoes

def fase_2_minimizacao(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int, log_passos: List[Dict]) -> Tuple[List[str], int]:
    expl_minima = list(expl_robusta)
    remocoes = 0
    
    if is_rejected:
        metricas = _get_abs_weights(modelo)
    else:
        metricas = np.abs(calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao))

    features_para_remover = sorted(
        [f.split(' = ')[0] for f in expl_minima],
        key=lambda nome: metricas[X_train.columns.get_loc(nome)],
        reverse=True 
    )

    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1: break
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        
        sucesso = False
        score_final = None
        metric_val = float(metricas[X_train.columns.get_loc(feat_nome)])

        if is_rejected:
            v1, s1 = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, 1)
            v2, s2 = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, 0)
            ok_n, ok_p = bool(v1), bool(v2)
            if v1 and v2: sucesso = True
            log_passos.append({'feat_nome': feat_nome, 'delta': metric_val, 'score_neg': s1, 'ok_neg': ok_n, 'score_pos': s2, 'ok_pos': ok_p, 'sucesso': sucesso})
        else:
            direcao = 1 if modelo.predict(instance_df)[0] == 1 else 0
            sucesso, score_final = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, direcao)
            log_passos.append({'feat_nome': feat_nome, 'delta': metric_val, 'score_perturbado': score_final, 'sucesso': sucesso})

        if sucesso:
            expl_minima = expl_temp
            remocoes += 1
            
    return expl_minima, remocoes

#==============================================================================
# FUNÇÕES COMPLETAS DE TREINAMENTO E MÉTRICAS
#==============================================================================

def gerar_explicacao_instancia(instancia_df: pd.DataFrame, modelo: Pipeline, X_train: pd.DataFrame, t_plus: float, t_minus: float) -> Tuple[List[str], List[str], int, int]:
    score = modelo.decision_function(instancia_df)[0]
    is_rejected = t_minus <= score <= t_plus
    log_fmt: List[str] = []
    emit_logs = TECHNICAL_LOGS and (X_train.shape[1] <= MAX_LOG_FEATURES)

    expl_inicial = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 1)

    if is_rejected:
        if emit_logs:
            log_fmt.append(LOG_TEMPLATES['rejeitada_analise'].format(t_minus=t_minus, t_plus=t_plus))
            log_fmt.append(LOG_TEMPLATES['rejeitada_prova_header'])
            
        expl_robusta, adicoes = fase_1_reforco(modelo, instancia_df, expl_inicial, X_train, t_plus, t_minus, True, 1)
        passos = []
        expl_final, remocoes = fase_2_minimizacao(modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, True, 1, passos)
        
        if emit_logs:
            for p in passos[:MAX_LOG_STEPS]:
                key = 'rejeitada_feat_header_sucesso' if p['sucesso'] else 'rejeitada_feat_header_falha'
                log_fmt.append(LOG_TEMPLATES[key].format(feat=p['feat_nome'], delta=p.get('delta',0)))

    else:
        pred_class = int(modelo.predict(instancia_df)[0])
        if emit_logs: log_fmt.append(LOG_TEMPLATES['classificada_analise'])
        
        expl_inicial = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, pred_class)
        
        if DISABLE_REFORCO_CLASSIFICADAS:
            expl_robusta, adicoes = expl_inicial, 0
        else:
            expl_robusta, adicoes = fase_1_reforco(modelo, instancia_df, expl_inicial, X_train, t_plus, t_minus, False, pred_class)
            
        passos = []
        expl_final, remocoes = fase_2_minimizacao(modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, False, pred_class, passos)

    return [f.split(' = ')[0] for f in expl_final], log_fmt, adicoes, remocoes

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

def coletar_metricas(resultados_instancias, y_test, modelo, X_test, feature_names, model_params):
    stats_pos_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 1]
    stats_neg_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 0]
    stats_rej_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 2]

    acc_sem_rej = float(np.mean(modelo.predict(X_test) == y_test) * 100)
    
    def get_stats(lst):
        return {'media': float(np.mean(lst)) if lst else 0.0, 'std_dev': float(np.std(lst)) if lst else 0.0, 'min': int(np.min(lst)) if lst else 0, 'max': int(np.max(lst)) if lst else 0}

    return {
        'acuracia_sem_rejeicao': acc_sem_rej,
        'taxa_rejeicao': float(len(stats_rej_list) / len(resultados_instancias) * 100),
        'stats_explicacao_positiva': get_stats(stats_pos_list),
        'stats_explicacao_negativa': get_stats(stats_neg_list),
        'stats_explicacao_rejeitada': get_stats(stats_rej_list),
    }

def gerar_relatorio_texto(filename_base, test_size, wr, t_plus, t_minus, metricas):
    output_path = os.path.join(OUTPUT_BASE_DIR, f"{filename_base}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO v2 - HÍBRIDO (Risco na Rejeição + Epsilon)\n")
        f.write(f"Dataset: {filename_base.replace('2M_peab_', '')} | WR: {wr} | T+: {t_plus:.4f} | T-: {t_minus:.4f}\n\n")
        f.write("[ ESTATÍSTICAS DE TAMANHO DE EXPLICAÇÃO ]\n")
        for cls, key in [("POSITIVA", 'stats_explicacao_positiva'), ("NEGATIVA", 'stats_explicacao_negativa'), ("REJEITADA", 'stats_explicacao_rejeitada')]:
            stats = metricas[key]
            f.write(f"  - {cls}: {stats['media']:.2f} ± {stats['std_dev']:.2f} (Min: {stats['min']}, Max: {stats['max']})\n")

#==============================================================================
# EXECUÇÃO PRINCIPAL
#==============================================================================
def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n===== EXECUÇÃO PEAB v2 (HÍBRIDO + EPSILON): {dataset_name.upper()} =====")
    
    todos_hiperparametros = carregar_hiperparametros()
    X, y, nomes_classes, rejection_cost_atual, test_size_atual = configurar_experimento(dataset_name)

    parametros_para_modelo = DEFAULT_LOGREG_PARAMS.copy()
    if dataset_name in todos_hiperparametros and 'params' in todos_hiperparametros[dataset_name]:
        parametros_para_modelo.update(todos_hiperparametros[dataset_name]['params'])

    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0:
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
        X_train_temp, X_test_temp, _, _ = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
        _, _, selected_features = aplicar_selecao_top_k_features(X_train_temp, X_test_temp, modelo_temp, top_k)
        X = X[selected_features]
    
    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
    
    print(f"[INFO] Gerando explicações v2...")
    resultados_instancias = []
    
    with ProgressBar(total=len(X_test), description=f"PEAB v2 {dataset_name}") as pbar:
        for i in range(len(X_test)):
            start = time.perf_counter()
            instancia_df = X_test.iloc[[i]]
            
            expl_final_nomes, log_fmt, adicoes, remocoes = gerar_explicacao_instancia(instancia_df, modelo, X_train, t_plus, t_minus)
            
            duration = time.perf_counter() - start
            
            score_final = modelo.decision_function(instancia_df)[0]
            if t_minus <= score_final <= t_plus: pred_code = 2
            elif score_final >= t_plus: pred_code = 1
            else: pred_code = 0
            
            resultados_instancias.append({
                'id': i,
                'classe_real': nomes_classes[y_test.iloc[i]],
                'predicao': 'REJEITADA' if pred_code == 2 else nomes_classes[pred_code],
                'pred_code': int(pred_code),
                'score': score_final,
                'explicacao': sorted(expl_final_nomes),
                'tamanho_explicacao': len(expl_final_nomes),
                'tempo_exec': duration
            })
            pbar.update()

    metricas_dict = coletar_metricas(resultados_instancias, y_test, modelo, X_test, X_train.columns, model_params)
    salvar_resultados_v2(dataset_name, metricas_dict) # Usando a função local
    gerar_relatorio_texto(f"2M_peab_{dataset_name}", test_size_atual, rejection_cost_atual, t_plus, t_minus, metricas_dict)
    
    print(f"\n[SUCESSO] Resultados salvos em:\n -> {JSON_RESULTS_FILE}\n -> {OUTPUT_BASE_DIR}/2M_peab_{dataset_name}.txt")

if __name__ == '__main__':
    nome_dataset, _, _, _, _ = selecionar_dataset_e_classe()
    if nome_dataset: executar_experimento_para_dataset(nome_dataset)