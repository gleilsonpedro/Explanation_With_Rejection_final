import os
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any, Set

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Imports do seu projeto
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
from utils.results_handler import update_method_results
from utils.progress_bar import ProgressBar

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================
RANDOM_STATE: int = 42

# Pastas de saída solicitadas
DIR_JSON = os.path.join('json', 'MLP')
DIR_REPORT = os.path.join('results', 'report', 'MLP')
os.makedirs(DIR_JSON, exist_ok=True)
os.makedirs(DIR_REPORT, exist_ok=True)

DATASET_CONFIG = {
    "mnist":                {'test_size': 0.3, 'rejection_cost': 0.24, 'subsample_size': 0.01},
    "breast_cancer":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "pima_indians_diabetes":{'test_size': 0.3, 'rejection_cost': 0.24},
    "vertebral_column":     {'test_size': 0.3, 'rejection_cost': 0.24},
    "sonar":                {'test_size': 0.3, 'rejection_cost': 0.24},
    "spambase":             {'test_size': 0.3, 'rejection_cost': 0.24},
    "banknote":             {'test_size': 0.3, 'rejection_cost': 0.24},
}

MLP_PARAMS = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'early_stopping': True,
    'n_iter_no_change': 10,
    'random_state': RANDOM_STATE
}

LOGREG_PARAMS = {
    'penalty': 'l2', 
    'C': 1.0, 
    'solver': 'liblinear', 
    'max_iter': 500
}

def sanitize_filename(filename: str) -> str:
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# ==============================================================================
# MOTOR DE EXPLICAÇÃO: MODELO SUBSTITUTO LOCAL (LOCAL SURROGATE)
# ==============================================================================

def gerar_vizinhanca_local(instancia: np.ndarray, std_train: np.ndarray, num_clones: int = 1000) -> pd.DataFrame:
    """Gera dados sintéticos ao redor da instância original usando a variância do treino."""
    # Cria ruído gaussiano baseado no desvio padrão de cada feature
    ruido = np.random.normal(0, std_train * 0.5, size=(num_clones, len(instancia)))
    clones = instancia + ruido
    
    # A primeira linha é sempre a instância original exata
    clones[0] = instancia
    return pd.DataFrame(clones)

def encontrar_thresholds_locais(modelo_lr: Pipeline, X_local: pd.DataFrame, y_local: np.ndarray, rejection_cost: float):
    """Encontra os melhores limiares (t+, t-) para o modelo linear na vizinhança local."""
    probas = np.clip(modelo_lr.predict_proba(X_local), 1e-9, 1 - 1e-9)
    decision_scores = np.log(probas[:, 1] / probas[:, 0])

    scores_neg = decision_scores[decision_scores < 0]
    scores_pos = decision_scores[decision_scores > 0]

    t_minus_grid = np.linspace(scores_neg.min(), -0.001, 20) if len(scores_neg) > 0 else np.array([-0.1])
    t_plus_grid  = np.linspace(0.001, scores_pos.max(), 20)  if len(scores_pos) > 0 else np.array([0.1])

    best_risk, best_t_plus, best_t_minus = float('inf'), 0.1, -0.1

    for tm in t_minus_grid:
        for tp in t_plus_grid:
            if not (tm < 0 < tp): continue
            acc_mask = (decision_scores >= tp) | (decision_scores <= tm)
            preds = np.full(y_local.shape, -1)
            preds[decision_scores >= tp] = 1
            preds[decision_scores <= tm] = 0

            error = np.mean(preds[acc_mask] != y_local[acc_mask]) if np.any(acc_mask) else 0.0
            rejection_rate = 1.0 - np.mean(acc_mask)
            risk = error + rejection_cost * rejection_rate

            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, tp, tm

    return best_t_plus, best_t_minus, decision_scores[0] # Retorna também o score da instância original

def explicar_instancia_surrogate(instancia_vals: np.ndarray, modelo_mlp: Pipeline, feature_names: list, 
                                 std_train: np.ndarray, rejection_cost: float) -> Tuple[List[str], int]:
    """
    Core do método: 
    1. Cria clones. 2. Passa na MLP. 3. Treina LogReg. 4. Extrai explicação do pior caso.
    """
    # 1. Cria a vizinhança e consulta o Oráculo (MLP)
    df_clones = gerar_vizinhanca_local(instancia_vals, std_train, num_clones=1000)
    df_clones.columns = feature_names
    y_oraculo = modelo_mlp.predict(df_clones)

    # Verifica se a MLP tem certeza absoluta na região (só previu 0 ou só previu 1)
    if len(np.unique(y_oraculo)) == 1:
        # Se for 100% puro, a explicação local é trivial (a região inteira é da mesma classe)
        feature_mais_importante = feature_names[np.argmax(np.abs(instancia_vals))]
        return [feature_mais_importante], y_oraculo[0]

    # 2. Treina o Modelo Substituto Local (Regressão Logística)
    modelo_local = Pipeline([
        ('scaler', MinMaxScaler()), 
        ('model', LogisticRegression(**LOGREG_PARAMS, random_state=RANDOM_STATE))
    ])
    modelo_local.fit(df_clones, y_oraculo)

    # 3. Calcula os thresholds ótimos na região local
    t_plus, t_minus, score_original = encontrar_thresholds_locais(modelo_local, df_clones, y_oraculo, rejection_cost)

    # 4. Extrai os pesos e calcula os piores casos (Lógica rápida do PEAB)
    scaler = modelo_local.named_steps['scaler']
    lr = modelo_local.named_steps['model']
    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    vals_s = scaler.transform([instancia_vals])[0]
    X_min_contribution = np.where(coefs > 0, 0.0, 1.0)
    X_max_contribution = np.where(coefs > 0, 1.0, 0.0)

    base_min_score = intercept + np.dot(coefs, X_min_contribution)
    base_max_score = intercept + np.dot(coefs, X_max_contribution)
    
    gains_from_min = (vals_s * coefs) - (X_min_contribution * coefs)
    losses_from_max = (X_max_contribution * coefs) - (vals_s * coefs)

    # Define o modo (Positivo, Negativo ou Rejeitado)
    if score_original >= t_plus: mode, sort_metric = 'positive', gains_from_min
    elif score_original <= t_minus: mode, sort_metric = 'negative', losses_from_max
    else: mode, sort_metric = 'rejected', gains_from_min # Padrão para rejeitado

    # Algoritmo Guloso (Fase 1 e Fase 2 fundidas para performance)
    expl_indices = set()
    for idx in np.argsort(-sort_metric):
        # Verifica worst case
        curr_gain = np.sum(gains_from_min[list(expl_indices)]) if expl_indices else 0.0
        curr_loss = np.sum(losses_from_max[list(expl_indices)]) if expl_indices else 0.0
        worst_min = base_min_score + curr_gain
        worst_max = base_max_score - curr_loss
        
        is_valid = False
        if mode == 'positive': is_valid = worst_min >= t_plus - 1e-5
        elif mode == 'negative': is_valid = worst_max <= t_minus + 1e-5
        elif mode == 'rejected': is_valid = (worst_min >= t_minus - 1e-5) and (worst_max <= t_plus + 1e-5)
        
        if is_valid: break
        expl_indices.add(int(idx))

    # Minimização (Pruning)
    for idx in sorted(list(expl_indices), key=lambda i: sort_metric[i]):
        if len(expl_indices) <= 1: break
        expl_indices.remove(idx)
        
        curr_gain = np.sum(gains_from_min[list(expl_indices)])
        curr_loss = np.sum(losses_from_max[list(expl_indices)])
        worst_min = base_min_score + curr_gain
        worst_max = base_max_score - curr_loss
        
        is_valid = False
        if mode == 'positive': is_valid = worst_min >= t_plus - 1e-5
        elif mode == 'negative': is_valid = worst_max <= t_minus + 1e-5
        elif mode == 'rejected': is_valid = (worst_min >= t_minus - 1e-5) and (worst_max <= t_plus + 1e-5)
        
        if not is_valid: expl_indices.add(idx)

    explicacao_final = [feature_names[i] for i in sorted(list(expl_indices))]
    pred_code = 1 if mode == 'positive' else (0 if mode == 'negative' else 2)
    
    return explicacao_final, pred_code

# ==============================================================================
# PIPELINE PRINCIPAL E EXPORTAÇÃO
# ==============================================================================

def executar_experimento(dataset_name: str):
    print(f"\n[INFO] Executando MINABRO SURROGATE para: {dataset_name.upper()}")
    
    X_full, y_full, _ = carregar_dataset(dataset_name)
    cfg = DATASET_CONFIG.get(dataset_name, {'test_size': 0.3, 'rejection_cost': 0.24})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=cfg['test_size'], random_state=RANDOM_STATE, stratify=y_full
    )

    if cfg.get('subsample_size') and cfg['subsample_size'] < 1.0:
        idx = np.random.choice(len(X_test), int(len(X_test) * cfg['subsample_size']), replace=False)
        X_test, y_test = X_test.iloc[idx], y_test.iloc[idx]

    # 1. Treina o Oráculo (MLP)
    print("[INFO] Treinando Oráculo (MLP)...")
    modelo_mlp = Pipeline([('scaler', MinMaxScaler()), ('mlp', MLPClassifier(**MLP_PARAMS))])
    modelo_mlp.fit(X_train, y_train)
    
    # Prepara métricas para a geração da vizinhança
    std_train = X_train.std().values
    feature_names = X_train.columns.tolist()
    X_test_vals = X_test.values

    # 2. Inicia as Explicações (CRONÔMETRO INICIA AQUI)
    print(f"[INFO] Explicando {len(X_test)} instâncias via Surrogate Local...")
    start_time_total = time.perf_counter()
    resultados = []

    with ProgressBar(total=len(X_test)) as pbar:
        for i in range(len(X_test)):
            start_inst = time.perf_counter()
            inst_vals = X_test_vals[i]
            
            explicacao, pred_code = explicar_instancia_surrogate(
                inst_vals, modelo_mlp, feature_names, std_train, cfg['rejection_cost']
            )
            
            duracao = time.perf_counter() - start_inst
            resultados.append({
                'id': str(X_test.index[i]),
                'y_true': int(y_test.iloc[i]),
                'pred_code': int(pred_code),
                'explicacao': explicacao,
                'tamanho_explicacao': len(explicacao),
                'tempo': duracao
            })
            pbar.update()

    # CRONÔMETRO PARA AQUI
    total_time = time.perf_counter() - start_time_total
    print(f"\n[INFO] Tempo Total: {total_time:.2f}s | Média: {total_time/len(X_test):.4f}s/inst")

    # 3. Empacota e Salva
    salvar_resultados_e_relatorio(dataset_name, resultados, X_test, y_test, total_time, cfg)

def salvar_resultados_e_relatorio(dataset_name, resultados, X_test, y_test, total_time, cfg):
    """Monta o JSON e cria o TXT imediatamente, sem amarrar o cronômetro."""
    preds_rej = np.array([r['pred_code'] for r in resultados])
    mask_rej = (preds_rej == 2)
    
    tamanhos_pos = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 1]
    tamanhos_neg = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 0]
    tamanhos_rej = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 2]

    def calc_stats(t):
        if not t: return {'count': 0, 'mean_length': 0.0, 'std_length': 0.0, 'min_length': 0, 'max_length': 0}
        return {'count': len(t), 'mean_length': float(np.mean(t)), 'std_length': float(np.std(t)), 'min_length': int(np.min(t)), 'max_length': int(np.max(t))}

    data = {
        'config': {
            'dataset_name': dataset_name, 'test_size': cfg['test_size'], 'rejection_cost': cfg['rejection_cost']
        },
        'performance': {
            'rejection_rate': float(np.mean(mask_rej) * 100),
            'num_test_instances': len(X_test), 'num_rejected': int(np.sum(mask_rej))
        },
        'explanation_stats': {
            'positive': calc_stats(tamanhos_pos), 'negative': calc_stats(tamanhos_neg), 'rejected': calc_stats(tamanhos_rej)
        },
        'computation_time': {
            'total': float(total_time), 'mean_per_instance': float(total_time / len(X_test))
        },
        'per_instance': resultados
    }

    # Salva JSON na pasta solicitada
    json_path = os.path.join(DIR_JSON, f"{sanitize_filename(dataset_name)}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    # Gera TXT na pasta solicitada
    txt_path = os.path.join(DIR_REPORT, f"report_{sanitize_filename(dataset_name)}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n          RELATÓRIO - MÉTODO SURROGATE LOCAL (MINABRO)\n" + "="*80 + "\n\n")
        f.write(f"Dataset: {dataset_name} | Instâncias: {len(X_test)} | Tempo Total: {total_time:.2f}s\n")
        f.write(f"Taxa de Rejeição: {data['performance']['rejection_rate']:.2f}%\n")
        for k in ['positive', 'negative', 'rejected']:
            st = data['explanation_stats'][k]
            f.write(f"\n{k.upper()}:\n  Qtd: {st['count']} | Tam. Médio: {st['mean_length']:.2f} | Std: {st['std_length']:.2f}")

    print(f"[OK] JSON salvo em: {json_path}")
    print(f"[OK] TXT salvo em: {txt_path}\n")

if __name__ == '__main__':
    resultado = selecionar_dataset_e_classe()
    if resultado[0] == '__MULTIPLE__':
        for dataset in resultado[4]:
            try: executar_experimento(dataset)
            except Exception as e: print(f"Erro em {dataset}: {e}")
    elif resultado[0]:
        executar_experimento(resultado[0])