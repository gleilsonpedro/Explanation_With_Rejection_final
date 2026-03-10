import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any, Set

# Silencia os avisos chatos do Sklearn para deixar o terminal limpo
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

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
# PRÉ-PROCESSAMENTO: ANTI-COLAPSO
# ==============================================================================
def balancear_dados_treino(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Faz um oversampling simples para evitar que a MLP foque só na classe majoritária."""
    df = pd.concat([X, y], axis=1)
    target_col = y.name
    
    contagem = df[target_col].value_counts()
    if len(contagem) < 2: return X, y
    
    classe_maioria = contagem.idxmax()
    classe_minoria = contagem.idxmin()
    
    df_maioria = df[df[target_col] == classe_maioria]
    df_minoria = df[df[target_col] == classe_minoria]
    
    # Duplica a classe minoritária para igualar o tamanho
    df_minoria_upsampled = df_minoria.sample(len(df_maioria), replace=True, random_state=RANDOM_STATE)
    df_balanceado = pd.concat([df_maioria, df_minoria_upsampled]).sample(frac=1, random_state=RANDOM_STATE)
    
    return df_balanceado.drop(target_col, axis=1), df_balanceado[target_col]

# ==============================================================================
# MOTOR DE EXPLICAÇÃO: SURROGATE LOCAL
# ==============================================================================

def gerar_vizinhanca_local_fronteira(instancia: np.ndarray, X_pool: np.ndarray, std_train: np.ndarray, modelo_mlp: Pipeline, feature_names: list, num_clones: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Gera clones interpolando uma linha reta entre a instância alvo e o vizinho mais próximo da classe oposta.
    (Ideia do 'Tiro de Sniper' sugerida pelo orientador).
    """
    # 1. Qual é a classe da instância original?
    df_inst = pd.DataFrame([instancia], columns=feature_names)
    classe_original = modelo_mlp.predict(df_inst)[0]
    
    # 2. Acha quem é da classe oposta no pool de dados (X_train)
    df_pool = pd.DataFrame(X_pool, columns=feature_names)
    preds_pool = modelo_mlp.predict(df_pool)
    opostos_idx = np.where(preds_pool != classe_original)[0]
    
    # Fallback de segurança (caso extremo onde o modelo previu só 1 classe no treino)
    if len(opostos_idx) == 0:
        ruido = np.random.normal(0, std_train, size=(num_clones, len(instancia)))
        clones = instancia + ruido
        df_clones = pd.DataFrame(clones, columns=feature_names)
        return df_clones, modelo_mlp.predict(df_clones)
        
    X_opostos = X_pool[opostos_idx]
    
    # 3. Calcula distância Euclidiana para achar o "inimigo mais próximo"
    distancias = np.linalg.norm(X_opostos - instancia, axis=1)
    inimigo_mais_proximo = X_opostos[np.argmin(distancias)]
    
    # 4. Interpolação Linear (A Rodovia)
    # Gera alphas de -0.1 até 1.1 (passando um pouquinho dos limites para garantir o corte da fronteira)
    alphas = np.linspace(-0.1, 1.1, num_clones)[:, np.newaxis]
    vetor_direcao = inimigo_mais_proximo - instancia
    clones_na_reta = instancia + alphas * vetor_direcao
    
    # 5. Adiciona um micro-ruído para criar um "cilindro" e permitir que a LogReg treine em n-dimensões
    ruido_cilindro = np.random.normal(0, std_train * 0.05, size=clones_na_reta.shape)
    clones_finais = clones_na_reta + ruido_cilindro
    
    # Garante que a instância original exata seja o ponto zero
    clones_finais[0] = instancia
    
    df_clones = pd.DataFrame(clones_finais, columns=feature_names)
    y_oraculo = modelo_mlp.predict(df_clones)
    
    return df_clones, y_oraculo

def encontrar_thresholds_locais(modelo_lr: Pipeline, X_local: pd.DataFrame, y_local: np.ndarray, rejection_cost: float):
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

    return best_t_plus, best_t_minus, decision_scores[0]

def check_fidelity_mlp(instancia_vals: np.ndarray, expl_indices: set, bounds: dict, modelo_mlp: Pipeline, original_pred: int) -> bool:
    """Verifica Fidelidade baseada no Pior Caso LOCAL (limites dos clones)"""
    inst_min_case = instancia_vals.copy()
    inst_max_case = instancia_vals.copy()
    
    for i in range(len(instancia_vals)):
        if i not in expl_indices:
            inst_min_case[i] = bounds['min_local'][i]
            inst_max_case[i] = bounds['max_local'][i]
            
    df_batch = pd.DataFrame([inst_min_case, inst_max_case], columns=bounds['feature_names'])
    preds = modelo_mlp.predict(df_batch)
    
    return bool((preds[0] == original_pred) and (preds[1] == original_pred))

def explicar_instancia_surrogate(instancia_vals: np.ndarray, X_pool_vals: np.ndarray, modelo_mlp: Pipeline, feature_names: list, 
                                 std_train: np.ndarray, rejection_cost: float) -> Tuple[List[str], int, bool]:
    
    # 1. Cria a vizinhança mirando na fronteira (Sniper)
    df_clones, y_oraculo = gerar_vizinhanca_local_fronteira(instancia_vals, X_pool_vals, std_train, modelo_mlp, feature_names, num_clones=1000)
    original_pred = y_oraculo[0]
    
    # Calcula os limites EXATOS dessa vizinhança para o teste de fidelidade local
    local_bounds = {
        'min_local': df_clones.min().values, 
        'max_local': df_clones.max().values, 
        'feature_names': feature_names
    }

    if len(np.unique(y_oraculo)) == 1:
        feature_mais_importante = feature_names[np.argmax(np.abs(instancia_vals))]
        is_faithful = check_fidelity_mlp(instancia_vals, {feature_names.index(feature_mais_importante)}, local_bounds, modelo_mlp, original_pred)
        return [feature_mais_importante], original_pred, is_faithful

    # 2. Treina LogReg
    modelo_local = Pipeline([('scaler', MinMaxScaler()), ('model', LogisticRegression(**LOGREG_PARAMS, random_state=RANDOM_STATE))])
    modelo_local.fit(df_clones, y_oraculo)

    t_plus, t_minus, score_original = encontrar_thresholds_locais(modelo_local, df_clones, y_oraculo, rejection_cost)

    scaler = modelo_local.named_steps['scaler']
    lr = modelo_local.named_steps['model']
    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    # Correção do Aviso do Sklearn (Passando um DataFrame com nomes)
    df_instancia = pd.DataFrame([instancia_vals], columns=feature_names)
    vals_s = scaler.transform(df_instancia)[0]
    
    X_min_contribution = np.where(coefs > 0, 0.0, 1.0)
    X_max_contribution = np.where(coefs > 0, 1.0, 0.0)

    base_min_score = intercept + np.dot(coefs, X_min_contribution)
    base_max_score = intercept + np.dot(coefs, X_max_contribution)
    
    gains_from_min = (vals_s * coefs) - (X_min_contribution * coefs)
    losses_from_max = (X_max_contribution * coefs) - (vals_s * coefs)

    if score_original >= t_plus: mode, sort_metric = 'positive', gains_from_min
    elif score_original <= t_minus: mode, sort_metric = 'negative', losses_from_max
    else: mode, sort_metric = 'rejected', gains_from_min 

    # FASE 1: Construção Gulosa
    expl_indices = set()
    for idx in np.argsort(-sort_metric):
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

    # FASE 2: Minimização
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
    
    # Avalia a Fidelidade com Pior Caso Local
    is_faithful = True
    if pred_code != 2: 
        is_faithful = check_fidelity_mlp(instancia_vals, expl_indices, local_bounds, modelo_mlp, original_pred)
    
    return explicacao_final, pred_code, is_faithful

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

    # Anti-Colapso
    X_train_bal, y_train_bal = balancear_dados_treino(X_train, y_train)

    # Treina Oráculo
    print("[INFO] Treinando Oráculo (MLP)...")
    modelo_mlp = Pipeline([('scaler', MinMaxScaler()), ('mlp', MLPClassifier(**MLP_PARAMS))])
    modelo_mlp.fit(X_train_bal, y_train_bal)
    X_train_vals = X_train_bal.values

    std_train = X_train.std().values
    feature_names = X_train.columns.tolist()
    X_test_vals = X_test.values

    # Inicia Explicações
    print(f"[INFO] Explicando {len(X_test)} instâncias via Surrogate Local...")
    start_time_total = time.perf_counter()
    resultados = []

    with ProgressBar(total=len(X_test)) as pbar:
        for i in range(len(X_test)):
            start_inst = time.perf_counter()
            inst_vals = X_test_vals[i]
            
            explicacao, pred_code, is_faithful = explicar_instancia_surrogate(
                inst_vals, X_train_vals, modelo_mlp, feature_names, std_train, cfg['rejection_cost']
            )
            
            duracao = time.perf_counter() - start_inst
            resultados.append({
                'id': str(X_test.index[i]),
                'y_true': int(y_test.iloc[i]),
                'pred_code': int(pred_code),
                'explicacao': explicacao,
                'tamanho_explicacao': len(explicacao),
                'faithful': is_faithful,
                'tempo': duracao
            })
            pbar.update()

    total_time = time.perf_counter() - start_time_total
    print(f"\n[INFO] Tempo Total: {total_time:.2f}s | Média: {total_time/len(X_test):.4f}s/inst")

    salvar_resultados_e_relatorio(dataset_name, resultados, X_test, y_test, total_time, cfg)

def salvar_resultados_e_relatorio(dataset_name, resultados, X_test, y_test, total_time, cfg):
    preds_rej = np.array([r['pred_code'] for r in resultados])
    mask_rej = (preds_rej == 2)
    
    tamanhos_pos = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 1]
    tamanhos_neg = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 0]
    tamanhos_rej = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 2]

    fidelidade_aceitos = [r['faithful'] for r in resultados if r['pred_code'] != 2]
    taxa_fidelidade = (np.mean(fidelidade_aceitos) * 100) if fidelidade_aceitos else 0.0

    # Extrai Top Features
    todas_features_pos = [f for r in resultados if r['pred_code'] == 1 for f in r['explicacao']]
    todas_features_neg = [f for r in resultados if r['pred_code'] == 0 for f in r['explicacao']]
    top10_pos = Counter(todas_features_pos).most_common(10)
    top10_neg = Counter(todas_features_neg).most_common(10)

    def calc_stats(t):
        if not t: return {'count': 0, 'mean_length': 0.0, 'std_length': 0.0, 'min_length': 0, 'max_length': 0}
        return {'count': len(t), 'mean_length': float(np.mean(t)), 'std_length': float(np.std(t)), 'min_length': int(np.min(t)), 'max_length': int(np.max(t))}

    data = {
        'config': {'dataset_name': dataset_name, 'test_size': cfg['test_size'], 'rejection_cost': cfg['rejection_cost']},
        'performance': {
            'rejection_rate': float(np.mean(mask_rej) * 100),
            'fidelity_rate_local': float(taxa_fidelidade),
            'num_test_instances': len(X_test), 'num_rejected': int(np.sum(mask_rej))
        },
        'explanation_stats': {'positive': calc_stats(tamanhos_pos), 'negative': calc_stats(tamanhos_neg), 'rejected': calc_stats(tamanhos_rej)},
        'computation_time': {'total': float(total_time), 'mean_per_instance': float(total_time / len(X_test))},
        'per_instance': resultados
    }

    json_path = os.path.join(DIR_JSON, f"{sanitize_filename(dataset_name)}.json")
    with open(json_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
    
    txt_path = os.path.join(DIR_REPORT, f"report_{sanitize_filename(dataset_name)}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n          RELATÓRIO - MÉTODO SURROGATE LOCAL (MINABRO)\n" + "="*80 + "\n\n")
        f.write(f"Dataset: {dataset_name} | Instâncias: {len(X_test)} | Tempo Total: {total_time:.2f}s\n")
        f.write(f"Taxa de Rejeição: {data['performance']['rejection_rate']:.2f}%\n")
        f.write(f"Fidelidade Local da Explicação: {data['performance']['fidelity_rate_local']:.2f}%\n")
        
        for k in ['positive', 'negative', 'rejected']:
            st = data['explanation_stats'][k]
            f.write(f"\n{k.upper()}:\n  Qtd: {st['count']} | Tam. Médio: {st['mean_length']:.2f} | Std: {st['std_length']:.2f}")
            
        f.write("\n\n" + "-"*80 + "\nTOP 10 FEATURES NAS EXPLICAÇÕES POSITIVAS\n" + "-"*80 + "\n")
        for feat, count in top10_pos: f.write(f"  {feat}: {count} vezes\n")
        
        f.write("\n" + "-"*80 + "\nTOP 10 FEATURES NAS EXPLICAÇÕES NEGATIVAS\n" + "-"*80 + "\n")
        for feat, count in top10_neg: f.write(f"  {feat}: {count} vezes\n")

    print(f"[OK] JSON salvo em: {json_path}\n[OK] TXT salvo em: {txt_path}\n")

if __name__ == '__main__':
    resultado = selecionar_dataset_e_classe()
    if resultado[0] == '__MULTIPLE__':
        for dataset in resultado[4]:
            try: executar_experimento(dataset)
            except Exception as e: print(f"Erro em {dataset}: {e}")
    elif resultado[0]:
        executar_experimento(resultado[0])