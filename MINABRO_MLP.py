

'''Treina a rede neural (MLP), define os limiares de rejeição (thresholds) 
e implementa a lógica de criar "clones" ao redor de uma instância 
para treinar uma Regressão Logística local. É aqui que a mágica da abdução 
acontece para encontrar o menor conjunto de características que explica a 
decisão (ou a dúvida) da MLP.'''


import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any, Set

# Silencia os avisos do Sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Imports do seu projeto (Mantenha sua estrutura)
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
from utils.results_handler import update_method_results
from utils.progress_bar import ProgressBar

# ==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
# ==============================================================================
RANDOM_STATE: int = 42

MNIST_CONFIG = {
    'feature_mode': 'raw',
    'digit_pair': (3, 8), # atebçao a mudança de pares de diguitos é feita no peab.py
    'top_k_features': None,
    'test_size': 0.3,
    'rejection_cost': 0.24,
    'subsample_size': 0.01
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
    "creditcard":           {'subsample_size': 0.03, 'test_size': 0.3, 'rejection_cost': 0.040},
    "covertype":            {'subsample_size': 0.005, 'test_size': 0.3, 'rejection_cost': 0.24},
    "gas_sensor":           {'subsample_size': 0.05, 'test_size': 0.3, 'rejection_cost': 0.045},
    "newsgroups":           {'subsample_size': 0.05, 'test_size': 0.3, 'rejection_cost': 0.24},
    "rcv1":                 {'subsample_size': 0.05, 'test_size': 0.3, 'rejection_cost': 0.24},
}

OUTPUT_BASE_DIR: str = 'results/report/minabro_mlp'
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

MLP_PARAMS = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'alpha': 0.0001,
    'learning_rate_init': 0.001,
    'early_stopping': True,
    'n_iter_no_change': 10,
    'validation_fraction': 0.1,
    'random_state': RANDOM_STATE
}

def sanitize_filename(filename: str) -> str:
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# ==============================================================================
# LEITURA DOS HIPERPARÂMETROS OTIMIZADOS (GRID SEARCH)
# ==============================================================================
def carregar_hiperparametros_locais(dataset_name: str) -> dict:
    caminho_json = os.path.join('json', 'best_hyperparameters.json')
    default_params = {'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear', 'max_iter': 500}
    
    if os.path.exists(caminho_json):
        try:
            with open(caminho_json, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            if dataset_name in dados:
                return dados[dataset_name]
        except Exception as e:
            print(f"[AVISO] Erro ao ler hiperparâmetros: {e}")
            
    return default_params

# ==============================================================================
# CLASSE EXPLAINER: SURROGATE LOCAL (TÉCNICA DO ESPELHO)
# ==============================================================================
class MinabroMLPSurrogateExplainer:
    def __init__(self, modelo_mlp: Pipeline, X_pool_df: pd.DataFrame, rejection_cost: float, logreg_params: dict):
        self.modelo_mlp = modelo_mlp
        self.X_pool_vals = X_pool_df.values
        self.feature_names = X_pool_df.columns.tolist()
        self.std_train = X_pool_df.std().values
        self.rejection_cost = rejection_cost
        self.logreg_params = logreg_params

    def _gerar_vizinhanca_local_fronteira(self, instancia: np.ndarray, num_clones: int = 1000) -> Tuple[np.ndarray, np.ndarray, dict]:
        # NUMPY PURO: Nada de pd.DataFrame aqui!
        classe_alvo = self.modelo_mlp.predict([instancia])[0]
        
        preds_pool = self.modelo_mlp.predict(self.X_pool_vals)
        opostos_idx = np.where(preds_pool != classe_alvo)[0]
        
        if len(opostos_idx) == 0:
            ruido = np.random.normal(0, self.std_train * 0.15, size=(num_clones, len(instancia)))
            clones_finais = instancia + ruido
            y_oraculo = self.modelo_mlp.predict(clones_finais)
        else:
            X_opostos = self.X_pool_vals[opostos_idx]
            distancias = np.linalg.norm(X_opostos - instancia, axis=1)
            inimigo_direcao = X_opostos[np.argmin(distancias)]
            
            passos = np.linspace(0, 1, 100)[:, np.newaxis]
            caminho = instancia + passos * (inimigo_direcao - instancia)
            
            preds_caminho = self.modelo_mlp.predict(caminho)
            
            mudancas = np.where(preds_caminho != classe_alvo)[0]
            if len(mudancas) > 0:
                idx_fronteira = mudancas[0]
                ponto_fronteira_exato = caminho[idx_fronteira]
                vetor_ate_fronteira = ponto_fronteira_exato - instancia
                inimigo_final = instancia + 2.0 * vetor_ate_fronteira
            else:
                inimigo_final = inimigo_direcao 
                
            alphas = np.linspace(0.0, 1.0, num_clones)[:, np.newaxis]
            vetor_balanceado = inimigo_final - instancia
            linha_central = instancia + alphas * vetor_balanceado
            
            ruido_gordo = np.random.normal(0, self.std_train * 0.15, size=linha_central.shape)
            clones_finais = linha_central + ruido_gordo
            
            clones_finais[0] = instancia 
            clones_finais[1] = inimigo_final
            
            y_oraculo = self.modelo_mlp.predict(clones_finais)

        local_bounds = {
            'min_local': clones_finais.min(axis=0), 
            'max_local': clones_finais.max(axis=0), 
            'feature_names': self.feature_names
        }
        
        return clones_finais, y_oraculo, local_bounds

    def _encontrar_thresholds_locais(self, modelo_lr: Pipeline, X_local: np.ndarray, y_local: np.ndarray):
        probas = np.clip(modelo_lr.predict_proba(X_local), 1e-9, 1 - 1e-9)
        decision_scores = np.log(probas[:, 1] / probas[:, 0])

        scores_neg = decision_scores[decision_scores < 0]
        scores_pos = decision_scores[decision_scores > 0]

        t_minus_grid = np.linspace(scores_neg.min(), -0.001, 20) if len(scores_neg) > 0 else np.array([-0.1])
        t_plus_grid  = np.linspace(0.001, scores_pos.max(), 20)  if len(scores_pos) > 0 else np.array([0.1])

        best_risk, best_t_plus, best_t_minus = float('inf'), 0.1, -0.1
        max_rejection_rate = 0.35  # <-- SIMETRIA: Mesma restrição do modelo global

        for tm in t_minus_grid:
            for tp in t_plus_grid:
                if not (tm < 0 < tp): continue
                acc_mask = (decision_scores >= tp) | (decision_scores <= tm)
                preds = np.full(y_local.shape, -1)
                preds[decision_scores >= tp] = 1
                preds[decision_scores <= tm] = 0

                error = np.mean(preds[acc_mask] != y_local[acc_mask]) if np.any(acc_mask) else 0.0
                rejection_rate = 1.0 - np.mean(acc_mask)
                risk = error + self.rejection_cost * rejection_rate

                # NOVA REGRA: O surrogate só pode adotar os limiares se não rejeitar excessivamente
                if risk < best_risk and rejection_rate <= max_rejection_rate:
                    best_risk, best_t_plus, best_t_minus = risk, tp, tm

        # Fallback de segurança caso a vizinhança seja caótica
        if best_risk == float('inf'):
            best_t_plus, best_t_minus = 0.01, -0.01

        return best_t_plus, best_t_minus, decision_scores[0]
# METODO DE TESTE DOI PROFESSOr - depois das features travadas na explicação ele gera os casos extremos (min e max) e testa se a MLP continua com a mesma predição, garantindo a fidelidade da explicação.
    def _check_fidelity_mlp(self, instancia_vals: np.ndarray, expl_indices: set, bounds: dict, original_pred: int) -> bool:
        if not expl_indices:
            return True
            
        inst_min_case = instancia_vals.copy()
        inst_max_case = instancia_vals.copy()
        
        for i in range(len(instancia_vals)):
            if i not in expl_indices:
                inst_min_case[i] = bounds['min_local'][i]
                inst_max_case[i] = bounds['max_local'][i]
                
        # NUMPY PURO
        preds = self.modelo_mlp.predict([inst_min_case, inst_max_case])
        return bool((preds[0] == original_pred) and (preds[1] == original_pred))

    def explain_instance(self, instancia_vals: np.ndarray) -> Tuple[List[str], int, bool]:
        clones_finais, y_oraculo, local_bounds = self._gerar_vizinhanca_local_fronteira(instancia_vals)
        original_pred = y_oraculo[0]
    
        if len(np.unique(y_oraculo)) == 1:
            return [], original_pred, True, local_bounds
    
        modelo_local = Pipeline([
            ('scaler', MinMaxScaler()), 
            ('model', LogisticRegression(**self.logreg_params, random_state=42))
        ])
        
        modelo_local.fit(clones_finais, y_oraculo)
    
        t_plus, t_minus, score_original = self._encontrar_thresholds_locais(modelo_local, clones_finais, y_oraculo)
    
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
    
        if score_original >= t_plus:
            mode, sort_metric = 'positive', gains_from_min
        elif score_original <= t_minus:
            mode, sort_metric = 'negative', losses_from_max
        else:
            mode, sort_metric = 'rejected', gains_from_min
    
        expl_indices = set()
        for idx in np.argsort(-sort_metric):
            curr_gain = np.sum(gains_from_min[list(expl_indices)]) if expl_indices else 0.0
            curr_loss = np.sum(losses_from_max[list(expl_indices)]) if expl_indices else 0.0
            worst_min = base_min_score + curr_gain
            worst_max = base_max_score - curr_loss
            
            is_valid = False
            if mode == 'positive':
                is_valid = worst_min >= t_plus - 1e-5
            elif mode == 'negative':
                is_valid = worst_max <= t_minus + 1e-5
            elif mode == 'rejected':
                is_valid = (worst_min >= t_minus - 1e-5) and (worst_max <= t_plus + 1e-5)
            
            if is_valid:
                break
            expl_indices.add(int(idx))
    
        for idx in sorted(list(expl_indices), key=lambda i: sort_metric[i]):
            if len(expl_indices) <= 1:
                break
            expl_indices.remove(idx)
            
            curr_gain = np.sum(gains_from_min[list(expl_indices)])
            curr_loss = np.sum(losses_from_max[list(expl_indices)])
            worst_min = base_min_score + curr_gain
            worst_max = base_max_score - curr_loss
            
            is_valid = False
            if mode == 'positive':
                is_valid = worst_min >= t_plus - 1e-5
            elif mode == 'negative':
                is_valid = worst_max <= t_minus + 1e-5
            elif mode == 'rejected':
                is_valid = (worst_min >= t_minus - 1e-5) and (worst_max <= t_plus + 1e-5)
            
            if not is_valid:
                expl_indices.add(idx)
    
        explicacao_final = [self.feature_names[i] for i in sorted(list(expl_indices))]
        pred_code = 1 if mode == 'positive' else (0 if mode == 'negative' else 2)
        
        is_faithful = True
        if pred_code != 2:
            is_faithful = self._check_fidelity_mlp(instancia_vals, expl_indices, local_bounds, original_pred)
        
        return explicacao_final, pred_code, is_faithful, local_bounds

# ==============================================================================
# PIPELINE DE TREINAMENTO E AVALIAÇÃO
# ==============================================================================
def configurar_experimento(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], float, float]:
    if dataset_name == 'mnist':
        from data import datasets as ds_module
        cfg = DATASET_CONFIG.get(dataset_name, {})
        ds_module.set_mnist_options(cfg.get('feature_mode', 'raw'), cfg.get('digit_pair', None))
    
    X, y, nomes_classes = carregar_dataset(dataset_name)
    cfg = DATASET_CONFIG.get(dataset_name, {'test_size': 0.3, 'rejection_cost': 0.24})
    return X, y, nomes_classes, cfg['rejection_cost'], cfg['test_size']

def aplicar_subsample_teste(X_test: pd.DataFrame, y_test: pd.Series, subsample_size: float) -> Tuple[pd.DataFrame, pd.Series]:
    if subsample_size and subsample_size < 1.0:
        idx = np.arange(len(y_test))
        sample_idx, _ = train_test_split(
            idx, test_size=(1 - subsample_size), random_state=RANDOM_STATE, stratify=y_test
        )
        X_test = X_test.iloc[sample_idx] if isinstance(X_test, pd.DataFrame) else X_test[sample_idx]
        y_test = y_test.iloc[sample_idx] if isinstance(y_test, pd.Series) else y_test[sample_idx]
        print(f"[SUBSAMPLE] Teste reduzido para {len(y_test)} instâncias.")
    return X_test, y_test

def treinar_modelo_mlp(X_train, y_train, mlp_params):
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', MLPClassifier(**mlp_params))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def detectar_e_corrigir_colapso(modelo, X_train, y_train, X_test, y_test, mlp_params):
    preds = modelo.predict(X_test)
    classes, counts = np.unique(preds, return_counts=True)
    taxa_dominante = counts.max() / counts.sum() if len(counts) > 0 else 1.0

    if taxa_dominante < 0.95:
        return modelo, X_train, y_train

    print("[AVISO] Colapso detectado! Balanceando X_train por oversampling...")
    classes_orig, counts_orig = np.unique(y_train, return_counts=True)
    n_max = counts_orig.max()
    X_parts, y_parts = [], []

    X_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.copy()
    y_s = pd.Series(y_train.values if hasattr(y_train, 'values') else y_train)

    for cls in classes_orig:
        mask = (y_s == cls)
        Xc, yc = X_df[mask.values], y_s[mask.values]
        if len(Xc) < n_max:
            Xc, yc = resample(Xc, yc, replace=True, n_samples=n_max, random_state=RANDOM_STATE)
        X_parts.append(Xc)
        y_parts.append(yc)

    X_bal = pd.concat(X_parts).reset_index(drop=True)
    y_bal = pd.concat(y_parts).reset_index(drop=True)

    mlp_params_bal = {**mlp_params, 'max_iter': 1000, 'learning_rate_init': 0.0005, 'n_iter_no_change': 20}
    modelo_bal = treinar_modelo_mlp(X_bal, y_bal, mlp_params_bal)
    
    return modelo_bal, X_bal, y_bal

def encontrar_thresholds_otimos(X_train, y_train, rejection_cost, modelo_fixo):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    probas = np.clip(modelo_fixo.predict_proba(X_val), 1e-9, 1 - 1e-9)
    decision_scores = np.log(probas[:, 1] / probas[:, 0])

    scores_neg = decision_scores[decision_scores < 0]
    scores_pos = decision_scores[decision_scores > 0]

    t_minus_grid = np.linspace(scores_neg.min(), -0.0001, 50) if len(scores_neg) > 0 else np.array([-0.1])
    t_plus_grid  = np.linspace(0.0001, scores_pos.max(), 50)  if len(scores_pos) > 0 else np.array([0.1])

    best_risk = float('inf')
    best_t_plus, best_t_minus = 0.1, -0.1
    max_rejection_rate = 0.35  # <-- NOVA REGRA: O sistema não pode rejeitar mais de 35%

    for tm in t_minus_grid:
        for tp in t_plus_grid:
            if not (tm < 0 < tp): continue
            accepted_mask = (decision_scores >= tp) | (decision_scores <= tm)
            preds = np.full(y_val.shape, -1)
            preds[decision_scores >= tp] = 1
            preds[decision_scores <= tm] = 0

            error = np.mean(preds[accepted_mask] != y_val.values[accepted_mask]) if np.any(accepted_mask) else 0.0
            rejection_rate = 1.0 - np.mean(accepted_mask)
            risk = error + rejection_cost * rejection_rate

            # NOVA REGRA: Só aceita o limiar se a rejeição for menor ou igual ao teto
            if risk < best_risk and rejection_rate <= max_rejection_rate:
                best_risk, best_t_plus, best_t_minus = risk, tp, tm

    # Fallback caso todas as combinações rejeitem demais (usa limiares bem fechados)
    if best_risk == float('inf'):
        best_t_plus, best_t_minus = 0.01, -0.01

    return float(best_t_plus), float(best_t_minus)

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n[INFO] Executando MINABRO MLP (Surrogate) para: {dataset_name.upper()}")
    
    X_full, y_full, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=RANDOM_STATE, stratify=y_full)
    
    cfg = DATASET_CONFIG.get(dataset_name, {})
    subsample_size = cfg.get('subsample_size', None)
    if subsample_size:
        X_test, y_test = aplicar_subsample_teste(X_test, y_test, subsample_size)

    # 1. Treino Inicial do Oráculo
    print("[INFO] Treinando Oráculo (MLP)...")
    modelo_mlp = treinar_modelo_mlp(X_train, y_train, MLP_PARAMS)
    
    # 2. Tratamento de Colapso
    X_train_orig = X_train.copy()
    modelo_mlp, X_train_para_threshold, y_train_para_threshold = detectar_e_corrigir_colapso(
        modelo_mlp, X_train, y_train, X_test, y_test, MLP_PARAMS
    )
    
    # 3. Limiares Globais do MLP (Avaliação da Caixa-Preta)
    t_plus_global, t_minus_global = encontrar_thresholds_otimos(
        X_train_para_threshold, y_train_para_threshold, rejection_cost, modelo_fixo=modelo_mlp
    )
    
    probas_mlp = np.clip(modelo_mlp.predict_proba(X_test), 1e-9, 1 - 1e-9)
    scores_mlp = np.log(probas_mlp[:, 1] / probas_mlp[:, 0])
    mask_rej_global = (scores_mlp > t_minus_global) & (scores_mlp < t_plus_global)
    y_pred_final = modelo_mlp.predict(X_test)
    
    print(f"[INFO] Thresholds Globais do MLP: T+={t_plus_global:.4f}, T-={t_minus_global:.4f}")

    # 4. Instanciar o Explainer Surrogate 
    logreg_params = carregar_hiperparametros_locais(dataset_name)
    print(f"[INFO] Hiperparâmetros locais carregados: {logreg_params}")
    explainer = MinabroMLPSurrogateExplainer(modelo_mlp, X_train_orig, rejection_cost, logreg_params)

    # 5. Loop de Explicação
    print(f"[INFO] Explicando {len(X_test)} instâncias via Surrogate Local...")
    
    resultados = []
    start_total = time.perf_counter()
    X_test_vals = X_test.values

    with ProgressBar(total=len(X_test)) as pbar:
        for i in range(len(X_test)):
            start_inst = time.perf_counter()
            inst_vals = X_test_vals[i]
            
            explicacao, pred_code, is_faithful, _ = explainer.explain_instance(inst_vals)
            
            duracao = time.perf_counter() - start_inst
            original_idx = str(X_test.index[i])
            
            resultados.append({
                'id': original_idx,
                'pred_code': int(pred_code),
                'explicacao': sorted(explicacao),
                'tamanho_explicacao': len(explicacao),
                'faithful': is_faithful,
                'tempo': duracao
            })
            pbar.update()

    total_time = time.perf_counter() - start_total
    print(f"\n[INFO] Tempo Total: {total_time:.2f}s | Média: {total_time/len(X_test):.4f}s/inst")

    # 6. Agrupar e Salvar (Estrutura Blindada)
    per_instance_data = []
    for res in resultados:
        idx = res['id']
        idx_int = int(X_test.index.get_loc(int(idx))) if str(idx).isdigit() else int(np.where(X_test.index == idx)[0][0])
        
        per_instance_data.append({
            'id': idx,
            'y_true': int(y_test.iloc[idx_int]),
            'y_pred': int(y_pred_final[idx_int]),
            'rejected': bool(res['pred_code'] == 2),
            'explanation': res['explicacao'],
            'explanation_size': res['tamanho_explicacao'],
            'computation_time': res.get('tempo', 0.0)
        })

    def calc_exp_stats(tamanhos):
        if not tamanhos: return {'count': 0, 'mean_length': 0.0, 'std_length': 0.0, 'min_length': 0, 'max_length': 0}
        return {
            'count': len(tamanhos), 'mean_length': float(np.mean(tamanhos)),
            'std_length': float(np.std(tamanhos)), 'min_length': int(np.min(tamanhos)), 'max_length': int(np.max(tamanhos))
        }

    tamanhos_pos = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 1]
    tamanhos_neg = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 0]
    tamanhos_rej = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 2]

    fidelidade_aceitos = [r['faithful'] for r in resultados if r['pred_code'] != 2]
    taxa_fidelidade = (np.mean(fidelidade_aceitos) * 100) if fidelidade_aceitos else 0.0

    results_data = {
        'config': {
            'dataset_name': dataset_name, 'test_size': test_size, 'random_state': RANDOM_STATE,
            'rejection_cost': rejection_cost, 'subsample_size': subsample_size
        },
        'thresholds_globais_mlp': {
            't_plus_global': float(t_plus_global), 
            't_minus_global': float(t_minus_global), 
            'rejection_zone_width': float(t_plus_global - t_minus_global)
        },
        'performance_oraculo_mlp': {
            'accuracy_without_rejection': float(np.mean(y_pred_final == y_test) * 100),
            'accuracy_with_rejection': float(np.mean(y_pred_final[~mask_rej_global] == y_test.iloc[~mask_rej_global]) * 100) if np.any(~mask_rej_global) else 100.0,
            'rejection_rate_global': float(np.mean(mask_rej_global) * 100),
            'num_test_instances': len(X_test), 
            'num_rejected': int(np.sum(mask_rej_global)), 
            'num_accepted': int(np.sum(~mask_rej_global))
        },
        'performance_explicacoes_locais': {
            'fidelity_rate_worst_case': float(taxa_fidelidade),
            'positive': calc_exp_stats(tamanhos_pos), 
            'negative': calc_exp_stats(tamanhos_neg), 
            'rejected': calc_exp_stats(tamanhos_rej)
        },
        'computation_time': {
            'total': float(total_time), 'mean_per_instance': float(total_time / len(X_test))
        },
        'model': {'params': MLP_PARAMS, 'num_features': len(X_train.columns)},
        'per_instance': per_instance_data
    }

    dataset_json_key_safe = sanitize_filename(dataset_name)
    
    filepath_json_salvo = update_method_results(method='minabro_mlp', dataset=dataset_json_key_safe, results=results_data)
    
    from minabro_mlp_rel import gerar_relatorio_do_json
    caminho_do_json_gerado = f"json/minabro_mlp/{dataset_json_key_safe}.json" 
    
    if os.path.exists(caminho_do_json_gerado):
        gerar_relatorio_do_json(caminho_do_json_gerado)
    else:
        print(f"[ERRO] JSON não encontrado no caminho esperado ({caminho_do_json_gerado}) para gerar o relatório.")

if __name__ == '__main__':
    resultado = selecionar_dataset_e_classe()
    if resultado[0] == '__MULTIPLE__':
        for dataset in resultado[4]:
            try: executar_experimento_para_dataset(dataset)
            except Exception as e: print(f"Erro em {dataset}: {e}")
    elif resultado[0]:
        executar_experimento_para_dataset(resultado[0])