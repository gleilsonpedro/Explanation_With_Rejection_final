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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Imports do seu projeto
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
from utils.results_handler import update_method_results
from utils.progress_bar import ProgressBar

# ==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
# ==============================================================================
RANDOM_STATE: int = 42

MNIST_CONFIG = {
    'feature_mode': 'raw',
    'digit_pair': (3, 8),
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
    "wine":                 {'subsample_size': 0.20, 'test_size': 0.3, 'rejection_cost': 0.24},
    "creditcard":           {'subsample_size': 0.03, 'test_size': 0.3, 'rejection_cost': 0.040},
    "covertype":            {'subsample_size': 0.005, 'test_size': 0.3, 'rejection_cost': 0.24},
    "gas_sensor":           {'subsample_size': 0.05, 'test_size': 0.3, 'rejection_cost': 0.045},
    "newsgroups":           {'subsample_size': 0.5, 'test_size': 0.3, 'rejection_cost': 0.24},
    "rcv1":                 {'subsample_size': 0.5, 'test_size': 0.3, 'rejection_cost': 0.24},
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
    'validation_fraction': 0.1
}

def sanitize_filename(filename: str) -> str:
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# ==============================================================================
# CLASSE EXPLAINER ORIENTADA A OBJETOS
# ==============================================================================
class MinabroMLPExplainer:
    """
    Gera explicações abdutivas com opção de rejeição para MLPs 
    baseadas em limites de pior caso global.
    """
    def __init__(self, modelo: Pipeline, X_train: pd.DataFrame, t_plus: float, t_minus: float):
        self.modelo = modelo
        self.t_plus = t_plus
        self.t_minus = t_minus
        
        self.bounds = {
            'min_orig': X_train.values.min(axis=0),
            'max_orig': X_train.values.max(axis=0),
            'feature_names': X_train.columns.tolist()
        }

    def _calcular_sorting_metric_local(self, instance_vals: np.ndarray, delta_frac: float = 0.01) -> np.ndarray:
        n_features = len(instance_vals)
        feature_names = self.bounds['feature_names']

        df_base = pd.DataFrame([instance_vals], columns=feature_names)
        probas_base = np.clip(self.modelo.predict_proba(df_base)[0], 1e-9, 1 - 1e-9)
        score_base = np.log(probas_base[1] / probas_base[0])

        perturbed_batch = np.tile(instance_vals, (n_features, 1))
        
        for j in range(n_features):
            val = instance_vals[j]
            min_val = self.bounds['min_orig'][j]
            max_val = self.bounds['max_orig'][j]
            
            # Tratamento especial para features binárias/categóricas
            if min_val == 0 and max_val == 1 and val in [0, 1]:
                perturbed_batch[j, j] = 1.0 - val 
            else:
                delta = max(abs(val) * delta_frac, 1e-6)
                perturbed_batch[j, j] = val + delta

        df_batch = pd.DataFrame(perturbed_batch, columns=feature_names)
        probas_batch = np.clip(self.modelo.predict_proba(df_batch), 1e-9, 1 - 1e-9)
        scores_batch = np.log(probas_batch[:, 1] / probas_batch[:, 0])

        return np.abs(scores_batch - score_base)

    def _check_validity_model_agnostic(self, fixed_indices: set, instance_vals: np.ndarray, mode: str) -> bool:
        n_features = len(instance_vals)
        inst_min_case = instance_vals.copy()
        inst_max_case = instance_vals.copy()

        for i in range(n_features):
            if i not in fixed_indices:
                inst_min_case[i] = self.bounds['min_orig'][i]
                inst_max_case[i] = self.bounds['max_orig'][i]

        df_batch = pd.DataFrame([inst_min_case, inst_max_case], columns=self.bounds['feature_names'])
        probas_batch = np.clip(self.modelo.predict_proba(df_batch), 1e-9, 1 - 1e-9)
        scores_batch = np.log(probas_batch[:, 1] / probas_batch[:, 0])

        score_min_case = scores_batch[0]
        score_max_case = scores_batch[1]
        EPSILON = 1e-5

        if mode == 'positive':
            return min(score_min_case, score_max_case) >= self.t_plus - EPSILON
        elif mode == 'negative':
            return max(score_min_case, score_max_case) <= self.t_minus + EPSILON
        elif mode == 'rejected':
            return (min(score_min_case, score_max_case) >= self.t_minus - EPSILON) and \
                   (max(score_min_case, score_max_case) <= self.t_plus + EPSILON)
        return False

    def _fase_1_reforco(self, instance_vals: np.ndarray, mode: str, sorting_metric: np.ndarray) -> set:
        expl_indices = set()
        for idx in np.argsort(-sorting_metric):
            if self._check_validity_model_agnostic(expl_indices, instance_vals, mode):
                break
            expl_indices.add(int(idx))
            
        if not self._check_validity_model_agnostic(expl_indices, instance_vals, mode):
            return set(range(len(instance_vals)))
        return expl_indices

    def _fase_2_minimizacao(self, instance_vals: np.ndarray, expl_indices_inicial: set, mode: str, sorting_metric: np.ndarray) -> set:
        expl_indices = expl_indices_inicial.copy()
        for idx in sorted(list(expl_indices), key=lambda i: sorting_metric[i]):
            if len(expl_indices) <= 1: break
            expl_indices.discard(idx)
            if not self._check_validity_model_agnostic(expl_indices, instance_vals, mode):
                expl_indices.add(idx)
        return expl_indices

    def explain_instance(self, instance_vals: np.ndarray) -> list:
        feature_names = self.bounds['feature_names']
        
        df_inst = pd.DataFrame([instance_vals], columns=feature_names)
        probas = np.clip(self.modelo.predict_proba(df_inst)[0], 1e-9, 1 - 1e-9)
        score_raw = np.log(probas[1] / probas[0])

        if score_raw >= self.t_plus: mode = 'positive'
        elif score_raw <= self.t_minus: mode = 'negative'
        else: mode = 'rejected'

        sorting_metric = self._calcular_sorting_metric_local(instance_vals)
        indices_robustos = self._fase_1_reforco(instance_vals, mode, sorting_metric)
        indices_minimos = self._fase_2_minimizacao(instance_vals, indices_robustos, mode, sorting_metric)

        if len(indices_minimos) == 0:
            indices_minimos = {int(np.argmax(sorting_metric))}

        return [feature_names[i] for i in sorted(list(indices_minimos))]

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
        ('model', MLPClassifier(random_state=RANDOM_STATE, **mlp_params))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def detectar_e_corrigir_colapso(modelo, X_train, y_train, X_test, y_test, mlp_params):
    preds = modelo.predict(X_test)
    classes, counts = np.unique(preds, return_counts=True)
    taxa_dominante = counts.max() / counts.sum()

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

def encontrar_thresholds_otimos(X_train, y_train, rejection_cost, mlp_params, modelo_fixo):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    probas = modelo_fixo.predict_proba(X_val)
    probas = np.clip(probas, 1e-9, 1 - 1e-9)
    decision_scores = np.log(probas[:, 1] / probas[:, 0])

    scores_neg = decision_scores[decision_scores < 0]
    scores_pos = decision_scores[decision_scores > 0]

    t_minus_grid = np.linspace(scores_neg.min(), -0.0001, 50) if len(scores_neg) > 0 else np.array([-0.1])
    t_plus_grid  = np.linspace(0.0001, scores_pos.max(), 50)  if len(scores_pos) > 0 else np.array([0.1])

    best_risk = float('inf')
    best_t_plus, best_t_minus = 0.1, -0.1

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

            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, tp, tm

    return float(best_t_plus), float(best_t_minus)

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def executar_experimento_para_dataset(dataset_name: str):
    print(f"\\n[INFO] Executando MINABRO MLP para: {dataset_name.upper()}")
    
    X_full, y_full, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=RANDOM_STATE, stratify=y_full)
    
    cfg = DATASET_CONFIG.get(dataset_name, {})
    subsample_size = cfg.get('subsample_size', None)
    if subsample_size:
        X_test, y_test = aplicar_subsample_teste(X_test, y_test, subsample_size)

    # 1. Treino Inicial
    print("[INFO] Treinando MLP...")
    modelo_mlp = treinar_modelo_mlp(X_train, y_train, MLP_PARAMS)
    
    # 2. Tratamento de Colapso
    X_train_orig = X_train.copy()
    modelo_mlp, X_train_para_threshold, y_train_para_threshold = detectar_e_corrigir_colapso(
        modelo_mlp, X_train, y_train, X_test, y_test, MLP_PARAMS
    )
    
    # 3. Limiares de Rejeição
    t_plus, t_minus = encontrar_thresholds_otimos(
        X_train_para_threshold, y_train_para_threshold, rejection_cost, MLP_PARAMS, modelo_fixo=modelo_mlp
    )
    print(f"[INFO] Thresholds Otimizados: T+={t_plus:.4f}, T-={t_minus:.4f}")

    # 4. Instanciar o Explainer
    explainer = MinabroMLPExplainer(modelo_mlp, X_train_orig, t_plus, t_minus)

    # 5. Loop de Explicação
    print(f"[INFO] Explicando {len(X_test)} instâncias...")
    
    probas = np.clip(modelo_mlp.predict_proba(X_test), 1e-9, 1 - 1e-9)
    scores = np.log(probas[:, 1] / probas[:, 0])
    preds = np.full(len(X_test), 2)
    preds[scores >= t_plus] = 1
    preds[scores <= t_minus] = 0
    
    resultados = []
    start_total = time.perf_counter()
    X_test_vals = X_test.values

    with ProgressBar(total=len(X_test)) as pbar:
        for i in range(len(X_test)):
            start_inst = time.perf_counter()
            inst_vals = X_test_vals[i]
            
            # Chamada super limpa usando a classe
            explicacao = explainer.explain_instance(inst_vals)
            
            duracao = time.perf_counter() - start_inst
            original_idx = str(X_test.index[i])
            
            resultados.append({
                'id': original_idx,
                'pred_code': int(preds[i]),
                'explicacao': sorted(explicacao),
                'tamanho_explicacao': len(explicacao),
                'tempo': duracao
            })
            pbar.update()

    total_time = time.perf_counter() - start_total
    print(f"\\n[INFO] Tempo Total: {total_time:.2f}s | Média: {total_time/len(X_test):.4f}s/inst")

    # 6. Agrupar e Salvar
    mask_rej = (preds == 2)
    y_pred_final = modelo_mlp.predict(X_test)
    
    per_instance_data = []
    for res in resultados:
        idx = res['id']
        idx_int = int(X_test.index.get_loc(int(idx))) if str(idx).isdigit() else int(np.where(X_test.index == idx)[0][0])
        
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

    def calc_exp_stats(tamanhos):
        if not tamanhos: return {'count': 0, 'mean_length': 0.0, 'std_length': 0.0, 'min_length': 0, 'max_length': 0}
        return {
            'count': len(tamanhos), 'mean_length': float(np.mean(tamanhos)),
            'std_length': float(np.std(tamanhos)), 'min_length': int(np.min(tamanhos)), 'max_length': int(np.max(tamanhos))
        }

    tamanhos_pos = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 1]
    tamanhos_neg = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 0]
    tamanhos_rej = [r['tamanho_explicacao'] for r in resultados if r['pred_code'] == 2]

    results_data = {
        'config': {
            'dataset_name': dataset_name, 'test_size': test_size, 'random_state': RANDOM_STATE,
            'rejection_cost': rejection_cost, 'subsample_size': subsample_size
        },
        'thresholds': {'t_plus': float(t_plus), 't_minus': float(t_minus), 'rejection_zone_width': float(t_plus - t_minus)},
        'performance': {
            'accuracy_without_rejection': float(np.mean(y_pred_final == y_test) * 100),
            'accuracy_with_rejection': float(np.mean(preds[~mask_rej] == y_test.iloc[~mask_rej]) * 100) if np.any(~mask_rej) else 100.0,
            'rejection_rate': float(np.mean(mask_rej) * 100),
            'num_test_instances': len(X_test), 'num_rejected': int(np.sum(mask_rej)), 'num_accepted': int(np.sum(~mask_rej))
        },
        'explanation_stats': {
            'positive': calc_exp_stats(tamanhos_pos), 'negative': calc_exp_stats(tamanhos_neg), 'rejected': calc_exp_stats(tamanhos_rej)
        },
        'computation_time': {
            'total': float(total_time), 'mean_per_instance': float(total_time / len(X_test))
        },
        'model': {'params': MLP_PARAMS, 'num_features': len(X_train.columns)},
        'per_instance': per_instance_data
    }

    dataset_json_key_safe = sanitize_filename(dataset_name)
    
    # 1. Salva o JSON usando a sua função padrão
    filepath_json_salvo = update_method_results(method='minabro_mlp', dataset=dataset_json_key_safe, results=results_data)
    
    # 2. Chama o NOVO gerador de relatórios passando o caminho do JSON
    from minabro_mlp_rel import gerar_relatorio_do_json
    
    # OBS: Verifique o caminho real que o update_method_results salva.
    # Geralmente é algo como: 'json/minabro_mlp/{dataset_json_key_safe}.json'
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