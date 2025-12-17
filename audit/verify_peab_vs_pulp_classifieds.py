import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Ensure project root on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.shared_training import get_shared_pipeline
from utils.results_handler import load_method_results

EPSILON = 1e-6

def worst_case_score_for_classified(modelo, coefs: np.ndarray, intercept: float, vals_s: np.ndarray,
                                    indices_fixos: List[int], direction: int, max_abs: float) -> float:
    """
    direction: 1 -> push DOWN (favor classe 0)
               0 -> push UP   (favor classe 1)
    Returns score normalized.
    """
    MIN_VEC = np.zeros_like(coefs)
    MAX_VEC = np.ones_like(coefs)
    if direction == 1:
        X_teste = np.where(coefs > 0, MIN_VEC, MAX_VEC)
    else:
        X_teste = np.where(coefs > 0, MAX_VEC, MIN_VEC)
    if indices_fixos:
        X_teste[indices_fixos] = vals_s[indices_fixos]
    s_raw = intercept + float(np.dot(X_teste, coefs))
    return s_raw / max_abs if max_abs > 0 else s_raw


def main(dataset_name: str = "pima_indians_diabetes"):
    modelo, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)
    max_abs = meta['model_params']['norm_params']['max_abs']

    peab = load_method_results('peab', dataset_name)
    pulp = load_method_results('pulp', dataset_name)
    if not peab or not pulp:
        print("❌ Resultados não encontrados.")
        return

    pulp_size = {str(e['indice']): int(e['tamanho']) for e in pulp.get('explicacoes', [])}
    feature_names = list(X_train.columns)
    col_to_idx = {c: i for i, c in enumerate(feature_names)}
    logreg = modelo.named_steps['model'] if 'model' in modelo.named_steps else modelo.named_steps['modelo']
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    scaler = modelo.named_steps['scaler']

    casos: List[Dict] = []

    for inst in peab.get('per_instance', []):
        rid = str(inst['id'])
        if rid not in pulp_size:
            continue
        if inst.get('rejected', False):
            continue
        size_peab = int(inst['explanation_size'])
        size_pulp = int(pulp_size[rid])
        if size_peab >= size_pulp:
            continue  # só interessam GAPs negativos

        y_pred = int(inst.get('y_pred', -1))
        if y_pred not in (0, 1):
            continue

        feat_names = [f for f in inst.get('explanation', [])]
        idx_fixos = [col_to_idx[f] for f in feat_names if f in col_to_idx]
        if int(rid) not in X_test.index:
            continue
        instancia = X_test.loc[[int(rid)]]
        vals_s = scaler.transform(instancia[feature_names])[0]

        # Para positivas, validar vs t_plus (push DOWN)
        # Para negativas, validar vs t_minus (push UP)
        if y_pred == 1:
            s_norm = worst_case_score_for_classified(modelo, coefs, intercept, vals_s, idx_fixos, 1, max_abs)
            ok = (s_norm >= t_plus - EPSILON)
        else:
            s_norm = worst_case_score_for_classified(modelo, coefs, intercept, vals_s, idx_fixos, 0, max_abs)
            ok = (s_norm <= t_minus + EPSILON)

        casos.append({
            'id': rid,
            'y_pred': y_pred,
            'size_peab': size_peab,
            'size_pulp': size_pulp,
            'score_norm': s_norm,
            't_plus': t_plus,
            't_minus': t_minus,
            'ok': ok,
            'features': feat_names,
        })

    if not casos:
        print("✅ Nenhum GAP negativo em classificadas.")
        return

    falhas = [c for c in casos if not c['ok']]
    print(f"\nClassificadas com GAP negativo: {len(casos)} | Falhas de validação PEAB: {len(falhas)}")
    for c in casos[:10]:
        lim = c['t_plus'] if c['y_pred'] == 1 else c['t_minus']
        cmp = ">=" if c['y_pred'] == 1 else "<="
        status = "OK" if c['ok'] else "FALHA"
        print(f"  • ID {c['id']}: PEAB={c['size_peab']} vs PuLP={c['size_pulp']} -> {status} | "
              f"score_norm={c['score_norm']:.4f} {cmp} {lim:.4f}")

if __name__ == "__main__":
    main()
