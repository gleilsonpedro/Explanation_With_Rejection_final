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

def worst_case_scores_for_reject(modelo, coefs: np.ndarray, intercept: float, vals_s: np.ndarray,
                                  indices_fixos: List[int], t_plus: float, t_minus: float, max_abs: float) -> Tuple[float, float, bool, bool]:
    # Vetores 0..1 no espaço do MinMaxScaler
    MIN_VEC = np.zeros_like(coefs)
    MAX_VEC = np.ones_like(coefs)

    # Baixo: tentar cair abaixo de t_minus
    X_baixo = np.where(coefs > 0, MIN_VEC, MAX_VEC)
    if indices_fixos:
        X_baixo[indices_fixos] = vals_s[indices_fixos]
    s_baixo_raw = intercept + float(np.dot(X_baixo, coefs))
    s_baixo = s_baixo_raw / max_abs if max_abs > 0 else s_baixo_raw

    # Cima: tentar subir acima de t_plus
    X_cima = np.where(coefs > 0, MAX_VEC, MIN_VEC)
    if indices_fixos:
        X_cima[indices_fixos] = vals_s[indices_fixos]
    s_cima_raw = intercept + float(np.dot(X_cima, coefs))
    s_cima = s_cima_raw / max_abs if max_abs > 0 else s_cima_raw

    ok_baixo = (s_baixo >= t_minus - EPSILON)
    ok_cima = (s_cima <= t_plus + EPSILON)
    return s_baixo, s_cima, ok_baixo, ok_cima


def main(dataset_name: str = "pima_indians_diabetes"):
    # Carrega pipeline compartilhado (mesmo do PuLP)
    modelo, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)
    max_abs = meta['model_params']['norm_params']['max_abs']

    # Carrega resultados PEAB e PuLP
    peab = load_method_results('peab', dataset_name)
    pulp = load_method_results('pulp', dataset_name)

    if not peab or not pulp:
        print("❌ Resultados não encontrados. Reexecute PEAB e PuLP primeiro.")
        return

    # Índice -> tamanho (PuLP)
    pulp_size = {str(e['indice']): int(e['tamanho']) for e in pulp.get('explicacoes', [])}

    # Preparar mapeamento nome->idx
    feature_names = list(X_train.columns)
    col_to_idx = {c: i for i, c in enumerate(feature_names)}

    # Coefs/params do modelo
    logreg = modelo.named_steps['model'] if 'model' in modelo.named_steps else modelo.named_steps['modelo']
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    scaler = modelo.named_steps['scaler']

    negativos: List[Dict] = []

    for inst in peab.get('per_instance', []):
        rid = str(inst['id'])
        if rid not in pulp_size:
            continue
        if not inst.get('rejected', False):
            continue

        size_peab = int(inst['explanation_size'])
        size_pulp = int(pulp_size[rid])
        if size_peab >= size_pulp:
            continue  # só queremos GAP negativo

        # Reconstruir indices da explicação do PEAB
        feat_names = [f for f in inst.get('explanation', [])]
        idx_fixos = [col_to_idx[f] for f in feat_names if f in col_to_idx]

        # Valores escalados da instância
        instancia = X_test.loc[[int(rid)]] if int(rid) in X_test.index else None
        if instancia is None:
            # Pode não estar no split; pula
            continue
        vals_s = scaler.transform(instancia)[0]

        s_baixo, s_cima, ok_baixo, ok_cima = worst_case_scores_for_reject(
            modelo, coefs, intercept, vals_s, idx_fixos, t_plus, t_minus, max_abs
        )

        negativos.append({
            'id': rid,
            'size_peab': size_peab,
            'size_pulp': size_pulp,
            'ok_baixo': ok_baixo,
            'ok_cima': ok_cima,
            's_baixo': s_baixo,
            's_cima': s_cima,
            't_minus': t_minus,
            't_plus': t_plus,
            'features': feat_names,
        })

    if not negativos:
        print("✅ Nenhum GAP negativo em rejeitadas (ou não encontrado no split atual).")
        return

    # Relatório rápido
    print(f"\nDiagnóstico para {dataset_name}: {len(negativos)} casos com GAP negativo em rejeitadas.")
    falhas = [n for n in negativos if not (n['ok_baixo'] and n['ok_cima'])]
    print(f" - Falharam validação PEAB sob o mesmo modelo: {len(falhas)}")
    print(f" - Passaram validação (potencial divergência de split/thresholds): {len(negativos) - len(falhas)}")

    # Mostrar alguns exemplos
    for n in negativos[:10]:
        status = "OK" if (n['ok_baixo'] and n['ok_cima']) else "FALHA"
        print(f"  • ID {n['id']}: PEAB={n['size_peab']} vs PuLP={n['size_pulp']} -> {status} | "
              f"s_baixo={n['s_baixo']:.4f} (>= {n['t_minus']:.4f}), "
              f"s_cima={n['s_cima']:.4f} (<= {n['t_plus']:.4f})")

if __name__ == "__main__":
    main()
