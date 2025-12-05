import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Importa utilidades do projeto
from peab import (
    configurar_experimento,
    treinar_e_avaliar_modelo,
    DATASET_CONFIG,
    RANDOM_STATE,
)

# Pequena busca de hiperparâmetros para MNIST visando rejeição > 0
# - Usa subsample_size da config do PEAB se existir; pode forçar 0.5/0.3
# - Avalia pares 3 vs 8 (via config MNIST)
# - Critério: maximiza acurácia com rejeição > 0 e taxa de rejeição alvo (~1% a ~10%)

C_GRID = [0.01, 0.1, 1.0, 3.0, 10.0]
SOLVERS = ["liblinear", "lbfgs", "saga"]
MAX_ITERS = [200, 500, 1000]

TARGET_REJ_MIN = 0.01
TARGET_REJ_MAX = 0.15


def avaliar_setup(X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], rejection_cost: float, test_size: float) -> Tuple[float, float, float, float, float]:
    """Treina e avalia retornando métricas principais.
    Retorna: (acc_sem_rej, acc_com_rej, taxa_rej, t_plus, t_minus)
    """
    # Split consistente com assinatura atual de treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    modelo, t_plus, t_minus, _ = treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params)
    scores = modelo.decision_function(X_test)
    y_pred = np.full(y_test.shape, -1, dtype=int)
    y_pred[scores >= t_plus] = 1
    y_pred[scores <= t_minus] = 0
    rejected = (y_pred == -1)
    y_pred_final = y_pred.copy()
    y_pred_final[rejected] = 2

    acc_sem_rej = float(np.mean(modelo.predict(X_test) == y_test) * 100)
    acc_com_rej = float(np.mean(y_pred_final[~rejected] == y_test.iloc[~rejected]) * 100) if np.any(~rejected) else 100.0
    taxa_rej = float(np.mean(rejected))
    return acc_sem_rej, acc_com_rej, taxa_rej, float(t_plus), float(t_minus)


def main():
    dataset_name = "mnist"
    # Puxa config atual e força par 3 vs 8, se aplicável
    cfg = DATASET_CONFIG.get(dataset_name, {}).copy()
    # Se desejar fixar subsample
    if cfg.get('subsample_size') is None:
        cfg['subsample_size'] = 0.5
    # Força par 3 vs 8 quando possível
    cfg['digit_pair'] = (3, 8)
    DATASET_CONFIG[dataset_name] = cfg

    # Carrega experimento com config aplicada
    X, y, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)

    melhor = {
        'score': -1.0,
        'params': None,
        'metrics': None
    }

    total_testados = 0
    print("== Busca MNIST (3 vs 8) visando rejeição > 0 ==\n")
    print(f"Config: subsample_size={cfg.get('subsample_size')}, test_size={test_size}, rejection_cost={rejection_cost}")

    for C in C_GRID:
        for solver in SOLVERS:
            if solver == 'liblinear' and X.shape[1] > 10000:
                # evita liblinear em espaços imensos
                continue
            for max_iter in MAX_ITERS:
                params = {
                    'penalty': 'l2',
                    'C': C,
                    'solver': solver,
                    'max_iter': max_iter
                }
                try:
                    acc_sem, acc_com, taxa_rej, t_plus, t_minus = avaliar_setup(X, y, params, rejection_cost, test_size)
                except Exception as e:
                    # Alguns solvers/combos podem falhar
                    print(f"[skip] C={C} solver={solver} max_iter={max_iter} erro: {e}")
                    continue
                total_testados += 1
                # Critério: rejeição dentro da faixa alvo e acurácia com rejeição alta
                if TARGET_REJ_MIN <= taxa_rej <= TARGET_REJ_MAX:
                    score = acc_com
                else:
                    # penaliza fora da faixa
                    dist = min(abs(taxa_rej - TARGET_REJ_MIN), abs(taxa_rej - TARGET_REJ_MAX))
                    score = acc_com - (dist * 100)

                print(f"C={C:<5} solver={solver:<9} it={max_iter:<4} | acc_sem={acc_sem:5.2f}% acc_com={acc_com:5.2f}% rej={taxa_rej:6.3f} t-={t_minus: .4f} t+={t_plus: .4f}")

                if score > melhor['score']:
                    melhor['score'] = score
                    melhor['params'] = params
                    melhor['metrics'] = {
                        'acc_sem_rej': acc_sem,
                        'acc_com_rej': acc_com,
                        'rejection_rate': taxa_rej,
                        't_plus': t_plus,
                        't_minus': t_minus
                    }

    print("\n== Resultado ==")
    if melhor['params'] is None:
        print("Nenhuma configuração adequada encontrada (tente ampliar a grade ou ajustar TARGET_REJ_*).")
        sys.exit(2)
    print(json.dumps({
        'best_params': melhor['params'],
        'metrics': melhor['metrics'],
        'tested': total_testados,
        'dataset': 'mnist_3_vs_8',
        'subsample_size': cfg.get('subsample_size')
    }, indent=2))


if __name__ == '__main__':
    main()
