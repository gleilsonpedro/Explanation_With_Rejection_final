from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

# Reutiliza a lógica do PEAB para garantir treino e thresholds idênticos
from peab_2 import (
    configurar_experimento,
    treinar_e_avaliar_modelo,
    carregar_hiperparametros,
    DEFAULT_LOGREG_PARAMS,
    RANDOM_STATE,
    DATASET_CONFIG,
)
from sklearn.model_selection import train_test_split


def get_shared_pipeline(dataset_name: str) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float, float, Dict[str, Any]]:
    """
    Treina um Pipeline(MinMaxScaler + LogisticRegression) e encontra t+ / t- exatamente
    como o PEAB, retornando também splits consistentes e metadados.

    Retorna:
      - pipeline: Pipeline treinado
      - X_train, X_test, y_train, y_test: splits determinísticos (mesmo seed do PEAB)
      - t_plus, t_minus: thresholds de rejeição otimizados
      - meta: dict com feature_names, rejection_cost, test_size, params, nomes_classes
    """
    # 1) Carregar configs/dataset como no PEAB
    hiperparams_todos = carregar_hiperparametros()
    X, y, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)

    # 2) Montar hiperparâmetros do modelo como no PEAB
    params_modelo = DEFAULT_LOGREG_PARAMS.copy()
    cfg_dataset = hiperparams_todos.get(dataset_name)
    if cfg_dataset and 'params' in cfg_dataset:
        from sklearn.linear_model import LogisticRegression
        valid_keys = LogisticRegression().get_params().keys()
        params_carregados = {k: v for k, v in cfg_dataset['params'].items() if k in valid_keys}
        params_modelo.update(params_carregados)

    # 3) Treinar e otimizar thresholds com a mesma função do PEAB
    pipeline, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(
        X=X, y=y, test_size=test_size, rejection_cost=rejection_cost, logreg_params=params_modelo
    )

    # 4) Reproduzir os mesmos splits de forma determinística
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    meta = {
        'feature_names': list(X.columns) if isinstance(X, pd.DataFrame) else [f'f{i}' for i in range(X.shape[1])],
        'rejection_cost': rejection_cost,
        'test_size': test_size,
        'params_modelo': params_modelo,
        'nomes_classes': nomes_classes,
        'model_params': model_params,
        'subsample_size': DATASET_CONFIG.get(dataset_name, {}).get('subsample_size', None)
    }

    return pipeline, X_train, X_test, y_train, y_test, float(t_plus), float(t_minus), meta
