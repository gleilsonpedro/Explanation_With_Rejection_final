from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

# Reutiliza a lógica do PEAB para garantir treino e thresholds idênticos
from peab import (
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
    
    IMPORTANTE: Se top_k_features estiver configurado, aplica redução de features
    idêntica ao PEAB, garantindo comparação justa entre PEAB, MinExp e Anchor.

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

    # 2.5) Aplicar redução de features (top-k) ANTES do treino, se configurado
    # [NOVO] Esta é a mesma lógica do PEAB para garantir comparação justa
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    
    if top_k and top_k > 0 and top_k < X.shape[1]:
        # Importar função de seleção do PEAB
        from peab import aplicar_selecao_top_k_features
        
        # Treinar modelo temporário para obter importâncias
        print(f"\n[INFO] [Shared Training] Aplicando seleção de top-{top_k} features...")
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X, y, test_size, rejection_cost, params_modelo)
        X_train_temp, X_test_temp, _, _ = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
        
        # Selecionar top-k features
        X_train_temp, X_test_temp, selected_features = aplicar_selecao_top_k_features(X_train_temp, X_test_temp, modelo_temp, top_k)
        
        # Reduzir X completo para apenas as features selecionadas
        X = X[selected_features]
        print(f"[INFO] [Shared Training] Dataset reduzido para {top_k} features (mesmo do PEAB).")

    # 3) Fazer split ANTES do treino (mesma ordem do PEAB)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # 4) Treinar e otimizar thresholds com a mesma função do PEAB
    pipeline, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(
        X_train=X_train, y_train=y_train, rejection_cost=rejection_cost, logreg_params=params_modelo
    )

    meta = {
        'feature_names': list(X.columns) if isinstance(X, pd.DataFrame) else [f'f{i}' for i in range(X.shape[1])],
        'rejection_cost': rejection_cost,
        'test_size': test_size,
        'params_modelo': params_modelo,
        'nomes_classes': nomes_classes,
        'model_params': model_params,
        'subsample_size': DATASET_CONFIG.get(dataset_name, {}).get('subsample_size', None),
        'top_k_features': top_k  # [NOVO] Informar se houve redução de features
    }

    return pipeline, X_train, X_test, y_train, y_test, float(t_plus), float(t_minus), meta
