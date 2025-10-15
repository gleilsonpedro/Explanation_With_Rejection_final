import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Tuple, List

def executar_logica_rejeicao(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, rejection_cost: float) -> Tuple[float, float, List, List, List]:
    """
    Executa o processo completo de rejeição:
    1. Encontra os thresholds ideais (t+ e t-) usando os dados de treino.
    2. Aplica esses thresholds para classificar/rejeitar os dados de teste.
    
    Retorna todos os resultados necessários de uma vez.
    """
    
    # --- Parte 1: Encontrar os Thresholds (Sua lógica, usando dados de TREINO) ---
    print(f"...Encontrando thresholds com custo de rejeição = {rejection_cost}...")
    decision_scores_train = model.decision_function(X_train)
    min_custo = float('inf')
    melhor_t_plus, melhor_t_minus = 0.1, -0.1
    score_max = np.max(decision_scores_train) if len(decision_scores_train) > 0 else 0.1
    score_min = np.min(decision_scores_train) if len(decision_scores_train) > 0 else -0.1
    pontos_busca = 100

    t_plus_candidatos = np.linspace(0.01, max(score_max, 0.1), pontos_busca)
    t_minus_candidatos = np.linspace(min(score_min, -0.1), -0.01, pontos_busca)

    for t_p in t_plus_candidatos:
        for t_m in t_minus_candidatos:
            if t_m >= t_p: continue
            
            rejeitadas = (decision_scores_train >= t_m) & (decision_scores_train <= t_p)
            aceitas = ~rejeitadas
            taxa_rejeicao = np.mean(rejeitadas)
            
            if np.sum(aceitas) == 0:
                taxa_erro_aceitas = 1.0
            else:
                # Certifique-se de que y_train seja um array numpy para comparação
                y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
                preds_aceitas = model.predict(X_train[aceitas])
                taxa_erro_aceitas = np.mean(preds_aceitas != y_train_array[aceitas])
            
            custo_total = taxa_erro_aceitas + rejection_cost * taxa_rejeicao
            
            if custo_total < min_custo:
                min_custo, melhor_t_plus, melhor_t_minus = custo_total, t_p, t_m
    
    # --- Parte 2: Aplicar os Thresholds (usando dados de TESTE) ---
    print(f"...Aplicando thresholds encontrados: t+ = {melhor_t_plus:.4f}, t- = {melhor_t_minus:.4f}...")
    decision_scores_test = model.decision_function(X_test)
    y_pred_com_rejeicao = []
    indices_aceitos, indices_rejeitados = [], []
    
    for i, score in enumerate(decision_scores_test):
        if score >= melhor_t_plus:
            y_pred_com_rejeicao.append(1)
            indices_aceitos.append(i)
        elif score <= melhor_t_minus:
            y_pred_com_rejeicao.append(0)
            indices_aceitos.append(i)
        else:
            y_pred_com_rejeicao.append(-1)
            indices_rejeitados.append(i)

    # Retorna TODOS os resultados de uma vez
    return melhor_t_minus, melhor_t_plus, y_pred_com_rejeicao, indices_aceitos, indices_rejeitados
    
