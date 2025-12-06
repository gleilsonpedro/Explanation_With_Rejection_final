"""
Teste focado: processar apenas UMA instância rejeitada do MNIST
para ver o debug output
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '.')

from data.datasets import carregar_dataset

RANDOM_STATE = 42

def _get_lr(modelo: Pipeline):
    if 'model' in modelo.named_steps:
        return modelo.named_steps['model']
    return modelo.named_steps['modelo']

def main():
    print("="*80)
    print("TESTE DEBUG: UMA INSTÂNCIA REJEITADA DO MNIST")
    print("="*80)
    
    # Configurar MNIST
    print("\n1. Configurando MNIST...")
    from data import datasets as ds_module
    ds_module.set_mnist_options('raw', (3, 8))
    
    X, y, _ = carregar_dataset('mnist')
    
    # Subsample
    idx = np.arange(len(y))
    sample_idx, _ = train_test_split(idx, test_size=0.7, 
                                     random_state=RANDOM_STATE, stratify=y)
    X = X.iloc[sample_idx]
    y = y.iloc[sample_idx]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Treinar
    print("\n2. Treinando modelo...")
    logreg_params = {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 200}
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params)),
    ])
    pipeline.fit(X_train, y_train)
    print("   Modelo treinado!")
    
    # Thresholds
    print("\n3. Encontrando thresholds...")
    decision_scores = pipeline.decision_function(X_train)
    qs = np.linspace(0, 1, 100)
    search_space = np.unique(np.quantile(decision_scores, qs))
    
    best_risk, best_t_plus, best_t_minus = float('inf'), 0.0, 0.0
    rejection_cost = 0.24
    
    for i in range(len(search_space)):
        for j in range(i, len(search_space)):
            t_minus, t_plus = float(search_space[i]), float(search_space[j])
            
            y_pred = np.full(y_train.shape, -1)
            accepted = (decision_scores >= t_plus) | (decision_scores <= t_minus)
            y_pred[decision_scores >= t_plus] = 1
            y_pred[decision_scores <= t_minus] = 0
            
            error_rate = np.mean(y_pred[accepted] != y_train[accepted]) if np.any(accepted) else 0.0
            rejection_rate = 1.0 - np.mean(accepted)
            risk = float(error_rate + rejection_cost * rejection_rate)
            
            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, t_plus, t_minus
    
    print(f"   t+ = {best_t_plus:.6f}")
    print(f"   t- = {best_t_minus:.6f}")
    print(f"   Largura: {best_t_plus - best_t_minus:.6f}")
    
    # Encontrar rejeitadas
    print("\n4. Identificando rejeitadas no teste...")
    scores_test = pipeline.decision_function(X_test)
    mask_rejeitadas = (scores_test > best_t_minus) & (scores_test < best_t_plus)
    num_rejeitadas = np.sum(mask_rejeitadas)
    
    print(f"   Total rejeitadas: {num_rejeitadas} ({100*num_rejeitadas/len(scores_test):.2f}%)")
    
    if num_rejeitadas == 0:
        print("\n[ERRO] Nenhuma rejeitada encontrada!")
        return
    
    # Pegar primeira rejeitada
    idx_rejeitada = np.where(mask_rejeitadas)[0][0]
    instance_df = X_test.iloc[[idx_rejeitada]]
    score_orig = pipeline.decision_function(instance_df)[0]
    
    print(f"\n5. Processando instância rejeitada (índice {idx_rejeitada})...")
    print(f"   Score original: {score_orig:.6f}")
    print(f"   Está em [{best_t_minus:.6f}, {best_t_plus:.6f}]")
    
    # Importar funções do peab
    from peab import gerar_explicacao_instancia
    
    print("\n6. Gerando explicação (DEBUG ATIVADO)...")
    print("-"*80)
    
    # Chamar função com benchmark_mode=False para ativar debug
    explicacao, logs, adicoes, remocoes = gerar_explicacao_instancia(
        instance_df, pipeline, X_train, best_t_plus, best_t_minus, 
        benchmark_mode=False
    )
    
    print("-"*80)
    print(f"\n7. RESULTADO:")
    print(f"   Explicação: {len(explicacao)} features")
    print(f"   Adições: {adicoes}")
    print(f"   Remoções: {remocoes}")
    
    if len(explicacao) == 0:
        print("\n   [PROBLEMA] Explicação vazia!")
    else:
        print(f"\n   Primeiras 10 features:")
        for feat in explicacao[:10]:
            print(f"     - {feat}")

if __name__ == '__main__':
    main()
