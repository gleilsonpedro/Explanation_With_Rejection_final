"""
Diagnóstico comparativo: PIMA vs MNIST
Para entender por que PIMA funciona e MNIST não
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data.datasets import carregar_dataset

RANDOM_STATE = 42

def _get_lr(modelo: Pipeline):
    if 'model' in modelo.named_steps:
        return modelo.named_steps['model']
    return modelo.named_steps['modelo']

def analisar_dataset(dataset_name, config):
    """Analisa características de um dataset"""
    print(f"\n{'='*80}")
    print(f"ANALISANDO: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Configurar e carregar
    if dataset_name == 'mnist':
        from data import datasets as ds_module
        ds_module.set_mnist_options('raw', (3, 8))
    
    X, y, _ = carregar_dataset(dataset_name)
    
    # Subsample se necessário
    if 'subsample_size' in config:
        frac = config['subsample_size']
        idx = np.arange(len(y))
        sample_idx, _ = train_test_split(idx, test_size=(1 - frac), 
                                         random_state=RANDOM_STATE, stratify=y)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], 
        random_state=RANDOM_STATE, stratify=y
    )
    
    # Treinar
    logreg_params = config.get('logreg_params', 
                               {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 200})
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params)),
    ])
    pipeline.fit(X_train, y_train)
    
    # Thresholds
    decision_scores = pipeline.decision_function(X_train)
    qs = np.linspace(0, 1, 100)
    search_space = np.unique(np.quantile(decision_scores, qs))
    
    best_risk, best_t_plus, best_t_minus = float('inf'), 0.0, 0.0
    rejection_cost = config['rejection_cost']
    
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
    
    # Estatísticas
    logreg = _get_lr(pipeline)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    
    print(f"\n--- DADOS ---")
    print(f"Train: {X_train.shape[0]} instâncias, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} instâncias")
    
    print(f"\n--- MODELO ---")
    print(f"Intercepto: {intercept:.6f}")
    print(f"Coeficientes positivos: {np.sum(coefs > 0)} ({100*np.sum(coefs > 0)/len(coefs):.1f}%)")
    print(f"Coeficientes negativos: {np.sum(coefs < 0)} ({100*np.sum(coefs < 0)/len(coefs):.1f}%)")
    print(f"Coeficientes zero: {np.sum(np.abs(coefs) < 1e-9)}")
    print(f"Max |coef|: {np.max(np.abs(coefs)):.6f}")
    print(f"Min |coef| (não-zero): {np.min(np.abs(coefs[np.abs(coefs) > 1e-9])):.6f}")
    print(f"Média |coef|: {np.mean(np.abs(coefs)):.6f}")
    print(f"Soma |coef|: {np.sum(np.abs(coefs)):.6f}")
    
    print(f"\n--- THRESHOLDS ---")
    print(f"t+: {best_t_plus:.6f}")
    print(f"t-: {best_t_minus:.6f}")
    print(f"Largura zona rejeição: {best_t_plus - best_t_minus:.6f}")
    
    # Rejeções
    scores_test = pipeline.decision_function(X_test)
    mask_rejeitadas = (scores_test > best_t_minus) & (scores_test < best_t_plus)
    num_rejeitadas = np.sum(mask_rejeitadas)
    
    print(f"\n--- REJEIÇÕES NO TESTE ---")
    print(f"Total rejeitadas: {num_rejeitadas} ({100*num_rejeitadas/len(scores_test):.2f}%)")
    
    if num_rejeitadas > 0:
        # Pegar uma rejeitada e analisar
        idx_rejeitada = np.where(mask_rejeitadas)[0][0]
        instance_df = X_test.iloc[[idx_rejeitada]]
        
        scaler = pipeline.named_steps['scaler']
        vals_s = scaler.transform(instance_df)[0]
        score_orig = pipeline.decision_function(instance_df)[0]
        
        print(f"\n--- ANÁLISE DE UMA REJEITADA (índice {idx_rejeitada}) ---")
        print(f"Score original: {score_orig:.6f}")
        
        # Calcular score worst-case empurrando para cima e para baixo
        MIN_VEC = np.zeros_like(coefs)
        MAX_VEC = np.ones_like(coefs)
        
        # Empurrar para baixo (dir=1)
        X_baixo = np.where(coefs > 0, MIN_VEC, MAX_VEC)
        score_baixo = intercept + np.dot(X_baixo, coefs)
        
        # Empurrar para cima (dir=0)
        X_cima = np.where(coefs > 0, MAX_VEC, MIN_VEC)
        score_cima = intercept + np.dot(X_cima, coefs)
        
        print(f"\nSEM FIXAR NENHUMA FEATURE (conjunto vazio):")
        print(f"  Score empurrando para BAIXO: {score_baixo:.6f} (precisa >= {best_t_minus:.6f})")
        print(f"    Distância: {score_baixo - best_t_minus:.6f}")
        print(f"  Score empurrando para CIMA: {score_cima:.6f} (precisa <= {best_t_plus:.6f})")
        print(f"    Distância: {score_cima - best_t_plus:.6f}")
        
        # Quanto precisamos "puxar" o score?
        puxar_baixo = best_t_minus - score_baixo if score_baixo < best_t_minus else 0
        puxar_cima = score_cima - best_t_plus if score_cima > best_t_plus else 0
        
        print(f"\nPRECISA PUXAR:")
        print(f"  Para entrar por baixo: {puxar_baixo:.6f}")
        print(f"  Para entrar por cima: {puxar_cima:.6f}")
        
        # Quantas features precisamos fixar?
        if puxar_baixo > 0:
            # Fixar features que aumentam o score
            deltas_positivos = coefs * vals_s
            deltas_que_ajudam = deltas_positivos[deltas_positivos > 0]
            deltas_ordenados = np.sort(deltas_que_ajudam)[::-1]
            
            soma_acum = np.cumsum(deltas_ordenados)
            num_necessarias = np.searchsorted(soma_acum, puxar_baixo) + 1
            
            print(f"\nPara BAIXO: precisaria fixar ~{num_necessarias} features das {len(deltas_que_ajudam)} que ajudam")
        
        if puxar_cima > 0:
            # Fixar features que diminuem o score
            deltas_negativos = coefs * vals_s
            deltas_que_ajudam = deltas_negativos[deltas_negativos < 0]
            deltas_ordenados = np.sort(np.abs(deltas_que_ajudam))[::-1]
            
            soma_acum = np.cumsum(deltas_ordenados)
            num_necessarias = np.searchsorted(soma_acum, puxar_cima) + 1
            
            print(f"Para CIMA: precisaria fixar ~{num_necessarias} features das {len(deltas_que_ajudam)} que ajudam")
        
        # Proporção features necessárias vs total
        if puxar_baixo > 0:
            prop = (num_necessarias / len(coefs)) * 100
            print(f"\nProporção de features necessárias: {prop:.1f}% do total")
            print(f"Viável? {'SIM' if prop < 50 else 'DIFÍCIL' if prop < 80 else 'IMPOSSÍVEL'}")

def main():
    # PIMA config
    pima_config = {
        'test_size': 0.3,
        'rejection_cost': 0.24,
        'logreg_params': {'penalty': 'l2', 'C': 10, 'solver': 'saga', 'max_iter': 200}
    }
    
    # MNIST config
    mnist_config = {
        'test_size': 0.3,
        'rejection_cost': 0.24,
        'subsample_size': 0.3,
        'logreg_params': {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 200}
    }
    
    analisar_dataset('pima_indians_diabetes', pima_config)
    analisar_dataset('mnist', mnist_config)
    
    print(f"\n\n{'#'*80}")
    print(f"# CONCLUSÃO")
    print(f"{'#'*80}")
    print("""
Se o MNIST requer fixar >50% das features para conter o score na zona,
enquanto o PIMA requer <20%, isso explica por que:
- PIMA: fase_1_reforco consegue encontrar conjunto robusto rapidamente
- MNIST: fase_1_reforco nunca consegue (precisaria de centenas de features)

A solução NÃO é corrigir a validação, mas mudar a ESTRATÉGIA para rejeitadas
em datasets de alta dimensionalidade.
    """)

if __name__ == '__main__':
    main()
