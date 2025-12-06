"""
Diagnóstico detalhado para entender por que as rejeitadas do MNIST têm tamanho 0
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data.datasets import carregar_dataset

# Configurações
RANDOM_STATE = 42
MNIST_CONFIG = {
    'feature_mode': 'raw',           
    'digit_pair': (3, 8),            
    'top_k_features': None,          
    'test_size': 0.3,                
    'rejection_cost': 0.24,          
    'subsample_size': 0.3 
}

def _get_lr(modelo: Pipeline):
    if 'model' in modelo.named_steps:
        return modelo.named_steps['model']
    return modelo.named_steps['modelo']

def setup_mnist():
    """Prepara dataset MNIST"""
    from data import datasets as ds_module
    ds_module.set_mnist_options(
        MNIST_CONFIG.get('feature_mode', 'raw'),
        MNIST_CONFIG.get('digit_pair', None)
    )
    
    X, y, _ = carregar_dataset('mnist')
    
    # Subsample
    frac = MNIST_CONFIG['subsample_size']
    idx = np.arange(len(y))
    sample_idx, _ = train_test_split(idx, test_size=(1 - frac), 
                                     random_state=RANDOM_STATE, stratify=y)
    X = X.iloc[sample_idx]
    y = y.iloc[sample_idx]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MNIST_CONFIG['test_size'], 
        random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Treina modelo LogReg"""
    logreg_params = {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 200}
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def find_thresholds(pipeline, X_train, y_train, rejection_cost=0.24):
    """Encontra thresholds ótimos"""
    decision_scores = pipeline.decision_function(X_train)
    qs = np.linspace(0, 1, 100)
    search_space = np.unique(np.quantile(decision_scores, qs))
    
    best_risk, best_t_plus, best_t_minus = float('inf'), 0.0, 0.0
    
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
    
    return best_t_plus, best_t_minus

def perturbar_e_validar(vals_s, coefs, indices_explicacao, intercept, 
                       t_plus, t_minus, direcao_override, is_rejected):
    """
    Versão simplificada de perturbar_e_validar_otimizado
    direcao_override: 0 = empurrar para CIMA, 1 = empurrar para BAIXO
    """
    MIN_VEC = np.zeros_like(coefs)
    MAX_VEC = np.ones_like(coefs)
    
    empurrar_para_baixo = (direcao_override == 1)
    
    if empurrar_para_baixo:
        X_teste = np.where(coefs > 0, MIN_VEC, MAX_VEC)
    else:
        X_teste = np.where(coefs > 0, MAX_VEC, MIN_VEC)
    
    if indices_explicacao:
        idx_fixos = list(indices_explicacao)
        X_teste[idx_fixos] = vals_s[idx_fixos]
    
    score_pert = intercept + np.dot(X_teste, coefs)
    EPSILON = 1e-6
    
    if is_rejected:
        # Se empurramos para baixo, deve ficar >= t_minus
        # Se empurramos para cima, deve ficar <= t_plus
        if empurrar_para_baixo:
            return (score_pert >= t_minus - EPSILON), score_pert
        else:
            return (score_pert <= t_plus + EPSILON), score_pert
    else:
        # Não deveria chegar aqui no nosso teste
        return False, score_pert

def diagnosticar_rejeitada(instance_df, modelo, X_train, t_plus, t_minus):
    """Diagnóstico detalhado de uma instância rejeitada"""
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    vals_s = scaler.transform(instance_df)[0]
    score_orig = modelo.decision_function(instance_df)[0]
    
    print(f"\n{'='*80}")
    print(f"DIAGNÓSTICO DE INSTÂNCIA REJEITADA")
    print(f"{'='*80}")
    print(f"Score original: {score_orig:.6f}")
    print(f"Thresholds: t- = {t_minus:.6f}, t+ = {t_plus:.6f}")
    print(f"Largura zona rejeição: {t_plus - t_minus:.6f}")
    print(f"Features total: {len(coefs)}")
    print(f"Intercepto: {intercept:.6f}")
    
    # Teste com conjunto vazio
    print(f"\n{'='*80}")
    print("TESTE 1: Conjunto vazio (apenas intercepto)")
    print(f"{'='*80}")
    
    indices_vazios = set()
    valido_cima, score_cima = perturbar_e_validar(vals_s, coefs, indices_vazios, intercept,
                                                   t_plus, t_minus, 0, True)
    valido_baixo, score_baixo = perturbar_e_validar(vals_s, coefs, indices_vazios, intercept,
                                                     t_plus, t_minus, 1, True)
    
    print(f"Teste empurrando para CIMA (dir=0): score_pert = {score_cima:.6f}")
    print(f"  Deve ser <= t+ ({t_plus:.6f}): {score_cima <= t_plus + 1e-6} {'✓' if valido_cima else '✗'}")
    print(f"Teste empurrando para BAIXO (dir=1): score_pert = {score_baixo:.6f}")
    print(f"  Deve ser >= t- ({t_minus:.6f}): {score_baixo >= t_minus - 1e-6} {'✓' if valido_baixo else '✗'}")
    print(f"Validação conjunta: {valido_cima and valido_baixo}")
    
    if valido_cima and valido_baixo:
        print("\n⚠️ PROBLEMA IDENTIFICADO: Conjunto vazio já é válido!")
        print("Isso significa que o intercepto sozinho mantém a rejeição em ambas as direções.")
        print("A fase de reforço termina imediatamente sem adicionar features.")
        return True, indices_vazios
    
    # Estatísticas dos coeficientes
    print(f"\n{'='*80}")
    print("ESTATÍSTICAS DOS COEFICIENTES")
    print(f"{'='*80}")
    print(f"Coeficientes positivos: {np.sum(coefs > 0)} ({100*np.sum(coefs > 0)/len(coefs):.1f}%)")
    print(f"Coeficientes negativos: {np.sum(coefs < 0)} ({100*np.sum(coefs < 0)/len(coefs):.1f}%)")
    print(f"Coeficientes zero: {np.sum(np.abs(coefs) < 1e-9)}")
    print(f"Max |coef|: {np.max(np.abs(coefs)):.6f}")
    print(f"Min |coef| (não-zero): {np.min(np.abs(coefs[np.abs(coefs) > 1e-9])):.6f}")
    print(f"Média |coef|: {np.mean(np.abs(coefs)):.6f}")
    
    # Teste com 1 feature
    print(f"\n{'='*80}")
    print("TESTE 2: Adicionar feature com maior |coef|")
    print(f"{'='*80}")
    
    idx_max = np.argmax(np.abs(coefs))
    feat_nome = X_train.columns[idx_max]
    indices_com_1 = {idx_max}
    
    valido_cima, score_cima = perturbar_e_validar(vals_s, coefs, indices_com_1, intercept,
                                                   t_plus, t_minus, 0, True)
    valido_baixo, score_baixo = perturbar_e_validar(vals_s, coefs, indices_com_1, intercept,
                                                     t_plus, t_minus, 1, True)
    
    print(f"Feature adicionada: {feat_nome} (idx={idx_max}, coef={coefs[idx_max]:.6f})")
    print(f"Valor original: {vals_s[idx_max]:.6f}")
    print(f"Teste empurrando para CIMA: score_pert = {score_cima:.6f}, válido = {valido_cima}")
    print(f"Teste empurrando para BAIXO: score_pert = {score_baixo:.6f}, válido = {valido_baixo}")
    print(f"Validação conjunta: {valido_cima and valido_baixo}")
    
    return False, indices_vazios

def main():
    print("Configurando MNIST...")
    X_train, X_test, y_train, y_test = setup_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\nTreinando modelo...")
    modelo = train_model(X_train, y_train)
    
    print("\nEncontrando thresholds...")
    t_plus, t_minus = find_thresholds(modelo, X_train, y_train)
    print(f"t+ = {t_plus:.6f}, t- = {t_minus:.6f}")
    print(f"Largura zona: {t_plus - t_minus:.6f}")
    
    print("\nIdentificando instâncias rejeitadas no teste...")
    scores_test = modelo.decision_function(X_test)
    mask_rejeitadas = (scores_test > t_minus) & (scores_test < t_plus)
    
    num_rejeitadas = np.sum(mask_rejeitadas)
    print(f"Total rejeitadas: {num_rejeitadas} ({100*num_rejeitadas/len(scores_test):.2f}%)")
    
    if num_rejeitadas == 0:
        print("\n⚠️ PROBLEMA: Nenhuma instância rejeitada encontrada!")
        print("Verificando distribuição de scores...")
        print(f"Min score: {scores_test.min():.6f}")
        print(f"Max score: {scores_test.max():.6f}")
        print(f"Scores < t-: {np.sum(scores_test <= t_minus)}")
        print(f"Scores > t+: {np.sum(scores_test >= t_plus)}")
        return
    
    # Diagnosticar primeira rejeitada
    idx_rejeitadas = np.where(mask_rejeitadas)[0]
    idx_primeira = idx_rejeitadas[0]
    
    print(f"\n\n{'#'*80}")
    print(f"# DIAGNÓSTICO DA PRIMEIRA INSTÂNCIA REJEITADA (índice {idx_primeira})")
    print(f"{'#'*80}")
    
    instance_df = X_test.iloc[[idx_primeira]]
    problema_vazio, _ = diagnosticar_rejeitada(instance_df, modelo, X_train, t_plus, t_minus)
    
    # Diagnosticar mais algumas se necessário
    if num_rejeitadas >= 3:
        print(f"\n\n{'#'*80}")
        print(f"# DIAGNÓSTICO DE MAIS 2 REJEITADAS")
        print(f"{'#'*80}")
        
        for i in [1, 2]:
            idx = idx_rejeitadas[i]
            instance_df = X_test.iloc[[idx]]
            problema_vazio, _ = diagnosticar_rejeitada(instance_df, modelo, X_train, t_plus, t_minus)

if __name__ == '__main__':
    main()
