"""
Diagnóstico ainda mais detalhado - simular exatamente o que fase_1_reforco faz
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

def calculate_deltas(modelo, instance_df, X_train, premis_class):
    """Calcula deltas como em peab.py"""
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    vals_s = scaler.transform(instance_df)[0]
    deltas = coefs * vals_s
    return deltas

def one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, premis_class):
    """Gera explicação formal como em peab.py"""
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    intercept = logreg.intercept_[0]
    
    deltas = calculate_deltas(modelo, instance_df, X_train, premis_class)
    indices_ordenados = np.argsort(-np.abs(deltas))
    
    target_score = t_plus if premis_class == 1 else t_minus
    soma_deltas_cumulativa = intercept
    explicacao = []
    EPSILON = 1e-9
    
    for i in indices_ordenados:
        feature_nome = X_train.columns[i]
        valor_original_feature = instance_df.iloc[0, X_train.columns.get_loc(feature_nome)]
        
        if abs(deltas[i]) > 1e-9:
             soma_deltas_cumulativa += deltas[i]
             explicacao.append(f"{feature_nome} = {valor_original_feature:.4f}")
        
        if premis_class == 1:
            if soma_deltas_cumulativa >= (target_score - EPSILON) and explicacao:
                break
        else:
            if soma_deltas_cumulativa <= (target_score + EPSILON) and explicacao:
                break
    
    if not explicacao and len(X_train.columns) > 0:
         idx_max = np.argmax(np.abs(logreg.coef_[0]))
         feat_nome = X_train.columns[idx_max]
         valor_feat = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
         explicacao.append(f"{feat_nome} = {valor_feat:.4f}")

    return explicacao

def perturbar_e_validar(vals_s, coefs, indices_explicacao, intercept, 
                       t_plus, t_minus, direcao_override, is_rejected):
    """
    Versão corrigida de perturbar_e_validar_otimizado
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
        if empurrar_para_baixo:
            return (score_pert >= t_minus - EPSILON), score_pert
        else:
            return (score_pert <= t_plus + EPSILON), score_pert
    else:
        return False, score_pert

def simular_fase_1_reforco(modelo, instance_df, expl_inicial, X_train, t_plus, t_minus, premisa_ordenacao):
    """Simula exatamente a fase_1_reforco"""
    scaler = modelo.named_steps['scaler']
    logreg = _get_lr(modelo)
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    vals_s = scaler.transform(instance_df)[0]
    
    col_to_idx = {c: i for i, c in enumerate(X_train.columns)}
    
    expl_robusta_indices = {col_to_idx[f.split(' = ')[0]] for f in expl_inicial}
    expl_robusta_str = list(expl_inicial)
    
    print(f"\n{'='*80}")
    print("SIMULANDO FASE 1: REFORÇO")
    print(f"{'='*80}")
    print(f"Explicação inicial: {len(expl_inicial)} features")
    print(f"Premisa de ordenação: classe {premisa_ordenacao}")
    
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premisa_ordenacao)
    indices_ordenados = np.argsort(-np.abs(deltas_para_ordenar))
    num_features_total = X_train.shape[1]
    
    iteracao = 0
    MAX_ITER = 20  # Limite para evitar loop infinito
    
    while iteracao < MAX_ITER:
        iteracao += 1
        print(f"\n--- Iteração {iteracao} (conjunto atual: {len(expl_robusta_indices)} features) ---")
        
        # Teste bidirecional
        valido_baixo, score_baixo = perturbar_e_validar(vals_s, coefs, expl_robusta_indices, 
                                                         intercept, t_plus, t_minus, 1, True)
        valido_cima, score_cima = perturbar_e_validar(vals_s, coefs, expl_robusta_indices, 
                                                       intercept, t_plus, t_minus, 0, True)
        
        print(f"  Teste BAIXO (dir=1): score={score_baixo:.6f}, deve >= {t_minus:.6f} -> {valido_baixo}")
        print(f"  Teste CIMA (dir=0): score={score_cima:.6f}, deve <= {t_plus:.6f} -> {valido_cima}")
        
        is_valid = valido_baixo and valido_cima
        print(f"  Validação conjunta: {is_valid}")
        
        if is_valid:
            print(f"\n[OK] Conjunto robusto encontrado com {len(expl_robusta_indices)} features")
            break
            
        if len(expl_robusta_indices) == num_features_total:
            print(f"\n[WARN] Todas as features ja foram adicionadas!")
            break

        # Adicionar próxima feature
        adicionou_feature = False
        for idx in indices_ordenados:
            if idx not in expl_robusta_indices:
                expl_robusta_indices.add(idx)
                feat_nome = X_train.columns[idx]
                val = instance_df.iloc[0, idx]
                expl_robusta_str.append(f"{feat_nome} = {val:.4f}")
                print(f"  -> Adicionando feature: {feat_nome} (idx={idx}, |delta|={abs(deltas_para_ordenar[idx]):.6f})")
                adicionou_feature = True
                break
        
        if not adicionou_feature:
            print(f"\n[WARN] Nao foi possivel adicionar mais features!")
            break
    
    if iteracao >= MAX_ITER:
        print(f"\n[WARN] Limite de iteracoes ({MAX_ITER}) atingido!")
    
    return expl_robusta_str, len(expl_robusta_str) - len(expl_inicial)

def main():
    print("Configurando MNIST...")
    X_train, X_test, y_train, y_test = setup_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\nTreinando modelo...")
    modelo = train_model(X_train, y_train)
    
    print("\nEncontrando thresholds...")
    t_plus, t_minus = find_thresholds(modelo, X_train, y_train)
    print(f"t+ = {t_plus:.6f}, t- = {t_minus:.6f}")
    
    print("\nIdentificando instâncias rejeitadas no teste...")
    scores_test = modelo.decision_function(X_test)
    mask_rejeitadas = (scores_test > t_minus) & (scores_test < t_plus)
    
    num_rejeitadas = np.sum(mask_rejeitadas)
    print(f"Total rejeitadas: {num_rejeitadas}")
    
    if num_rejeitadas == 0:
        print("\n[WARN] Nenhuma instancia rejeitada encontrada!")
        return
    
    # Pegar primeira rejeitada
    idx_rejeitadas = np.where(mask_rejeitadas)[0]
    idx_primeira = idx_rejeitadas[0]
    
    print(f"\n\n{'#'*80}")
    print(f"# SIMULANDO GERAÇÃO DE EXPLICAÇÃO PARA REJEITADA (índice {idx_primeira})")
    print(f"{'#'*80}")
    
    instance_df = X_test.iloc[[idx_primeira]]
    score_orig = modelo.decision_function(instance_df)[0]
    print(f"Score original: {score_orig:.6f}")
    
    # Gerar explicações iniciais com cada premisa
    print(f"\n{'='*80}")
    print("GERANDO EXPLICAÇÕES INICIAIS (one_explanation_formal)")
    print(f"{'='*80}")
    
    expl_inicial_p1 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 1)
    print(f"\nPremisa classe 1: {len(expl_inicial_p1)} features")
    if len(expl_inicial_p1) <= 5:
        for feat in expl_inicial_p1:
            print(f"  - {feat}")
    
    expl_inicial_p2 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 0)
    print(f"\nPremisa classe 0: {len(expl_inicial_p2)} features")
    if len(expl_inicial_p2) <= 5:
        for feat in expl_inicial_p2:
            print(f"  - {feat}")
    
    # Simular fase 1 para premisa 1
    print(f"\n\n{'#'*80}")
    print(f"# FASE 1 COM PREMISA CLASSE 1")
    print(f"{'#'*80}")
    expl_robusta_p1, adicoes1 = simular_fase_1_reforco(modelo, instance_df, expl_inicial_p1, 
                                                        X_train, t_plus, t_minus, 1)
    print(f"\nResultado: {len(expl_robusta_p1)} features ({adicoes1} adicionadas)")
    
    # Simular fase 1 para premisa 0
    print(f"\n\n{'#'*80}")
    print(f"# FASE 1 COM PREMISA CLASSE 0")
    print(f"{'#'*80}")
    expl_robusta_p2, adicoes2 = simular_fase_1_reforco(modelo, instance_df, expl_inicial_p2, 
                                                        X_train, t_plus, t_minus, 0)
    print(f"\nResultado: {len(expl_robusta_p2)} features ({adicoes2} adicionadas)")
    
    print(f"\n\n{'='*80}")
    print("RESUMO FINAL")
    print(f"{'='*80}")
    print(f"Premisa 1: {len(expl_robusta_p1)} features")
    print(f"Premisa 0: {len(expl_robusta_p2)} features")
    print(f"Menor explicação: {min(len(expl_robusta_p1), len(expl_robusta_p2))} features")

if __name__ == '__main__':
    main()
