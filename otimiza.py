"""
================================================================================
SCRIPT: Otimização de Explicações (Validação do PEAB)
================================================================================

OBJETIVO:
    Validar a eficiência do método heurístico PEAB comparando-o com uma 
    solução exata baseada em Programação Linear Inteira (MILP) usando PuLP.

FUNCIONAMENTO:
    1. Treina o mesmo modelo (LogisticRegression) usado no PEAB.
    2. Para um conjunto de instâncias de teste:
       a. Gera a explicação heurística (PEAB).
       b. Gera a explicação ótima (MILP Solver).
    3. Compara os tamanhos dos conjuntos de features encontrados.

MATEMÁTICA (MILP):
    Minimizar: sum(z_i)  onde z_i ∈ {0, 1}
    Sujeito a: Score(x_mascarado) >= Threshold (para classe positiva)
    
    Onde "x_mascarado" assume o pior valor possível para as features onde z_i = 0.

AUTOR: GitHub Copilot (Baseado na solicitação do usuário)
DATA: 27/11/2025
================================================================================
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import pulp
from typing import List, Dict, Any, Tuple

# Importações do projeto existente
from peab import (
    configurar_experimento, 
    treinar_e_avaliar_modelo, 
    gerar_explicacao_instancia,
    DEFAULT_LOGREG_PARAMS,
    RANDOM_STATE
)

def solve_exact_explanation(
    instance_idx: int,
    X_instance: np.ndarray,
    coefs: np.ndarray,
    intercept: float,
    threshold: float,
    target_class: int,
    feature_names: List[str],
    time_limit: int = 60
) -> Dict[str, Any]:
    """
    Encontra o subconjunto MÍNIMO de features usando solver MILP (PuLP).
    
    Considera o cenário de 'worst-case' para features removidas (missingness),
    assumindo que os dados estão normalizados [0, 1] pelo MinMaxScaler.
    """
    num_features = len(coefs)
    
    # Cria o problema de minimização
    prob = pulp.LpProblem(f"Min_Explanation_Idx_{instance_idx}", pulp.LpMinimize)
    
    # Variáveis de decisão binárias: z_i = 1 se a feature i é mantida, 0 se removida
    z = pulp.LpVariable.dicts("z", range(num_features), cat="Binary")
    
    # Função Objetivo: Minimizar a quantidade de features mantidas
    prob += pulp.lpSum([z[i] for i in range(num_features)])
    
    # --- RESTRIÇÃO DE ROBUSTEZ (WORST-CASE) ---
    # O modelo é linear: Score = Σ(w_i * x_i) + b
    # Se mantemos a feature (z_i=1), ela contribui com w_i * x_instance_i
    # Se removemos a feature (z_i=0), ela contribui com o PIOR valor possível (worst_val)
    # Como usamos MinMaxScaler, o range é [0, 1].
    
    # Cálculo dos termos constantes para a restrição
    # Queremos: Σ [z_i * (w_i * x_i) + (1-z_i) * (w_i * x_worst)] + b >= Threshold (se classe 1)
    #           Σ [z_i * (w_i * x_i) + (1-z_i) * (w_i * x_worst)] + b <= Threshold (se classe 0)
    
    constraint_terms = []
    constant_sum = intercept
    
    for i in range(num_features):
        w_i = coefs[i]
        x_val = X_instance[i]
        
        # Determinar o pior valor (worst_case) para a feature i
        # Se queremos provar Classe 1 (Score alto), o pior valor é aquele que joga o score pra baixo.
        # Se w_i > 0, pior val = 0. Se w_i < 0, pior val = 1.
        
        if target_class == 1:
            # Queremos Score > T. Pior cenário: feature assume valor que minimiza w_i * val
            x_worst = 0.0 if w_i > 0 else 1.0
            
            # Restrição: Score >= Threshold
            # Termo da feature i: z_i * (w_i * x_val) + (1 - z_i) * (w_i * x_worst)
            # = z_i * (w_i * x_val - w_i * x_worst) + w_i * x_worst
            
            term_variable = z[i] * (w_i * x_val - w_i * x_worst)
            term_constant = w_i * x_worst
            
            constraint_terms.append(term_variable)
            constant_sum += term_constant
            
        else: # target_class == 0
            # Queremos Score < T. Pior cenário: feature assume valor que maximiza w_i * val
            # Se w_i > 0, pior val = 1. Se w_i < 0, pior val = 0.
            x_worst = 1.0 if w_i > 0 else 0.0
            
            # Restrição: Score <= Threshold
            term_variable = z[i] * (w_i * x_val - w_i * x_worst)
            term_constant = w_i * x_worst
            
            constraint_terms.append(term_variable)
            constant_sum += term_constant

    # Adicionar a restrição ao solver
    if target_class == 1:
        prob += (pulp.lpSum(constraint_terms) + constant_sum >= threshold), "Robustness_Constraint"
    else:
        prob += (pulp.lpSum(constraint_terms) + constant_sum <= threshold), "Robustness_Constraint"

    # Resolver
    # Usar CBC solver (padrão do PuLP) com silenciamento de log
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    
    if status != "Optimal":
        return {
            "status": status,
            "size": -1,
            "features": []
        }
    
    # Extrair solução
    selected_indices = [i for i in range(num_features) if pulp.value(z[i]) > 0.5]
    selected_features = [feature_names[i] for i in selected_indices]
    
    return {
        "status": status,
        "size": len(selected_features),
        "features": selected_features
    }

def main():
    print("\n" + "="*80)
    print("COMPARADOR: PEAB (Heurística) vs MILP (Otimização Exata)")
    print("="*80 + "\n")
    
    # 1. Configuração (Pode alterar hardcoded ou usar input)
    dataset_name = "mnist" # Ex: 'mnist', 'breast_cancer', 'wine'
    
    # Se for MNIST, define o par aqui para garantir consistência
    if dataset_name == 'mnist':
        from data import datasets
        # Forçando o par que você estava usando (8 vs 3) ou (5 vs 6)
        # datasets.set_mnist_options('raw', (8, 3)) 
        # O peab.py já carrega a config do datasets.py ou do próprio script.
        pass

    print(f"[1/4] Carregando dataset: {dataset_name}...")
    try:
        X, y, class_names, rejection_cost, test_size = configurar_experimento(dataset_name)
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        return

    # 2. Treinamento
    print(f"[2/4] Treinando modelo (mesmo pipeline do PEAB)...")
    pipeline, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(
        X, y, test_size, rejection_cost, DEFAULT_LOGREG_PARAMS
    )
    
    # Extrair dados do modelo para o otimizador
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    feature_names = list(X.columns)
    
    # Preparar dados de teste transformados (para o otimizador não precisar escalar)
    # O PEAB trabalha com dados brutos e escala internamente, mas para o MILP
    # precisamos saber o valor exato que entra na multiplicação w*x.
    # Vamos pegar o X_test do split
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    
    # Transformar X_test usando o scaler treinado
    X_test_scaled = scaler.transform(X_test)
    
    # Calcular scores para filtrar
    decision_scores = model.decision_function(X_test_scaled)
    y_pred = model.predict(X_test_scaled)
    
    # 3. Seleção de Instâncias para Comparação
    # Vamos pegar algumas aleatórias e algumas "difíceis" (perto do threshold)
    n_samples = 10
    indices_to_test = np.random.choice(len(X_test), n_samples, replace=False)
    
    print(f"\n[3/4] Iniciando comparação em {n_samples} instâncias aleatórias...\n")
    
    results = []
    
    print(f"{'ID':<6} | {'Classe':<6} | {'Score':<8} | {'PEAB':<6} | {'OTIMIZA':<8} | {'Diff':<6} | {'Status':<10}")
    print("-" * 70)
    
    for i, idx in enumerate(indices_to_test):
        instance_raw = X_test.iloc[idx] # Pandas Series (bruto)
        instance_scaled = X_test_scaled[idx] # Numpy array (escalado)
        score = decision_scores[idx]
        pred_class = int(y_pred[idx])
        
        # Definir threshold alvo baseada na predição
        # Se pred=1, score deve ser > t_plus. Se pred=0, score deve ser < t_minus.
        # Se estiver na zona de rejeição, o PEAB rejeita. O otimizador pode tentar forçar uma classe?
        # Vamos focar nas ACEITAS pelo modelo.
        
        is_rejected = (score > t_minus) and (score < t_plus)
        
        if is_rejected:
            print(f"{idx:<6} | {'REJ':<6} | {score:+.4f} | {'-':<6} | {'-':<8} | {'-':<6} | {'Skipped'}")
            continue
            
        target_threshold = t_plus if pred_class == 1 else t_minus
        
        # --- A. Executar PEAB (Heurística) ---
        start_peab = time.time()
        # Nota: gerar_explicacao_instancia espera a instância bruta e usa o pipeline
        expl_peab = gerar_explicacao_instancia(
            instance_raw, 
            pipeline, 
            feature_names, 
            t_plus, 
            t_minus, 
            pred_class
        )
        time_peab = time.time() - start_peab
        size_peab = expl_peab['tamanho_explicacao']
        
        # --- B. Executar Otimizador (Exato) ---
        start_opt = time.time()
        res_opt = solve_exact_explanation(
            idx,
            instance_scaled,
            coefs,
            intercept,
            target_threshold,
            pred_class,
            feature_names
        )
        time_opt = time.time() - start_opt
        size_opt = res_opt['size']
        
        # Comparação
        diff = size_peab - size_opt
        status_icon = "✅" if diff == 0 else ("⚠️" if diff > 0 else "❌") # ❌ seria estranho (heurística melhor que ótimo?)
        
        print(f"{idx:<6} | {pred_class:<6} | {score:+.2f}   | {size_peab:<6} | {size_opt:<8} | {diff:<6} | {status_icon}")
        
        results.append({
            'id': idx,
            'peab_size': size_peab,
            'opt_size': size_opt,
            'diff': diff,
            'peab_features': expl_peab['explicacao'],
            'opt_features': res_opt['features']
        })

    # 4. Resumo
    print("-" * 70)
    print("\n[4/4] Análise Final:")
    total_diff = sum(r['diff'] for r in results)
    perfect_matches = sum(1 for r in results if r['diff'] == 0)
    
    print(f"  - Total de instâncias comparadas: {len(results)}")
    print(f"  - PEAB encontrou o ótimo global em: {perfect_matches} ({perfect_matches/len(results)*100:.1f}%)")
    print(f"  - Diferença total de features (excesso do PEAB): {total_diff}")
    
    if total_diff == 0:
        print("\n🏆 CONCLUSÃO: O PEAB foi perfeito em todas as instâncias testadas!")
    else:
        print(f"\n💡 CONCLUSÃO: O PEAB ficou próximo, com média de {total_diff/len(results):.2f} features a mais que o ótimo.")

if __name__ == "__main__":
    main()
