
'''Executa o MINABRO, LIME e SHAP em uma amostra de instâncias 
e imprime uma tabela comparando Tempo de Execução e Fidelidade. 
É um script de "sanidade" para ver se o custo computacional do 
MINABRO compensa o ganho em fidelidade.'''

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Importações para as Baselines
import shap
import lime
import lime.lime_tabular

# Importações do seu projeto
from data.datasets import carregar_dataset
from sklearn.model_selection import train_test_split
from MINABRO_MLP import (
    treinar_modelo_mlp, encontrar_thresholds_otimos, 
    MinabroMLPSurrogateExplainer, MLP_PARAMS, RANDOM_STATE, carregar_hiperparametros_locais
)

def run_baseline_comparison():
    print("="*70)
    print(" INICIANDO COMPARAÇÃO: MINABRO vs LIME vs SHAP")
    print(" Dataset: Pima Indians Diabetes | Amostra: 30 Instâncias Aceites")
    print("="*70)

    # 1. Preparar Dados e Treinar Oráculo
    X, y, _ = carregar_dataset('pima_indians_diabetes')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    
    print("\n[1/4] Treinando Oráculo (MLP)...")
    modelo_mlp = treinar_modelo_mlp(X_train, y_train, MLP_PARAMS)
    t_plus, t_minus = encontrar_thresholds_otimos(X_train, y_train, 0.24, modelo_mlp)
    
    # Filtrar apenas instâncias que NÃO foram rejeitadas globalmente
    probas = np.clip(modelo_mlp.predict_proba(X_test), 1e-9, 1 - 1e-9)
    scores = np.log(probas[:, 1] / probas[:, 0])
    mask_aceites = ~((scores > t_minus) & (scores < t_plus))
    X_test_aceites = X_test[mask_aceites]
    
    # Selecionar uma amostra de 30 instâncias para o teste não demorar horas (SHAP é lento)
    amostra_size = min(30, len(X_test_aceites))
    X_sample = X_test_aceites.sample(n=amostra_size, random_state=RANDOM_STATE)
    X_sample_vals = X_sample.values
    feature_names = X.columns.tolist()

    # 2. Inicializar Explicadores
    print("[2/4] Inicializando Explicadores (MINABRO, LIME, SHAP)...")
    
    # MINABRO
    logreg_params = carregar_hiperparametros_locais('pima_indians_diabetes')
    explainer_minabro = MinabroMLPSurrogateExplainer(modelo_mlp, X_train, 0.24, logreg_params)
    
    # LIME
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=feature_names, 
        class_names=['Negativo', 'Positivo'], discretize_continuous=False
    )
    
    # SHAP (Usamos um background menor do KernelExplainer para não travar)
    X_background = shap.kmeans(X_train.values, 20)
    explainer_shap = shap.KernelExplainer(modelo_mlp.predict_proba, X_background)

    resultados = []

    print(f"\n[3/4] Explicando {amostra_size} instâncias lado a lado...\n")
    
    for i in range(amostra_size):
        inst_vals = X_sample_vals[i]
        
        # --- MINABRO ---
        start_m = time.perf_counter()
        exp_minabro, _, _ = explainer_minabro.explain_instance(inst_vals)
        tempo_minabro = time.perf_counter() - start_m
        
        # Recupera os limites locais para testar LIME e SHAP no mesmo "campo de batalha"
        _, _, bounds = explainer_minabro._gerar_vizinhanca_local_fronteira(inst_vals)
        original_pred = modelo_mlp.predict([inst_vals])[0]
        K = len(exp_minabro) # O tamanho ótimo ditado pelo MINABRO
        
        # Se for decisão incondicional (K=0), pulamos a comparação para sermos justos
        if K == 0:
            continue

        # --- LIME ---
        start_l = time.perf_counter()
        exp_l = explainer_lime.explain_instance(inst_vals, modelo_mlp.predict_proba, num_features=K)
        tempo_lime = time.perf_counter() - start_l
        exp_lime_features = [feature_names.index(f[0]) for f in exp_l.as_list()]
        
        # --- SHAP ---
        start_s = time.perf_counter()
        shap_values = explainer_shap.shap_values(inst_vals, nsamples=500, silent=True)
        tempo_shap = time.perf_counter() - start_s
        # Pega os índices das K features com maior valor absoluto de SHAP
        shap_importances = np.abs(shap_values[original_pred]) if isinstance(shap_values, list) else np.abs(shap_values)
        exp_shap_features = np.argsort(-shap_importances)[:K].tolist()

        # --- TESTE DE FIDELIDADE (O JUÍZO FINAL) ---
        # Converte nomes do MINABRO para índices para a função de teste
        minabro_indices = {feature_names.index(f) for f in exp_minabro}
        lime_indices = set(exp_lime_features)
        shap_indices = set(exp_shap_features)

        fid_minabro = explainer_minabro._check_fidelity_mlp(inst_vals, minabro_indices, bounds, original_pred)
        fid_lime = explainer_minabro._check_fidelity_mlp(inst_vals, lime_indices, bounds, original_pred)
        fid_shap = explainer_minabro._check_fidelity_mlp(inst_vals, shap_indices, bounds, original_pred)

        resultados.append({
            'Tamanho (K)': K,
            'Tempo_MINABRO': tempo_minabro, 'Fidelidade_MINABRO': fid_minabro,
            'Tempo_LIME': tempo_lime, 'Fidelidade_LIME': fid_lime,
            'Tempo_SHAP': tempo_shap, 'Fidelidade_SHAP': fid_shap
        })
        
        print(f"Instância {i+1}: K={K} | Fidelidade -> MINABRO: {fid_minabro} | LIME: {fid_lime} | SHAP: {fid_shap}")

    # 4. Consolidar Resultados
    print("\n[4/4] Resultados Finais (Médias):")
    df_res = pd.DataFrame(resultados)
    
    print("-" * 50)
    print(f"Métrica               | MINABRO | LIME    | SHAP")
    print("-" * 50)
    print(f"Tamanho Médio (Fixo)  | {df_res['Tamanho (K)'].mean():.2f}    | {df_res['Tamanho (K)'].mean():.2f}    | {df_res['Tamanho (K)'].mean():.2f}")
    print(f"Fidelidade Pior Caso  | {df_res['Fidelidade_MINABRO'].mean()*100:.1f}%  | {df_res['Fidelidade_LIME'].mean()*100:.1f}%  | {df_res['Fidelidade_SHAP'].mean()*100:.1f}%")
    print(f"Tempo por Instância   | {df_res['Tempo_MINABRO'].mean():.3f}s  | {df_res['Tempo_LIME'].mean():.3f}s  | {df_res['Tempo_SHAP'].mean():.3f}s")
    print("-" * 50)
    print("\nSalvo em results/baseline_comparison.csv")
    
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/baseline_comparison.csv', index=False)

if __name__ == '__main__':
    run_baseline_comparison()