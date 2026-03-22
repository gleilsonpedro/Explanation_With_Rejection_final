import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import shap
import lime
import lime.lime_tabular

from data.datasets import carregar_dataset
from sklearn.model_selection import train_test_split
from MINABRO_MLP import (
    treinar_modelo_mlp, encontrar_thresholds_otimos, 
    MinabroMLPSurrogateExplainer, MLP_PARAMS, RANDOM_STATE, carregar_hiperparametros_locais
)

# [Solução Problema 4] Classe Wrapper para isolar o escopo do SHAP de forma robusta
class ShapWrapper:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def __call__(self, X_data):
        if isinstance(X_data, np.ndarray):
            X_data = pd.DataFrame(X_data, columns=self.feature_names)
        return self.model.predict_proba(X_data)

def executar_batalha_dos_titas():
    # [Solução Novos Datasets] Lista expandida para peso estatístico
    datasets_alvo = [
        'banknote', 'pima_indians_diabetes', 'breast_cancer', 
        'sonar', 'spambase', 'heart_disease', 'vertebral_column'
    ]
    resultados_globais = []
    
    print("="*95)
    print(" BATALHA DOS TITAS: MINABRO vs LIME vs SHAP (Multi-Dataset)")
    print(" Objetivo: Avaliar Fidelidade Abdutiva no Pior Cenario (Robustez Mestre)")
    print("="*95)

    for nome_dataset in datasets_alvo:
        print(f"\n>>> Iniciando analise: {nome_dataset.upper()}")
        try:
            X, y, _ = carregar_dataset(nome_dataset)
        except Exception as e:
            print(f"    [AVISO] Dataset '{nome_dataset}' nao encontrado ou com erro de nome. Pulando...")
            continue
            
        feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
        
        modelo_mlp = treinar_modelo_mlp(X_train, y_train, MLP_PARAMS)
        
        logreg_params = carregar_hiperparametros_locais(nome_dataset)
        explainer_minabro = MinabroMLPSurrogateExplainer(modelo_mlp, X_train, 0.24, logreg_params)
        
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_train.values, feature_names=feature_names, 
            class_names=['Classe 0', 'Classe 1'], discretize_continuous=False
        )
        
        X_background = shap.kmeans(X_train.values, 20)
        wrapper = ShapWrapper(modelo_mlp, feature_names)
        explainer_shap = shap.KernelExplainer(wrapper, X_background)

        # [Solução Problema 3] 50 instâncias para peso estatístico de dissertação
        meta_instancias = 50
        instancias_coletadas = 0
        instancias_rejeitadas = 0 # [Solução Problema 1] Rastreio de rejeições
        X_test_vals = X_test.values 
        
        tempos_minabro, tempos_lime, tempos_shap = [], [], []
        fidelidades_minabro, fidelidades_lime, fidelidades_shap = [], [], []

        print(f"    Progresso: Coletando {meta_instancias} explicacoes fies...")
        for i in range(len(X_test_vals)):
            if instancias_coletadas >= meta_instancias:
                break
                
            inst_vals = X_test_vals[i]
            original_pred = modelo_mlp.predict(pd.DataFrame([inst_vals], columns=feature_names))[0]
            _, _, bounds = explainer_minabro._gerar_vizinhanca_local_fronteira(inst_vals)
            
            start_m = time.perf_counter()
            exp_minabro, pred_code, is_faithful = explainer_minabro.explain_instance(inst_vals)
            tempo_minabro = time.perf_counter() - start_m
            
            K = len(exp_minabro)
            # Regra de Rejeição Honesta
            if pred_code == 2 or K == 0: 
                instancias_rejeitadas += 1
                continue

            start_l = time.perf_counter()
            exp_l = explainer_lime.explain_instance(inst_vals, modelo_mlp.predict_proba, num_features=K)
            tempo_lime = time.perf_counter() - start_l
            
            lime_map = exp_l.as_map()
            lime_class = list(lime_map.keys())[0]
            exp_lime_features = [x[0] for x in lime_map[lime_class]]
            
            start_s = time.perf_counter()
            # [Solução Problema 3] SHAP nsamples=100 para viabilidade de tempo
            shap_values = explainer_shap.shap_values(inst_vals, nsamples=100, silent=True)
            tempo_shap = time.perf_counter() - start_s
            
            shap_importances = np.abs(shap_values[original_pred]) if isinstance(shap_values, list) else np.abs(shap_values)
            if len(shap_importances.shape) > 1:
                shap_importances = shap_importances[0]
            exp_shap_features = np.argsort(-shap_importances)[:K].tolist()

            minabro_indices = {feature_names.index(f) for f in exp_minabro}
            lime_indices = set(exp_lime_features)
            shap_indices = set(exp_shap_features)

            fid_minabro = explainer_minabro._check_fidelity_mlp(inst_vals, minabro_indices, bounds, original_pred)
            fid_lime = explainer_minabro._check_fidelity_mlp(inst_vals, lime_indices, bounds, original_pred)
            fid_shap = explainer_minabro._check_fidelity_mlp(inst_vals, shap_indices, bounds, original_pred)

            tempos_minabro.append(tempo_minabro)
            tempos_lime.append(tempo_lime)
            tempos_shap.append(tempo_shap)
            
            fidelidades_minabro.append(int(fid_minabro))
            fidelidades_lime.append(int(fid_lime))
            fidelidades_shap.append(int(fid_shap))
            
            instancias_coletadas += 1

        if instancias_coletadas > 0:
            resultados_globais.append({
                'Dataset': nome_dataset,
                'Coletadas': instancias_coletadas,
                'Rejeitadas': instancias_rejeitadas,
                'Fid_MINABRO': np.mean(fidelidades_minabro) * 100,
                'Fid_LIME': np.mean(fidelidades_lime) * 100,
                'Fid_SHAP': np.mean(fidelidades_shap) * 100,
                'Tempo_MINABRO': np.mean(tempos_minabro),
                'Tempo_LIME': np.mean(tempos_lime),
                'Tempo_SHAP': np.mean(tempos_shap)
            })
            print(f"    [CONCLUIDO] {instancias_coletadas} coletadas | {instancias_rejeitadas} rejeitadas por seguranca.")

    print("\n" + "="*110)
    print(" TABELA CONSOLIDADA: RESULTADOS FINAIS PARA A DISSERTACAO")
    print("="*110)
    print(f"{'Dataset':<22} | {'Validas':<7} | {'Rejeit.':<7} | {'Fid MINABRO':<11} | {'Fid LIME':<10} | {'Fid SHAP':<10} | {'T. MINABRO':<10} | {'T. LIME':<10} | {'T. SHAP':<10}")
    print("-" * 110)
    
    for res in resultados_globais:
        print(f"{res['Dataset']:<22} | {res['Coletadas']:>7} | {res['Rejeitadas']:>7} | {res['Fid_MINABRO']:>9.1f}% | {res['Fid_LIME']:>8.1f}% | {res['Fid_SHAP']:>8.1f}% | {res['Tempo_MINABRO']:>8.4f}s | {res['Tempo_LIME']:>8.4f}s | {res['Tempo_SHAP']:>8.4f}s")
    print("=" * 110)

if __name__ == '__main__':
    executar_batalha_dos_titas()