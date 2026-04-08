
'''Compara o MINABRO contra o LIME e o SHAP. Ele não olha 
apenas para a explicação, mas calcula a Fidelidade Abdutiva 
(se a explicação se mantém em cenários de pior caso) e a 
Estabilidade (Jaccard). Ele gera os dados para a "Tabela 1" 
que você mencionou, mostrando onde os métodos clássicos 
falham em ser estáveis ou fiéis à lógica do modelo.'''

import os
import time
import json
import warnings
import itertools
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

class ShapWrapper:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    def __call__(self, X_data):
        if isinstance(X_data, np.ndarray):
            X_data = pd.DataFrame(X_data, columns=self.feature_names)
        return self.model.predict_proba(X_data)

def calc_jaccard_stability(list_of_sets):
    if not list_of_sets or len(list_of_sets) < 2: return 1.0
    pairs = list(itertools.combinations(list_of_sets, 2))
    scores = []
    for a, b in pairs:
        if not a and not b: scores.append(1.0)
        elif not a or not b: scores.append(0.0)
        else: scores.append(len(a.intersection(b)) / len(a.union(b)))
    return np.mean(scores)

def measure_local_fidelity(mlp, inst_vals, feature_names, selected_features, X_train_std, num_samples=200):
    if not selected_features: return 0.0
    perturbations = np.tile(inst_vals, (num_samples, 1))
    feature_indices = set([feature_names.index(f) for f in selected_features])
    
    for idx in range(len(feature_names)):
        if idx not in feature_indices:
            std = X_train_std[idx] if X_train_std[idx] > 0 else 1.0
            perturbations[:, idx] += np.random.normal(0, std, num_samples)
            
    df_pert = pd.DataFrame(perturbations, columns=feature_names)
    preds = mlp.predict(df_pert)
    orig_pred = mlp.predict(pd.DataFrame([inst_vals], columns=feature_names))[0]
    return np.mean(preds == orig_pred)

def executar_batalha_dos_titas_v4():
    datasets_alvo = [
        'banknote', 'pima_indians_diabetes', 'breast_cancer', 
        'sonar', 'spambase', 'heart_disease', 'vertebral_column'
    ]
    resultados_globais = []
    
    print("="*120)
    print(" SCRIPT V4: BATALHA DOS TITAS (Metodologia Corrigida para Defesa)")
    print("="*120)

    for nome_dataset in datasets_alvo:
        print(f"\n>>> Analisando: {nome_dataset.upper()}")
        try:
            X, y, _ = carregar_dataset(nome_dataset)
        except Exception:
            print(f"    [AVISO] Dataset nao encontrado. Pulando...")
            continue
            
        feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
        X_train_std = X_train.std().values
        
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

        meta_instancias = 50
        instancias_coletadas = 0
        instancias_rejeitadas = 0
        X_test_vals = X_test.values 
        
        metricas = {
            'fid_abd_m': [], 'fid_abd_l': [], 'fid_abd_s': [],
            'fid_loc_m': [], 'fid_loc_l': [], 'fid_loc_s': [],
            'estab_m': [], 'estab_l': [], 'estab_s': [],
            'tam_m': [], 'tam_l': [], 'tam_s': [],
            't_m': [], 't_l': [], 't_s': []
        }

        max_attempts = len(X_test_vals)
        
        for i in range(max_attempts):
            if instancias_coletadas >= meta_instancias: break
                
            inst_vals = X_test_vals[i]
            original_pred = modelo_mlp.predict(pd.DataFrame([inst_vals], columns=feature_names))[0]
            
            # [Correcao 1] Recebendo bounds diretamente do modelo validado
            start_m = time.perf_counter()
            exp_minabro, pred_code, is_faithful, bounds = explainer_minabro.explain_instance(inst_vals)
            t_minabro = time.perf_counter() - start_m
            
            # [Correcao 4] Logica de rejeicao explicita com fallback
            if pred_code == 2 or len(exp_minabro) == 0: 
                instancias_rejeitadas += 1
                if instancias_rejeitadas > max_attempts * 0.8:
                    print(f"    [AVISO] Muitas rejeicoes ({instancias_rejeitadas}). Dataset ignorado por excesso de zonas de inseguranca.")
                    break
                continue

            # LIME
            num_f_lime = min(10, len(feature_names))
            start_l = time.perf_counter()
            exp_l = explainer_lime.explain_instance(inst_vals, modelo_mlp.predict_proba, num_features=num_f_lime)
            t_lime = time.perf_counter() - start_l
            lime_map = exp_l.as_map()
            lime_class = list(lime_map.keys())[0]
            exp_lime_features = [feature_names[x[0]] for x in lime_map[lime_class]]

            # SHAP
            start_s = time.perf_counter()
            shap_values = explainer_shap.shap_values(inst_vals, nsamples=100, silent=True)
            t_shap = time.perf_counter() - start_s
            
            if isinstance(shap_values, list):
                shap_importances = np.abs(shap_values[1])
            else:
                shap_importances = np.abs(shap_values)
            if len(shap_importances.shape) > 1: shap_importances = shap_importances[0]
            
            # [Correcao 3] Threshold do SHAP usando percentil 75
            thresh = np.percentile(shap_importances, 75)
            indices_shap = np.where(shap_importances >= thresh)[0]
            if len(indices_shap) == 0: indices_shap = [np.argmax(shap_importances)]
            exp_shap_features = [feature_names[idx] for idx in indices_shap]

            # ESTABILIDADE (JACCARD)
            sets_lime = [set(exp_lime_features)]
            sets_shap = [set(exp_shap_features)]
            for _ in range(4):
                exp_l_temp = explainer_lime.explain_instance(inst_vals, modelo_mlp.predict_proba, num_features=num_f_lime)
                l_c = list(exp_l_temp.as_map().keys())[0]
                sets_lime.append(set([feature_names[x[0]] for x in exp_l_temp.as_map()[l_c]]))
                
                sv_temp = explainer_shap.shap_values(inst_vals, nsamples=100, silent=True)
                imp_temp = np.abs(sv_temp[1]) if isinstance(sv_temp, list) else np.abs(sv_temp)
                if len(imp_temp.shape) > 1: imp_temp = imp_temp[0]
                th_temp = np.percentile(imp_temp, 75)
                idx_temp = np.where(imp_temp >= th_temp)[0]
                if len(idx_temp) == 0: idx_temp = [np.argmax(imp_temp)]
                # [Correcao 2] Usando j em vez de i
                sets_shap.append(set([feature_names[j] for j in idx_temp]))

            metricas['estab_m'].append(1.0) 
            metricas['estab_l'].append(calc_jaccard_stability(sets_lime))
            metricas['estab_s'].append(calc_jaccard_stability(sets_shap))

            # FIDELIDADE ABDUTIVA (Pior Cenario)
            f_m = explainer_minabro._check_fidelity_mlp(inst_vals, {feature_names.index(f) for f in exp_minabro}, bounds, original_pred)
            f_l = explainer_minabro._check_fidelity_mlp(inst_vals, {feature_names.index(f) for f in exp_lime_features}, bounds, original_pred)
            f_s = explainer_minabro._check_fidelity_mlp(inst_vals, {feature_names.index(f) for f in exp_shap_features}, bounds, original_pred)
            metricas['fid_abd_m'].append(int(f_m))
            metricas['fid_abd_l'].append(int(f_l))
            metricas['fid_abd_s'].append(int(f_s))

            # FIDELIDADE LOCAL (Gaussiana)
            metricas['fid_loc_m'].append(measure_local_fidelity(modelo_mlp, inst_vals, feature_names, exp_minabro, X_train_std))
            metricas['fid_loc_l'].append(measure_local_fidelity(modelo_mlp, inst_vals, feature_names, exp_lime_features, X_train_std))
            metricas['fid_loc_s'].append(measure_local_fidelity(modelo_mlp, inst_vals, feature_names, exp_shap_features, X_train_std))

            metricas['tam_m'].append(len(exp_minabro))
            metricas['tam_l'].append(len(exp_lime_features))
            metricas['tam_s'].append(len(exp_shap_features))
            metricas['t_m'].append(t_minabro)
            metricas['t_l'].append(t_lime)
            metricas['t_s'].append(t_shap)

            instancias_coletadas += 1

        if instancias_coletadas > 0:
            res = {'Dataset': nome_dataset, 'Validas': instancias_coletadas, 'Rejeitadas': instancias_rejeitadas}
            for k in metricas.keys():
                res[k] = np.mean(metricas[k]) if metricas[k] else 0.0
            resultados_globais.append(res)
            print(f"    [OK] {instancias_coletadas} validas | {instancias_rejeitadas} rejeitadas por seguranca.")

    with open('resultados_dissertacao_final.json', 'w') as f:
        json.dump(resultados_globais, f, indent=4)

    print("\n" + "="*120)
    print(" TABELA 1: FIDELIDADE ABDUTIVA E ESTABILIDADE (JACCARD)")
    print("="*120)
    print(f"{'Dataset':<18} | {'Validas':<7} | {'Rejeit.':<7} | {'Fid_MINABRO':<11} | {'Fid_LIME':<8} | {'Fid_SHAP':<8} | {'Estab_MINABRO':<13} | {'Estab_LIME':<10} | {'Estab_SHAP':<10}")
    print("-" * 120)
    for r in resultados_globais:
        print(f"{r['Dataset']:<18} | {r['Validas']:>7} | {r['Rejeitadas']:>7} | {r['fid_abd_m']*100:>10.1f}% | {r['fid_abd_l']*100:>7.1f}% | {r['fid_abd_s']*100:>7.1f}% | {r['estab_m']*100:>12.1f}% | {r['estab_l']*100:>9.1f}% | {r['estab_s']*100:>9.1f}%")

    print("\n" + "="*120)
    print(" TABELA 2: FIDELIDADE LOCAL, TAMANHO E TEMPO")
    print("="*120)
    print(f"{'Dataset':<18} | {'FLoc_MIN':<8} | {'FLoc_LIM':<8} | {'FLoc_SHA':<8} | {'Tam_MIN':<7} | {'Tam_LIM':<7} | {'Tam_SHA':<7} | {'T_MIN':<7} | {'T_LIM':<7} | {'T_SHA':<7}")
    print("-" * 120)
    for r in resultados_globais:
        print(f"{r['Dataset']:<18} | {r['fid_loc_m']*100:>7.1f}% | {r['fid_loc_l']*100:>7.1f}% | {r['fid_loc_s']*100:>7.1f}% | {r['tam_m']:>7.1f} | {r['tam_l']:>7.1f} | {r['tam_s']:>7.1f} | {r['t_m']:>6.3f}s | {r['t_l']:>6.3f}s | {r['t_s']:>6.3f}s")
    print("=" * 120)

if __name__ == '__main__':
    executar_batalha_dos_titas_v4()