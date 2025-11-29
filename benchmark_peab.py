import time
import pulp
import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- IMPORTANDO AS FUNÇÕES DO SEU SCRIPT ORIGINAL ---
# Certifique-se que o arquivo peab.py está na mesma pasta
from peab import (
    carregar_hiperparametros,
    configurar_experimento,
    treinar_e_avaliar_modelo,
    aplicar_selecao_top_k_features,
    gerar_explicacao_instancia, # SEU MÉTODO PEAB
    selecionar_dataset_e_classe,
    DATASET_CONFIG,
    DEFAULT_LOGREG_PARAMS,
    RANDOM_STATE,
    _get_lr
)
from utils.progress_bar import ProgressBar

# =============================================================================
#  PARTE 1: O "JUIZ" (Cálculo do Mínimo Matemático com PuLP)
# =============================================================================
def calcular_minimo_exato_pulp(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float) -> int:
    """
    Calcula APENAS o tamanho mínimo necessário (cardinalidade) usando Otimização Inteira.
    """
    logreg = _get_lr(modelo)
    scaler = modelo.named_steps['scaler']
    
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    
    vals_scaled = scaler.transform(instance_df)[0]
    X_train_scaled = scaler.transform(X_train)
    min_scaled = X_train_scaled.min(axis=0)
    max_scaled = X_train_scaled.max(axis=0)
    
    score_atual = modelo.decision_function(instance_df)[0]
    if score_atual >= t_plus: estado = 1
    elif score_atual <= t_minus: estado = 0
    else: estado = 2 # Rejeição

    prob = pulp.LpProblem("JuizMinimalidade", pulp.LpMinimize)
    
    # Variáveis: z_i = 1 (Fixa feature), z_i = 0 (Livre)
    z = [pulp.LpVariable(f"z_{i}", cat='Binary') for i in range(len(coefs))]
    
    # Objetivo: Menor número de features
    prob += pulp.lpSum(z)
    
    # Restrições de Robustez (Pior Caso)
    base_worst_min = intercept
    base_worst_max = intercept
    termos_min = []
    termos_max = []
    
    for i, w in enumerate(coefs):
        v_worst_min = min_scaled[i] if w > 0 else max_scaled[i]
        v_worst_max = max_scaled[i] if w > 0 else min_scaled[i]
        
        contrib_worst_min = v_worst_min * w
        contrib_worst_max = v_worst_max * w
        contrib_real = vals_scaled[i] * w
        
        base_worst_min += contrib_worst_min
        base_worst_max += contrib_worst_max
        
        termos_min.append(z[i] * (contrib_real - contrib_worst_min))
        termos_max.append(z[i] * (contrib_real - contrib_worst_max))

    if estado == 1:
        prob += (base_worst_min + pulp.lpSum(termos_min)) >= t_plus
    elif estado == 0:
        prob += (base_worst_max + pulp.lpSum(termos_max)) <= t_minus
    else: # Rejeição
        prob += (base_worst_max + pulp.lpSum(termos_max)) <= t_plus
        prob += (base_worst_min + pulp.lpSum(termos_min)) >= t_minus

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        return int(pulp.value(prob.objective))
    else:
        return -1 

# =============================================================================
#  PARTE 2: O ORGANIZADOR DO EXPERIMENTO
# =============================================================================
def executar_benchmark():
    dataset_name, _, _, _, _ = selecionar_dataset_e_classe()
    
    if not dataset_name:
        print("Nenhum dataset selecionado. Encerrando.")
        return

    print(f"\n========================================================")
    print(f"   BENCHMARK CIENTÍFICO: {dataset_name.upper()}")
    print(f"========================================================\n")

    todos_params = carregar_hiperparametros()
    X, y, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)
    
    params = DEFAULT_LOGREG_PARAMS.copy()
    if dataset_name in todos_params and 'params' in todos_params[dataset_name]:
        params.update(todos_params[dataset_name]['params'])

    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0:
        print(f"[BENCH] Aplicando redução Top-{top_k} features...")
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X, y, test_size, rejection_cost, params)
        X_train_tmp, X_test_tmp, _, _ = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
        _, _, selected_feats = aplicar_selecao_top_k_features(X_train_tmp, X_test_tmp, modelo_temp, top_k)
        X = X[selected_feats]

    print("[BENCH] Treinando modelo...")
    modelo, t_plus, t_minus, _ = treinar_e_avaliar_modelo(X, y, test_size, rejection_cost, params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    
    print(f"[BENCH] T+: {t_plus:.4f}, T-: {t_minus:.4f}")

    resultados = []
    
    with ProgressBar(total=len(X_test), description=f"Comparando {dataset_name}") as pbar:
        for i in range(len(X_test)):
            instancia = X_test.iloc[[i]]
            
            # --- [NOVO] Identifica se é REJEIÇÃO ou CLASSIFICAÇÃO ---
            score = modelo.decision_function(instancia)[0]
            if t_minus <= score <= t_plus:
                tipo_predicao = "REJEITADA"
            else:
                tipo_predicao = "CLASSIFICADA"
            
            # --- PEAB ---
            start_peab = time.perf_counter()
            expl_peab, _, _, _ = gerar_explicacao_instancia(instancia, modelo, X_train, t_plus, t_minus)
            time_peab = time.perf_counter() - start_peab
            
            # --- OTIMIZAÇÃO ---
            start_opt = time.perf_counter()
            tamanho_optimo = calcular_minimo_exato_pulp(modelo, instancia, X_train, t_plus, t_minus)
            time_opt = time.perf_counter() - start_opt
            
            # Fallback seguro
            if tamanho_optimo == -1: tamanho_optimo = len(expl_peab)
            
            gap = len(expl_peab) - tamanho_optimo
            
            resultados.append({
                'id': i,
                'classe_real': nomes_classes[y_test.iloc[i]],
                'tipo_predicao': tipo_predicao, # <--- COLUNA NOVA
                'tamanho_PEAB': len(expl_peab),
                'tamanho_OPTIMO': tamanho_optimo,
                'GAP': gap,
                'tempo_PEAB': time_peab,
                'tempo_OPTIMO': time_opt,
                'is_optimal': (gap == 0)
            })
            pbar.update()

    df = pd.DataFrame(resultados)
    
    # Resumo Rápido no Terminal
    print(f"\n{'='*60}")
    print(f"RESUMO POR TIPO DE PREDIÇÃO ({dataset_name})")
    print(f"{'='*60}")
    # Agrupa por tipo para mostrar onde o PEAB é melhor/pior
    print(df.groupby('tipo_predicao').agg({
        'GAP': 'mean',
        'is_optimal': lambda x: f"{x.mean()*100:.1f}%",
        'tempo_PEAB': lambda x: f"{x.mean():.4f}s"
    }).rename(columns={'is_optimal': 'Taxa Otimalidade', 'tempo_PEAB': 'Tempo Médio'}).to_string())
    print(f"{'='*60}")
    
    filename = f"results/benchmark/bench_{dataset_name}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[ARQUIVO] CSV salvo em: {filename}")

if __name__ == "__main__":
    executar_benchmark()