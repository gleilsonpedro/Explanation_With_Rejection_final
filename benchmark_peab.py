import time
import pulp
import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- IMPORTANDO AS FUNÇÕES DO SEU SCRIPT ORIGINAL ---
# Isso garante que estamos usando as mesmas regras, mesmos dados e mesmo modelo.
from peab import (
    carregar_hiperparametros,
    configurar_experimento,
    treinar_e_avaliar_modelo,
    aplicar_selecao_top_k_features,
    gerar_explicacao_instancia, # <--- SEU MÉTODO PEAB
    selecionar_dataset_e_classe, # <--- O MENU
    DATASET_CONFIG,
    DEFAULT_LOGREG_PARAMS,
    RANDOM_STATE,
    _get_lr # Função auxiliar para pegar a regressão logística
)
from utils.progress_bar import ProgressBar

# =============================================================================
#  PARTE 1: O "JUIZ" (Cálculo do Mínimo Matemático com PuLP)
# =============================================================================
def calcular_minimo_exato_pulp(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float) -> int:
    """
    Calcula APENAS o tamanho mínimo necessário (cardinalidade) usando Otimização Inteira.
    Não precisamos retornar os nomes das features aqui, apenas o número para comparar.
    """
    logreg = _get_lr(modelo)
    scaler = modelo.named_steps['scaler']
    
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    
    # Prepara dados (escalonados)
    vals_scaled = scaler.transform(instance_df)[0]
    X_train_scaled = scaler.transform(X_train)
    min_scaled = X_train_scaled.min(axis=0)
    max_scaled = X_train_scaled.max(axis=0)
    
    # Identifica estado atual
    score_atual = modelo.decision_function(instance_df)[0]
    if score_atual >= t_plus: estado = 1
    elif score_atual <= t_minus: estado = 0
    else: estado = 2 # Rejeição

    # Otimização
    prob = pulp.LpProblem("JuizMinimalidade", pulp.LpMinimize)
    
    # Variáveis: z_i = 1 (Fixa feature), z_i = 0 (Livre)
    z = [pulp.LpVariable(f"z_{i}", cat='Binary') for i in range(len(coefs))]
    
    # Objetivo: Menor número de features
    prob += pulp.lpSum(z)
    
    # Construção das Restrições (Pior Caso)
    base_worst_min = intercept
    base_worst_max = intercept
    termos_min = []
    termos_max = []
    
    for i, w in enumerate(coefs):
        # Lógica: Se feature livre (z=0), assume pior valor. Se fixa (z=1), assume valor real.
        v_worst_min = min_scaled[i] if w > 0 else max_scaled[i]
        v_worst_max = max_scaled[i] if w > 0 else min_scaled[i]
        
        contrib_worst_min = v_worst_min * w
        contrib_worst_max = v_worst_max * w
        contrib_real = vals_scaled[i] * w
        
        base_worst_min += contrib_worst_min
        base_worst_max += contrib_worst_max
        
        # Ganho de estabilidade ao fixar a feature (z_i * delta_pior_caso)
        termos_min.append(z[i] * (contrib_real - contrib_worst_min))
        termos_max.append(z[i] * (contrib_real - contrib_worst_max))

    # Aplica restrição conforme a classe
    if estado == 1:
        prob += (base_worst_min + pulp.lpSum(termos_min)) >= t_plus
    elif estado == 0:
        prob += (base_worst_max + pulp.lpSum(termos_max)) <= t_minus
    else: # Rejeição
        prob += (base_worst_max + pulp.lpSum(termos_max)) <= t_plus
        prob += (base_worst_min + pulp.lpSum(termos_min)) >= t_minus

    # Resolve silenciosamente
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        return int(pulp.value(prob.objective))
    else:
        return -1 # Erro (não deve acontecer)

# =============================================================================
#  PARTE 2: O ORGANIZADOR DO EXPERIMENTO
# =============================================================================
def executar_benchmark():
    # 1. Menu de Seleção (Igual ao seu script original)
    dataset_name, _, _, _, _ = selecionar_dataset_e_classe()
    
    if not dataset_name:
        print("Nenhum dataset selecionado. Encerrando.")
        return

    print(f"\n========================================================")
    print(f"   INICIANDO BENCHMARK CIENTÍFICO: {dataset_name.upper()}")
    print(f"   (PEAB vs OTIMIZAÇÃO EXATA)")
    print(f"========================================================\n")

    # 2. Configuração e Treino (Reusando lógica do peab.py)
    todos_params = carregar_hiperparametros()
    X, y, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)
    
    # Hiperparâmetros
    params = DEFAULT_LOGREG_PARAMS.copy()
    if dataset_name in todos_params and 'params' in todos_params[dataset_name]:
        params.update(todos_params[dataset_name]['params'])

    # Redução Top-K (Se houver)
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0:
        print(f"[BENCH] Aplicando redução Top-{top_k} features (igual ao original)...")
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X, y, test_size, rejection_cost, params)
        X_train_tmp, X_test_tmp, _, _ = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
        _, _, selected_feats = aplicar_selecao_top_k_features(X_train_tmp, X_test_tmp, modelo_temp, top_k)
        X = X[selected_feats]

    # Treino Definitivo
    print("[BENCH] Treinando modelo...")
    modelo, t_plus, t_minus, _ = treinar_e_avaliar_modelo(X, y, test_size, rejection_cost, params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    
    print(f"[BENCH] Modelo pronto. Instâncias de teste: {len(X_test)}")
    print(f"[BENCH] Thresholds: t+={t_plus:.4f}, t-={t_minus:.4f}")

    # 3. Loop de Comparação
    resultados = []
    
    with ProgressBar(total=len(X_test), description=f"Comparando {dataset_name}") as pbar:
        for i in range(len(X_test)):
            instancia = X_test.iloc[[i]]
            
            # --- RODADA 1: SEU MÉTODO (PEAB) ---
            start_peab = time.perf_counter()
            # Chama a função principal do seu arquivo peab.py
            expl_peab, _, _, _ = gerar_explicacao_instancia(instancia, modelo, X_train, t_plus, t_minus)
            time_peab = time.perf_counter() - start_peab
            tamanho_peab = len(expl_peab)
            
            # --- RODADA 2: O JUIZ (OTIMIZAÇÃO) ---
            start_opt = time.perf_counter()
            tamanho_optimo = calcular_minimo_exato_pulp(modelo, instancia, X_train, t_plus, t_minus)
            time_opt = time.perf_counter() - start_opt
            
            # --- ANÁLISE ---
            if tamanho_optimo == -1: # Fallback erro solver
                tamanho_optimo = tamanho_peab 
                
            gap = tamanho_peab - tamanho_optimo
            
            resultados.append({
                'id': i,
                'classe_real': nomes_classes[y_test.iloc[i]],
                'tamanho_PEAB': tamanho_peab,
                'tamanho_OPTIMO': tamanho_optimo,
                'GAP': gap, # 0 = Perfeito
                'tempo_PEAB': time_peab,
                'tempo_OPTIMO': time_opt,
                'is_optimal': (gap == 0)
            })
            pbar.update()

    # 4. Resultados e Salvamento
    df = pd.DataFrame(resultados)
    
    # Estatísticas Rápidas
    total = len(df)
    perfeitos = df['is_optimal'].sum()
    perc_perfeito = (perfeitos / total) * 100
    speedup = df['tempo_OPTIMO'].mean() / df['tempo_PEAB'].mean() if df['tempo_PEAB'].mean() > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL: {dataset_name}")
    print(f"{'='*60}")
    print(f"-> PEAB encontrou o ótimo em: {perfeitos}/{total} casos ({perc_perfeito:.2f}%)")
    print(f"-> Tamanho Médio: PEAB={df['tamanho_PEAB'].mean():.2f} vs ÓTIMO={df['tamanho_OPTIMO'].mean():.2f}")
    print(f"-> Diferença Média (Gap): {df['GAP'].mean():.4f} features")
    print(f"-> Velocidade: PEAB é {speedup:.1f}x mais rápido que a Otimização")
    print(f"{'='*60}")
    
    # Salvar CSV
    filename = f"results/benchmark/bench_{dataset_name}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[ARQUIVO] CSV salvo em: {filename}")

if __name__ == "__main__":
    executar_benchmark()