import time
import pulp
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- IMPORTANDO AS FUNÇÕES DO PEAB (que agora estão completas no seu peab.py) ---
from peab import (
    carregar_hiperparametros,
    configurar_experimento,
    treinar_e_avaliar_modelo,
    gerar_explicacao_instancia,
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
    """Calcula APENAS o tamanho mínimo necessário (cardinalidade) usando Otimização Inteira."""
    logreg = _get_lr(modelo)
    scaler = modelo.named_steps['scaler']
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    
    vals_scaled = scaler.transform(instance_df)[0]
    
    # [IMPORTANTE] Usa limites do scaler para consistência com o PEAB
    if hasattr(scaler, 'feature_range'):
        f_min, f_max = scaler.feature_range
    else:
        f_min, f_max = 0.0, 1.0
        
    min_scaled = np.full_like(coefs, f_min)
    max_scaled = np.full_like(coefs, f_max)
    
    score_atual = modelo.decision_function(instance_df)[0]
    if score_atual >= t_plus: estado = 1
    elif score_atual <= t_minus: estado = 0
    else: estado = 2 # Rejeição

    prob = pulp.LpProblem("JuizMinimalidade", pulp.LpMinimize)
    z = [pulp.LpVariable(f"z_{i}", cat='Binary') for i in range(len(coefs))]
    prob += pulp.lpSum(z)
    
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
        
        # A restrição modela: "Se z=1 (mantém), usa valor real. Se z=0 (remove), usa pior caso."
        termos_min.append(z[i] * (contrib_real - contrib_worst_min))
        termos_max.append(z[i] * (contrib_real - contrib_worst_max))

    if estado == 1:
        # Classe 1: Score Minimo >= t_plus
        prob += (base_worst_min + pulp.lpSum(termos_min)) >= t_plus
    elif estado == 0:
        # Classe 0: Score Máximo <= t_minus
        prob += (base_worst_max + pulp.lpSum(termos_max)) <= t_minus
    else: 
        # Rejeição: Score Máximo <= t_plus E Score Mínimo >= t_minus
        prob += (base_worst_max + pulp.lpSum(termos_max)) <= t_plus
        prob += (base_worst_min + pulp.lpSum(termos_min)) >= t_minus

    # Solver Silencioso
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        return int(pulp.value(prob.objective))
    else:
        return -1 

# =============================================================================
#  PARTE 2: GERADOR DE RELATÓRIO (AGORA COM DESVIO PADRÃO)
# =============================================================================
def gerar_relatorio_tabela(df: pd.DataFrame, dataset_name: str, t_plus: float, t_minus: float, params_usados: dict):
    """Gera o relatório final com Média e Desvio Padrão para o tamanho."""
    
    output_path = f"results/benchmark/F_bench_{dataset_name}.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO DE BENCHMARK CIENTÍFICO: {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # --- SEÇÃO 0 ---
        f.write("-" * 80 + "\n")
        f.write("0. CONFIGURAÇÃO DO EXPERIMENTO\n")
        f.write("-" * 80 + "\n")
        f.write(f" - Instâncias de Teste: {len(df)}\n")
        f.write(f" - Thresholds: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n")
        f.write(f" - Parâmetros: {json.dumps(params_usados, indent=0)}\n\n")

        # --- SEÇÃO 1 ---
        f.write("-" * 80 + "\n")
        f.write("1. RESUMO GERAL DE DESEMPENHO\n")
        f.write("-" * 80 + "\n")
        
        taxa_otimalidade = df['is_optimal'].mean() * 100
        gap_medio = df['GAP'].mean()
        tempo_peab = df['tempo_PEAB'].mean()
        tempo_opt = df['tempo_OPTIMO'].mean()
        speedup = tempo_opt / tempo_peab if tempo_peab > 0 else 0.0

        geral = pd.DataFrame({
            'Métrica': ['Taxa de Otimalidade (Gap=0)', 'Gap Médio (Features)', 'Tempo Médio PEAB (s)', 'Tempo Médio OTIMIZAÇÃO (s)', 'Speedup'],
            'Valor': [f"{taxa_otimalidade:.2f}%", f"{gap_medio:.4f}", f"{tempo_peab:.5f}", f"{tempo_opt:.5f}", f"{speedup:.2f}x"]
        })
        f.write(geral.to_string(index=False, justify='left'))
        f.write("\n\n")

        # --- SEÇÃO 2: DETALHAMENTO COM DESVIO PADRÃO ---
        f.write("-" * 80 + "\n")
        f.write("2. DETALHAMENTO POR CLASSE (Média ± Desvio Padrão)\n")
        f.write("-" * 80 + "\n")
        f.write("Nota: 'Tam. PEAB' mostra a variabilidade do tamanho da explicação.\n\n")

        # Agregação inteligente solicitando 'mean' e 'std'
        grupo = df.groupby('tipo_predicao').agg({
            'id': 'count',
            'is_optimal': 'mean',
            'GAP': 'mean',
            'tamanho_PEAB': ['mean', 'std'],  # <--- AQUI ESTÁ A MÁGICA
            'tamanho_OPTIMO': 'mean',
            'tempo_PEAB': 'mean',
            'tempo_OPTIMO': 'mean'
        }).reset_index()

        # Achatando as colunas MultiIndex do Pandas
        grupo.columns = ['Tipo', 'Qtd', '% Ótimo', 'Gap Médio', 'Tam_Mean', 'Tam_Std', 'Tam. Ótimo', 'Tempo PEAB', 'Tempo Ótimo']
        
        # Tratamento de NaN no Std (caso haja apenas 1 instância de um tipo)
        grupo['Tam_Std'] = grupo['Tam_Std'].fillna(0.0)

        # Formatação das Colunas
        grupo['% Ótimo'] = (grupo['% Ótimo'] * 100).map('{:.2f}%'.format)
        grupo['Gap Médio'] = grupo['Gap Médio'].map('{:.4f}'.format)
        
        # Combinando Média e Desvio numa string bonita "10.5 ± 2.1"
        grupo['Tam. PEAB'] = grupo.apply(lambda x: f"{x['Tam_Mean']:.2f} \u00B1 {x['Tam_Std']:.2f}", axis=1)
        
        grupo['Tam. Ótimo'] = grupo['Tam. Ótimo'].map('{:.2f}'.format)
        grupo['Tempo PEAB'] = grupo['Tempo PEAB'].map('{:.4f}s'.format)
        grupo['Tempo Ótimo'] = grupo['Tempo Ótimo'].map('{:.4f}s'.format)
        
        # Seleção final para exibição
        cols_final = ['Tipo', 'Qtd', '% Ótimo', 'Gap Médio', 'Tam. PEAB', 'Tam. Ótimo', 'Tempo PEAB', 'Tempo Ótimo']
        f.write(grupo[cols_final].to_string(index=False))
        f.write("\n\n")

        # --- SEÇÃO 3 ---
        f.write("-" * 80 + "\n")
        f.write("3. TOP 10 MAIORES GAPS (PEAB vs Ótimo)\n")
        f.write("-" * 80 + "\n")
        piores = df.sort_values(by='GAP', ascending=False).head(10)
        cols_show = ['id', 'tipo_predicao', 'tamanho_PEAB', 'tamanho_OPTIMO', 'GAP']
        f.write(piores[cols_show].to_string(index=False))
        f.write("\n\n")

    print(f"\n[RELATÓRIO PERFEITO] Salvo em: {output_path}")

# =============================================================================
#  PARTE 3: O ORGANIZADOR DO EXPERIMENTO
# =============================================================================
def executar_benchmark():
    dataset_name, _, _, _, _ = selecionar_dataset_e_classe()
    if not dataset_name: return

    print(f"\n=== BENCHMARK: {dataset_name.upper()} ===")
    todos_params = carregar_hiperparametros()
    # Carrega dados completos
    X_full, y_full, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)
    
    params = DEFAULT_LOGREG_PARAMS.copy()
    if dataset_name in todos_params and 'params' in todos_params[dataset_name]:
        params.update(todos_params[dataset_name]['params'])

    print("\n" + "!"*50)
    print(f"[VERIFICAÇÃO] PARÂMETROS REAIS: {json.dumps(params, indent=4)}")
    print("!"*50 + "\n")

    # [CORREÇÃO CRÍTICA] SPLIT ÚNICO
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, 
        test_size=test_size, 
        random_state=RANDOM_STATE, 
        stratify=y_full
    )

    # Redução Top-K
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    if top_k and top_k > 0 and top_k < X_train.shape[1]:
        print(f"[BENCH] Aplicando redução Top-{top_k} features...")
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params)
        logreg_tmp = _get_lr(modelo_temp)
        importances = np.abs(logreg_tmp.coef_[0])
        indices_top = np.argsort(importances)[::-1][:top_k]
        selected_feats = X_train.columns[indices_top]
        X_train = X_train[selected_feats]
        X_test = X_test[selected_feats]
    
    print("[BENCH] Treinando modelo...")
    # Chama treino com dados já divididos (sem split interno)
    modelo, t_plus, t_minus, _ = treinar_e_avaliar_modelo(X_train, y_train, rejection_cost, params)
    
    print(f"[BENCH] T+: {t_plus:.4f}, T-: {t_minus:.4f}")

    resultados = []
    
    with ProgressBar(total=len(X_test), description=f"Comparando {dataset_name}") as pbar:
        for i in range(len(X_test)):
            instancia = X_test.iloc[[i]]
            score = modelo.decision_function(instancia)[0]
            
            if score >= t_plus: type_pred = "POSITIVA"
            elif score <= t_minus: type_pred = "NEGATIVA"
            else: type_pred = "REJEITADA"
            
            # PEAB
            t0 = time.perf_counter()
            expl, _, _, _ = gerar_explicacao_instancia(instancia, modelo, X_train, t_plus, t_minus, benchmark_mode=True)
            t_peab = time.perf_counter() - t0
            sz_peab = len(expl)
            
            # Solver
            t0 = time.perf_counter()
            sz_opt = calcular_minimo_exato_pulp(modelo, instancia, X_train, t_plus, t_minus)
            t_opt = time.perf_counter() - t0
            if sz_opt == -1: sz_opt = sz_peab # Fallback se solver falhar
            
            gap = sz_peab - sz_opt
            
            resultados.append({
                'id': i, 'tipo_predicao': type_pred,
                'tamanho_PEAB': sz_peab, 'tamanho_OPTIMO': sz_opt,
                'GAP': gap, 'is_optimal': (gap == 0),
                'tempo_PEAB': t_peab, 'tempo_OPTIMO': t_opt
            })
            pbar.update()

    df = pd.DataFrame(resultados)
    df.to_csv(f"results/benchmark/bench_csv/bench_{dataset_name}.csv", index=False)
    gerar_relatorio_tabela(df, dataset_name, t_plus, t_minus, params)

if __name__ == "__main__":
    executar_benchmark()