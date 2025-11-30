import time
import pulp
import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- IMPORTANDO AS FUNÇÕES DO SEU SCRIPT ORIGINAL ---
from peab import (
    carregar_hiperparametros,
    configurar_experimento,
    treinar_e_avaliar_modelo,
    aplicar_selecao_top_k_features,
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
    X_train_scaled = scaler.transform(X_train)
    min_scaled = X_train_scaled.min(axis=0)
    max_scaled = X_train_scaled.max(axis=0)
    
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
#  PARTE 2: GERADOR DE RELATÓRIO (TABELAS)
# =============================================================================
def gerar_relatorio_tabela(df: pd.DataFrame, dataset_name: str, t_plus: float, t_minus: float):
    """Gera um arquivo de texto com tabelas formatadas comparando os métodos."""
    
    output_path = f"results/benchmark/relatorio_bench_{dataset_name}.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO DE BENCHMARK CIENTÍFICO: {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Parâmetros do Experimento:\n")
        f.write(f" - Instâncias de Teste: {len(df)}\n")
        f.write(f" - Thresholds: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n\n")

        # --- TABELA 1: RESUMO GERAL ---
        f.write("-" * 80 + "\n")
        f.write("1. RESUMO GERAL DE DESEMPENHO (PEAB vs OTIMIZAÇÃO)\n")
        f.write("-" * 80 + "\n")
        
        geral = pd.DataFrame({
            'Métrica': [
                'Taxa de Otimalidade (Gap=0)', 
                'Gap Médio (Features Excedentes)', 
                'Tempo Médio PEAB (s)', 
                'Tempo Médio OTIMIZAÇÃO (s)', 
                'Speedup (x vezes mais rápido)'
            ],
            'Valor': [
                f"{df['is_optimal'].mean()*100:.2f}%",
                f"{df['GAP'].mean():.4f}",
                f"{df['tempo_PEAB'].mean():.6f}",
                f"{df['tempo_OPTIMO'].mean():.6f}",
                f"{df['tempo_OPTIMO'].mean() / df['tempo_PEAB'].mean():.2f}x"
            ]
        })
        f.write(geral.to_string(index=False))
        f.write("\n\n")

        # --- TABELA 2: ANÁLISE DETALHADA POR TIPO (POSITIVA / NEGATIVA / REJEITADA) ---
        f.write("-" * 80 + "\n")
        f.write("2. DETALHAMENTO POR CLASSE DE PREDIÇÃO\n")
        f.write("-" * 80 + "\n")
        f.write("Esta tabela permite identificar se o gargalo está nas Rejeitadas.\n\n")

        # Agrupamento
        grupo = df.groupby('tipo_predicao').agg({
            'id': 'count',
            'is_optimal': 'mean',
            'GAP': 'mean',
            'tamanho_PEAB': 'mean',
            'tamanho_OPTIMO': 'mean',
            'tempo_PEAB': 'mean'
        }).reset_index()

        # Renomeando colunas para ficar bonito
        grupo.columns = ['Tipo', 'Qtd', '% Ótimo', 'Gap Médio', 'Tam. PEAB', 'Tam. Ótimo', 'Tempo PEAB']
        
        # Formatação
        grupo['% Ótimo'] = (grupo['% Ótimo'] * 100).map('{:.2f}%'.format)
        grupo['Gap Médio'] = grupo['Gap Médio'].map('{:.4f}'.format)
        grupo['Tam. PEAB'] = grupo['Tam. PEAB'].map('{:.2f}'.format)
        grupo['Tam. Ótimo'] = grupo['Tam. Ótimo'].map('{:.2f}'.format)
        grupo['Tempo PEAB'] = grupo['Tempo PEAB'].map('{:.5f}s'.format)
        
        f.write(grupo.to_string(index=False))
        f.write("\n\n")

        # --- TABELA 3: OS 10 PIORES CASOS ---
        f.write("-" * 80 + "\n")
        f.write("3. TOP 10 MAIORES ERROS (GAPS)\n")
        f.write("-" * 80 + "\n")
        f.write("Instâncias onde o PEAB ficou mais longe da solução matemática ideal.\n\n")
        
        piores = df.sort_values(by='GAP', ascending=False).head(10)
        cols_show = ['id', 'tipo_predicao', 'tamanho_PEAB', 'tamanho_OPTIMO', 'GAP']
        f.write(piores[cols_show].to_string(index=False))
        f.write("\n\n")

    print(f"\n[RELATÓRIO] Relatório detalhado salvo em: {output_path}")
    print(f"            (Abra este arquivo para ver as tabelas formatadas)")

# =============================================================================
#  PARTE 3: O ORGANIZADOR DO EXPERIMENTO
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
            
            # --- [ATUALIZADO] Classificação Granular: POS / NEG / REJ ---
            score = modelo.decision_function(instancia)[0]
            
            if score >= t_plus:
                tipo_predicao = "POSITIVA"
            elif score <= t_minus:
                tipo_predicao = "NEGATIVA"
            else:
                tipo_predicao = "REJEITADA"
            
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
                'tipo_predicao': tipo_predicao, 
                'tamanho_PEAB': len(expl_peab),
                'tamanho_OPTIMO': tamanho_optimo,
                'GAP': gap,
                'tempo_PEAB': time_peab,
                'tempo_OPTIMO': time_opt,
                'is_optimal': (gap == 0)
            })
            pbar.update()

    df = pd.DataFrame(resultados)
    
    # 1. Salvar CSV (Dados Brutos)
    csv_filename = f"results/benchmark/bench_{dataset_name}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    print(f"\n[ARQUIVO] CSV bruto salvo em: {csv_filename}")
    
    # 2. Gerar Relatório de Tabela (Dados Formatados)
    gerar_relatorio_tabela(df, dataset_name, t_plus, t_minus)

if __name__ == "__main__":
    executar_benchmark()