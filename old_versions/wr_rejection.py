import os
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

# --- Importações do seu rejeita_6.py ---
# Certifique-se que rejeita_6.py está acessível (mesmo diretório ou PYTHONPATH)
try:
    from rejeita_6 import (
        carregar_dataset,
        calcular_thresholds,
        carregar_hiperparametros,
        DEFAULT_LOGREG_PARAMS, # Usado como base para params do modelo
        RANDOM_STATE,          # Para consistência nos splits
        TEST_SIZE,             # Para consistência nos splits
        DIRETORIO_VISUALIZACOES # Para salvar plots de otimização
    )
except ImportError as e:
    print(f"Erro ao importar de rejeita_6.py: {e}")
    print("Certifique-se que rejeita_6.py está no mesmo diretório ou no PYTHONPATH.")
    exit()

# --- Constantes para o Script de Otimização ---
WR_CANDIDATES = list(np.linspace(0.05, 0.60, num=12)) # Ex: 12 valores entre 0.05 e 0.60
# Adicionar 0.24 se não estiver presente, para comparar com o valor do artigo
if 0.24 not in WR_CANDIDATES:
    WR_CANDIDATES.append(0.24)
WR_CANDIDATES.sort()

OUTPUT_JSON_FILE = "wr_rejection.json" # Nome do arquivo de saída sugerido

# Configuração para processamento de datasets (especialmente para binarização)
# Adicione ou modifique conforme os datasets que você quer testar.
# 'class_0_original_idx' e 'class_1_original_idx' são os índices das classes NO DATASET ORIGINAL.
# Para datasets já binários (e que carregar_dataset já trata como 0/1),
# as chaves de índice de classe podem ser omitidas.
DATASET_CONFIGS = {
    'iris': {'class_0_original_idx': 0, 'class_1_original_idx': 1, 'description': 'Iris Setosa vs Versicolor'},
    'wine': {'class_0_original_idx': 0, 'class_1_original_idx': 1, 'description': 'Wine Class 0 vs Class 1'},
    'seeds': {'class_0_original_idx': 0, 'class_1_original_idx': 1, 'description': 'Seeds Kama vs Rosa'},
    'pima_indians_diabetes': {'description': 'Originalmente Binário'},
    'breast_cancer': {'description': 'Originalmente Binário'},
    'banknote': {'description': 'Originalmente Binário'},
    'heart_disease': {'description': 'Processado para Binário em carregar_dataset'},
    'wine_quality': {'description': 'Processado para Binário em carregar_dataset'},
    'haberman': {'description': 'Originalmente Binário (1 e 2 -> 0 e 1)'},
    'creditcard': {'description': 'Amostra Binária'}, # Já é amostrado e binário
    # Adicione mais datasets aqui conforme necessário
}

# --- Funções Auxiliares ---

def preparar_dados_binarios_para_otimizacao(
    nome_dataset: str,
    config: Dict[str, Any],
    X_original: pd.DataFrame,
    y_original: pd.Series,
    classes_originais_nomes: List[str]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]:
    """
    Prepara (filtra e binariza) os dados para o processo de otimização.
    """
    if y_original is None or X_original is None:
        print(f"Dados originais nulos para {nome_dataset}.")
        return None, None, None

    y_original_series = pd.Series(y_original) if not isinstance(y_original, pd.Series) else y_original

    # Caso 1: Dataset multi-classe que precisa de binarização explícita via config
    if 'class_0_original_idx' in config and 'class_1_original_idx' in config:
        idx0 = config['class_0_original_idx']
        idx1 = config['class_1_original_idx']

        if not (0 <= idx0 < len(classes_originais_nomes) and 0 <= idx1 < len(classes_originais_nomes) and idx0 != idx1):
            print(f"Erro: Índices de classe inválidos ({idx0}, {idx1}) para {nome_dataset} com {len(classes_originais_nomes)} classes.")
            return None, None, None

        nome_classe_0 = classes_originais_nomes[idx0]
        nome_classe_1 = classes_originais_nomes[idx1]

        mascara_c0 = (y_original_series == idx0)
        mascara_c1 = (y_original_series == idx1)
        mascara_combinada = mascara_c0 | mascara_c1

        if np.sum(mascara_combinada) == 0:
            print(f"Erro: Nenhuma amostra encontrada para as classes {idx0} ou {idx1} no dataset {nome_dataset}.")
            return None, None, None
            
        X_filtrado = X_original[mascara_combinada].copy()
        y_filtrado = y_original_series[mascara_combinada].copy()

        y_binario_np = np.where(y_filtrado == idx0, 0, 1)
        y_binario = pd.Series(y_binario_np, index=X_filtrado.index, name='target_binario')
        nomes_classes_binarias_finais = [nome_classe_0, nome_classe_1]
        
        print(f"Dataset '{nome_dataset}' ({config.get('description', '')}) binarizado: '{nome_classe_0}' (0) vs '{nome_classe_1}' (1). Amostras: {len(X_filtrado)}")
        return X_filtrado, y_binario, nomes_classes_binarias_finais
    
    # Caso 2: Dataset já é binário (espera-se que y seja 0 e 1)
    elif len(np.unique(y_original_series)) == 2:
        unique_y_values = sorted(np.unique(y_original_series))
        # Se já for 0 e 1, ótimo.
        if unique_y_values[0] == 0 and unique_y_values[1] == 1:
            print(f"Dataset '{nome_dataset}' ({config.get('description', '')}) já é binário (0,1). Amostras: {len(X_original)}")
            return X_original.copy(), y_original_series.copy(), list(classes_originais_nomes)
        else:
            # Tenta mapear para 0 e 1 se for binário mas com outros rótulos (ex: 1 e 2)
            print(f"Dataset '{nome_dataset}' ({config.get('description', '')}) é binário com rótulos {unique_y_values}. Mapeando para 0 e 1.")
            class_map = {unique_y_values[0]: 0, unique_y_values[1]: 1}
            y_binario = y_original_series.map(class_map).astype(int)
             # Tenta preservar os nomes das classes se disponíveis e com 2 elementos
            nomes_binarios = list(classes_originais_nomes) if len(classes_originais_nomes) == 2 else [f"Classe_{unique_y_values[0]}", f"Classe_{unique_y_values[1]}"]
            return X_original.copy(), y_binario, nomes_binarios
    
    else:
        print(f"Dataset '{nome_dataset}' não parece ser binário ou configurável para binário. Classes únicas: {np.unique(y_original_series)}. Nomes: {classes_originais_nomes}")
        return None, None, None


def encontrar_melhor_wr_rejection_cost(
    modelo: LogisticRegression,
    X_treino_df: pd.DataFrame,
    y_treino_series: pd.Series,
    X_val_df: pd.DataFrame,
    y_val_series: pd.Series,
    wr_cost_candidates: List[float],
    nome_dataset: str # Para salvar o plot com nome específico
) -> Tuple[float, float, Dict[float, float]]:
    """
    Encontra o melhor WR_REJECTION_COST dentre uma lista de candidatos.
    """
    melhor_wr_cost = -1.0
    menor_custo_total_val = float('inf')
    resultados_por_wr = {} # {wr_cand: {'custo_total': X, 'erro_aceitas': Y, 'taxa_rej': Z, 't+': A, 't-': B}}


    print(f"\n--- Otimizando WR_REJECTION_COST para o dataset: {nome_dataset} ---")
    print(f"Testando {len(wr_cost_candidates)} candidatos para WR: { [round(c,3) for c in wr_cost_candidates] }")

    for wr_cand in wr_cost_candidates:
        # 1. Calcular thresholds
        t_plus_cand, t_minus_cand = calcular_thresholds(
            modelo, X_treino_df, y_treino_series, custo_rejeicao_wr=wr_cand
        )

        # 2. Avaliar no conjunto de validação
        if X_val_df.empty:
            print(f"    AVISO: Conjunto de validação vazio para wr_cand={wr_cand:.4f}. Pulando avaliação.")
            resultados_por_wr[wr_cand] = {'custo_total': float('inf'), 'erro_aceitas': 0, 'taxa_rej': 0, 't+': t_plus_cand, 't-': t_minus_cand}
            continue

        decision_scores_val = modelo.decision_function(X_val_df)
        
        rejeitadas_mask_val = (decision_scores_val <= t_plus_cand) & (decision_scores_val >= t_minus_cand)
        aceitas_mask_val = ~rejeitadas_mask_val
        
        taxa_rejeicao_val = np.mean(rejeitadas_mask_val)
        taxa_erro_aceitas_val = 0.0
        num_aceitas = np.sum(aceitas_mask_val)

        if num_aceitas > 0:
            predicoes_aceitas_val = modelo.predict(X_val_df[aceitas_mask_val])
            y_verdadeiro_aceitas_val = y_val_series[aceitas_mask_val]
            if not y_verdadeiro_aceitas_val.empty:
                 taxa_erro_aceitas_val = np.mean(predicoes_aceitas_val != y_verdadeiro_aceitas_val.values)
        
        custo_total_val_atual = taxa_erro_aceitas_val + wr_cand * taxa_rejeicao_val
        resultados_por_wr[wr_cand] = {
            'custo_total': custo_total_val_atual,
            'erro_aceitas': taxa_erro_aceitas_val,
            'taxa_rej': taxa_rejeicao_val,
            't+': t_plus_cand,
            't-': t_minus_cand
        }
        
        print(f"  WR={wr_cand:.4f} | t+:{t_plus_cand:7.4f}, t-:{t_minus_cand:7.4f} | ErrAceitas:{taxa_erro_aceitas_val:6.4f}, TaxaRej:{taxa_rejeicao_val:6.4f} | CustoTotal:{custo_total_val_atual:7.4f}")

        if custo_total_val_atual < menor_custo_total_val:
            menor_custo_total_val = custo_total_val_atual
            melhor_wr_cost = wr_cand
    
    print("-" * 70)
    if melhor_wr_cost != -1.0:
        print(f"Melhor WR_REJECTION_COST para '{nome_dataset}': {melhor_wr_cost:.4f} (Custo Total Val: {menor_custo_total_val:.4f})")
    else:
        print(f"Não foi possível determinar um melhor WR_REJECTION_COST para '{nome_dataset}'.")
    print("-" * 70)

    # Plotar e salvar a curva de custo
    try:
        if resultados_por_wr and DIRETORIO_VISUALIZACOES:
            costs = {wr: data['custo_total'] for wr, data in resultados_por_wr.items() if data['custo_total'] != float('inf')}
            if costs: # Apenas plota se houver dados válidos
                plt.figure(figsize=(12, 7))
                plt.plot(list(costs.keys()), list(costs.values()), marker='o', linestyle='-')
                plt.xlabel("WR_REJECTION_COST Candidato")
                plt.ylabel("Custo Total no Conjunto de Validação")
                plt.title(f"Otimização de WR_REJECTION_COST para {nome_dataset}")
                plt.grid(True)
                if melhor_wr_cost != -1.0:
                    plt.scatter([melhor_wr_cost], [menor_custo_total_val], color='red', s=100, zorder=5, label=f'Melhor ({melhor_wr_cost:.2f})')
                if 0.24 in costs: # Valor do artigo
                    plt.scatter([0.24], [costs[0.24]], color='green', s=100, zorder=5, label='Ref. (0.24)')
                plt.legend()
                
                os.makedirs(DIRETORIO_VISUALIZACOES, exist_ok=True)
                plot_filename = os.path.join(DIRETORIO_VISUALIZACOES, f"otim_wr_cost_{nome_dataset}.png")
                plt.savefig(plot_filename)
                print(f"Plot da otimização salvo em: {plot_filename}")
                plt.close() # Fecha a figura para liberar memória
    except Exception as e:
        print(f"Erro ao gerar/salvar plot para {nome_dataset}: {e}")
        
    return melhor_wr_cost, menor_custo_total_val, resultados_por_wr


# --- Script Principal ---
if __name__ == "__main__":
    resultados_finais_otimizacao = {}
    datasets_para_processar = list(DATASET_CONFIGS.keys())

    print(f"Iniciando otimização de WR_REJECTION_COST para {len(datasets_para_processar)} datasets.")
    print(f"Candidatos WR_REJECTION_COST: {[round(c,3) for c in WR_CANDIDATES]}")


    for nome_ds in datasets_para_processar:
        print(f"\n================ Processando Dataset: {nome_ds} ================")
        dataset_config = DATASET_CONFIGS[nome_ds]

        # 1. Carregar dados originais
        X_orig, y_orig_series, classes_orig_nomes = carregar_dataset(nome_ds)
        if X_orig is None or y_orig_series is None:
            print(f"Falha ao carregar {nome_ds}. Pulando.")
            resultados_finais_otimizacao[nome_ds] = {'melhor_wr': None, 'custo': float('inf'), 'error': 'Falha ao carregar'}
            continue

        # 2. Preparar dados para binário (filtrar/mapear classes)
        X_data, y_data, nomes_classes_bin = preparar_dados_binarios_para_otimizacao(
            nome_ds, dataset_config, X_orig, y_orig_series, classes_orig_nomes
        )
        if X_data is None or y_data is None or X_data.empty:
            print(f"Falha ao preparar dados binários para {nome_ds}. Pulando.")
            resultados_finais_otimizacao[nome_ds] = {'melhor_wr': None, 'custo': float('inf'), 'error': 'Falha ao preparar dados binários'}
            continue
        
        # 3. Dividir em treino e validação (usaremos a nomenclatura de teste aqui para consistência com rejeita_6)
        # Usando TEST_SIZE e RANDOM_STATE de rejeita_6 para consistência, se desejado.
        X_treino, X_val, y_treino, y_val = train_test_split(
            X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_data
        )
        X_treino_df = pd.DataFrame(X_treino, columns=X_data.columns)
        X_val_df = pd.DataFrame(X_val, columns=X_data.columns)
        y_treino_series = pd.Series(y_treino, name=y_data.name)
        y_val_series = pd.Series(y_val, name=y_data.name)

        if X_treino_df.empty or X_val_df.empty:
            print(f"Conjunto de treino ou validação vazio para {nome_ds} após split. Pulando.")
            resultados_finais_otimizacao[nome_ds] = {'melhor_wr': None, 'custo': float('inf'), 'error': 'Split resultou em dataframes vazios'}
            continue

        # 4. Carregar hiperparâmetros do modelo e treinar
        hiperparams_dataset = carregar_hiperparametros(nome_ds) # de hiperparam.json
        params_modelo = DEFAULT_LOGREG_PARAMS.copy()
        params_modelo.update(hiperparams_dataset)
        
        modelo = LogisticRegression(**params_modelo)
        try:
            modelo.fit(X_treino_df, y_treino_series)
        except Exception as e:
            print(f"Erro ao treinar modelo para {nome_ds}: {e}. Pulando.")
            resultados_finais_otimizacao[nome_ds] = {'melhor_wr': None, 'custo': float('inf'), 'error': f'Erro no treino: {e}'}
            continue

        # 5. Encontrar melhor WR_REJECTION_COST
        melhor_wr, menor_custo, _ = encontrar_melhor_wr_rejection_cost(
            modelo, X_treino_df, y_treino_series, X_val_df, y_val_series, WR_CANDIDATES, nome_ds
        )
        
        if melhor_wr != -1.0:
            resultados_finais_otimizacao[nome_ds] = {'melhor_wr': melhor_wr, 'custo_val_minimo': menor_custo}
        else:
             resultados_finais_otimizacao[nome_ds] = {'melhor_wr': None, 'custo_val_minimo': float('inf'), 'nota': 'Nenhum WR ótimo encontrado'}


    # 6. Salvar resultados da otimização em JSON
    print(f"\nSalvando resultados da otimização em: {OUTPUT_JSON_FILE}")
    try:
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(resultados_finais_otimizacao, f, indent=4)
        print("Resultados salvos com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar resultados em JSON: {e}")

    print("\nOtimização de WR_REJECTION_COST concluída para todos os datasets.")