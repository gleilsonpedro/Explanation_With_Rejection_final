# Imports de bibliotecas padrão
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
import warnings
import time
import os
import sys
import matplotlib.pyplot as plt

# Garante que o Python consiga encontrar pastas no diretório atual
sys.path.append(os.getcwd())

# --- Módulos do projeto ---
try:
    # Importa o Anchor da biblioteca ALIBI
    from alibi.explainers import AnchorTabular
    from data.datasets import selecionar_dataset_e_classe
    # Importa as funções de rejeição do nosso arquivo auxiliar
    from utils.rejection_logic import executar_logica_rejeicao

    from utils.results_handler import update_method_results
except ImportError as e:
    print(f"ERRO CRÍTICO AO IMPORTAR MÓDULO: {e}")
    print("Verifique se você instalou 'alibi[tabular]' (pip install alibi[tabular]) e se todas as pastas e arquivos do projeto estão corretos.")
    exit()

#==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
#==============================================================================
RANDOM_STATE = 107

### MUDANÇA 1: Alinhando o threshold de precisão com o de Mateus ###
# O valor foi alterado de 0.95 para 1.0 para exigir 100% de confiança.
ANCHOR_PRECISION_THRESHOLD = 1.0

# Dicionário de configuração para todos os datasets
DATASET_CONFIG = {
    "iris":             {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine":             {'test_size': 0.3, 'rejection_cost': 0.24},
    "pima_indians_diabetes": {'test_size': 0.3, 'rejection_cost': 0.24},
    "sonar":            {'test_size': 0.3, 'rejection_cost': 0.24},
    "vertebral_column": {'test_size': 0.3, 'rejection_cost': 0.24},
    "breast_cancer":    {'test_size': 0.3, 'rejection_cost': 0.24},
    "spambase":         {'test_size': 0.2, 'rejection_cost': 0.24},
    "banknote_auth":    {'test_size': 0.2, 'rejection_cost': 0.24},
    "heart_disease":    {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine_quality":     {'test_size': 0.2, 'rejection_cost': 0.24},
    "creditcard":       {'subsample_size': 0.1, 'test_size': 0.3, 'rejection_cost': 0.24}
}

warnings.filterwarnings("ignore", category=FutureWarning)

def formatar_log_anchor(metricas):
    """
    Formata o log final para o método Anchor, espelhando o formato do PEAB.
    """
    stats_pos, stats_neg, stats_rej = "N/A", "N/A", "N/A"
    if metricas.get('stats_explicacao_positiva', {}).get('instancias', 0) > 0:
        stats = metricas['stats_explicacao_positiva']
        stats_pos = f"Tamanho Explicação (Média ± Desv. Padrão): {stats['media']:.2f} ± {stats['std_dev']:.2f}  (Min: {stats['min']}, Max: {stats['max']})"
    if metricas.get('stats_explicacao_negativa', {}).get('instancias', 0) > 0:
        stats = metricas['stats_explicacao_negativa']
        stats_neg = f"Tamanho Explicação (Média ± Desv. Padrão): {stats['media']:.2f} ± {stats['std_dev']:.2f}  (Min: {stats['min']}, Max: {stats['max']})"
    if metricas.get('stats_explicacao_rejeitada', {}).get('instancias', 0) > 0:
        stats = metricas['stats_explicacao_rejeitada']
        stats_rej = f"Tamanho Explicação (Média ± Desv. Padrão): {stats['media']:.2f} ± {stats['std_dev']:.2f}  (Min: {stats['min']}, Max: {stats['max']})"

    features_frequentes_str = "\n".join([f"    - {feat}: {count} vezes" for feat, count in metricas.get('features_frequentes', [])])

    log = f"""[CONFIGURAÇÕES GERAIS E DO MODELO]
  - Dataset: {metricas.get('dataset_name', 'N/A')}
  - Classificador Usado: LogisticRegression (explicado por Anchor)
  - Tamanho do Conjunto de Teste: {metricas.get('test_size_atual', 0):.0%}
  - Total de Instâncias de Teste: {metricas.get('total_instancias_teste', 'N/A')}
  - Número de Features do Modelo: {metricas.get('num_features', 'N/A')}
  - Acurácia do Modelo (teste, sem rejeição): {metricas.get('acuracia_sem_rejeicao', 0):.2f}%
  - Thresholds de Rejeição (Score): t+ = {metricas.get('t_upper', 0):.4f}, t- = {metricas.get('t_lower', 0):.4f}
  - Custo de Rejeição usado para Thresholds: {metricas.get('rejection_cost', 'N/A'):.2f}
  - Tempo Total de Geração das Explicações: {metricas.get('tempo_total', 0):.4f} segundos
  - Tempo Médio por Instância (Geral): {metricas.get('tempo_medio_instancia', 0):.4f} segundos

[Métricas de Desempenho do Modelo]
  - Taxa de Rejeição no Teste: {metricas.get('taxa_rejeicao_teste', 0):.2f}% ({metricas.get('num_rejeitadas_teste', 0)} de {metricas.get('total_instancias_teste', 0)} instâncias)
  - Acurácia com Opção de Rejeição (nas {metricas.get('num_aceitas_teste', 0)} instâncias aceitas): {metricas.get('acuracia_com_rejeicao', 0):.2f}%

[Estatísticas do Tamanho das Explicações (Nº de Regras)]
  - Classe Positiva (Aceitas): ({metricas.get('stats_explicacao_positiva', {}).get('instancias', 0)} instâncias)
    - {stats_pos}
  - Classe Negativa (Aceitas): ({metricas.get('stats_explicacao_negativa', {}).get('instancias', 0)} instâncias)
    - {stats_neg}
  - Classe REJEITADA: ({metricas.get('stats_explicacao_rejeitada', {}).get('instancias', 0)} instâncias)
    - {stats_rej}

[Análise de Importância de Features (Frequência nas Âncoras)]
  - Top 10 Features Mais Frequentes:
{features_frequentes_str}

[Custo Computacional por Classe]
  - Tempo Médio (Positivas Aceitas): {metricas.get('tempo_medio_positivas', 0):.4f} segundos
  - Tempo Médio (Negativas Aceitas): {metricas.get('tempo_medio_negativas', 0):.4f} segundos
  - Tempo Médio (Rejeitadas): {metricas.get('tempo_medio_rejeitadas', 0):.4f} segundos
"""
    return log

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == '__main__':
    print("Chamando menu de seleção de dataset...")
    nome_dataset_original, nome_classe_positiva, dados_completos, alvos_completos, nomes_classes_binarias = selecionar_dataset_e_classe()

    if dados_completos is not None:
        
        config_atual = DATASET_CONFIG.get(nome_dataset_original, {})
        test_size_atual = config_atual.get('test_size', 0.3)
        rejection_cost_atual = config_atual.get('rejection_cost', 0.25)

        if 'subsample_size' in config_atual:
            print(f"Aplicando amostragem estratificada de {config_atual['subsample_size']*100}%...")
            dados_processar, _, alvos_processar, _ = train_test_split(
                dados_completos, alvos_completos, 
                train_size=config_atual['subsample_size'], 
                random_state=RANDOM_STATE, 
                stratify=alvos_completos
            )
        else:
            dados_processar, alvos_processar = dados_completos, alvos_completos
            
        nome_relatorio = f"{nome_dataset_original}_{nome_classe_positiva}_vs_rest"
        print(f"\n--- Iniciando análise com ANCHOR (COM REJEIÇÃO) para: {nome_relatorio} ---")
        
        metricas = {'dataset_name': nome_relatorio, 'test_size_atual': test_size_atual}
        
        if isinstance(dados_processar, pd.DataFrame):
            nomes_features = dados_completos.columns.tolist()
            dados_numpy = dados_processar.values
        else:
            dados_numpy = np.array(dados_processar)
            nomes_features = [f'feature_{i}' for i in range(dados_numpy.shape[1])]
        alvos_numpy = np.array(alvos_processar)

        X_train, X_test, y_train, y_test = train_test_split(
            dados_numpy, alvos_numpy, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=alvos_numpy
        )
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        modelo = LogisticRegression(random_state=RANDOM_STATE)
        modelo.fit(X_train_scaled, y_train)

        metricas['total_instancias_teste'] = len(y_test)
        metricas['num_features'] = dados_numpy.shape[1]
        y_pred_sem_rejeicao = modelo.predict(X_test_scaled)
        metricas['acuracia_sem_rejeicao'] = accuracy_score(y_test, y_pred_sem_rejeicao) * 100

        t_inferior, t_superior, y_pred_com_rejeicao, indices_aceitos, indices_rejeitados = executar_logica_rejeicao(
            modelo, X_train_scaled, y_train, X_test_scaled, rejection_cost_atual
        )
        metricas['t_lower'], metricas['t_upper'] = t_inferior, t_superior
        metricas['rejection_cost'] = rejection_cost_atual
        
        y_test_aceitos = y_test[indices_aceitos]
        y_pred_aceitos = np.array(y_pred_com_rejeicao)[indices_aceitos]
        metricas['num_rejeitadas_teste'] = len(indices_rejeitados)
        metricas['num_aceitas_teste'] = len(indices_aceitos)
        metricas['taxa_rejeicao_teste'] = (metricas['num_rejeitadas_teste'] / metricas['total_instancias_teste']) * 100
        metricas['acuracia_com_rejeicao'] = accuracy_score(y_test_aceitos, y_pred_aceitos) * 100 if len(y_test_aceitos) > 0 else 100

        # --- GERAÇÃO DAS EXPLICAÇÕES ANCHOR ---

        # [CORREÇÃO] A função de predição para o Alibi DEVE ser a de probabilidade.
        # Isso permite que o Anchor calcule a precisão de suas regras corretamente.
        predict_fn = modelo.predict_proba

        # [CORREÇÃO] Inicializa o explainer com a função de probabilidade correta.
        explainer = AnchorTabular(predict_fn, feature_names=nomes_features)
        # --------------------------------------------------------------------
                
        # O Alibi precisa 'aprender' a distribuição dos dados de treino
        explainer.fit(X_train_scaled, disc_perc=(25, 50, 75))
        
        print(f"Gerando explicações com Anchor (Alibi) para as {len(X_test_scaled)} instâncias de teste...")
        tempos_total, tempos_pos, tempos_neg, tempos_rej = [], [], [], []
        explicacoes = {}
        
        y_pred_final = np.array(y_pred_com_rejeicao)

        for i in range(len(X_test_scaled)):
            start_time = time.time()
            # A chamada da explicação continua a mesma, mas agora usa a nova lógica interna
            explanation = explainer.explain(X_test_scaled[i], threshold=ANCHOR_PRECISION_THRESHOLD)
            runtime = time.time() - start_time
            tempos_total.append(runtime)
            explicacoes[i] = explanation.anchor

            # Atribui o tempo à classe correta (positiva, negativa ou rejeitada)
            if i in indices_rejeitados:
                tempos_rej.append(runtime)
            elif y_pred_final[i] == 1: # Classe positiva
                tempos_pos.append(runtime)
            else: # Classe negativa
                tempos_neg.append(runtime)

        # --- O RESTO DO CÓDIGO DE CÁLCULO DE MÉTRICAS E PLOT ---
        metricas['tempo_total'] = sum(tempos_total)
        metricas['tempo_medio_instancia'] = np.mean(tempos_total) if tempos_total else 0
        metricas['tempo_medio_positivas'] = np.mean(tempos_pos) if tempos_pos else 0
        metricas['tempo_medio_negativas'] = np.mean(tempos_neg) if tempos_neg else 0
        metricas['tempo_medio_rejeitadas'] = np.mean(tempos_rej) if tempos_rej else 0
        
        feature_counts = Counter()
        exp_lengths_pos, exp_lengths_neg, exp_lengths_rej = [], [], []
        for i in range(len(y_pred_final)):
            exp_len = len(explicacoes.get(i, []))
            if i in indices_rejeitados: exp_lengths_rej.append(exp_len)
            elif y_pred_final[i] == 1: exp_lengths_pos.append(exp_len)
            else: exp_lengths_neg.append(exp_len)
            for regra in explicacoes.get(i, []):
                for feature_name in nomes_features:
                    if feature_name in regra: feature_counts[feature_name] += 1
        
        stats_pos = {
            'instancias': len(exp_lengths_pos),
            'min': min(exp_lengths_pos) if exp_lengths_pos else 0,
            'media': np.mean(exp_lengths_pos) if exp_lengths_pos else 0,
            'max': max(exp_lengths_pos) if exp_lengths_pos else 0,
            'std_dev': np.std(exp_lengths_pos) if exp_lengths_pos else 0
        }
        
        stats_neg = {
            'instancias': len(exp_lengths_neg),
            'min': min(exp_lengths_neg) if exp_lengths_neg else 0,
            'media': np.mean(exp_lengths_neg) if exp_lengths_neg else 0,
            'max': max(exp_lengths_neg) if exp_lengths_neg else 0,
            'std_dev': np.std(exp_lengths_neg) if exp_lengths_neg else 0
        }
        
        stats_rej = {
            'instancias': len(exp_lengths_rej),
            'min': min(exp_lengths_rej) if exp_lengths_rej else 0,
            'media': np.mean(exp_lengths_rej) if exp_lengths_rej else 0,
            'max': max(exp_lengths_rej) if exp_lengths_rej else 0,
            'std_dev': np.std(exp_lengths_rej) if exp_lengths_rej else 0
        }
        
        metricas['stats_explicacao_positiva'] = stats_pos
        metricas['stats_explicacao_negativa'] = stats_neg
        metricas['stats_explicacao_rejeitada'] = stats_rej
        metricas['features_frequentes'] = feature_counts.most_common(10)
        
        log_final = formatar_log_anchor(metricas)
        base_dir = os.path.join('results','report', 'anchor')
        os.makedirs(base_dir, exist_ok=True)
        caminho_arquivo = os.path.join(base_dir, f'anchor_{nome_relatorio}.txt')
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write(log_final)
        print(f"\nANÁLISE CONCLUÍDA. Relatório salvo em: {caminho_arquivo}")
        
        dataset_key = nome_dataset_original
        
        results_data = {
            "config": {
                "dataset_name": nome_dataset_original,
                "test_size": test_size_atual,
                "random_state": RANDOM_STATE,
                "rejection_cost": rejection_cost_atual
            },
            "thresholds": {
                "t_plus": float(t_superior),
                "t_minus": float(t_inferior)
            },
            "performance": {
                "accuracy_without_rejection": float(metricas['acuracia_sem_rejeicao']),
                "accuracy_with_rejection": float(metricas['acuracia_com_rejeicao']),
                "rejection_rate": float(metricas['taxa_rejeicao_teste'])
            },
            "explanation_stats": {
                "positive": stats_pos,
                "negative": stats_neg,
                "rejected": stats_rej
            },
            "computation_time": {
                "total": float(metricas['tempo_total']),
                "mean_per_instance": float(metricas['tempo_medio_instancia']),
                "positive": float(metricas['tempo_medio_positivas']),
                "negative": float(metricas['tempo_medio_negativas']),
                "rejected": float(metricas['tempo_medio_rejeitadas'])
            },
            "top_features": [
                {"feature": feat, "count": count} 
                for feat, count in metricas['features_frequentes']
            ]
        }
        
        update_method_results("anchor", dataset_key, results_data)

        # Salvar plot também dentro de results/anchor/plots
        plot_dir = os.path.join(base_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(tempos_total)), tempos_total, alpha=0.6)
        plt.title(f'Tempo de Geração de Explicação (Anchor) por Instância\nDataset: {nome_relatorio}')
        plt.xlabel('Índice da Instância no Conjunto de Teste')
        plt.ylabel('Tempo de Execução (segundos)')
        plt.grid(True)
        caminho_plot = os.path.join(plot_dir, f'plot_tempo_anchor_{nome_relatorio}.png')
        plt.savefig(caminho_plot)
        print(f"Gráfico de dispersão salvo em: {caminho_plot}")

    else:
        print("Nenhum dataset selecionado. Encerrando o programa.")