# Imports de bibliotecas padrão
import numpy as np
import pandas as pd
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
    from utils.shared_training import get_shared_pipeline
    from utils.results_handler import update_method_results
except ImportError as e:
    print(f"ERRO CRÍTICO AO IMPORTAR MÓDULO: {e}")
    print("Verifique se você instalou 'alibi[tabular]' (pip install alibi[tabular]) e se todas as pastas e arquivos do projeto estão corretos.")
    exit()

#==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
#==============================================================================
RANDOM_STATE = 107

# Exigir alta precisão nas âncoras
ANCHOR_PRECISION_THRESHOLD = 1.0  # valor padrão; pode ser ajustado para datasets de alta dimensionalidade

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
    nome_dataset_original, nome_classe_positiva, _, _, nomes_classes_binarias = selecionar_dataset_e_classe()

    if nome_dataset_original is not None:

        # Usa treino e thresholds exatamente como no PEAB
        pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(nome_dataset_original)

        # Nome do relatório: usa as classes efetivas do pipeline compartilhado
        nome_relatorio = f"{nome_dataset_original}_{meta['nomes_classes'][0]}_vs_{meta['nomes_classes'][1]}"
        print(f"\n--- Iniciando análise com ANCHOR (COM REJEIÇÃO) para: {nome_relatorio} ---")

        metricas = {'dataset_name': nome_relatorio, 'test_size_atual': meta['test_size']}
        nomes_features = meta['feature_names']
        # Ajuste de precisão para MNIST (acelera geração): usar 0.95
        local_anchor_threshold = 0.95 if len(nomes_features) >= 500 else ANCHOR_PRECISION_THRESHOLD

        # Métricas do modelo sem rejeição (mesmo pipeline do PEAB)
        metricas['total_instancias_teste'] = len(y_test)
        metricas['num_features'] = len(nomes_features)
        y_pred_sem_rejeicao = pipeline.predict(X_test)
        metricas['acuracia_sem_rejeicao'] = accuracy_score(y_test, y_pred_sem_rejeicao) * 100

        # Thresholds e custo (idênticos ao PEAB)
        metricas['t_lower'], metricas['t_upper'] = float(t_minus), float(t_plus)
        metricas['rejection_cost'] = float(meta['rejection_cost'])

        # Aplicar rejeição com os mesmos thresholds
        decision_scores_test = pipeline.decision_function(X_test)
        y_pred_com_rejeicao = []
        indices_aceitos, indices_rejeitados = [], []
        for i, s in enumerate(decision_scores_test):
            if s >= t_plus:
                y_pred_com_rejeicao.append(1)
                indices_aceitos.append(i)
            elif s <= t_minus:
                y_pred_com_rejeicao.append(0)
                indices_aceitos.append(i)
            else:
                y_pred_com_rejeicao.append(-1)
                indices_rejeitados.append(i)

        y_pred_final = np.array(y_pred_com_rejeicao)
        y_test_aceitos = y_test.iloc[indices_aceitos] if isinstance(y_test, pd.Series) else y_test[indices_aceitos]
        y_pred_aceitos = y_pred_final[indices_aceitos]
        metricas['num_rejeitadas_teste'] = len(indices_rejeitados)
        metricas['num_aceitas_teste'] = len(indices_aceitos)
        metricas['taxa_rejeicao_teste'] = (metricas['num_rejeitadas_teste'] / metricas['total_instancias_teste']) * 100
        metricas['acuracia_com_rejeicao'] = accuracy_score(y_test_aceitos, y_pred_aceitos) * 100 if len(indices_aceitos) > 0 else 100

        # --- GERAÇÃO DAS EXPLICAÇÕES ANCHOR ---
        # Anchor deve usar a probabilidade do MESMO pipeline (igual PEAB)
        # Criar um wrapper para garantir que o pipeline receba DataFrame com nomes de features
        def predict_fn_wrapper(x: np.ndarray):
            # Alibi frequentemente passa arrays numpy; o pipeline (MinMaxScaler) foi ajustado
            # com nomes de features (DataFrame). Para evitar warning e garantir alinhamento,
            # convertemos para DataFrame quando necessário.
            if isinstance(x, np.ndarray):
                try:
                    df = pd.DataFrame(x, columns=nomes_features)
                except Exception:
                    # Se a forma não encaixar (e.g., única instância 1D), tentar reshape
                    arr = x.reshape(1, -1) if x.ndim == 1 else x
                    df = pd.DataFrame(arr, columns=nomes_features)
                return pipeline.predict_proba(df)
            else:
                return pipeline.predict_proba(x)

        explainer = AnchorTabular(predict_fn_wrapper, feature_names=nomes_features)
        # O explainer aprende a distribuição dos dados de treino (dados crus; pipeline faz o scaling)
        # Alibi espera um array numpy para discretização por percentis
        explainer.fit(X_train.values if hasattr(X_train, 'values') else X_train, disc_perc=(25, 50, 75))

        # Para datasets muito grandes/alto-dimensional (ex.: MNIST) limitamos explicações
        # Contudo, se o experimento já aplicou 'subsample_size' (via peap_2.DATASET_CONFIG),
        # NÃO aplicamos limite extra — queremos que PEAB/MinExp/Anchor façam a mesma análise.
        max_instances_to_explain = len(X_test)
        subsample = meta.get('subsample_size', None)
        if subsample is None and len(nomes_features) >= 500:
            # limite seguro para evitar estouro de memória em Alibi (quando NÃO houve subamostragem)
            max_instances_to_explain = min(200, len(X_test))
            print(f"Dataset de alta dimensionalidade detectado ({len(nomes_features)} features). "
                  f"Explicando apenas as primeiras {max_instances_to_explain} instâncias para evitar estouro de memória.")

        print(f"Gerando explicações com Anchor (Alibi) para as {max_instances_to_explain} instâncias de teste (de {len(X_test)} totais)...")
        tempos_total, tempos_pos, tempos_neg, tempos_rej = [], [], [], []
        explicacoes = {}
        # Parâmetros low-memory para alta dimensionalidade (ex.: MNIST)
        low_memory = len(nomes_features) >= 500
        anchor_params = {}
        if low_memory:
            anchor_params = {
                'batch_size': 32,
                'beam_size': 1
            }

        for i in range(max_instances_to_explain):
            start_time = time.time()
            instance_arr = X_test.iloc[i].values if hasattr(X_test, 'iloc') else X_test[i]
            try:
                explanation = explainer.explain(instance_arr, threshold=local_anchor_threshold, **anchor_params)
            except (MemoryError, OverflowError) as mem_err:
                # Fallback: tentar reduzir ainda mais memória e continuar
                if low_memory and anchor_params.get('batch_size', 32) > 8:
                    print(f"MEMORY ERROR na instância {i}. Reduzindo batch_size e tentando novamente...")
                    try:
                        explanation = explainer.explain(instance_arr, threshold=local_anchor_threshold, batch_size=8, beam_size=1)
                    except Exception as e:
                        print(f"Falha ao re-tentar com batch_size=8: {e}. Pulando esta instância.")
                        continue
                else:
                    print(f"MEMORY ERROR ao explicar instância {i}: {mem_err}. Pulando esta instância.")
                    continue
            except Exception as e:
                print(f"Erro ao explicar instância {i} com Anchor: {e}. Pulando esta instância.")
                continue

            runtime = time.time() - start_time
            tempos_total.append(runtime)
            explicacoes[i] = explanation.anchor
            if i in indices_rejeitados:
                tempos_rej.append(runtime)
            elif y_pred_final[i] == 1:
                tempos_pos.append(runtime)
            else:
                tempos_neg.append(runtime)

        metricas['tempo_total'] = sum(tempos_total)
        metricas['tempo_medio_instancia'] = np.mean(tempos_total) if tempos_total else 0
        metricas['tempo_medio_positivas'] = np.mean(tempos_pos) if tempos_pos else 0
        metricas['tempo_medio_negativas'] = np.mean(tempos_neg) if tempos_neg else 0
        metricas['tempo_medio_rejeitadas'] = np.mean(tempos_rej) if tempos_rej else 0
        feature_counts = Counter()
        exp_lengths_pos, exp_lengths_neg, exp_lengths_rej = [], [], []
        for i in range(len(y_pred_final)):
            exp_len = len(explicacoes.get(i, []))
            if i in indices_rejeitados:
                exp_lengths_rej.append(exp_len)
            elif y_pred_final[i] == 1:
                exp_lengths_pos.append(exp_len)
            else:
                exp_lengths_neg.append(exp_len)
            for regra in explicacoes.get(i, []):
                for feature_name in nomes_features:
                    if feature_name in regra:
                        feature_counts[feature_name] += 1

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
        base_dir = os.path.join('results', 'report', 'anchor')
        os.makedirs(base_dir, exist_ok=True)
        caminho_arquivo = os.path.join(base_dir, f'anchor_{nome_relatorio}.txt')
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write(log_final)
        print(f"\nANÁLISE CONCLUÍDA. Relatório salvo em: {caminho_arquivo}")

        dataset_key = nome_dataset_original
        # JSON detalhado compatível com PEAB
        X_test_dict = {str(col): [float(x) for x in X_test[col].tolist()] for col in X_test.columns}
        y_test_list = [int(v) for v in (y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test))]
        rejected_mask = np.array([i in indices_rejeitados for i in range(len(y_test))])

        def extract_feats(rules):
            feats = set()
            for regra in rules:
                for fname in nomes_features:
                    if fname in regra:
                        feats.add(fname)
            return sorted(list(feats))

        per_instance = []
        explicacoes_local = locals().get('explicacoes', {}) if isinstance(locals().get('explicacoes', {}), dict) else {}
        for i in range(len(y_test_list)):
            y_pred_i = int(y_pred_final[i]) if int(y_pred_final[i]) in (0, 1) else -1
            rules = explicacoes_local.get(i, [])
            feats_list = extract_feats(rules)
            per_instance.append({
                'id': str(i),
                'y_true': int(y_test_list[i]),
                'y_pred': y_pred_i,
                'rejected': bool(rejected_mask[i]),
                'decision_score': float(decision_scores_test[i]),
                'explanation': feats_list,
                'explanation_size': int(len(feats_list))
            })

        def conv_stats(d):
            return {
                'count': int(d.get('instancias', 0)),
                'min_length': int(d.get('min', 0)),
                'mean_length': float(d.get('media', 0.0)),
                'max_length': int(d.get('max', 0)),
                'std_length': float(d.get('std_dev', 0.0))
            }

        explanation_stats = {
            'positive': conv_stats(stats_pos),
            'negative': conv_stats(stats_neg),
            'rejected': conv_stats(stats_rej)
        }

        model_params = meta.get('model_params', {})
        coefs_ordered = [float(model_params.get('coefs', {}).get(col, 0.0)) for col in nomes_features]
        intercepto = float(model_params.get('intercepto', float(pipeline.named_steps['model'].intercept_[0])))
        scaler_params = model_params.get('scaler_params', {
            'min': [float(v) for v in pipeline.named_steps['scaler'].min_],
            'scale': [float(v) for v in pipeline.named_steps['scaler'].scale_]
        })

        # Metadados extras para MNIST
        mnist_meta = {}
        if nome_dataset_original == 'mnist':
            try:
                from data.datasets import MNIST_FEATURE_MODE, MNIST_SELECTED_PAIR
                mnist_meta = {
                    'mnist_feature_mode': MNIST_FEATURE_MODE,
                    'mnist_digit_pair': list(MNIST_SELECTED_PAIR) if MNIST_SELECTED_PAIR is not None else None
                }
            except Exception:
                mnist_meta = {}

        dataset_cache = {
            'config': {
                'dataset_name': dataset_key,
                'test_size': float(meta['test_size']),
                'random_state': RANDOM_STATE,
                'rejection_cost': float(meta['rejection_cost']),
                **mnist_meta
            },
            'thresholds': {
                't_plus': float(t_plus),
                't_minus': float(t_minus)
            },
            'performance': {
                'accuracy_without_rejection': float(metricas['acuracia_sem_rejeicao']),
                'accuracy_with_rejection': float(metricas['acuracia_com_rejeicao']),
                'rejection_rate': float(metricas['taxa_rejeicao_teste'])
            },
            'explanation_stats': explanation_stats,
            'computation_time': {
                'total': float(metricas.get('tempo_total', 0.0)),
                'mean_per_instance': float(metricas.get('tempo_medio_instancia', 0.0)),
                'positive': float(metricas.get('tempo_medio_positivas', 0.0)),
                'negative': float(metricas.get('tempo_medio_negativas', 0.0)),
                'rejected': float(metricas.get('tempo_medio_rejeitadas', 0.0))
            },
            'top_features': [
                {"feature": feat, "count": int(count)}
                for feat, count in metricas['features_frequentes']
            ],
            'data': {
                'feature_names': list(nomes_features),
                'class_names': list(meta['nomes_classes']),
                'X_test': X_test_dict,
                'y_test': y_test_list
            },
            'model': {
                'coefs': coefs_ordered,
                'intercept': intercepto,
                'scaler_params': scaler_params
            },
            'per_instance': per_instance
        }

        update_method_results("anchor", dataset_key, dataset_cache)

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


# Runner programático para automação (sem menu)
def run_anchor_for_dataset(dataset_name: str) -> dict:
    """
    Executa o Anchor usando o MESMO pipeline e thresholds do PEAB para um dataset específico.
    Retorna um dicionário com métricas principais e caminhos de saída.
    """
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)

    nomes_features = meta['feature_names']
    # Usa o rótulo positivo reportado em meta (índice 1) apenas para nomear o relatório
    nome_relatorio = f"{dataset_name}_{meta['nomes_classes'][0]}_vs_{meta['nomes_classes'][1]}"

    metricas = {'dataset_name': nome_relatorio, 'test_size_atual': meta['test_size']}
    metricas['total_instancias_teste'] = len(y_test)
    metricas['num_features'] = len(nomes_features)
    # Ajuste de precisão para MNIST (acelera geração): usar 0.95
    local_anchor_threshold = 0.95 if len(nomes_features) >= 500 else ANCHOR_PRECISION_THRESHOLD
    y_pred_sem_rejeicao = pipeline.predict(X_test)
    metricas['acuracia_sem_rejeicao'] = accuracy_score(y_test, y_pred_sem_rejeicao) * 100
    metricas['t_lower'], metricas['t_upper'] = float(t_minus), float(t_plus)
    metricas['rejection_cost'] = float(meta['rejection_cost'])

    # Aplicar rejeição com os mesmos thresholds
    decision_scores_test = pipeline.decision_function(X_test)
    y_pred_com_rejeicao = []
    indices_aceitos, indices_rejeitados = [], []
    for i, s in enumerate(decision_scores_test):
        if s >= t_plus:
            y_pred_com_rejeicao.append(1)
            indices_aceitos.append(i)
        elif s <= t_minus:
            y_pred_com_rejeicao.append(0)
            indices_aceitos.append(i)
        else:
            y_pred_com_rejeicao.append(-1)
            indices_rejeitados.append(i)
    y_pred_final = np.array(y_pred_com_rejeicao)
    y_test_aceitos = y_test.iloc[indices_aceitos] if isinstance(y_test, pd.Series) else y_test[indices_aceitos]
    y_pred_aceitos = y_pred_final[indices_aceitos]
    metricas['num_rejeitadas_teste'] = len(indices_rejeitados)
    metricas['num_aceitas_teste'] = len(indices_aceitos)
    metricas['taxa_rejeicao_teste'] = (metricas['num_rejeitadas_teste'] / metricas['total_instancias_teste']) * 100
    metricas['acuracia_com_rejeicao'] = accuracy_score(y_test_aceitos, y_pred_aceitos) * 100 if len(indices_aceitos) > 0 else 100

    # Anchor
    def predict_fn_wrapper(x: np.ndarray):
        if isinstance(x, np.ndarray):
            try:
                df = pd.DataFrame(x, columns=nomes_features)
            except Exception:
                arr = x.reshape(1, -1) if x.ndim == 1 else x
                df = pd.DataFrame(arr, columns=nomes_features)
            return pipeline.predict_proba(df)
        else:
            return pipeline.predict_proba(x)

    explainer = AnchorTabular(predict_fn_wrapper, feature_names=nomes_features)
    explainer.fit(X_train.values if hasattr(X_train, 'values') else X_train, disc_perc=(25, 50, 75))

    # Limitar número de explicações para datasets de alta dimensionalidade
    # Contudo, se houver subamostragem aplicada (meta['subsample_size']), respeitamos isso
    max_instances_to_explain = len(X_test)
    subsample = meta.get('subsample_size', None)
    if subsample is None and len(nomes_features) >= 500:
        max_instances_to_explain = min(200, len(X_test))
        print(f"Dataset de alta dimensionalidade detectado ({len(nomes_features)} features). "
              f"Explicando apenas as primeiras {max_instances_to_explain} instâncias para evitar estouro de memória.")

    tempos_total, tempos_pos, tempos_neg, tempos_rej = [], [], [], []
    explicacoes = {}
    # Parâmetros low-memory para alta dimensionalidade (ex.: MNIST)
    low_memory = len(nomes_features) >= 500
    anchor_params = {}
    if low_memory:
        anchor_params = {
            'batch_size': 32,
            'beam_size': 1
        }

    for i in range(max_instances_to_explain):
        start_time = time.time()
        instance_arr = X_test.iloc[i].values if hasattr(X_test, 'iloc') else X_test[i]
        try:
            explanation = explainer.explain(instance_arr, threshold=local_anchor_threshold, **anchor_params)
        except (MemoryError, OverflowError) as mem_err:
            if low_memory and anchor_params.get('batch_size', 32) > 8:
                print(f"MEMORY ERROR na instância {i}. Reduzindo batch_size e tentando novamente...")
                try:
                    explanation = explainer.explain(instance_arr, threshold=local_anchor_threshold, batch_size=8, beam_size=1)
                except Exception as e:
                    print(f"Falha ao re-tentar com batch_size=8: {e}. Pulando esta instância.")
                    continue
            else:
                print(f"MEMORY ERROR ao explicar instância {i}: {mem_err}. Pulando esta instância.")
                continue
        except Exception as e:
            print(f"Erro ao explicar instância {i} com Anchor: {e}. Pulando esta instância.")
            continue

        runtime = time.time() - start_time
        tempos_total.append(runtime)
        explicacoes[i] = explanation.anchor
        if i in indices_rejeitados:
            tempos_rej.append(runtime)
        elif y_pred_final[i] == 1:
            tempos_pos.append(runtime)
        else:
            tempos_neg.append(runtime)

    metricas['tempo_total'] = sum(tempos_total)
    metricas['tempo_medio_instancia'] = np.mean(tempos_total) if tempos_total else 0
    metricas['tempo_medio_positivas'] = np.mean(tempos_pos) if tempos_pos else 0
    metricas['tempo_medio_negativas'] = np.mean(tempos_neg) if tempos_neg else 0
    metricas['tempo_medio_rejeitadas'] = np.mean(tempos_rej) if tempos_rej else 0

    feature_counts = Counter()
    exp_lengths_pos, exp_lengths_neg, exp_lengths_rej = [], [], []
    for i in range(len(y_pred_final)):
        exp_len = len(explicacoes.get(i, []))
        if i in indices_rejeitados:
            exp_lengths_rej.append(exp_len)
        elif y_pred_final[i] == 1:
            exp_lengths_pos.append(exp_len)
        else:
            exp_lengths_neg.append(exp_len)
        for regra in explicacoes.get(i, []):
            for feature_name in nomes_features:
                if feature_name in regra:
                    feature_counts[feature_name] += 1

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

    # Relatório e JSON
    log_final = formatar_log_anchor(metricas)
    base_dir = os.path.join('results', 'report', 'anchor')
    os.makedirs(base_dir, exist_ok=True)
    caminho_arquivo = os.path.join(base_dir, f'anchor_{nome_relatorio}.txt')
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write(log_final)

    def _conv_stats(d):
        return {
            'count': int(d.get('instancias', 0)),
            'min_length': int(d.get('min', 0)),
            'mean_length': float(d.get('media', 0.0)),
            'max_length': int(d.get('max', 0)),
            'std_length': float(d.get('std_dev', 0.0))
        }

    results_data = {
        "config": {
            "dataset_name": dataset_name,
            "test_size": float(meta['test_size']),
            "random_state": RANDOM_STATE,
            "rejection_cost": float(meta['rejection_cost'])
        },
        "thresholds": {
            "t_plus": float(t_plus),
            "t_minus": float(t_minus)
        },
        "performance": {
            "accuracy_without_rejection": float(metricas['acuracia_sem_rejeicao']),
            "accuracy_with_rejection": float(metricas['acuracia_com_rejeicao']),
            "rejection_rate": float(metricas['taxa_rejeicao_teste'])
        },
        "explanation_stats": {
            "positive": _conv_stats(stats_pos),
            "negative": _conv_stats(stats_neg),
            "rejected": _conv_stats(stats_rej)
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
    update_method_results("anchor", dataset_name, results_data)

    return {
        'report_path': caminho_arquivo,
        'json_updated_for': dataset_name,
        'metrics': metricas
    }