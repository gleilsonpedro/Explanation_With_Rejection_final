# Imports Python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import warnings
import sys
import time
import os

# --- Módulos do projeto ---
try:
    import utils.svm_explainer
    import utils.utility
    from data.datasets import selecionar_dataset_e_classe
    ### ADICIONADO: Importar a função para salvar os resultados no JSON ###
    from utils.results_handler import update_method_results
except ImportError as e:
    print(f"ERRO CRÍTICO AO IMPORTAR MÓDULO: {e}")
    print("Verifique se os scripts 'svm_explainer.py', 'utility.py' e a pasta 'datasets' com seu conteúdo estão no lugar correto.")
    exit()

warnings.filterwarnings("ignore", message="Overwriting previously set objective.")

def formatar_log(metricas):
    """
    Pega todas as métricas coletadas e formata a string do log final.
    """
    stats_pos, stats_neg, stats_rej = "N/A", "N/A", "N/A"
    
    # CORREÇÃO: Usando as chaves corretas ('count', 'mean_length', etc.)
    if metricas.get('stats_explicacao_positive', {}).get('count', 0) > 0:
        stats = metricas['stats_explicacao_positive']
        stats_pos = f"Tamanho (Média ± Desv. Padrão): {stats['mean_length']:.2f} ± {stats['std_length']:.2f} (Min: {stats['min_length']}, Max: {stats['max_length']})"
    
    if metricas.get('stats_explicacao_negative', {}).get('count', 0) > 0:
        stats = metricas['stats_explicacao_negative']
        stats_neg = f"Tamanho (Média ± Desv. Padrão): {stats['mean_length']:.2f} ± {stats['std_length']:.2f} (Min: {stats['min_length']}, Max: {stats['max_length']})"
        
    if metricas.get('stats_explicacao_rejected', {}).get('count', 0) > 0:
        stats = metricas['stats_explicacao_rejected']
        stats_rej = f"Tamanho (Média ± Desv. Padrão): {stats['mean_length']:.2f} ± {stats['std_length']:.2f} (Min: {stats['min_length']}, Max: {stats['max_length']})"
    
    features_frequentes_str = "\n".join([f"    - {feat}: {count} vezes" for feat, count in metricas.get('features_frequentes', [])[:10]])

    log = f"""[CONFIGURAÇÕES GERAIS E DO MODELO]
  - Dataset: {metricas.get('dataset_name', 'N/A')}
  - Classificador: SVM (kernel='linear')
  - Tamanho do Conjunto de Teste: {metricas.get('test_size', 0):.0%}
  - Total de Instâncias de Teste: {metricas.get('total_instancias_teste', 'N/A')}
  - Número de Features do Modelo: {metricas.get('num_features', 'N/A')}
  - Acurácia do Modelo (teste, sem rejeição): {metricas.get('acuracia_sem_rejeicao', 0):.2f}%
  - Thresholds de Rejeição: t+ = {metricas.get('t_upper', 0):.4f}, t- = {metricas.get('t_lower', 0):.4f}
  - Custo de Rejeição (wr): {metricas.get('rejection_cost', 0):.2f}
  - Tempo Total de Geração das Explicações: {metricas.get('tempo_total', 0):.4f} segundos
  - Tempo Médio por Instância (Geral): {metricas.get('tempo_medio_instancia', 0):.4f} segundos

[Métricas de Desempenho do Modelo]
  - Taxa de Rejeição no Teste: {metricas.get('taxa_rejeicao_teste', 0):.2f}% ({metricas.get('num_rejeitadas_teste', 0)} de {metricas.get('total_instancias_teste', 0)} instâncias)
  - Acurácia com Opção de Rejeição (nas {metricas.get('num_aceitas_teste', 0)} instâncias aceitas): {metricas.get('acuracia_com_rejeicao', 0):.2f}%

[Estatísticas do Tamanho das Explicações (Nº de Regras)]
  - Classe Positiva ({metricas.get('stats_explicacao_positive', {}).get('count', 0)} instâncias):
    - {stats_pos}
  - Classe Negativa ({metricas.get('stats_explicacao_negative', {}).get('count', 0)} instâncias):
    - {stats_neg}
  - Classe REJEITADA ({metricas.get('stats_explicacao_rejected', {}).get('count', 0)} instâncias):
    - {stats_rej}

[Análise de Importância de Features (Frequência nas Explicações)]
  - Top 10 Features Mais Frequentes:
{features_frequentes_str}

[Custo Computacional por Classe (Tempo Médio)]
  - Positivas: {metricas.get('tempo_medio_positivas', 0):.4f} segundos
  - Negativas: {metricas.get('tempo_medio_negativas', 0):.4f} segundos
  - Rejeitadas: {metricas.get('tempo_medio_rejeitadas', 0):.4f} segundos
"""
    return log

RANDOM_STATE = 107
# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == '__main__':
    params_config = {
        "iris":             {'wr': [0.24], 'test_size': 0.3},
        "wine":             {'wr': [0.24], 'test_size': 0.3},
        "pima_indians_diabetes": {'wr': [0.24], 'test_size': 0.3},
        "sonar":            {'wr': [0.24], 'test_size': 0.3},
        "vertebral_column": {'wr': [0.24], 'test_size': 0.3},
        "breast_cancer":    {'wr': [0.24], 'test_size': 0.3},
        "spambase":         {'wr': [0.24], 'test_size': 0.2},
        "banknote_auth":    {'wr': [0.24], 'test_size': 0.2},
        "heart_disease":    {'wr': [0.24], 'test_size': 0.3},
        "wine_quality":     {'wr': [0.24], 'test_size': 0.2},
        "creditcard":       {'wr': [0.24], 'test_size': 0.3, 'subsample_size': 0.1}
    }

    print("Chamando menu de seleção de dataset...")
    nome_dataset_original, nome_classe_positiva, dados_completos, alvos_completos, _ = selecionar_dataset_e_classe()

    if dados_completos is not None:
        config_atual = params_config.get(nome_dataset_original, {'wr': [0.25], 'test_size': 0.3})
        wr_param = config_atual['wr']
        test_size_atual = config_atual['test_size']
        
        # Lógica de amostragem para datasets muito grandes
        if 'subsample_size' in config_atual:
            print(f"Aplicando amostragem estratificada de {config_atual['subsample_size']*100}%...")
            dados_prontos, _, alvos_prontos, _ = train_test_split(
                dados_completos, alvos_completos, 
                train_size=config_atual['subsample_size'], 
                random_state= RANDOM_STATE, 
                stratify=alvos_completos
            )
        else:
            dados_prontos, alvos_prontos = dados_completos, alvos_completos
      
        nome_relatorio = f"{nome_dataset_original}_{nome_classe_positiva}_vs_rest"
        nomes_features = dados_completos.columns.tolist() if isinstance(dados_completos, pd.DataFrame) else [f'feature_{i}' for i in range(dados_completos.shape[1])]

        print(f"\n--- Iniciando análise (MinExp - SVM) para: {nome_relatorio} ---")
        metricas = {'dataset_name': nome_relatorio, 'test_size': test_size_atual, 'rejection_cost': wr_param[0]}
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(dados_prontos))
        metricas['num_features'] = scaled_data.shape[1]
        final_targets = utils.utility.check_targets(np.array(alvos_prontos))
        
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_data, final_targets, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=final_targets
        )
        metricas['total_instancias_teste'] = len(y_test)
        
        # Explicações são geradas para o conjunto de teste para comparação justa
        X_exp, y_exp = X_test, y_test

        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)

        metricas['acuracia_sem_rejeicao'] = metrics.accuracy_score(y_test, clf.predict(X_test)) * 100
        metricas['intercepto'] = clf.intercept_[0]

        t_upper, t_lower = utils.utility.find_thresholds(clf, X_train, y_train, wr=wr_param)
        metricas['t_upper'], metricas['t_lower'] = t_upper, t_lower
        
        pos_idx, neg_idx, rej_idx = utils.utility.find_indexes(clf, X_exp, t_upper, t_lower)
        all_explanations = {}
        
        tempo_total_explicacoes = 0
        
        start_time_neg = time.time()
        if len(neg_idx) > 0:
            explanations = utils.svm_explainer.svm_explanation_binary(
                dual_coef=clf.dual_coef_, support_vectors=clf.support_vectors_, intercept=clf.intercept_,
                t_lower=t_lower, t_upper=t_upper, lower_bound=scaled_data.min(), upper_bound=scaled_data.max(),
                show_log=0, n_threads=4, data=X_exp[neg_idx], classified="Negative"
            )
            all_explanations.update({idx: exp for idx, exp in zip(neg_idx, explanations)})
        runtime_neg = time.time() - start_time_neg
        tempo_total_explicacoes += runtime_neg
        metricas['tempo_medio_negativas'] = runtime_neg / len(neg_idx) if len(neg_idx) > 0 else 0

        start_time_pos = time.time()
        if len(pos_idx) > 0:
            explanations = utils.svm_explainer.svm_explanation_binary(
                dual_coef=clf.dual_coef_, support_vectors=clf.support_vectors_, intercept=clf.intercept_,
                t_lower=t_lower, t_upper=t_upper, lower_bound=scaled_data.min(), upper_bound=scaled_data.max(),
                show_log=0, n_threads=4, data=X_exp[pos_idx], classified="Positive"
            )
            all_explanations.update({idx: exp for idx, exp in zip(pos_idx, explanations)})
        runtime_pos = time.time() - start_time_pos
        tempo_total_explicacoes += runtime_pos
        metricas['tempo_medio_positivas'] = runtime_pos / len(pos_idx) if len(pos_idx) > 0 else 0

        start_time_rej = time.time()
        if len(rej_idx) > 0:
            explanations = utils.svm_explainer.svm_explanation_rejected(
                dual_coef=clf.dual_coef_, support_vectors=clf.support_vectors_, intercept=clf.intercept_,
                t_lower=t_lower, t_upper=t_upper, lower_bound=scaled_data.min(), upper_bound=scaled_data.max(),
                data=X_exp[rej_idx], show_log=0, n_threads=4
            )
            all_explanations.update({idx: exp for idx, exp in zip(rej_idx, explanations)})
        runtime_rej = time.time() - start_time_rej
        tempo_total_explicacoes += runtime_rej
        metricas['tempo_medio_rejeitadas'] = runtime_rej / len(rej_idx) if len(rej_idx) > 0 else 0

        metricas['tempo_total'] = tempo_total_explicacoes
        metricas['tempo_medio_instancia'] = tempo_total_explicacoes / len(y_exp) if len(y_exp) > 0 else 0
        
        metricas['num_rejeitadas_teste'] = len(rej_idx)
        metricas['num_aceitas_teste'] = len(y_exp) - len(rej_idx)
        metricas['taxa_rejeicao_teste'] = (len(rej_idx) / len(y_exp)) * 100 if len(y_exp) > 0 else 0
        metricas['acuracia_com_rejeicao'] = utils.utility.calculate_accuracy(clf, t_upper, t_lower, X_test, y_test) * 100
        
        feature_counts = {name: 0 for name in nomes_features}
      
      # Listas para guardar os tamanhos das explicações por classe
        exp_lengths_pos, exp_lengths_neg, exp_lengths_rej = [], [], []

      # Primeiro, itera sobre todas as explicações para coletar os tamanhos e contar as features
        for idx, exp in all_explanations.items():
            exp_len = len(exp)

            # Adiciona o tamanho à lista da classe correta
            if idx in pos_idx:
                exp_lengths_pos.append(exp_len)
            elif idx in neg_idx:
                exp_lengths_neg.append(exp_len)
            elif idx in rej_idx:
                exp_lengths_rej.append(exp_len)

            # Conta a frequência das features
            for item_explicacao in exp:
                feature_idx = item_explicacao[0]
                if feature_idx < len(nomes_features):
                    feature_counts[nomes_features[feature_idx]] += 1

      # Agora, calcula as estatísticas para cada classe usando as listas preenchidas
        for class_key, exp_lengths in [("positive", exp_lengths_pos), ("negative",    exp_lengths_neg), ("rejected", exp_lengths_rej)]:
            stats = {'count': len(exp_lengths)}
            if exp_lengths:
                stats.update({
                    'min_length': int(np.min(exp_lengths)),
                    'mean_length': float(np.mean(exp_lengths)),
                    'max_length': int(np.max(exp_lengths)),
                    'std_length': float(np.std(exp_lengths))
                })
            else:
                stats.update({'min_length': 0, 'mean_length': 0, 'max_length': 0, 'std_length': 0})
            metricas[f'stats_explicacao_{class_key}'] = stats

        metricas['features_frequentes'] = sorted(feature_counts.items(), key=lambda item: item  [1], reverse=True)

        log_final = formatar_log(metricas)
        if not os.path.exists('report'): os.makedirs('report')
        caminho_arquivo = os.path.join('report', f'MinExp_{nome_relatorio}.txt')
        with open(caminho_arquivo, 'w', encoding='utf-8') as f: f.write(log_final)
        print(f"\nANÁLISE CONCLUÍDA. Relatório salvo em: {caminho_arquivo}")
        
        ### ADICIONADO: Bloco inteiro para formatar e salvar os resultados no JSON ###
        dataset_key = nome_dataset_original
        results_data = {
            "config": {
                "dataset_name": dataset_key,
                "test_size": test_size_atual,
                "random_state": RANDOM_STATE,
                "rejection_cost": wr_param[0]
            },
            "thresholds": {
                "t_plus": t_upper,
                "t_minus": t_lower
            },
            "performance": {
                "accuracy_without_rejection": metricas['acuracia_sem_rejeicao'],
                "accuracy_with_rejection": metricas['acuracia_com_rejeicao'],
                "rejection_rate": metricas['taxa_rejeicao_teste']
            },
            "explanation_stats": {
                "positive": metricas['stats_explicacao_positive'],
                "negative": metricas['stats_explicacao_negative'],
                "rejected": metricas['stats_explicacao_rejected']
            },
            "computation_time": {
                "total": metricas['tempo_total'],
                "mean_per_instance": metricas['tempo_medio_instancia'],
                "positive": metricas['tempo_medio_positivas'],
                "negative": metricas['tempo_medio_negativas'],
                "rejected": metricas['tempo_medio_rejeitadas']
            },
            "top_features": [
                {"feature": feat, "count": count} 
                for feat, count in metricas['features_frequentes']
            ]
        }
        update_method_results("MinExp", dataset_key, results_data)
        
    else:
        print("Nenhum dataset foi selecionado. Encerrando o programa.")