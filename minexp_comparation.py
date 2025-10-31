# Imports Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import sys
import time
import os

# --- Módulos do projeto ---
try:
    import utils.svm_explainer
    import utils.utility
    from utils.shared_training import get_shared_pipeline
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
    # Config local não é mais necessária para treino/thresholds; ficam a cargo do shared trainer

    print("Chamando menu de seleção de dataset...")
    nome_dataset_original, nome_classe_positiva, dados_completos, alvos_completos, _ = selecionar_dataset_e_classe()

    if dados_completos is not None:
        nome_relatorio = f"{nome_dataset_original}_{nome_classe_positiva}_vs_rest"
        print(f"\n--- Iniciando análise (MinExp) para: {nome_relatorio} ---")

        # 1) Obter pipeline e thresholds idênticos ao PEAB
        pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(nome_dataset_original)
        nomes_features = meta['feature_names']

        # 2) Métricas básicas
        metricas = {
            'dataset_name': nome_relatorio,
            'test_size': meta['test_size'],
            'rejection_cost': meta['rejection_cost'],
            'num_features': len(nomes_features),
            'total_instancias_teste': len(y_test)
        }
        metricas['acuracia_sem_rejeicao'] = metrics.accuracy_score(y_test, pipeline.predict(X_test)) * 100

        # 3) Índices por thresholds compartilhados
        t_upper, t_lower = float(t_plus), float(t_minus)
        decfun_test = pipeline.decision_function(X_test)
        pos_idx = np.where(decfun_test > t_upper)[0]
        neg_idx = np.where(decfun_test < t_lower)[0]
        rej_idx = np.where((decfun_test <= t_upper) & (decfun_test >= t_lower))[0]

        # 4) Adapter LR -> forma esperada pelo solver MinExp (linear)
        w = pipeline.named_steps['model'].coef_[0]
        b = pipeline.named_steps['model'].intercept_[0]
        support_vectors = np.array([w])        # shape (1, d)
        dual_coef = np.array([[1.0]])         # shape (1, 1)
        intercept = np.array([b])             # shape (1,)

        # 5) Dados escalados para o solver (MinMaxScaler)
        X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)
        lower_bound, upper_bound = 0.0, 1.0

        all_explanations = {}
        tempo_total_explicacoes = 0.0
        
        start_time_neg = time.time()
        if len(neg_idx) > 0:
            explanations = utils.svm_explainer.svm_explanation_binary(
                dual_coef=dual_coef,
                support_vectors=support_vectors,
                intercept=intercept,
                t_lower=t_lower,
                t_upper=t_upper,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                show_log=0,
                n_threads=4,
                data=X_test_scaled[neg_idx],
                classified="Negative"
            )
            all_explanations.update({idx: exp for idx, exp in zip(neg_idx, explanations)})
        runtime_neg = time.time() - start_time_neg
        tempo_total_explicacoes += runtime_neg
        metricas['tempo_medio_negativas'] = runtime_neg / len(neg_idx) if len(neg_idx) > 0 else 0

        start_time_pos = time.time()
        if len(pos_idx) > 0:
            explanations = utils.svm_explainer.svm_explanation_binary(
                dual_coef=dual_coef,
                support_vectors=support_vectors,
                intercept=intercept,
                t_lower=t_lower,
                t_upper=t_upper,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                show_log=0,
                n_threads=4,
                data=X_test_scaled[pos_idx],
                classified="Positive"
            )
            all_explanations.update({idx: exp for idx, exp in zip(pos_idx, explanations)})
        runtime_pos = time.time() - start_time_pos
        tempo_total_explicacoes += runtime_pos
        metricas['tempo_medio_positivas'] = runtime_pos / len(pos_idx) if len(pos_idx) > 0 else 0

        start_time_rej = time.time()
        if len(rej_idx) > 0:
            explanations = utils.svm_explainer.svm_explanation_rejected(
                dual_coef=dual_coef,
                support_vectors=support_vectors,
                intercept=intercept,
                t_lower=t_lower,
                t_upper=t_upper,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                data=X_test_scaled[rej_idx],
                show_log=0,
                n_threads=4
            )
            all_explanations.update({idx: exp for idx, exp in zip(rej_idx, explanations)})
        runtime_rej = time.time() - start_time_rej
        tempo_total_explicacoes += runtime_rej
        metricas['tempo_medio_rejeitadas'] = runtime_rej / len(rej_idx) if len(rej_idx) > 0 else 0

        metricas['tempo_total'] = tempo_total_explicacoes
        metricas['tempo_medio_instancia'] = tempo_total_explicacoes / len(y_test) if len(y_test) > 0 else 0

        metricas['num_rejeitadas_teste'] = len(rej_idx)
        metricas['num_aceitas_teste'] = len(y_test) - len(rej_idx)
        metricas['taxa_rejeicao_teste'] = (len(rej_idx) / len(y_test)) * 100 if len(y_test) > 0 else 0
        metricas['acuracia_com_rejeicao'] = utils.utility.calculate_accuracy(pipeline, t_upper, t_lower, X_test, y_test) * 100

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
        for class_key, exp_lengths in [("positive", exp_lengths_pos), ("negative", exp_lengths_neg), ("rejected", exp_lengths_rej)]:
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

        metricas['features_frequentes'] = sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)

        log_final = formatar_log(metricas)
        base_dir = os.path.join('results', 'report', 'minexp')
        os.makedirs(base_dir, exist_ok=True)
        caminho_arquivo = os.path.join(base_dir, f'minexp_{nome_relatorio}.txt')
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write(log_final)
        print(f"\nANÁLISE CONCLUÍDA. Relatório salvo em: {caminho_arquivo}")

        # Salvar resultados consolidados no JSON
        dataset_key = nome_dataset_original
        results_data = {
            "config": {
                "dataset_name": dataset_key,
                "test_size": float(meta['test_size']),
                "random_state": RANDOM_STATE,
                "rejection_cost": float(meta['rejection_cost'])
            },
            "thresholds": {
                "t_plus": float(t_upper),
                "t_minus": float(t_lower)
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


def run_minexp_for_dataset(dataset_name: str) -> dict:
    """
    Executa o MinExp usando o MESMO pipeline e thresholds do PEAB para um dataset específico.
    Retorna um dicionário com métricas principais e caminho do relatório.
    """
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)
    nomes_features = meta['feature_names']

    nome_relatorio = f"{dataset_name}_{meta['nomes_classes'][1]}_vs_rest"

    metricas = {
        'dataset_name': nome_relatorio,
        'test_size': meta['test_size'],
        'rejection_cost': meta['rejection_cost'],
        'num_features': len(nomes_features),
        'total_instancias_teste': len(y_test)
    }
    metricas['acuracia_sem_rejeicao'] = metrics.accuracy_score(y_test, pipeline.predict(X_test)) * 100

    # Índices com thresholds compartilhados
    t_upper, t_lower = float(t_plus), float(t_minus)
    decfun_test = pipeline.decision_function(X_test)
    pos_idx = np.where(decfun_test > t_upper)[0]
    neg_idx = np.where(decfun_test < t_lower)[0]
    rej_idx = np.where((decfun_test <= t_upper) & (decfun_test >= t_lower))[0]

    # Adapter LR -> solver
    w = pipeline.named_steps['model'].coef_[0]
    b = pipeline.named_steps['model'].intercept_[0]
    support_vectors = np.array([w])
    dual_coef = np.array([[1.0]])
    intercept = np.array([b])

    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)
    lower_bound, upper_bound = 0.0, 1.0

    all_explanations = {}
    tempo_total_explicacoes = 0.0

    start_time_neg = time.time()
    if len(neg_idx) > 0:
        time_limit = 1.5 if len(nomes_features) >= 500 else None
        explanations = utils.svm_explainer.svm_explanation_binary(
            dual_coef=dual_coef,
            support_vectors=support_vectors,
            intercept=intercept,
            t_lower=t_lower,
            t_upper=t_upper,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            show_log=0,
            n_threads=4,
            data=X_test_scaled[neg_idx],
            classified="Negative",
            time_limit=time_limit
        )
        all_explanations.update({idx: exp for idx, exp in zip(neg_idx, explanations)})
    runtime_neg = time.time() - start_time_neg
    tempo_total_explicacoes += runtime_neg
    metricas['tempo_medio_negativas'] = runtime_neg / len(neg_idx) if len(neg_idx) > 0 else 0

    start_time_pos = time.time()
    if len(pos_idx) > 0:
        time_limit = 1.5 if len(nomes_features) >= 500 else None
        explanations = utils.svm_explainer.svm_explanation_binary(
            dual_coef=dual_coef,
            support_vectors=support_vectors,
            intercept=intercept,
            t_lower=t_lower,
            t_upper=t_upper,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            show_log=0,
            n_threads=4,
            data=X_test_scaled[pos_idx],
            classified="Positive",
            time_limit=time_limit
        )
        all_explanations.update({idx: exp for idx, exp in zip(pos_idx, explanations)})
    runtime_pos = time.time() - start_time_pos
    tempo_total_explicacoes += runtime_pos
    metricas['tempo_medio_positivas'] = runtime_pos / len(pos_idx) if len(pos_idx) > 0 else 0

    start_time_rej = time.time()
    if len(rej_idx) > 0:
        time_limit = 1.5 if len(nomes_features) >= 500 else None
        explanations = utils.svm_explainer.svm_explanation_rejected(
            dual_coef=dual_coef,
            support_vectors=support_vectors,
            intercept=intercept,
            t_lower=t_lower,
            t_upper=t_upper,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            data=X_test_scaled[rej_idx],
            show_log=0,
            n_threads=4,
            time_limit=time_limit
        )
        all_explanations.update({idx: exp for idx, exp in zip(rej_idx, explanations)})
    runtime_rej = time.time() - start_time_rej
    tempo_total_explicacoes += runtime_rej
    metricas['tempo_medio_rejeitadas'] = runtime_rej / len(rej_idx) if len(rej_idx) > 0 else 0

    metricas['tempo_total'] = tempo_total_explicacoes
    metricas['tempo_medio_instancia'] = tempo_total_explicacoes / len(y_test) if len(y_test) > 0 else 0
    metricas['num_rejeitadas_teste'] = len(rej_idx)
    metricas['num_aceitas_teste'] = len(y_test) - len(rej_idx)
    metricas['taxa_rejeicao_teste'] = (len(rej_idx) / len(y_test)) * 100 if len(y_test) > 0 else 0
    metricas['acuracia_com_rejeicao'] = utils.utility.calculate_accuracy(pipeline, t_upper, t_lower, X_test, y_test) * 100

    feature_counts = {name: 0 for name in nomes_features}
    exp_lengths_pos, exp_lengths_neg, exp_lengths_rej = [], [], []
    for idx, exp in all_explanations.items():
        exp_len = len(exp)
        if idx in pos_idx:
            exp_lengths_pos.append(exp_len)
        elif idx in neg_idx:
            exp_lengths_neg.append(exp_len)
        elif idx in rej_idx:
            exp_lengths_rej.append(exp_len)
        for item_explicacao in exp:
            feature_idx = item_explicacao[0]
            if feature_idx < len(nomes_features):
                feature_counts[nomes_features[feature_idx]] += 1

    for class_key, exp_lengths in [("positive", exp_lengths_pos), ("negative", exp_lengths_neg), ("rejected", exp_lengths_rej)]:
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

    metricas['features_frequentes'] = sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)

    log_final = formatar_log(metricas)
    base_dir = os.path.join('results', 'report', 'minexp')
    os.makedirs(base_dir, exist_ok=True)
    caminho_arquivo = os.path.join(base_dir, f'minexp_{nome_relatorio}.txt')
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write(log_final)

    dataset_key = dataset_name
    results_data = {
        "config": {
            "dataset_name": dataset_key,
            "test_size": float(meta['test_size']),
            "random_state": RANDOM_STATE,
            "rejection_cost": float(meta['rejection_cost'])
        },
        "thresholds": {
            "t_plus": float(t_upper),
            "t_minus": float(t_lower)
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

    return {
        'report_path': caminho_arquivo,
        'json_updated_for': dataset_key,
        'metrics': metricas
    }