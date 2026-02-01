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

def sanitize_filename(filename: str) -> str:
    """Remove caracteres inválidos para nomes de arquivo no Windows."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

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
  - Classificador: LogisticRegression (via get_shared_pipeline, mesmo do PEAB)
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

RANDOM_STATE = 42
# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == '__main__':
    # Config local não é mais necessária para treino/thresholds; ficam a cargo do shared trainer

    print("Chamando menu de seleção de dataset...")
    nome_dataset_original, nome_classe_positiva, dados_completos, alvos_completos, _ = selecionar_dataset_e_classe()

    if dados_completos is not None:
        # 1) Obter pipeline e thresholds idênticos ao PEAB
        pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(nome_dataset_original)
        nomes_features = meta['feature_names']
        # Definir nome do relatório com classes efetivas
        nome_relatorio = f"{nome_dataset_original}_{meta['nomes_classes'][0]}_vs_{meta['nomes_classes'][1]}"
        print(f"\n--- Iniciando análise (MinExp) para: {nome_relatorio} ---")

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
        # Importante: o MinMaxScaler pode gerar valores fora de [0,1] em teste (se ultrapassarem min/max do treino).
        # Como o solver assume variáveis em [0,1], fazemos clipping aqui para evitar erros de warm start no LP.
        X_test_scaled = np.clip(X_test_scaled, 0.0, 1.0)
        lower_bound, upper_bound = 0.0, 1.0

        # 5.1) O solver MinExp usará o conjunto completo de features fornecido.
        # A seleção de features (Top-K) agora é controlada centralmente pelo `get_shared_pipeline`.
        X_test_solver = X_test_scaled
        w_solver = w
        topk_idx = None  # Garantir que o remapeamento não ocorra
        hi_dim = len(nomes_features) >= 500 # Mantido para controle de time_limit
        
        # [OTIMIZAÇÃO] Configuração adaptativa por tamanho de dataset E número de features
        num_instances = len(y_test)
        num_features = len(nomes_features)
        
        # Regra: Mais features = chunks menores (LP fica mais pesado)
        if num_features >= 150:  # MNIST pool2x2 (196), etc
            default_chunk_size = 10  # Processa 10 por vez
            default_time_limit = 20.0  # 20s por chunk de 10 = ~2s/instância
            default_threads = 4
            print(f"[MinExp] Dataset com {num_features} features: chunk_size=10, threads=4")
        elif num_instances > 1000:  # Datasets grandes mas poucas features
            default_chunk_size = 50
            default_time_limit = 60.0
            default_threads = 4
        else:
            default_chunk_size = 50  # Reduzido de 200 para 50
            default_time_limit = 30.0
            default_threads = 2


        print(f"\n[INFO] Gerando explicações para {len(y_test)} instâncias de teste...")
        
        # Importar barra de progresso e suprimir warnings
        from utils.progress_bar import ProgressBar, suppress_library_warnings
        suppress_library_warnings()
        
        all_explanations = {}
        tempo_total_explicacoes = 0.0
        
        # Criar barra de progresso global
        total_instances = len(neg_idx) + len(pos_idx) + len(rej_idx)
        pbar = ProgressBar(total=total_instances, description=f"MinExp Explicando {nome_relatorio}")
        
        # [AUDITORIA] Dicionário para armazenar tempo por instância
        tempos_individuais = {}
        
        # Helper: explicar em chunks para reduzir uso de memória
        def explain_in_chunks(idx_array, classified_label):
            if len(idx_array) == 0:
                return
            time_limit = 30.0 if hi_dim else default_time_limit
            chunk_size = 20 if hi_dim else default_chunk_size
            for start in range(0, len(idx_array), chunk_size):
                sl = slice(start, start + chunk_size)
                sel_idx = idx_array[sl]
                
                # [AUDITORIA] Medir APENAS o tempo do solver (excluir overhead)
                start_chunk = time.perf_counter()
                try:
                    explanations_local = utils.svm_explainer.svm_explanation_binary(
                        dual_coef=dual_coef,
                        support_vectors=np.array([w_solver]),
                        intercept=intercept,
                        t_lower=t_lower,
                        t_upper=t_upper,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        show_log=0,
                        n_threads=default_threads,
                        data=X_test_solver[sel_idx],
                        classified=classified_label,
                        time_limit=time_limit
                    )
                    # [AUDITORIA] Tempo do solver dividido igualmente entre instâncias
                    tempo_chunk = time.perf_counter() - start_chunk
                    tempo_por_instancia = tempo_chunk / len(sel_idx) if len(sel_idx) > 0 else 0.0
                    
                    # Se reduzimos dimensão, remapear índices para espaço original
                    if topk_idx is not None:
                        remapped = []
                        for exp in explanations_local:
                            remapped.append([(int(topk_idx[int(item[0])]), item[1]) for item in exp])
                        explanations_local = remapped
                    all_explanations.update({idx: exp for idx, exp in zip(sel_idx, explanations_local)})
                    
                    # [AUDITORIA] Armazenar tempo para cada instância
                    for idx in sel_idx:
                        tempos_individuais[idx] = tempo_por_instancia
                    
                    # Atualizar progresso
                    pbar.update(len(sel_idx))
                except Exception:
                    # [AUDITORIA] Em caso de erro, tempo = 0
                    for idx in sel_idx:
                        tempos_individuais[idx] = 0.0
                    # Silenciar erros e atualizar progresso
                    pbar.update(len(sel_idx))

        # [AUDITORIA] Processar instâncias negativas
        if len(neg_idx) > 0:
            explain_in_chunks(neg_idx, "Negative")
        
        # [AUDITORIA] Processar instâncias positivas
        if len(pos_idx) > 0:
            explain_in_chunks(pos_idx, "Positive")

        # [AUDITORIA] Processar instâncias rejeitadas
        if len(rej_idx) > 0:
            time_limit = 30.0 if hi_dim else default_time_limit
            chunk_size = 20 if hi_dim else default_chunk_size
            for start in range(0, len(rej_idx), chunk_size):
                sl = slice(start, start + chunk_size)
                sel_idx = rej_idx[sl]
                
                # [AUDITORIA] Medir APENAS o tempo do solver
                start_chunk = time.perf_counter()
                try:
                    explanations_local = utils.svm_explainer.svm_explanation_rejected(
                        dual_coef=dual_coef,
                        support_vectors=np.array([w_solver]),
                        intercept=intercept,
                        t_lower=t_lower,
                        t_upper=t_upper,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        data=X_test_solver[sel_idx],
                        show_log=0,
                        n_threads=default_threads,
                        time_limit=time_limit
                    )
                    # [AUDITORIA] Tempo dividido igualmente entre instâncias
                    tempo_chunk = time.perf_counter() - start_chunk
                    tempo_por_instancia = tempo_chunk / len(sel_idx) if len(sel_idx) > 0 else 0.0
                    
                    if topk_idx is not None:
                        remapped = []
                        for exp in explanations_local:
                            remapped.append([(int(topk_idx[int(item[0])]), item[1]) for item in exp])
                        explanations_local = remapped
                    all_explanations.update({idx: exp for idx, exp in zip(sel_idx, explanations_local)})
                    
                    # [AUDITORIA] Armazenar tempo para cada instância
                    for idx in sel_idx:
                        tempos_individuais[idx] = tempo_por_instancia
                    
                    # Atualizar progresso
                    pbar.update(len(sel_idx))
                except Exception:
                    # [AUDITORIA] Em caso de erro, tempo = 0
                    for idx in sel_idx:
                        tempos_individuais[idx] = 0.0
                    # Silenciar erros e atualizar progresso
                    pbar.update(len(sel_idx))
        
        # Fechar barra de progresso
        pbar.close()
        
        # [AUDITORIA] Calcular métricas de tempo a partir dos tempos individuais
        tempos_neg = [tempos_individuais.get(i, 0.0) for i in neg_idx]
        tempos_pos = [tempos_individuais.get(i, 0.0) for i in pos_idx]
        tempos_rej = [tempos_individuais.get(i, 0.0) for i in rej_idx]
        todos_tempos = [tempos_individuais.get(i, 0.0) for i in range(len(y_test))]
        
        metricas['tempo_medio_negativas'] = float(np.mean(tempos_neg)) if tempos_neg else 0.0
        metricas['tempo_medio_positivas'] = float(np.mean(tempos_pos)) if tempos_pos else 0.0
        metricas['tempo_medio_rejeitadas'] = float(np.mean(tempos_rej)) if tempos_rej else 0.0
        metricas['tempo_total'] = float(sum(todos_tempos))
        metricas['tempo_medio_instancia'] = float(np.mean(todos_tempos)) if todos_tempos else 0.0

        metricas['num_rejeitadas_teste'] = len(rej_idx)
        metricas['num_aceitas_teste'] = len(y_test) - len(rej_idx)
        metricas['taxa_rejeicao_teste'] = (len(rej_idx) / len(y_test)) * 100 if len(y_test) > 0 else 0
        metricas['acuracia_com_rejeicao'] = utils.utility.calculate_accuracy(pipeline, t_upper, t_lower, X_test, y_test) * 100

        feature_counts = {name: 0 for name in nomes_features}

        # Listas para guardar os tamanhos das explicações por classe (contabiliza TODAS as instâncias)
        exp_lengths_pos, exp_lengths_neg, exp_lengths_rej = [], [], []

        # Itera sobre todas as instâncias e usa explicações quando disponíveis (senão tamanho=0)
        total_n = len(y_test)
        for i in range(total_n):
            exp = all_explanations.get(i, [])
            exp_len = len(exp)

            if i in pos_idx:
                exp_lengths_pos.append(exp_len)
            elif i in neg_idx:
                exp_lengths_neg.append(exp_len)
            elif i in rej_idx:
                exp_lengths_rej.append(exp_len)

            # Conta a frequência das features apenas quando houver explicação
            for item_explicacao in exp:
                feature_idx = item_explicacao[0]
                if 0 <= feature_idx < len(nomes_features):
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
        nome_arquivo_seguro = sanitize_filename(nome_relatorio)
        caminho_arquivo = os.path.join(base_dir, f'minexp_{nome_arquivo_seguro}.txt')
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write(log_final)
        print(f"\nANÁLISE CONCLUÍDA. Relatório salvo em: {caminho_arquivo}")

        # Salvar resultados detalhados no JSON (compatível com PEAB)
        dataset_key = nome_dataset_original
        # Predições finais por threshold
        y_pred_final = np.full(len(y_test), -1, dtype=int)
        y_pred_final[pos_idx] = 1
        y_pred_final[neg_idx] = 0
        rejected_mask = np.array([i in rej_idx for i in range(len(y_test))])

        # Dados e rótulos
        X_test_dict = {}
        for col in X_test.columns:
            X_test_dict[str(col)] = [float(x) for x in X_test[col].tolist()]
        y_test_list = [int(v) for v in (y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test))]

        def extract_feats_minexp(exp):
            feats = set()
            for item in exp:
                try:
                    fidx = int(item[0])
                except Exception:
                    continue
                if 0 <= fidx < len(nomes_features):
                    feats.add(nomes_features[fidx])
            return sorted(list(feats))

        per_instance = []
        for i in range(len(y_test_list)):
            feats_list = extract_feats_minexp(all_explanations.get(i, []))
            per_instance.append({
                'id': str(i),
                'y_true': int(y_test_list[i]),
                'y_pred': int(y_pred_final[i]) if int(y_pred_final[i]) in (0, 1) else -1,
                'rejected': bool(rejected_mask[i]),
                'decision_score': float(decfun_test[i]),
                'explanation': feats_list,
                'explanation_size': int(len(feats_list)),
                'tempo_segundos': float(tempos_individuais.get(i, 0.0))  # [AUDITORIA] Tempo por instância
            })

        # Metadados MNIST
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

        subsample = meta.get('subsample_size')

        dataset_cache = {
            'config': {
                'dataset_name': dataset_key,
                'test_size': float(meta['test_size']),
                'random_state': RANDOM_STATE,
                'rejection_cost': float(meta['rejection_cost']),
                'subsample_size': float(subsample) if subsample else None,
                **mnist_meta
            },
            'thresholds': {
                't_plus': float(t_upper),
                't_minus': float(t_lower),
                'rejection_zone_width': float(t_upper - t_lower)
            },
            'performance': {
                'accuracy_without_rejection': float(metricas['acuracia_sem_rejeicao']),
                'accuracy_with_rejection': float(metricas['acuracia_com_rejeicao']),
                'rejection_rate': float(metricas['taxa_rejeicao_teste']),
                'num_test_instances': int(metricas.get('total_instancias_teste', len(y_test))),
                'num_rejected': int(metricas.get('num_rejeitadas_teste', len(rej_idx))),
                'num_accepted': int(metricas.get('num_aceitas_teste', len(pos_idx) + len(neg_idx)))
            },
            'explanation_stats': {
                'positive': metricas['stats_explicacao_positive'],
                'negative': metricas['stats_explicacao_negative'],
                'rejected': metricas['stats_explicacao_rejected']
            },
            'computation_time': {
                'total': float(metricas['tempo_total']),
                'mean_per_instance': float(metricas['tempo_medio_instancia']),
                'positive': float(metricas['tempo_medio_positivas']),
                'negative': float(metricas['tempo_medio_negativas']),
                'rejected': float(metricas['tempo_medio_rejeitadas'])
            },
            'top_features': [
                {"feature": feat, "count": int(count)}
                for feat, count in metricas['features_frequentes'][:20]  # Top 20 em vez de todas
            ],
            'model': {
                'type': 'LogisticRegression',
                'num_features': len(nomes_features),
                'class_names': list(meta['nomes_classes']),
                'coefs': [float(c) for c in pipeline.named_steps['model'].coef_[0].tolist()],
                'intercept': float(pipeline.named_steps['model'].intercept_[0]),
                'scaler_params': {
                    'min': [float(v) for v in pipeline.named_steps['scaler'].min_],
                    'scale': [float(v) for v in pipeline.named_steps['scaler'].scale_]
                }
            }
        }
        dataset_key_safe = sanitize_filename(dataset_key)
        update_method_results("MinExp", dataset_key_safe, dataset_cache)
        
    else:
        print("Nenhum dataset foi selecionado. Encerrando o programa.")


def run_minexp_for_dataset(dataset_name: str) -> dict:
    """
    Executa o MinExp usando o MESMO pipeline e thresholds do PEAB para um dataset específico.
    Retorna um dicionário com métricas principais e caminho do relatório.
    """
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)
    nomes_features = meta['feature_names']

    nome_relatorio = f"{dataset_name}_{meta['nomes_classes'][0]}_vs_{meta['nomes_classes'][1]}"

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
    # Clipping para manter dentro de [0,1] esperado pelo solver
    X_test_scaled = np.clip(X_test_scaled, 0.0, 1.0)
    lower_bound, upper_bound = 0.0, 1.0

    # Alta dimensionalidade: manter MESMAS features do PEAB (sem redução aqui).
    # Nota: get_shared_pipeline já aplica top_k_features quando configurado no PEAB.
    # Portanto, para 1:1, não reduzimos novamente nesta função.
    hi_dim = len(nomes_features) >= 500
    topk_idx = None
    X_test_solver = X_test_scaled
    w_solver = w
    
    # [OTIMIZAÇÃO] Configuração adaptativa por tamanho de dataset E número de features
    num_instances = len(y_test)
    num_features = len(nomes_features)
    
    # Regra: Mais features = chunks menores (LP fica mais pesado)
    if num_features >= 150:  # MNIST pool2x2 (196), etc
        default_chunk_size = 10  # Processa 10 por vez
        default_time_limit = 20.0  # 20s por chunk de 10 = ~2s/instância
        default_threads = 4
        print(f"[MinExp] Dataset com {num_features} features: chunk_size=10, threads=4")
    elif num_instances > 1000:  # Datasets grandes mas poucas features
        default_chunk_size = 50
        default_time_limit = 60.0
        default_threads = 4
    else:
        default_chunk_size = 50  # Reduzido de 200 para 50
        default_time_limit = 30.0
        default_threads = 2

    all_explanations = {}
    # [AUDITORIA] Dicionário para armazenar tempo por instância (run_minexp_for_dataset)
    tempos_individuais = {}
    
    # Importar barra de progresso
    from utils.progress_bar import ProgressBar, suppress_library_warnings
    suppress_library_warnings()
    
    # Criar barra de progresso global
    total_instances = len(neg_idx) + len(pos_idx) + len(rej_idx)
    pbar = ProgressBar(total=total_instances, description=f"MinExp Explicando {nome_relatorio}")

    # Runner: usar mesma estratégia de chunks e time_limit maior para alta dimensionalidade
    def explain_in_chunks(idx_array, classified_label):
        if len(idx_array) == 0:
            return
        time_limit = 30.0 if hi_dim else default_time_limit
        chunk_size = 20 if hi_dim else default_chunk_size
        for start in range(0, len(idx_array), chunk_size):
            sl = slice(start, start + chunk_size)
            sel_idx = idx_array[sl]
            
            # [AUDITORIA] Medir APENAS o tempo do solver (excluir overhead)
            start_chunk = time.perf_counter()
            try:
                explanations_local = utils.svm_explainer.svm_explanation_binary(
                    dual_coef=dual_coef,
                    support_vectors=np.array([w_solver]),
                    intercept=intercept,
                    t_lower=t_lower,
                    t_upper=t_upper,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    show_log=0,
                    n_threads=default_threads,
                    data=X_test_solver[sel_idx],
                    classified=classified_label,
                    time_limit=time_limit
                )
                # [AUDITORIA] Tempo do solver dividido igualmente entre instâncias
                tempo_chunk = time.perf_counter() - start_chunk
                tempo_por_instancia = tempo_chunk / len(sel_idx) if len(sel_idx) > 0 else 0.0
                
                if topk_idx is not None:
                    remapped = []
                    for exp in explanations_local:
                        remapped.append([(int(topk_idx[int(item[0])]), item[1]) for item in exp])
                    explanations_local = remapped
                all_explanations.update({idx: exp for idx, exp in zip(sel_idx, explanations_local)})
                
                # [AUDITORIA] Armazenar tempo para cada instância do chunk
                for idx in sel_idx:
                    tempos_individuais[idx] = tempo_por_instancia
                
                # Atualizar barra de progresso
                pbar.update(len(sel_idx))
                    
            except Exception as e:
                print(f"[MinExp] Solver falhou (runner) para {classified_label.lower()} (chunk {start}:{start+chunk_size}): {e}. Prosseguindo.")
                # Atualizar progresso mesmo em caso de erro
                pbar.update(len(sel_idx))

    # [AUDITORIA] Processar cada tipo (não medir aqui, já medido dentro do explain_in_chunks)
    if len(neg_idx) > 0:
        explain_in_chunks(neg_idx, "Negative")
    if len(pos_idx) > 0:
        explain_in_chunks(pos_idx, "Positive")
    
    # [AUDITORIA] Rejeitadas: mesmo padrão com timer no solver
    if len(rej_idx) > 0:
        time_limit = 30.0 if hi_dim else default_time_limit
        chunk_size = 20 if hi_dim else default_chunk_size
        for start in range(0, len(rej_idx), chunk_size):
            sl = slice(start, start + chunk_size)
            sel_idx = rej_idx[sl]
            
            # [AUDITORIA] Medir APENAS o tempo do solver
            start_chunk = time.perf_counter()
            try:
                explanations_local = utils.svm_explainer.svm_explanation_rejected(
                    dual_coef=dual_coef,
                    support_vectors=np.array([w_solver]),
                    intercept=intercept,
                    t_lower=t_lower,
                    t_upper=t_upper,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    data=X_test_solver[sel_idx],
                    show_log=0,
                    n_threads=default_threads,
                    time_limit=time_limit
                )
                # [AUDITORIA] Tempo do solver dividido igualmente entre instâncias
                tempo_chunk = time.perf_counter() - start_chunk
                tempo_por_instancia = tempo_chunk / len(sel_idx) if len(sel_idx) > 0 else 0.0
                
                if topk_idx is not None:
                    remapped = []
                    for exp in explanations_local:
                        remapped.append([(int(topk_idx[int(item[0])]), item[1]) for item in exp])
                    explanations_local = remapped
                all_explanations.update({idx: exp for idx, exp in zip(sel_idx, explanations_local)})
                
                # [AUDITORIA] Armazenar tempo para cada instância do chunk
                for idx in sel_idx:
                    tempos_individuais[idx] = tempo_por_instancia
                
                # Atualizar barra de progresso
                pbar.update(len(sel_idx))
                    
            except Exception as e:
                print(f"[MinExp] Solver falhou (runner) para rejeitadas (chunk {start}:{start+chunk_size}): {e}. Prosseguindo.")
                # Atualizar progresso mesmo em caso de erro
                pbar.update(len(sel_idx))

    # [AUDITORIA] Calcular métricas a partir dos tempos individuais
    tempos_neg = [tempos_individuais.get(i, 0.0) for i in neg_idx]
    tempos_pos = [tempos_individuais.get(i, 0.0) for i in pos_idx]
    tempos_rej = [tempos_individuais.get(i, 0.0) for i in rej_idx]
    todos_tempos = [tempos_individuais.get(i, 0.0) for i in range(len(y_test))]
    
    metricas['tempo_medio_negativas'] = float(np.mean(tempos_neg)) if tempos_neg else 0.0
    metricas['tempo_medio_positivas'] = float(np.mean(tempos_pos)) if tempos_pos else 0.0
    metricas['tempo_medio_rejeitadas'] = float(np.mean(tempos_rej)) if tempos_rej else 0.0
    metricas['tempo_total'] = float(sum(todos_tempos))
    metricas['tempo_medio_instancia'] = float(np.mean(todos_tempos)) if todos_tempos else 0.0
    metricas['num_rejeitadas_teste'] = len(rej_idx)
    metricas['num_aceitas_teste'] = len(y_test) - len(rej_idx)
    metricas['taxa_rejeicao_teste'] = (len(rej_idx) / len(y_test)) * 100 if len(y_test) > 0 else 0
    metricas['acuracia_com_rejeicao'] = utils.utility.calculate_accuracy(pipeline, t_upper, t_lower, X_test, y_test) * 100

    feature_counts = {name: 0 for name in nomes_features}
    exp_lengths_pos, exp_lengths_neg, exp_lengths_rej = [], [], []
    total_n = len(y_test)
    for i in range(total_n):
        exp = all_explanations.get(i, [])
        exp_len = len(exp)
        if i in pos_idx:
            exp_lengths_pos.append(exp_len)
        elif i in neg_idx:
            exp_lengths_neg.append(exp_len)
        elif i in rej_idx:
            exp_lengths_rej.append(exp_len)
        for item_explicacao in exp:
            feature_idx = item_explicacao[0]
            if 0 <= feature_idx < len(nomes_features):
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
    nome_arquivo_seguro = sanitize_filename(nome_relatorio)
    caminho_arquivo = os.path.join(base_dir, f'minexp_{nome_arquivo_seguro}.txt')
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write(log_final)

    dataset_key = dataset_name
    # Predições finais por threshold
    y_pred_final = np.full(len(y_test), -1, dtype=int)
    y_pred_final[pos_idx] = 1
    y_pred_final[neg_idx] = 0
    rejected_mask = np.array([i in rej_idx for i in range(len(y_test))])

    X_test_dict = {}
    for col in X_test.columns:
        X_test_dict[str(col)] = [float(x) for x in X_test[col].tolist()]
    y_test_list = [int(v) for v in (y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test))]

    def extract_feats_minexp(exp):
        feats = set()
        for item in exp:
            try:
                fidx = int(item[0])
            except Exception:
                continue
            if 0 <= fidx < len(nomes_features):
                feats.add(nomes_features[fidx])
        return sorted(list(feats))

    # Metadados MNIST
    mnist_meta = {}
    if dataset_name == 'mnist':
        try:
            from data.datasets import MNIST_FEATURE_MODE, MNIST_SELECTED_PAIR
            mnist_meta = {
                'mnist_feature_mode': MNIST_FEATURE_MODE,
                'mnist_digit_pair': list(MNIST_SELECTED_PAIR) if MNIST_SELECTED_PAIR is not None else None
            }
        except Exception:
            mnist_meta = {}

    subsample = meta.get('subsample_size')

    dataset_cache = {
        'config': {
            'dataset_name': dataset_key,
            'test_size': float(meta['test_size']),
            'random_state': RANDOM_STATE,
            'rejection_cost': float(meta['rejection_cost']),
            'subsample_size': float(subsample) if subsample else None,
            **mnist_meta
        },
        'thresholds': {
            't_plus': float(t_upper),
            't_minus': float(t_lower),
            'rejection_zone_width': float(t_upper - t_lower)
        },
        'performance': {
            'accuracy_without_rejection': float(metricas['acuracia_sem_rejeicao']),
            'accuracy_with_rejection': float(metricas['acuracia_com_rejeicao']),
            'rejection_rate': float(metricas['taxa_rejeicao_teste']),
            'num_test_instances': int(metricas.get('total_instancias_teste', len(y_test))),
            'num_rejected': int(metricas.get('num_rejeitadas_teste', len(rej_idx))),
            'num_accepted': int(metricas.get('num_aceitas_teste', len(pos_idx) + len(neg_idx)))
        },
        'explanation_stats': {
            'positive': metricas['stats_explicacao_positive'],
            'negative': metricas['stats_explicacao_negative'],
            'rejected': metricas['stats_explicacao_rejected']
        },
        'computation_time': {
            'total': float(metricas['tempo_total']),
            'mean_per_instance': float(metricas['tempo_medio_instancia']),
            'positive': float(metricas['tempo_medio_positivas']),
            'negative': float(metricas['tempo_medio_negativas']),
            'rejected': float(metricas['tempo_medio_rejeitadas'])
        },
        'top_features': [
            {"feature": feat, "count": int(count)}
            for feat, count in metricas['features_frequentes'][:20]
        ],
        'model': {
            'type': 'LogisticRegression',
            'num_features': len(nomes_features),
            'class_names': list(meta['nomes_classes']),
            'coefs': [float(c) for c in pipeline.named_steps['model'].coef_[0].tolist()],
            'intercept': float(pipeline.named_steps['model'].intercept_[0]),
            'scaler_params': {
                'min': [float(v) for v in pipeline.named_steps['scaler'].min_],
                'scale': [float(v) for v in pipeline.named_steps['scaler'].scale_]
            }
        }
    }
    dataset_key_safe = sanitize_filename(dataset_key)
    update_method_results("MinExp", dataset_key_safe, dataset_cache)

    return {
        'report_path': caminho_arquivo,
        'json_updated_for': dataset_key,
        'metrics': metricas
    }