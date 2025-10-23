import os
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
from utils.results_handler import update_method_results


# INDICES DOS COMENTÁRIOS
# [TIME] - tempo de execução esta medindo o custo de gerar as explicações
# [TÓPICO GERAL] - Comentários sobre a lógica geral e conceitos dos artigos
# [CLASSE POS/NEG] - Lógica específica para instâncias classificadas (positivas ou negativas)
# [CLASSE REJEITADA] - Lógica específica para instâncias rejeitadas
# [REJEITADA FORMAL] - Passo a passo do modo formal de iniciação (método do artigo)
# [MODIFICAÇÃO IMPORTANTE] O script agora importa a função do seu novo arquivo datasets.py

#==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
#==============================================================================
RANDOM_STATE: int = 42 # Semente aleatória única para garantir reprodutibilidade
# Dicionário de configuração para parâmetros do experimento por dataset
DATASET_CONFIG = {
    "iris":                 {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine":                 {'test_size': 0.3, 'rejection_cost': 0.24},
    "breast_cancer":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "pima_indians_diabetes":{'test_size': 0.3, 'rejection_cost': 0.24},
    "vertebral_column":     {'test_size': 0.3, 'rejection_cost': 0.24},
    "sonar":                {'test_size': 0.3, 'rejection_cost': 0.24}
}
OUTPUT_BASE_DIR: str = 'results/report/peab'
DETAILED_LOG_ENABLED: bool = True # Controla a geração de logs detalhados por instância

#==============================================================================
# [NOVA SEÇÃO] TEMPLATES PARA O RELATÓRIO DE LOG
#==============================================================================
LOG_TEMPLATES = {
    'classificada_analise': "\n   --> ANÁLISE: O score da instância está {posicao}. O algoritmo buscará o menor conjunto de features que garante essa condição.",
    'classificada_processo_header': "\n   [ PROCESSO DE MINIMIZAÇÃO (Garantir Suficiência) ]",
    'classificada_inicio': "\n     - Explicação Inicial: {num_features} features.",
    'classificada_tentativa': "     - Tentativa de remoção da feature '{feat_nome}'...",
    'classificada_pior_cenario': "       - Pior Cenário (Score Perturbado): {score:.4f}",
    'classificada_verificacao_sucesso': "       - Verificação: {score:.4f} {condicao} -> SUCESSO. Remoção validada.",
    'classificada_verificacao_falha': "       - Verificação: {score:.4f} {condicao} -> FALHA. A feature é essencial.",
    'classificada_atualizacao_expl': "       - Explicação atualizada com {num_features} features.",
    'classificada_manter_expl': "       - '{feat_nome}' é mantida na explicação.",

    'rejeitada_analise': "\n   --> ANÁLISE: O score da instância está na zona de rejeição [{t_minus:.4f}, {t_plus:.4f}].\n       Para encontrar a explicação de tamanho mínimo, o algoritmo executa uma\n       otimização com duas ordens de avaliação e seleciona o melhor resultado.",
    'rejeitada_caminho_header': "\n   [ CAMINHO {num_caminho}: Ordem de avaliação {ordem} ]\n",
    'rejeitada_missao_a_header': "     -> Missão A: Garantir que o score > t- ({t_minus:.4f})",
    'rejeitada_missao_a_processo': "        - Processo de Minimização (Anti-Classe 0)...",
    'rejeitada_missao_a_resultado': "        - Resultado da Missão A: {explicacao} ({num_features} features)\n",
    'rejeitada_missao_b_header': "     -> Missão B: Garantir que o score < t+ ({t_plus:.4f})",
    'rejeitada_missao_b_processo': "        - Processo de Minimização (Anti-Classe 1)...",
    'rejeitada_missao_b_resultado': "        - Resultado da Missão B: {explicacao} ({num_features} features)\n",
    'rejeitada_uniao': "     -> UNIÃO (A U B): A explicação para este caminho precisa de {num_features} features.",
    'rejeitada_explicacao_caminho': "        - Explicação do Caminho {num_caminho}: {explicacao}",
}

#==============================================================================
# FUNÇÕES AUXILIARES
#==============================================================================
def configurar_experimento(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], float, float]:
    """Carrega o dataset e as configurações específicas do experimento (modo programático)."""
    # Usa o carregador direto (sem menu) e aplica a configuração do dicionário
    X, y, nomes_classes = carregar_dataset(dataset_name)
    if X is None or y is None or nomes_classes is None:
        raise ValueError(f"Falha ao carregar dataset '{dataset_name}'")
    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        raise KeyError(f"Configuração ausente para dataset '{dataset_name}' em DATASET_CONFIG")
    return X, y, nomes_classes, cfg['rejection_cost'], cfg['test_size']

def treinar_e_avaliar_modelo(X: pd.DataFrame, y: pd.Series, test_size: float, rejection_cost: float) -> Tuple[Pipeline, float, float, Dict[str, Any]]:
    """Treina o modelo, otimiza os limiares de rejeição e retorna o modelo treinado e os limiares."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='liblinear'))
    ])
    pipeline.fit(X_train, y_train)

    decision_scores = pipeline.decision_function(X_train)
    
    # [TÓPICO GERAL] Otimização dos limiares t+ e t-
    search_space = np.unique(np.quantile(decision_scores, np.linspace(0, 1, 100)))
    best_risk, best_t_plus, best_t_minus = float('inf'), 0, 0

    for i in range(len(search_space)):
        for j in range(i, len(search_space)):
            t_minus, t_plus = search_space[i], search_space[j]
            
            y_pred = np.full(y_train.shape, -1)
            accepted_mask = (decision_scores >= t_plus) | (decision_scores <= t_minus)
            
            y_pred[decision_scores >= t_plus] = 1
            y_pred[decision_scores <= t_minus] = 0
            
            error_rate = np.mean(y_pred[accepted_mask] != y_train[accepted_mask]) if np.any(accepted_mask) else 0
            rejection_rate = 1 - np.mean(accepted_mask)
            
            risk = error_rate + rejection_cost * rejection_rate
            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, t_plus, t_minus

    model_params = {
        'coefs': {col: coef for col, coef in zip(X.columns, pipeline.named_steps['model'].coef_[0])},
        'intercepto': pipeline.named_steps['model'].intercept_[0],
        'scaler_params': {'min': pipeline.named_steps['scaler'].min_, 'scale': pipeline.named_steps['scaler'].scale_}
    }
    
    return pipeline, best_t_plus, best_t_minus, model_params

#==============================================================================
# GERAÇÃO DAS EXPLICAÇÕES (LÓGICA)
#==============================================================================
# [CLASSE POS/NEG]
def gerar_explicacao_instancia_classificada(instancia: pd.Series, modelo: Pipeline, t_plus: float, t_minus: float, pred: int) -> Tuple[List[str], Dict[str, Any]]:
    """[LÓGICA] Gera a explicação para uma instância classificada (não rejeitada)."""
    
    # [CLASSE POS/NEG] Determina o objetivo: garantir que o score fique acima de t+ ou abaixo de t-
    objetivo = "manter_acima_de" if pred == 1 else "manter_abaixo_de"
    limiar = t_plus if pred == 1 else t_minus

    # Executa a minimização com a primeira ordem de features
    expl_minima_1, log_data_1 = minimizar_explicacao(instancia, modelo, objetivo, limiar, 'ordem_A')
    
    # [MODIFICAÇÃO IMPORTANTE] Executa uma segunda vez com outra ordenação para buscar uma explicação menor
    expl_minima_2, log_data_2 = minimizar_explicacao(instancia, modelo, objetivo, limiar, 'ordem_B')

    # Seleciona o melhor resultado
    if len(expl_minima_2) < len(expl_minima_1):
        return expl_minima_2, log_data_2
    else:
        return expl_minima_1, log_data_1

# [CLASSE REJEITADA]
def gerar_explicacao_instancia_rejeitada(instancia: pd.Series, modelo: Pipeline, t_plus: float, t_minus: float) -> Tuple[List[str], Dict[str, Any]]:
    """[LÓGICA] Gera a explicação para uma instância rejeitada."""
    
    log_data = {'caminhos': []}

    # [REJEITADA FORMAL] Caminho 1
    expl_A_up, _ = minimizar_explicacao(instancia, modelo, "manter_acima_de", t_minus, 'ordem_A')
    expl_A_down, _ = minimizar_explicacao(instancia, modelo, "manter_abaixo_de", t_plus, 'ordem_A')
    uniao_A = sorted(list(set(expl_A_up) | set(expl_A_down)))
    log_data['caminhos'].append({
        'num': 1, 'ordem': 'A', 'missao_a_resultado': expl_A_up,
        'missao_b_resultado': expl_A_down, 'uniao': uniao_A
    })

    # [REJEITADA FORMAL] Caminho 2
    expl_B_up, _ = minimizar_explicacao(instancia, modelo, "manter_acima_de", t_minus, 'ordem_B')
    expl_B_down, _ = minimizar_explicacao(instancia, modelo, "manter_abaixo_de", t_plus, 'ordem_B')
    uniao_B = sorted(list(set(expl_B_up) | set(expl_B_down)))
    log_data['caminhos'].append({
        'num': 2, 'ordem': 'B', 'missao_a_resultado': expl_B_up,
        'missao_b_resultado': expl_B_down, 'uniao': uniao_B
    })

    expl_final = uniao_A if len(uniao_A) < len(uniao_B) else uniao_B
    return expl_final, log_data

def minimizar_explicacao(instancia: pd.Series, modelo: Pipeline, objetivo: str, limiar: float, ordem: str) -> Tuple[List[str], Dict[str, Any]]:
    """[LÓGICA] Função central que realiza a minimização subtrativa."""
    expl_minima = list(instancia.index)
    
    log_data = {'passos': [], 'num_features_inicial': len(expl_minima)}
    
    # [TÓPICO GERAL] A ordem de teste das features pode levar a mínimos locais diferentes.
    if ordem == 'ordem_A':
        feats_para_testar = sorted(expl_minima, key=lambda f: abs(modelo.named_steps['model'].coef_[0][instancia.index.get_loc(f)] * instancia[f]), reverse=True)
    else:
        feats_para_testar = sorted(expl_minima, key=lambda f: abs(modelo.named_steps['model'].coef_[0][instancia.index.get_loc(f)] * instancia[f]), reverse=False)

    for feat_nome in feats_para_testar:
        if feat_nome not in expl_minima: continue
        
        expl_temp = [f for f in expl_minima if f != feat_nome]
        if not expl_temp: continue
        
        score_perturbado = calcular_score_pior_caso(expl_temp, objetivo, modelo, instancia)
        
        remocao_bem_sucedida = (objetivo == "manter_acima_de" and score_perturbado > limiar) or \
                               (objetivo == "manter_abaixo_de" and score_perturbado < limiar)

        log_passo = {'feat_nome': feat_nome, 'score_perturbado': score_perturbado, 'sucesso': remocao_bem_sucedida}
        
        if remocao_bem_sucedida:
            expl_minima = expl_temp
        
        log_data['passos'].append(log_passo)
            
    return expl_minima, log_data

def calcular_score_pior_caso(explicacao_atual: List[str], objetivo: str, modelo: Pipeline, instancia_original: pd.Series) -> float:
    """[LÓGICA] Calcula o score da instância sob o pior cenário de perturbação."""
    scaler = modelo.named_steps['scaler']
    logreg = modelo.named_steps['model']
    
    score_fixo = logreg.intercept_[0]
    for feat_nome in explicacao_atual:
        idx = instancia_original.index.get_loc(feat_nome)
        score_fixo += logreg.coef_[0][idx] * instancia_original[feat_nome]

    score_perturbado = score_fixo
    features_perturbaveis = [f for f in instancia_original.index if f not in explicacao_atual]

    for feat_nome in features_perturbaveis:
        idx = instancia_original.index.get_loc(feat_nome)
        peso = logreg.coef_[0][idx]
        
        # [CLASSE POS/NEG] O pior caso depende do objetivo e do sinal do peso
        if (objetivo == "manter_acima_de" and peso < 0) or (objetivo == "manter_abaixo_de" and peso > 0):
            # Queremos o maior valor possível para a feature (valor 1.0 após scaling)
            score_perturbado += peso * 1.0
        else:
            # Queremos o menor valor possível para a feature (valor 0.0 após scaling)
            score_perturbado += peso * 0.0
            
    return score_perturbado

#==============================================================================
# FORMATAÇÃO DOS LOGS (APRESENTAÇÃO)
#==============================================================================
def formatar_log_classificada(log_data: Dict[str, Any], pred: int, t_plus: float, t_minus: float) -> List[str]:
    """[APRESENTAÇÃO] Formata o log para uma instância classificada."""
    log_formatado = []
    limiar = t_plus if pred == 1 else t_minus
    posicao = 'acima de t+' if pred == 1 else 'abaixo de t-'
    log_formatado.append(LOG_TEMPLATES['classificada_analise'].format(posicao=posicao))
    log_formatado.append(LOG_TEMPLATES['classificada_processo_header'])
    log_formatado.append(LOG_TEMPLATES['classificada_inicio'].format(num_features=log_data['num_features_inicial']))

    expl_atual = log_data['num_features_inicial']
    for passo in log_data['passos']:
        log_formatado.append(LOG_TEMPLATES['classificada_tentativa'].format(feat_nome=passo['feat_nome']))
        log_formatado.append(LOG_TEMPLATES['classificada_pior_cenario'].format(score=passo['score_perturbado']))
        
        condicao_texto = f"< t- ({limiar:.4f})" if pred == 0 else f"> t+ ({limiar:.4f})"

        if passo['sucesso']:
            expl_atual -= 1
            log_formatado.append(LOG_TEMPLATES['classificada_verificacao_sucesso'].format(score=passo['score_perturbado'], condicao=condicao_texto))
            log_formatado.append(LOG_TEMPLATES['classificada_atualizacao_expl'].format(num_features=expl_atual))
        else:
            log_formatado.append(LOG_TEMPLATES['classificada_verificacao_falha'].format(score=passo['score_perturbado'], condicao=condicao_texto))
            log_formatado.append(LOG_TEMPLATES['classificada_manter_expl'].format(feat_nome=passo['feat_nome']))

    return log_formatado

def formatar_log_rejeitada(log_data: Dict[str, Any], t_plus: float, t_minus: float) -> List[str]:
    """[APRESENTAÇÃO] Formata o log para uma instância rejeitada."""
    log_formatado = []
    log_formatado.append(LOG_TEMPLATES['rejeitada_analise'].format(t_minus=t_minus, t_plus=t_plus))

    for caminho in log_data['caminhos']:
        log_formatado.append(LOG_TEMPLATES['rejeitada_caminho_header'].format(num_caminho=caminho['num'], ordem=caminho['ordem']))
        
        log_formatado.append(LOG_TEMPLATES['rejeitada_missao_a_header'].format(t_minus=t_minus))
        log_formatado.append(LOG_TEMPLATES['rejeitada_missao_a_processo'])
        log_formatado.append(LOG_TEMPLATES['rejeitada_missao_a_resultado'].format(explicacao=caminho['missao_a_resultado'], num_features=len(caminho['missao_a_resultado'])))

        log_formatado.append(LOG_TEMPLATES['rejeitada_missao_b_header'].format(t_plus=t_plus))
        log_formatado.append(LOG_TEMPLATES['rejeitada_missao_b_processo'])
        log_formatado.append(LOG_TEMPLATES['rejeitada_missao_b_resultado'].format(explicacao=caminho['missao_b_resultado'], num_features=len(caminho['missao_b_resultado'])))
        
        log_formatado.append(LOG_TEMPLATES['rejeitada_uniao'].format(num_features=len(caminho['uniao'])))
        log_formatado.append(LOG_TEMPLATES['rejeitada_explicacao_caminho'].format(num_caminho=caminho['num'], explicacao=caminho['uniao']))

    return log_formatado

#==============================================================================
# EXECUÇÃO PRINCIPAL
#==============================================================================
def executar_experimento_para_dataset(dataset_name: str, rejection_cost: float, test_size: float):
    """Função principal que orquestra todo o processo para um único dataset."""
    
    print(f" DATASET: {dataset_name.upper()}")

    X, y, nomes_classes, WR_REJECTION_COST, test_size_atual = configurar_experimento(dataset_name)
    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE)
    
    decision_scores_test = modelo.decision_function(X_test)
    y_pred_test = np.full(y_test.shape, -1, dtype=int)
    y_pred_test[decision_scores_test > t_plus] = 1
    y_pred_test[decision_scores_test < t_minus] = 0
    rejected_mask = (y_pred_test == -1)
    y_pred_test[rejected_mask] = 2 # 2 representa rejeição

    # [TIME] Início da contagem de tempo para geração das explicações
    start_time = time.time()
    
    resultados_instancias = []
    for i in range(len(X_test)):
        instancia_original = X_test.iloc[i]
        # Mantém nomes de colunas para evitar o warning do sklearn
        instancia_df = instancia_original.to_frame().T
        instancia_scaled = modelo.named_steps['scaler'].transform(instancia_df).iloc[0].values if hasattr(modelo.named_steps['scaler'].transform(instancia_df), 'iloc') else modelo.named_steps['scaler'].transform(instancia_df)[0]
        instancia_scaled_series = pd.Series(instancia_scaled, index=X_test.columns)
        
        pred_class = y_pred_test[i]
        
        if pred_class == 2: # Rejeitada
            expl, log_data = gerar_explicacao_instancia_rejeitada(instancia_scaled_series, modelo, t_plus, t_minus)
            log_formatado = formatar_log_rejeitada(log_data, t_plus, t_minus)
        else: # Classificada
            expl, log_data = gerar_explicacao_instancia_classificada(instancia_scaled_series, modelo, t_plus, t_minus, pred_class)
            log_formatado = formatar_log_classificada(log_data, pred_class, t_plus, t_minus)
        
        resultados_instancias.append({
            'id': X_test.index[i],
            'classe_real': nomes_classes[y_test.iloc[i]],
            'predicao': 'REJEITADA' if pred_class == 2 else nomes_classes[pred_class],
            'pred_code': int(pred_class),
            'score': decision_scores_test[i],
            'explicacao': expl,
            'tamanho_explicacao': len(expl),
            'log_detalhado': log_formatado
        })
    
    # [TIME] Fim da contagem de tempo
    end_time = time.time()
    tempo_total = end_time - start_time
    
    # Coleta de métricas
    metricas_dict = coletar_metricas(resultados_instancias, y_test, y_pred_test, rejected_mask, tempo_total, model_params, modelo, X_test)

    # Geração do relatório de texto
    gerar_relatorio_texto(dataset_name, test_size_atual, WR_REJECTION_COST, modelo, t_plus, t_minus, y_test.shape[0], metricas_dict, resultados_instancias)
    
    # Atualiza o JSON cumulativo
    dataset_cache_para_json = montar_dataset_cache(dataset_name, X_train, X_test, y_train, y_test, nomes_classes, t_plus, t_minus, WR_REJECTION_COST, test_size_atual, model_params, metricas_dict, y_pred_test, decision_scores_test, rejected_mask, resultados_instancias)
    update_method_results('peab', dataset_name, dataset_cache_para_json)

def coletar_metricas(resultados_instancias, y_test, y_pred_test, rejected_mask, tempo_total, model_params, modelo: Pipeline, X_test: pd.DataFrame):
    """Coleta e calcula todas as métricas do experimento."""
    
    stats_pos = [r['tamanho_explicacao'] for r in resultados_instancias if r.get('pred_code') == 1]
    stats_neg = [r['tamanho_explicacao'] for r in resultados_instancias if r.get('pred_code') == 0]
    stats_rej = [r['tamanho_explicacao'] for r in resultados_instancias if r.get('pred_code') == 2]

    acc_sem_rej = float(np.mean(modelo.predict(X_test) == y_test) * 100)
    acc_com_rej = float(np.mean(y_pred_test[~rejected_mask] == y_test[~rejected_mask]) * 100) if np.any(~rejected_mask) else 100.0
    taxa_rej = float(np.mean(rejected_mask) * 100)

    features_todas_explicacoes = [feat for r in resultados_instancias for feat in r['explicacao']]
    
    return {
    'acuracia_sem_rejeicao': acc_sem_rej,
    'acuracia_com_rejeicao': acc_com_rej,
    'taxa_rejeicao': taxa_rej,
        'stats_explicacao_positiva': {'instancias': len(stats_pos), 'media': np.mean(stats_pos) if stats_pos else 0, 'std_dev': np.std(stats_pos) if stats_pos else 0, 'min': np.min(stats_pos) if stats_pos else 0, 'max': np.max(stats_pos) if stats_pos else 0},
        'stats_explicacao_negativa': {'instancias': len(stats_neg), 'media': np.mean(stats_neg) if stats_neg else 0, 'std_dev': np.std(stats_neg) if stats_neg else 0, 'min': np.min(stats_neg) if stats_neg else 0, 'max': np.max(stats_neg) if stats_neg else 0},
        'stats_explicacao_rejeitada': {'instancias': len(stats_rej), 'media': np.mean(stats_rej) if stats_rej else 0, 'std_dev': np.std(stats_rej) if stats_rej else 0, 'min': np.min(stats_rej) if stats_rej else 0, 'max': np.max(stats_rej) if stats_rej else 0},
    'tempo_total': float(tempo_total),
    'tempo_medio_instancia': float(tempo_total / len(y_test) if len(y_test) > 0 else 0),
        'features_frequentes': Counter(features_todas_explicacoes).most_common(),
    'pesos_modelo': sorted(((k, float(v)) for k, v in model_params['coefs'].items()), key=lambda item: abs(item[1]), reverse=True)
    }

def gerar_relatorio_texto(dataset_name, test_size, wr, modelo, t_plus, t_minus, num_test, metricas, resultados_instancias):
    """[APRESENTAÇÃO] Gera o arquivo de texto com o relatório completo."""
    
    output_path = os.path.join(OUTPUT_BASE_DIR, f"peab_{dataset_name}.txt")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("          RELATÓRIO DE ANÁLISE - MÉTODO PEAB (EXPLAINABLE AI)\n")
        f.write("="*80 + "\n\n")

        f.write("[ PARÂMETROS DO EXPERIMENTO ]\n\n")
        f.write(f"  - Dataset: {dataset_name}\n")
        f.write(f"  - Proporção de Teste: {int(test_size*100)}%\n")
        f.write(f"  - Total de Instâncias de Teste: {num_test}\n")
        f.write(f"  - Custo de Rejeição (wr): {wr:.2f}\n\n")

        f.write("[ PARÂMETROS DO MODELO DE REGRESSÃO LOGÍSTICA ]\n\n")
        f.write(f"  - Número de Features: {len(modelo.named_steps['model'].coef_[0])}\n")
        f.write(f"  - Acurácia (sem rejeição): {metricas['acuracia_sem_rejeicao']:.2f}%\n")
        f.write(f"  - Limiar Superior (t+): {t_plus:.4f}\n")
        f.write(f"  - Limiar Inferior (t-): {t_minus:.4f}\n")
        f.write(f"  - Intercepto (w0): {modelo.named_steps['model'].intercept_[0]:.4f}\n\n")

        f.write("[ DESEMPENHO COMPUTACIONAL ]\n\n")
        f.write(f"  - Tempo Total de Geração das Explicações: {metricas['tempo_total']:.2f} segundos\n")
        f.write(f"  - Tempo Médio por Instância: {metricas['tempo_medio_instancia']:.4f} segundos\n\n")

        f.write("="*80 + "\n")
        f.write("                ANÁLISE DETALHADA POR INSTÂNCIA\n")
        f.write("="*80 + "\n\n")

        # Separa os resultados por tipo de predição
        rejeitadas = [r for r in resultados_instancias if r['predicao'] == 'REJEITADA']
        classificadas = [r for r in resultados_instancias if r['predicao'] != 'REJEITADA']
        
        if rejeitadas:
            f.write("-" * 80 + "\n")
            f.write("                         SEÇÃO A: INSTÂNCIAS REJEITADAS\n")
            f.write("-" * 80 + "\n")
            for r in rejeitadas[:5]: # Limita a 5 exemplos para não poluir o log
                f.write(f"--- INSTÂNCIA #{r['id']} | CLASSE REAL: {r['classe_real']} | SCORE ORIGINAL: {r['score']:.4f} ---\n")
                for log_line in r['log_detalhado']:
                    f.write(f"{log_line}\n")
                f.write(f"\n   --> RESULTADO FINAL (Instância #{r['id']}):\n")
                f.write(f"       - EXPLICAÇÃO ABDUTIVA (Tamanho: {r['tamanho_explicacao']}): {sorted(r['explicacao'])}\n\n")

        if classificadas:
            f.write("-" * 80 + "\n")
            f.write("                       SEÇÃO B: INSTÂNCIAS CLASSIFICADAS\n")
            f.write("-" * 80 + "\n")
            for r in classificadas[:5]: # Limita a 5 exemplos
                pred_class_num = 0 if r['predicao'].endswith('0') else 1
                f.write(f"--- INSTÂNCIA #{r['id']} | CLASSE REAL: {r['classe_real']} | PREDIÇÃO: CLASSE {pred_class_num} | SCORE: {r['score']:.4f} ---\n")
                for log_line in r['log_detalhado']:
                    f.write(f"{log_line}\n")
                f.write(f"\n   --> RESULTADO FINAL (Instância #{r['id']}):\n")
                f.write(f"       - EXPLICAÇÃO ABDUTIVA (Tamanho: {r['tamanho_explicacao']}): {sorted(r['explicacao'])}\n\n")
        
        f.write("="*80 + "\n")
        f.write("                    SUMÁRIO ESTATÍSTICO DAS EXPLICAÇÕES\n")
        f.write("="*80 + "\n\n")

        f.write("[ ESTATÍSTICAS POR CLASSE PREVISTA ]\n\n")
        pos_stats = metricas['stats_explicacao_positiva']
        f.write(f"  - CLASSE POSITIVA ({pos_stats['instancias']} instâncias):\n")
        f.write(f"    - Tamanho (Média ± Desv. Padrão): {pos_stats['media']:.2f} ± {pos_stats['std_dev']:.2f} (Min: {pos_stats['min']}, Max: {pos_stats['max']})\n\n")

        neg_stats = metricas['stats_explicacao_negativa']
        f.write(f"  - CLASSE NEGATIVA ({neg_stats['instancias']} instâncias):\n")
        f.write(f"    - Tamanho (Média ± Desv. Padrão): {neg_stats['media']:.2f} ± {neg_stats['std_dev']:.2f} (Min: {neg_stats['min']}, Max: {neg_stats['max']})\n\n")

        rej_stats = metricas['stats_explicacao_rejeitada']
        f.write(f"  - REJEITADA ({rej_stats['instancias']} instâncias):\n")
        f.write(f"    - Tamanho (Média ± Desv. Padrão): {rej_stats['media']:.2f} ± {rej_stats['std_dev']:.2f} (Min: {rej_stats['min']}, Max: {rej_stats['max']})\n\n")

        f.write("[ ANÁLISE DE FREQUÊNCIA DAS FEATURES NAS EXPLICAÇÕES ]\n\n")
        f.write("  - Features Mais Frequentes (em todas as explicações):\n")
        for feat, count in metricas['features_frequentes']:
            f.write(f"    - {feat}: {count} vezes\n")
        f.write("\n")

        f.write("[ PESOS DO MODELO (COEFICIENTES GLOBAIS) ]\n\n")
        f.write("  - Features Mais Influentes (por valor absoluto do peso):\n")
        for feat, peso in metricas['pesos_modelo']:
            f.write(f"    - {feat:<25}: {peso:.4f}\n")
        f.write("\n")

def montar_dataset_cache(dataset_name: str,
                         X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series,
                         nomes_classes: List[str],
                         t_plus: float,
                         t_minus: float,
                         WR_REJECTION_COST: float,
                         test_size_atual: float,
                         model_params: Dict[str, Any],
                         metricas_dict: Dict[str, Any],
                         y_pred_test: np.ndarray,
                         decision_scores_test: np.ndarray,
                         rejected_mask: np.ndarray,
                         resultados_instancias: List[Dict[str, Any]]):
    """Monta um dicionário com todos os dados relevantes do experimento para salvar em JSON (cumulativo)."""
    try:
        # Extrai os parâmetros do scaler para o JSON
        scaler_params = {
            'min': [float(v) for v in model_params['scaler_params']['min']],
            'scale': [float(v) for v in model_params['scaler_params']['scale']]
        }
        
        # Converte os coeficientes para um formato serializável (lista alinhada às colunas)
        feature_names = list(X_train.columns)
        coefs_ordered = [float(model_params['coefs'][col]) for col in feature_names]
        intercepto = float(model_params['intercepto'])

        # Converte dados de X/Y para JSON (test set suficiente para explicações)
        X_test_dict = {str(col): [float(x) for x in X_test[col].tolist()] for col in X_test.columns}
        y_test_list = [int(v) for v in y_test.tolist()]
        
        # Per-instance details
        per_instance = []
        for i, rid in enumerate(X_test.index):
            per_instance.append({
                'id': str(rid),
                'y_true': int(y_test.iloc[i]),
                'y_pred': int(y_pred_test[i]) if int(y_pred_test[i]) in (0,1) else -1,
                'rejected': bool(rejected_mask[i]),
                'decision_score': float(decision_scores_test[i]),
                'explanation': resultados_instancias[i]['explicacao'],
                'explanation_size': int(resultados_instancias[i]['tamanho_explicacao'])
            })
        
        dataset_cache = {
            'config': {
                'dataset_name': dataset_name,
                'test_size': float(test_size_atual),
                'random_state': RANDOM_STATE,
                'rejection_cost': float(WR_REJECTION_COST)
            },
            'thresholds': {
                't_plus': float(t_plus),
                't_minus': float(t_minus)
            },
            'performance': {
                'accuracy_without_rejection': metricas_dict['acuracia_sem_rejeicao'],
                'accuracy_with_rejection': metricas_dict['acuracia_com_rejeicao'],
                'rejection_rate': metricas_dict['taxa_rejeicao']
            },
            'explanation_stats': {
                'positive': {
                    'count': metricas_dict['stats_explicacao_positiva']['instancias'],
                    'min_length': metricas_dict['stats_explicacao_positiva']['min'],
                    'mean_length': metricas_dict['stats_explicacao_positiva']['media'],
                    'max_length': metricas_dict['stats_explicacao_positiva']['max'],
                    'std_length': metricas_dict['stats_explicacao_positiva']['std_dev']
                },
                'negative': {
                    'count': metricas_dict['stats_explicacao_negativa']['instancias'],
                    'min_length': metricas_dict['stats_explicacao_negativa']['min'],
                    'mean_length': metricas_dict['stats_explicacao_negativa']['media'],
                    'max_length': metricas_dict['stats_explicacao_negativa']['max'],
                    'std_length': metricas_dict['stats_explicacao_negativa']['std_dev']
                },
                'rejected': {
                    'count': metricas_dict['stats_explicacao_rejeitada']['instancias'],
                    'min_length': metricas_dict['stats_explicacao_rejeitada']['min'],
                    'mean_length': metricas_dict['stats_explicacao_rejeitada']['media'],
                    'max_length': metricas_dict['stats_explicacao_rejeitada']['max'],
                    'std_length': metricas_dict['stats_explicacao_rejeitada']['std_dev']
                }
            },
            'computation_time': {
                'total': metricas_dict['tempo_total'],
                'mean_per_instance': metricas_dict['tempo_medio_instancia'],
            },
            'top_features': [
                {"feature": feat, "count": count} 
                for feat, count in metricas_dict['features_frequentes']
            ],
            'data': {
                'feature_names': feature_names,
                'class_names': list(nomes_classes),
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
        return dataset_cache
    except Exception as e:
        print(f"Erro ao montar dataset_cache para o JSON cumulativo: {e}")
        return None

if __name__ == '__main__':
    # Modo interativo: menu para escolher o dataset (como no peab_comparation)
    nome_dataset, _, X_sel, y_sel, nomes_classes_sel = selecionar_dataset_e_classe()
    if not nome_dataset:
        print("Operação cancelada pelo usuário.")
    else:
        cfg = DATASET_CONFIG.get(nome_dataset)
        if not cfg:
            raise KeyError(f"Configuração não encontrada para '{nome_dataset}' em DATASET_CONFIG")
        # Executa usando o fluxo programático (carregará novamente via carregar_dataset para consistência)
        executar_experimento_para_dataset(nome_dataset, cfg['rejection_cost'], cfg['test_size'])
        print("\n\nProcesso concluído.")