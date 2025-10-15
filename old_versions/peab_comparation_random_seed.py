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

from datasets.datasets_Mateus_comparation_complete import selecionar_dataset_e_classe

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
#RANDOM_STATE: int = 42 # Semente aleatória única para garantir reprodutibilidade
#RANDOM_SEEDS: list = [42, 107, 777, 1234, 9876]
RANDOM_SEEDS: list = [107]
# Dicionário de configuração para parâmetros do experimento por dataset
DATASET_CONFIG = {
    "iris":                 {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine":                 {'test_size': 0.3, 'rejection_cost': 0.23},
    "pima_indians_diabetes":{'test_size': 0.1, 'rejection_cost': 0.25},
    "sonar":                {'test_size': 0.3, 'rejection_cost': 0.49},
    "vertebral_column":     {'test_size': 0.3, 'rejection_cost': 0.33},
    "breast_cancer":        {'test_size': 0.3, 'rejection_cost': 0.22},
    "spambase":             {'test_size': 0.1, 'rejection_cost': 0.49},
    "banknote_auth":        {'test_size': 0.2, 'rejection_cost': 0.25}, # Exemplo de valor
    "heart_disease":        {'test_size': 0.3, 'rejection_cost': 0.25}, # Exemplo de valor
    "wine_quality":         {'test_size': 0.2, 'rejection_cost': 0.25}, # Exemplo de valor
    "creditcard":           {'subsample_size': 0.1, 'test_size': 0.3, 'rejection_cost': 0.25} # Exemplo de valor
}

# Constante global que a sua função 'calcular_thresholds' espera encontrar
# O valor dela será atualizado dinamicamente pela função main
WR_REJECTION_COST: float = 0.0

# Suas outras constantes
EPSILON: float = 1e-9
DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'liblinear',
    'max_iter': 200
}

#==============================================================================
# LÓGICA CENTRAL DE EXPLICAÇÃO (Funções base)
#==============================================================================
def carregar_hiperparametros(caminho_arquivo: str = 'hiperparametros.json') -> dict:
    """
    Carrega o arquivo JSON com os hiperparâmetros otimizados.
    """
    try:
        with open(caminho_arquivo, 'r') as f:
            params = json.load(f)
        print(f"Arquivo de hiperparâmetros '{caminho_arquivo}' carregado com sucesso.")
        return params
    except FileNotFoundError:
        print(f"AVISO: Arquivo '{caminho_arquivo}' não encontrado. Usando parâmetros padrão.")
        return {}
    except json.JSONDecodeError:
        print(f"ERRO: Arquivo '{caminho_arquivo}' está corrompido ou mal formatado.")
        return {}

def calcular_thresholds(modelo: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[float, float]:
    """
    Calcula os limiares da zona de rejeição (t+ e t-).
    """
    decision_scores = modelo.decision_function(X_train)
    min_custo = float('inf')
    melhor_t_plus, melhor_t_minus = 0.1, -0.1
    score_max = np.max(decision_scores) if len(decision_scores) > 0 else 0.1
    score_min = np.min(decision_scores) if len(decision_scores) > 0 else -0.1
    pontos_busca = 100
    t_plus_candidatos = np.linspace(0.01, max(score_max, 0.1), pontos_busca)
    t_minus_candidatos = np.linspace(min(score_min, -0.1), -0.01, pontos_busca)
    for t_p in t_plus_candidatos:
        for t_m in t_minus_candidatos:
            if t_m >= t_p: continue
            rejeitadas = (decision_scores >= t_m) & (decision_scores <= t_p)
            aceitas = ~rejeitadas
            taxa_rejeicao = np.mean(rejeitadas)
            if np.sum(aceitas) == 0:
                taxa_erro_aceitas = 1.0
            else:
                preds_aceitas = modelo.predict(X_train[aceitas])
                taxa_erro_aceitas = np.mean(preds_aceitas != y_train[aceitas].values)
            custo_total = taxa_erro_aceitas + WR_REJECTION_COST * taxa_rejeicao
            if custo_total < min_custo:
                min_custo, melhor_t_plus, melhor_t_minus = custo_total, t_p, t_m
    return melhor_t_plus, melhor_t_minus

# [TÓPICO GERAL] 01: CÁLCULO DO DELTA (δ) - O CORAÇÃO DO MÉTODO
# Esta função é a implementação central do conceito de 'delta' (δ) do artigo de Marques-Silva et al.
# O delta mede o "poder" ou "impacto" de uma feature para uma determinada predição.
# Ele calcula a diferença entre a contribuição atual da feature e a sua pior contribuição possível.
# O parâmetro 'premis_class' é crucial, pois define qual é o "pior cenário" (se é virar classe 0 ou 1).
def calculate_deltas(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, premis_class: int) -> np.ndarray:
    scaler = modelo.named_steps['scaler']
    logreg = modelo.named_steps['modelo']
    coefs = logreg.coef_[0]
    scaled_instance_vals = scaler.transform(instance_df)[0]
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled_min = X_train_scaled.min(axis=0)
    X_train_scaled_max = X_train_scaled.max(axis=0)
    deltas = np.zeros_like(coefs)

    for i, (coef, scaled_val) in enumerate(zip(coefs, scaled_instance_vals)):
        if premis_class == 1: # Pior caso para classe 1 é o score ir para o lado negativo (classe 0)
            pior_valor_escalonado = X_train_scaled_min[i] if coef > 0 else X_train_scaled_max[i]
        else: # premis_class == 0. Pior caso para classe 0 é o score ir para o lado positivo (classe 1)
            pior_valor_escalonado = X_train_scaled_max[i] if coef > 0 else X_train_scaled_min[i]
        deltas[i] = (scaled_val - pior_valor_escalonado) * coef
    return deltas

# [CLASSE REJEITADA] 01: MÉTODO DE INICIAÇÃO FORMAL (ARTIGO)
# [REJEITADA FORMAL] - Passo a passo da iniciação formal
# Esta função gera uma explicação inicial seguindo a lógica formal do artigo.
# Ela calcula o "Pior Score Possível" (score_base) e adiciona features (em ordem de |delta|)
# até que o score cumulativo ultrapasse o limiar de rejeição (t+ ou t-).
# É usada tanto para instâncias classificadas quanto para a iniciação formal das rejeitadas.
def one_explanation_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, premis_class: int, log_collector: List[str]):
    score = modelo.decision_function(instance_df)[0]
    deltas = calculate_deltas(modelo, instance_df, X_train, premis_class)
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []
    # [REJEITADA FORMAL] - 1. Calcula o "Pior Score Possível" (score_base ou Γ^ω). Este é o ponto de partida, a "dívida" a ser paga.
    score_base = score - np.sum(deltas)
    soma_deltas_cumulativa = score_base
    
    #[log] log_collector.append(f"   [Fase 0 - Formal] Calculando explicação com premissa para Classe {premis_class}:")
    #[log] log_collector.append(f"     - Pior Score Possível (Score Base): {score_base:.4f}")
    # [REJEITADA FORMAL] - 2. Define o alvo a ser atingido. Não é o score da instância, mas o limiar da zona de rejeição (t+ ou t-).
    target_score = t_plus if premis_class == 1 else t_minus
    #[log] log_collector.append(f"     - Objetivo: Score cumulativo {' >' if premis_class == 1 else ' <'} {target_score:.4f}")
    # [REJEITADA FORMAL] - 3. Loop para "pagar a dívida". Adiciona features de maior impacto e soma seus deltas (com sinal) ao score_base.
    for i in indices_ordenados:
        # [REJEITADA FORMAL] - 4. Condição de parada: o loop para se o score cumulativo cruzar o alvo.
        if (premis_class == 1 and soma_deltas_cumulativa > target_score and explicacao) or \
           (premis_class == 0 and soma_deltas_cumulativa < target_score and explicacao):
            break

        feature_nome = X_train.columns[i]
        soma_deltas_cumulativa += deltas[i]
        explicacao.append(f"{feature_nome} = {instance_df.iloc[0, i]:.4f}")
       #[log]  log_collector.append(f"       - Adicionando '{feature_nome}'. Delta = {deltas[i]:.4f}. Score cumulativo = {soma_deltas_cumulativa:.4f}.")

    if not explicacao and len(deltas) > 0:
        explicacao.append(f"{X_train.columns[indices_ordenados[0]]} = {instance_df.iloc[0, indices_ordenados[0]]:.4f}")
    # [REJEITADA FORMAL] - 5. Retorna a explicação inicial, que pode ter várias features se a "dívida" for grande
    
    log_collector.append(f"   [Passo 1] Geração da Explicação Inicial:")
    log_collector.append(f"     - Explicação Inicial Gerada ({len(explicacao)} feats): {explicacao}")

    return explicacao

# [CLASSE REJEITADA] 03: GERAÇÃO DAS DUAS INICIAÇÕES FORMAIS
# Esta função orquestra a geração das duas explicações iniciais formais para uma instância rejeitada.
# Uma é gerada com a premissa de evitar a Classe 0, e a outra com a premissa de evitar a Classe 1.
# Essas duas explicações servirão de ponto de partida para os dois caminhos da busca otimizada.
def gerar_explicacao_inicial_rejeitada_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, log_collector: List[str]):
    log_collector.append("\n==================== ESTRATÉGIA DE BUSCA 1 (Objetivo: Evitar Classe 0) ====================")
    expl_inicial_path1 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 1, log_collector)

    log_collector.append("\n==================== ESTRATÉGIA DE BUSCA 2 (Objetivo: Evitar Classe 1) ====================")
    expl_inicial_path2 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 0, log_collector)

    return expl_inicial_path1, expl_inicial_path2

# [TÓPICO GERAL] 02: VALIDAÇÃO DE ROBUSTEZ (O JUIZ)
# Esta é a função mais importante para garantir a corretude. Ela testa se uma explicação é robusta.
# Para isso, ela perturba todas as features que NÃO estão na explicação para o seu pior cenário possível
# e verifica se, mesmo assim, a predição original se mantém (ou se a instância permanece na rejeição).
def perturbar_e_validar_com_log(modelo: Pipeline, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int) -> Tuple[str, str, List[str]]:
    calc_log = [f"          (Log Perturbação: Explicação atual com {len(explicacao)} features fixas)"]
    inst_pert = instance_df.copy()
    features_explicacao = {f.split(' = ')[0] for f in explicacao}
    
    perturbar_para_diminuir_score = (direcao_override == 1)
    modelo_interno = modelo.named_steps['modelo']
    
    for feat_idx, feat_nome in enumerate(X_train.columns):
        if feat_nome in features_explicacao:
            continue
        coef = modelo_interno.coef_[0][feat_idx]
        train_min = X_train[feat_nome].min()
        train_max = X_train[feat_nome].max()
        
        if perturbar_para_diminuir_score:
            valor_pert = train_min if coef > 0 else train_max
        else: # perturbar para aumentar
            valor_pert = train_max if coef > 0 else train_min
        
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
    
    score_pert = modelo.decision_function(inst_pert)[0]
    pred_pert = modelo.predict(inst_pert)[0]
    pert_rejeitada = t_minus <= score_pert <= t_plus
    
    calc_log.append(f"          -> Score da Instância Perturbada: {score_pert:.4f} {'Sim' if pert_rejeitada else 'Não'} ∈ [t- ({t_minus:.4f}), t+ ({t_plus:.4f})]")
    
    is_original_rejected = t_minus <= modelo.decision_function(instance_df)[0] <= t_plus
    
    if is_original_rejected:
        status = "VÁLIDA" if pert_rejeitada else "INVÁLIDA"
    else:
        pred_original_class = modelo.predict(instance_df)[0]
        status = "VÁLIDA" if pred_pert == pred_original_class and not pert_rejeitada else "INVÁLIDA"
        
    return status, "", calc_log

#==============================================================================
# FUNÇÕES DE FASE (ADITIVA -> SUBTRATIVA)
#==============================================================================

# [CLASSE REJEITADA] 04: REFORÇO BIDIRECIONAL (ADICIONANDO FEATURES)
# Esta função implementa a Fase 1 (reforço) para instâncias rejeitadas.
# O ponto crucial é o parâmetro 'premissa_ordenacao', que dita a estratégia.
# A função calcula os deltas dinamicamente com base nessa premissa e adiciona
# features até que a explicação se torne robusta nas DUAS direções.
def executar_fase_1_reforco_bidirecional(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str], premissa_ordenacao: int) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo (Premissa de ordenação: evitar Classe {1 - premissa_ordenacao})")
    expl_robusta = list(expl_inicial)
    adicoes = 0
    
    # [CLASSE REJEITADA] 04.1: CÁLCULO DINÂMICO DO DELTA PARA A ESTRATÉGIA ATUAL
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premissa_ordenacao)
    indices_ordenados = np.argsort(-np.abs(deltas_para_ordenar))

    while True:
        log_collector.append(f"\n     - Testando robustez da explicação atual com {len(expl_robusta)} features...")
        
        status1, _, calc_log1 = perturbar_e_validar_com_log(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, class_names, 1)
        log_collector.append(f"       -> Teste vs Classe 0 (Diminuir Score)...")
        log_collector.extend(calc_log1)

        status2 = ""
        if status1.startswith("VÁLIDA"):
            log_collector.append(f"       -> SUCESSO PARCIAL. Robusto contra Classe 0. Verificando agora contra Classe 1...")
            status2, _, calc_log2 = perturbar_e_validar_com_log(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, class_names, 0)
            log_collector.extend(calc_log2)
        else:
            log_collector.append(f"       -> FALHA IMEDIATA. Não é necessário testar a segunda direção.")

        if status1.startswith("VÁLIDA") and status2.startswith("VÁLIDA"):
            log_collector.append(f"     -> SUCESSO: Explicação é BI-DIRECIONALMENTE robusta.")
            break
        
        log_collector.append(f"     -> FALHA: Explicação não é robusta.")
        if len(expl_robusta) == X_train.shape[1]:
            log_collector.append("     -> ATENÇÃO: Todas as features já foram adicionadas. Impossível reforçar mais.")
            break

        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta}
        adicionou_feature = False
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set:
                log_collector.append(f"     -> REFORÇANDO: Adicionando a próxima feature de maior impacto: '{feat_nome}'.")
                expl_robusta.append(f"{feat_nome} = {instance_df.iloc[0, idx]:.4f}")
                adicoes += 1
                adicionou_feature = True
                break
        
        if not adicionou_feature:
             break
    
    log_collector.append(f"\n   -> Fim da Fase 1. Explicação robusta final tem {len(expl_robusta)} features.")
    return expl_robusta, adicoes

# [CLASSE REJEITADA] 05: MINIMIZAÇÃO BIDIRECIONAL (REMOVENDO FEATURES)
# Esta função implementa a Fase 2 (minimização) para instâncias rejeitadas.
# Ela tenta remover as features de MENOR impacto (menor |delta|) primeiro.
# Uma feature só é removida se a explicação restante continuar robusta nas duas direções.
# Isso garante que a explicação final seja mínima (subset-minimal).
def executar_fase_2_minimizacao_bidirecional(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str], premissa_ordenacao: int) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 2] Início da Minimização Subtrativa (Premissa de ordenação: evitar Classe {1-premissa_ordenacao})")
    expl_minima = list(expl_robusta)
    remocoes = 0

    # [CLASSE REJEITADA] 05.1: CÁLCULO DINÂMICO DO DELTA PARA A ESTRATÉGIA ATUAL DE REMOÇÃO
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premissa_ordenacao)
    
    features_para_remover = sorted([f.split(' = ')[0] for f in expl_minima], key=lambda nome: abs(deltas_para_ordenar[X_train.columns.get_loc(nome)]))
    log_collector.append(f"     - Ordem de tentativa de remoção (do menor |delta| para o maior): {features_para_remover}")

    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1:
            log_collector.append("     -> Parando minimização para manter ao menos uma feature.")
            break
        
        log_collector.append(f"\n     - TENTANDO REMOVER: '{feat_nome}'...")
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        
        status1, _, calc_log1 = perturbar_e_validar_com_log(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, class_names, 1)
        log_collector.append("       -> Teste de remoção vs Classe 0...")
        log_collector.extend(calc_log1)

        remocao_bem_sucedida = False
        if status1.startswith("VÁLIDA"):
            log_collector.append("       -> SUCESSO PARCIAL. Verificando a segunda direção...")
            status2, _, calc_log2 = perturbar_e_validar_com_log(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, class_names, 0)
            log_collector.extend(calc_log2)
            if status2.startswith("VÁLIDA"):
                remocao_bem_sucedida = True
        
        if remocao_bem_sucedida:
            log_collector.append(f"       -> SUCESSO TOTAL: A remoção de '{feat_nome}' manteve a robustez bi-direcional. Explicação agora com {len(expl_temp)} features.")
            expl_minima = expl_temp
            remocoes += 1
        else:
            log_collector.append(f"       -> FALHA: A remoção de '{feat_nome}' quebrou a robustez. Mantendo a feature.")
            
    log_collector.append(f"\n   -> Fim da Fase 2. Explicação mínima final tem {len(expl_minima)} features.")
    return expl_minima, remocoes

# [CLASSE POS/NEG] 01: REFORÇO UNIDIRECIONAL
# Lógica de reforço para instâncias já classificadas. É mais simples porque a batalha é de um lado só.
# O objetivo é apenas garantir que a instância não caia na zona de rejeição.
def executar_fase_1_reforco_unidirecional(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo Uni-Direcional")
    expl_robusta = list(expl_inicial)
    adicoes = 0
    pred_class = modelo.predict(instance_df)[0]
    direcao = 1 if pred_class == 1 else 0

    # [CLASSE POS/NEG] 01.1: CÁLCULO DO DELTA BASEADO NA CLASSE DA INSTÂNCIA
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, pred_class)
    indices_ordenados = np.argsort(-np.abs(deltas_para_ordenar))

    while True:
        log_collector.append(f"     - Testando robustez da explicação atual com {len(expl_robusta)} features...")
        status, _, calc_log = perturbar_e_validar_com_log(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, class_names, direcao)
        log_collector.extend(calc_log)
        
        if status.startswith("VÁLIDA"):
            log_collector.append(f"     -> SUCESSO: Explicação é robusta.")
            break
        
        log_collector.append(f"     -> FALHA: Explicação não é robusta.")
        if len(expl_robusta) == X_train.shape[1]:
            break

        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta}
        adicionou_feature = False
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set:
                expl_robusta.append(f"{feat_nome} = {instance_df.iloc[0, idx]:.4f}")
                adicoes += 1
                adicionou_feature = True
                break
        if not adicionou_feature:
            break

    log_collector.append(f"\n   -> Fim da Fase 1. Explicação robusta final tem {len(expl_robusta)} features.")
    return expl_robusta, adicoes

# [CLASSE POS/NEG] 02: MINIMIZAÇÃO UNIDIRECIONAL
# Lógica de minimização para instâncias já classificadas.
def executar_fase_2_minimizacao_unidirecional(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 2] Início da Minimização Subtrativa Uni-Direcional")
    expl_minima = list(expl_robusta)
    remocoes = 0
    pred_class = modelo.predict(instance_df)[0]
    direcao = 1 if pred_class == 1 else 0

    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, pred_class)
    features_para_remover = sorted([f.split(' = ')[0] for f in expl_minima], key=lambda nome: abs(deltas_para_ordenar[X_train.columns.get_loc(nome)]))

    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1:
            break
        
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        status, _, _ = perturbar_e_validar_com_log(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, class_names, direcao)
        
        if status.startswith("VÁLIDA"):
            expl_minima = expl_temp
            remocoes += 1

    log_collector.append(f"\n   -> Fim da Fase 2. Explicação mínima final tem {len(expl_minima)} features (removidas: {remocoes}).")
    return expl_minima, remocoes

# [CLASSE REJEITADA] 06: BUSCA OTIMIZADA (A GENERAL)
# Esta função é a "general" que comanda a batalha para instâncias rejeitadas.
# Ela executa os dois caminhos de busca (evitar Classe 0 e evitar Classe 1)
# e, no final, compara os resultados para escolher a explicação menor.
def encontrar_explicacao_otimizada_para_rejeitada(
    modelo: Pipeline, 
    instance_df: pd.DataFrame, 
    expl_inicial_path1: List[str], 
    expl_inicial_path2: List[str], 
    X_train: pd.DataFrame, 
    t_plus: float, 
    t_minus: float, 
    class_names: List[str], 
    log_collector: List[str]
) -> Tuple[List[str], int, int, int, float, float]:
    
    # --- CAMINHO 1: Ordenação baseada em evitar a Classe 0 (premissa_ordenacao=1) ---
    start_path1 = time.perf_counter() #[TIME] adicionando variavel para medir tempo
    log_collector.append("\n==================== ESTRATÉGIA DE BUSCA 1 (Evitar Classe 0) ====================")
    expl_robusta_1, adicoes_1 = executar_fase_1_reforco_bidirecional(modelo, instance_df, expl_inicial_path1, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=1)
    expl_final_1, remocoes_1 = executar_fase_2_minimizacao_bidirecional(modelo, instance_df, expl_robusta_1, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=1)
    end_path1 = time.perf_counter() #[TIME] finzliando contagem do passo 1 e calculando
    duration_path1 = end_path1 - start_path1
    log_collector.append(f"   --> Resultado da Estratégia 1: Explicação com {len(expl_final_1)} features (calculado em {duration_path1:.4f}s).")

    # --- CAMINHO 2: Ordenação baseada em evitar a Classe 1 (premissa_ordenacao=0) ---
    start_path2 = time.perf_counter()
    log_collector.append("\n==================== ESTRATÉGIA DE BUSCA 2 (Evitar Classe 1) ====================")
    expl_robusta_2, adicoes_2 = executar_fase_1_reforco_bidirecional(modelo, instance_df, expl_inicial_path2, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=0)
    expl_final_2, remocoes_2 = executar_fase_2_minimizacao_bidirecional(modelo, instance_df, expl_robusta_2, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=0)
    end_path2 = time.perf_counter()
    duration_path2 = end_path2 - start_path2
    log_collector.append(f"   --> Resultado da Estratégia 2: Explicação com {len(expl_final_2)} features (calculado em {duration_path2:.4f}s).")

    log_collector.append("\n------------------------- COMPARAÇÃO E DECISÃO FINAL -------------------------")

    if len(expl_final_1) <= len(expl_final_2):
        log_collector.append("   -> ESCOLHIDO: Caminho 1 resultou na explicação menor ou de igual tamanho.")
        log_collector.append("==================== FIM DA BUSCA OTIMIZADA ====================\n")
        return expl_final_1, adicoes_1, remocoes_1, 1, duration_path1, duration_path2
    else:
        log_collector.append("   -> ESCOLHIDO: Caminho 2 resultou na explicação estritamente menor.")
        log_collector.append("==================== FIM DA BUSCA OTIMIZADA ====================\n")
        return expl_final_2, adicoes_2, remocoes_2, 0, duration_path1, duration_path2

#==============================================================================
# GERAÇÃO DE RELATÓRIO (Função Principal)
#==============================================================================

def gerar_relatorio_consolidado(modelo: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float, rejection_cost, test_size_atual, silent_mode = False):
    if not os.path.exists("report"):
        os.makedirs("report")

    caminho_relatorio = os.path.join("report",f"peab_comp_mat_{nome_dataset}.txt")
    
    scores_teste = modelo.decision_function(X_test)
    modelo_interno = modelo.named_steps['modelo']
    intercepto = modelo_interno.intercept_[0]
    pesos = modelo_interno.coef_[0]

    logs_rejeitadas = []
    logs_classificadas = []
    stats = {
        "tamanhos_expl_neg": [], "tamanhos_expl_pos": [], "tamanhos_expl_rej": [],
        "all_features_in_exps": [], 
        "adicoes_fase1": [], "remocoes_fase2": []
    }
    # [TIME] Listas para armazenar os tempos de cada classe
    times_pos = []
    times_neg = []
    times_rej = []
    times_rej_path1 = []
    times_rej_path2 = []

    print(f"Processando {len(X_test)} instâncias de teste para gerar o relatório...")
    
    # [TIME] Inicia o cronometro antes do loop principal
    start_time = time.perf_counter()

    for i in range(len(X_test)):
        # [TIME] Inicia um cronômetro para esta instância específica
        inst_start_time = time.perf_counter()
        log_instancia_atual = []
        inst_df = X_test.iloc[[i]]
        score = scores_teste[i]
        pred_class = modelo.predict(inst_df)[0]
        rejeitada = t_minus <= score <= t_plus
        
        pred_str = f"REJEITADA (Score: {score:.4f})" if rejeitada else f"CLASSE {pred_class} (Score: {score:.4f})"
        log_instancia_atual.append(f"--- INSTÂNCIA #{i} | Predição Original: {pred_str} | Classe Real: {class_names[y_test.iloc[i]]} ---")

        if rejeitada:
            log_instancia_atual.append("\n   --> Esta instância caiu na 'zona de rejeição'. Para explicá-la, o sistema testa duas estratégias de busca e escolhe a que gera a explicação mais curta.")
        else:
            log_instancia_atual.append("\n   --> O objetivo é encontrar a explicação mais simples que mantém a classificação original.")


        expl_final = []
        premissa_final = pred_class

        if rejeitada:
            expl_inicial_f1, expl_inicial_f2 = gerar_explicacao_inicial_rejeitada_formal(modelo, inst_df, X_train, t_plus, t_minus, log_instancia_atual)
            # [TIME] adicionando chamada para receber novos valores duration
            expl_final, adicoes, remocoes, premissa_vencedora, duration_p1, duration_p2 = encontrar_explicacao_otimizada_para_rejeitada(modelo, inst_df, expl_inicial_f1, expl_inicial_f2, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            # [TIME] ADICIONANDO TEMPO NAS NOVAS LISTAS
            times_rej_path1.append(duration_p1)
            times_rej_path2.append(duration_p2)

            stats['adicoes_fase1'].append(adicoes)
            stats['remocoes_fase2'].append(remocoes)
            premissa_final = premissa_vencedora
            stats['tamanhos_expl_rej'].append(len(expl_final))

        else: # Instância Classificada
            expl_inicial = one_explanation_formal(modelo, inst_df, X_train, t_plus, t_minus, pred_class, log_instancia_atual)
            expl_robusta, adicoes_classif = executar_fase_1_reforco_unidirecional(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            expl_final, remocoes_classif = executar_fase_2_minimizacao_unidirecional(modelo, inst_df, expl_robusta, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            
            if pred_class == 1:
                stats['tamanhos_expl_pos'].append(len(expl_final))
            else:
                stats['tamanhos_expl_neg'].append(len(expl_final))
            
            stats['adicoes_fase1'].append(adicoes_classif)
            stats['remocoes_fase2'].append(remocoes_classif)
            
        log_instancia_atual.append(f"\n  >> RESULTADO FINAL DA INSTÂNCIA #{i}:")
        log_instancia_atual.append(f"     - PI-EXPLICAÇÃO FINAL (Tamanho: {len(expl_final)}):")
        
        if expl_final:
            deltas_finais = calculate_deltas(modelo, inst_df, X_train, premis_class=premissa_final)
            for feat_explicacao in expl_final:
                nome_feat = feat_explicacao.split(' = ')[0]
                idx_feat = X_train.columns.get_loc(nome_feat)
                delta_val = deltas_finais[idx_feat]
                log_instancia_atual.append(f"       - {feat_explicacao}  (Delta: {delta_val:.4f})")
        else:
            log_instancia_atual.append("       - (Explicação vazia)")
        # [TIME] O tempo total da instância rejeitada é a soma dos dois caminhos
        inst_end_time = time.perf_counter()
        if rejeitada:
            logs_rejeitadas.append("\n".join(log_instancia_atual))
        else:
            logs_classificadas.append("\n".join(log_instancia_atual))
        
        stats['all_features_in_exps'].extend([feat.split(' = ')[0] for feat in expl_final])

                #seu comentario-> para o cronômetro da instância
        inst_end_time = time.perf_counter()
        inst_duration = inst_end_time - inst_start_time
        
        #seu comentario-> adiciona a duração à lista correta
        if rejeitada:
            times_rej.append(inst_duration)
        elif pred_class == 1:
            times_pos.append(inst_duration)
        else:
            times_neg.append(inst_duration)

    # [TIME] para a contagem e calcula o tempo
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    avg_duration_per_instance = total_duration / len(X_test) if len(X_test) > 0 else 0

    # [TIME] adiciona os tempos calculando.
    avg_time_pos = np.mean(times_pos) if times_pos else 0
    avg_time_neg = np.mean(times_neg) if times_neg else 0
    avg_time_rej = np.mean(times_rej) if times_rej else 0

    # Calcula as médias de tempo para cada caminho das instâncias rejeitadas
    avg_time_rej_path1 = np.mean(times_rej_path1) if times_rej_path1 else 0
    avg_time_rej_path2 = np.mean(times_rej_path2) if times_rej_path2 else 0
    if not silent_mode:

        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write ("============================================================================   ====\n")
            f.write("       RELATÓRIO DE ANÁLISE DE EXPLICAÇÕES ABDUTIVAS(MÉTODO FORMAL     BIDIRECIONAL)\n")
            f.write ("============================================================================   ====\n\n")

            f.write("[CONFIGURAÇÕES GERAIS E DO MODELO]\n\n")
            f.write(f"  - Dataset: {nome_dataset}\n")
            f.write(f"  - Total de Instâncias de Teste: {len(X_test)}\n")
            f.write(f"  - Tamanho do Conjunto de Teste: {test_size_atual:.0%}\n")
            f.write(f"  - Número de Features do Modelo: {len(X_train.columns)}\n")
            acuracia_geral = modelo.score(X_test, y_test)
            f.write(f"  - Acurácia do Modelo (teste, sem rejeição): {acuracia_geral:.2%}    \n")
            f.write(f"  - Thresholds de Rejeição: t+ = {t_plus:.4f}, t- = {t_minus:.4f} \n")
            f.write(f"  - Custo de Rejeição WR: {rejection_cost:.2f}\n")
            f.write(f"  - Intercepto do Modelo (w0): {intercepto:.4f}\n\n")
            f.write(f"  - Tempo Total de Geração das Explicações: {total_duration:.2f}  segundos\n")
            f.write(f"  - Tempo Médio por Instância: {avg_duration_per_instance:.4f}    segundos\n\n")



            f.write ("============================================================================   ====\n")
            f.write("                    ANÁLISE DETALHADA POR INSTÂNCIA\n")
            f.write ("============================================================================   ====\n\n")

            f.write ("----------------------------------------------------------------------------   ----\n")
            f.write("                         SEÇÃO A: INSTÂNCIAS REJEITADAS\n")
            f.write ("----------------------------------------------------------------------------   ----\n")
            f.write("\n\n".join(logs_rejeitadas))

            f.write ("\n\n------------------------------------------------------------------------   --------\n")
            f.write("                       SEÇÃO B: INSTÂNCIAS CLASSIFICADAS\n")
            f.write ("----------------------------------------------------------------------------   ----\n")
            f.write("\n\n".join(logs_classificadas))

            f.write ("\n\n========================================================================   ========\n")
            f.write("                         RESUMO ESTATÍSTICO GERAL\n")
            f.write ("============================================================================   ====\n\n")

            aceitas_mask = ~((scores_teste >= t_minus) & (scores_teste <= t_plus))
            taxa_rejeicao = 1 - np.mean(aceitas_mask)
            acuracia_com_rejeicao = modelo.score(X_test[aceitas_mask], y_test   [aceitas_mask]) if np.sum(aceitas_mask) > 0 else "N/A"

            f.write("[Métricas de Desempenho do Modelo]\n")
            f.write(f"  - Acurácia Geral (sem rejeição): {acuracia_geral:.2%}\n")
            f.write(f"  - Taxa de Rejeição no Teste: {taxa_rejeicao:.2%} ({len  (logs_rejeitadas)} de {len(X_test)} instâncias)\n")
            f.write(f"  - Acurácia com Opção de Rejeição (nas {np.sum(aceitas_mask)}    instâncias aceitas): {acuracia_com_rejeicao if isinstance  (acuracia_com_rejeicao, str) else f'{acuracia_com_rejeicao:.2%}'}\n\n")

            f.write("[Estatísticas do Tamanho das Explicações]\n")
            total_exps = len(X_test)
            nomes_para_stats = {1: "Positiva", 0: "Negativa"}
            for tipo_pred, lista, nome_classe in [(1, stats['tamanhos_expl_pos'],   nomes_para_stats.get(1)), (0, stats['tamanhos_expl_neg'], nomes_para_stats.get    (0)), ("Rejeitada", stats['tamanhos_expl_rej'], "REJEITADA")]:
                n = len(lista)
                perc = (n / total_exps * 100) if total_exps > 0 else 0
                f.write(f"  - Classe {nome_classe} ({n} instâncias - {perc:.1f}% do total)  :\n")
                if n > 0:
                    f.write(f"    - Tamanho Explicação (Min / Média / Max): {np.min (lista)} / {np.mean(lista):.2f} / {np.max(lista)}\n")
                    f.write(f"    - Tamanho Explicação (Média ± Desv. Padrão): {np.mean (lista):.2f} ± {np.std(lista):.2f}  (Min: {np.min(lista)}, Max: {np. max(lista)})\n")
            f.write("\n")

            f.write("[Análise de Importância de Features]\n")
            f.write("  - Top 10 Features Mais Frequentes em Todas as Explicações:\n")
            if stats['all_features_in_exps']:
                for feat, count in Counter(stats['all_features_in_exps']).most_common(10):
                    f.write(f"    - {feat}: {count} vezes\n")

            f.write("\n  - Top 10 Pesos (Coeficientes) do Modelo (por valor absoluto):\n")
            pesos_df = pd.DataFrame({'feature': X_train.columns, 'peso': pesos,     'abs_peso': np.abs(pesos)}).sort_values(by='abs_peso', ascending=False)
            for _, row in pesos_df.head(10).iterrows():
                f.write(f"    - {row['feature']:<25}: {row['peso']:.4f}\n")
            f.write("\n")

            f.write("[Análise do Processo de Geração da Explicação (Método Formal)]\n")
            total_exps_f = len(stats['adicoes_fase1'])
            if total_exps_f > 0:
                instancias_com_adicao = sum(1 for x in stats['adicoes_fase1'] if x > 0)
                f.write(f"  - Instâncias que precisaram de reforço na Fase 1:   {instancias_com_adicao} ({instancias_com_adicao/total_exps_f*100:.2f}%)   \n")
                if instancias_com_adicao > 0:
                    media_adicoes = np.mean([x for x in stats['adicoes_fase1'] if x > 0])
                    f.write(f"    - Média de features adicionadas (quando houve reforço):   {media_adicoes:.2f}\n")

                instancias_com_remocao = sum(1 for x in stats['remocoes_fase2'] if x > 0)
                f.write(f"  - Instâncias com remoção efetiva de features na Fase 2:     {instancias_com_remocao} ({instancias_com_remocao/total_exps_f*100:.2f}%)   \n")
                if instancias_com_remocao > 0:
                    media_remocoes = np.mean([x for x in stats['remocoes_fase2'] if x >     0])
                    f.write(f"    - Média de features removidas (quando houve remoção):     {media_remocoes:.2f}\n")
            f.write("\n")

            # [TIME] custo computacional .
            f.write("\n  - Custo Computacional por Classe:\n")
            f.write(f"    - Tempo Médio (Positivas): {avg_time_pos:.4f} segundos\n")
            f.write(f"    - Tempo Médio (Negativas): {avg_time_neg:.4f} segundos\n")
            f.write(f"    - Tempo Médio (Rejeitadas): {avg_time_rej:.4f} segundos\n")
            f.write(f"        - Tempo Médio (Rej. - Apenas Caminho 1):  {avg_time_rej_path1:.4f} segundos\n")
            f.write(f"        - Tempo Médio (Rej. - Apenas Caminho 2):  {avg_time_rej_path2:.4f} segundos\n")        
            f.write("\n")

    # Calcula as métricas que precisamos para o resumo de estabilidade
    aceitas_mask = ~((scores_teste >= t_minus) & (scores_teste <= t_plus))
    taxa_rejeicao = 1 - np.mean(aceitas_mask)
    acuracia_com_rejeicao = modelo.score(X_test[aceitas_mask], y_test[aceitas_mask]) if np.sum(aceitas_mask) > 0 else 1.0

    tamanhos_explicacao_geral = stats['tamanhos_expl_pos'] + stats['tamanhos_expl_neg'] + stats['tamanhos_expl_rej']
    media_explicacao_geral = np.mean(tamanhos_explicacao_geral) if tamanhos_explicacao_geral else 0

    metricas_da_rodada = {
        'acuracia_com_rejeicao': acuracia_com_rejeicao,
        'taxa_rejeicao_teste': taxa_rejeicao,
        'media_explicacao_geral': media_explicacao_geral,
        'tempo_medio_instancia': avg_duration_per_instance
    }
    
    print(f"\nRelatório final salvo em: {caminho_relatorio}")
    return metricas_da_rodada

def main():
    """
    Função principal que orquestra a análise de estabilidade e a geração
    do relatório final detalhado.
    """
    global WR_REJECTION_COST
    todos_hiperparametros = carregar_hiperparametros()

    selecao_result = selecionar_dataset_e_classe()
    if not selecao_result or selecao_result[0] is None:
        print("Nenhum dataset selecionado. Encerrando.")
        return
    
    nome_dataset, nome_classe_positiva, X_data, y_data, nomes_classes = selecao_result

    # --- ETAPA 1: ANÁLISE DE ESTABILIDADE (LOOP) ---
    print("\n==========================================================")
    print("           INICIANDO ANÁLISE DE ESTABILIDADE")
    print("==========================================================")
    
    resultados_das_rodadas = []
    for seed in RANDOM_SEEDS:
        print(f"\n--- Executando rodada com Seed: {seed} ---")
        
        # Pega as configurações para o dataset atual
        config_experimento = DATASET_CONFIG.get(nome_dataset, {})
        test_size_atual = config_experimento.get('test_size', 0.3)
        rejection_cost_atual = config_experimento.get('rejection_cost', 0.25)
        
        WR_REJECTION_COST = rejection_cost_atual
        
        # Pega os parâmetros do modelo
        parametros_para_modelo = DEFAULT_LOGREG_PARAMS
        config_do_modelo = todos_hiperparametros.get(nome_dataset, {})
        if 'params' in config_do_modelo:
            parametros_para_modelo = config_do_modelo['params']

        pipeline_modelo = Pipeline([
            ('scaler', MinMaxScaler()),
            ('modelo', LogisticRegression(**parametros_para_modelo, random_state=seed))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size_atual, random_state=seed, stratify=y_data
        )
        
        pipeline_modelo.fit(X_train, y_train)
        t_plus, t_minus = calcular_thresholds(pipeline_modelo, X_train, y_train)
        
        # Chama a função em "modo silencioso" para apenas coletar as métricas
        metricas = gerar_relatorio_consolidado(
            pipeline_modelo, X_test, y_test, X_train, y_train, nomes_classes, 
            nome_dataset, t_plus, t_minus, WR_REJECTION_COST, test_size_atual, 
            silent_mode=True # <-- Importante: não salva o arquivo aqui
        )
        resultados_das_rodadas.append(metricas)

    # --- ETAPA 2: IMPRIMIR RELATÓRIO RESUMIDO DE ESTABILIDADE ---
    print("\n==========================================================")
    print(f"        RELATÓRIO DE ESTABILIDADE ({len(RANDOM_SEEDS)} RODADAS)")
    print("==========================================================")
    
    if resultados_das_rodadas:
        acuracias = [res['acuracia_com_rejeicao'] * 100 for res in resultados_das_rodadas]
        rejeicoes = [res['taxa_rejeicao_teste'] * 100 for res in resultados_das_rodadas]
        tamanhos = [res['media_explicacao_geral'] for res in resultados_das_rodadas]
        
        print(f"\nDataset: {nome_dataset}")
        print("-" * 58)
        print(f"Acurácia c/ Rejeição (Média ± Desv. Padrão): {np.mean(acuracias):.2f}% ± {np.std(acuracias):.2f}%")
        print(f"Taxa de Rejeição (Média ± Desv. Padrão):      {np.mean(rejeicoes):.2f}% ± {np.std(rejeicoes):.2f}%")
        print(f"Tamanho Médio da Explicação (Média ± Desv. Padrão): {np.mean(tamanhos):.2f} ± {np.std(tamanhos):.2f}")
        print("=" * 58)

    # --- ETAPA 3: GERAR RELATÓRIO PRINCIPAL DETALHADO ---
    print(f"\nGerando relatório principal detalhado com a semente oficial ({RANDOM_SEEDS[0]})...")
    
    # Roda o experimento mais uma vez com a primeira semente da lista
    semente_oficial = RANDOM_SEEDS[0]
    config_oficial = DATASET_CONFIG.get(nome_dataset, {})
    test_size_oficial = config_oficial.get('test_size', 0.3)
    rejection_cost_oficial = config_oficial.get('rejection_cost', 0.25)
    
    WR_REJECTION_COST = rejection_cost_oficial
    
    pipeline_oficial = Pipeline([
        ('scaler', MinMaxScaler()),
        ('modelo', LogisticRegression(**parametros_para_modelo, random_state=semente_oficial))
    ])
    
    X_train_oficial, X_test_oficial, y_train_oficial, y_test_oficial = train_test_split(
        X_data, y_data, test_size=test_size_oficial, random_state=semente_oficial, stratify=y_data
    )
    
    pipeline_oficial.fit(X_train_oficial, y_train_oficial)
    t_plus_oficial, t_minus_oficial = calcular_thresholds(pipeline_oficial, X_train_oficial, y_train_oficial)
    
    # Chama a função no modo normal (NÃO silencioso) para salvar o arquivo
    gerar_relatorio_consolidado(
        pipeline_oficial, X_test_oficial, y_test_oficial, X_train_oficial, y_train_oficial, nomes_classes, 
        nome_dataset, t_plus_oficial, t_minus_oficial, WR_REJECTION_COST, test_size_oficial,
        silent_mode=False # <-- Importante: agora salva o arquivo
    )
    
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()