import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any

from datasets.datasets import selecionar_dataset_e_classe

# INDICES DOS COMENTÁRIOS
# [TÓPICO GERAL] - Comentários sobre a lógica geral e conceitos dos artigos
# [CLASSE POS/NEG] - Lógica específica para instâncias classificadas (positivas ou negativas)
# [CLASSE REJEITADA] - Lógica específica para instâncias rejeitadas
# [REJEITADA HEURÍSTICO] - 6 partes - passo-a-paso do modo heuristico calculo inicial das rejeitadas (método simples)
# [REJEITADA FORMAL] - Passo a passo do modo formal de iniciação (método do artigo)

#==============================================================================
# CONSTANTES GLOBAIS
#==============================================================================
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
WR_REJECTION_COST: float = 0.24
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

def calcular_thresholds(modelo: LogisticRegression, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[float, float]:
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
    
    log_collector.append(f"   [Fase 0 - Formal] Calculando explicação com premissa para Classe {premis_class}:")
    log_collector.append(f"     - Pior Score Possível (Score Base): {score_base:.4f}")
    # [REJEITADA FORMAL] - 2. Define o alvo a ser atingido. Não é o score da instância, mas o limiar da zona de rejeição (t+ ou t-).
    target_score = t_plus if premis_class == 1 else t_minus
    log_collector.append(f"     - Objetivo: Score cumulativo {' >' if premis_class == 1 else ' <'} {target_score:.4f}")
    # [REJEITADA FORMAL] - 3. Loop para "pagar a dívida". Adiciona features de maior impacto e soma seus deltas (com sinal) ao score_base.
    for i in indices_ordenados:
        # [REJEITADA FORMAL] - 4. Condição de parada: o loop para se o score cumulativo cruzar o alvo.
        if (premis_class == 1 and soma_deltas_cumulativa > target_score and explicacao) or \
           (premis_class == 0 and soma_deltas_cumulativa < target_score and explicacao):
            break

        feature_nome = X_train.columns[i]
        soma_deltas_cumulativa += deltas[i]
        explicacao.append(f"{feature_nome} = {instance_df.iloc[0, i]:.4f}")
        log_collector.append(f"       - Adicionando '{feature_nome}'. Delta = {deltas[i]:.4f}. Score cumulativo = {soma_deltas_cumulativa:.4f}.")

    if not explicacao and len(deltas) > 0:
        explicacao.append(f"{X_train.columns[indices_ordenados[0]]} = {instance_df.iloc[0, indices_ordenados[0]]:.4f}")
    # [REJEITADA FORMAL] - 5. Retorna a explicação inicial, que pode ter várias features se a "dívida" for grande
    log_collector.append(f"   -> Explicação Inicial Formal Gerada ({len(explicacao)} feats): {explicacao}")
    return explicacao

# [CLASSE REJEITADA] 02: MÉTODO DE INICIAÇÃO HEURÍSTICO (SIMPLES)
# Esta função gera uma explicação inicial para rejeitadas usando a heurística simples.
# O objetivo é apenas que a soma dos |deltas| ultrapasse o |score| da instância.
# É um método mais rápido, mas menos "inteligente" que o formal.
def gerar_explicacao_inicial_rejeitada_heuristica(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, log_collector: List[str]):
    log_collector.append("   [Fase 0 - Heurística] Calculando explicação inicial para Rejeitada:")
    score = modelo.decision_function(instance_df)[0] # [REJEITADA HEURÍSTICO] - 1. Calcula o score da instância (antes de qualquer perturbação)
    deltas = calculate_deltas(modelo, instance_df, X_train, 1) # [REJEITADA HEURÍSTICO] - 2. Calcula os deltas para todas as features (usando premis_class=1 como referência) 
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []
    
    soma_deltas_cumulativa = 0.0
    # [REJEITADA HEURÍSTICO] - 3. a baixo define o alvo a ser atingido: o valor absoluto do score da instância. É um alvo baixo e fácil de alcançar.
    target_delta_sum = abs(score)
    log_collector.append(f"     - Objetivo (Heurístico): Soma de |deltas| >= |score| ({target_delta_sum:.4f})")
    # [REJEITADA HEURÍSTICO] - 4. Loop para atingir o alvo. Adiciona features de maior impacto (maior |delta|).
    for i in indices_ordenados:
        # [REJEITADA HEURÍSTICO] - 5. Condição de parada: o loop para assim que a soma dos |deltas| ultrapassa o alvo
        if soma_deltas_cumulativa > target_delta_sum and explicacao:
            break
        feature_nome = X_train.columns[i]
        soma_deltas_cumulativa += abs(deltas[i])
        explicacao.append(f"{feature_nome} = {instance_df.iloc[0, i]:.4f}")
    # [REJEITADA HEURÍSTICO] - 6. Retorna a explicação inicial, que geralmente é pequena (frequentemente 1 feature).
    log_collector.append(f"   -> Explicação Inicial Heurística Gerada ({len(explicacao)} feats): {explicacao}")
    return explicacao, explicacao

# [CLASSE REJEITADA] 03: GERAÇÃO DAS DUAS INICIAÇÕES FORMAIS
# Esta função orquestra a geração das duas explicações iniciais formais para uma instância rejeitada.
# Uma é gerada com a premissa de evitar a Classe 0, e a outra com a premissa de evitar a Classe 1.
# Essas duas explicações servirão de ponto de partida para os dois caminhos da busca otimizada.
def gerar_explicacao_inicial_rejeitada_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, log_collector: List[str]):
    log_collector.append("\n     --- Gerando start para Caminho 1 (premissa: Classe 1) ---")
    expl_inicial_path1 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 1, log_collector)

    log_collector.append("\n     --- Gerando start para Caminho 2 (premissa: Classe 0) ---")
    expl_inicial_path2 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 0, log_collector)

    return expl_inicial_path1, expl_inicial_path2

# [TÓPICO GERAL] 02: VALIDAÇÃO DE ROBUSTEZ (O JUIZ)
# Esta é a função mais importante para garantir a corretude. Ela testa se uma explicação é robusta.
# Para isso, ela perturba todas as features que NÃO estão na explicação para o seu pior cenário possível
# e verifica se, mesmo assim, a predição original se mantém (ou se a instância permanece na rejeição).
def perturbar_e_validar_com_log(modelo: LogisticRegression, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int) -> Tuple[str, str, List[str]]:
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
def executar_fase_1_reforco_unidirecional(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
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
def executar_fase_2_minimizacao_unidirecional(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
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
) -> Tuple[List[str], int, int, int]:
    log_collector.append("\n==================== INÍCIO DA BUSCA OTIMIZADA PARA REJEITADA ====================")
    
    # --- CAMINHO 1: Ordenação baseada em evitar a Classe 0 (premissa_ordenacao=1) ---
    log_collector.append("\n--- [CAMINHO 1] Testando com ordenação para evitar a CLASSE 0 ---")
    log_collector.append(f"   (Usando explicação inicial com {len(expl_inicial_path1)} feats: {expl_inicial_path1})")
    expl_robusta_1, adicoes_1 = executar_fase_1_reforco_bidirecional(modelo, instance_df, expl_inicial_path1, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=1)
    expl_final_1, remocoes_1 = executar_fase_2_minimizacao_bidirecional(modelo, instance_df, expl_robusta_1, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=1)
    log_collector.append(f"   -> Resultado do Caminho 1: {len(expl_final_1)} features. Adições: {adicoes_1}, Remoções: {remocoes_1}")

    # --- CAMINHO 2: Ordenação baseada em evitar a Classe 1 (premissa_ordenacao=0) ---
    log_collector.append("\n--- [CAMINHO 2] Testando com ordenação para evitar a CLASSE 1 ---")
    log_collector.append(f"   (Usando explicação inicial com {len(expl_inicial_path2)} feats: {expl_inicial_path2})")
    expl_robusta_2, adicoes_2 = executar_fase_1_reforco_bidirecional(modelo, instance_df, expl_inicial_path2, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=0)
    expl_final_2, remocoes_2 = executar_fase_2_minimizacao_bidirecional(modelo, instance_df, expl_robusta_2, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=0)
    log_collector.append(f"   -> Resultado do Caminho 2: {len(expl_final_2)} features. Adições: {adicoes_2}, Remoções: {remocoes_2}")

    log_collector.append("\n------------------------- COMPARAÇÃO FINAL -------------------------")
    log_collector.append(f"Tamanho da Explicação - Caminho 1: {len(expl_final_1)}")
    log_collector.append(f"Tamanho da Explicação - Caminho 2: {len(expl_final_2)}")

    if len(expl_final_1) <= len(expl_final_2):
        log_collector.append("   -> ESCOLHIDO: Caminho 1 resultou na explicação menor ou de igual tamanho.")
        log_collector.append("==================== FIM DA BUSCA OTIMIZADA ====================\n")
        return expl_final_1, adicoes_1, remocoes_1, 1
    else:
        log_collector.append("   -> ESCOLHIDO: Caminho 2 resultou na explicação estritamente menor.")
        log_collector.append("==================== FIM DA BUSCA OTIMIZADA ====================\n")
        return expl_final_2, adicoes_2, remocoes_2, 0

#==============================================================================
# GERAÇÃO DE RELATÓRIO (Função Principal)
#==============================================================================

def gerar_relatorio_consolidado(modelo: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float):
    
    caminho_relatorio = os.path.join("report",f"PEAB_{nome_dataset}.txt")
    
    scores_teste = modelo.decision_function(X_test)
    modelo_interno = modelo.named_steps['modelo']
    intercepto = modelo_interno.intercept_[0]
    pesos = modelo_interno.coef_[0]

    logs_rejeitadas = []
    logs_classificadas = []
    stats = {
        "tamanhos_expl_neg": [], "tamanhos_expl_pos": [], "tamanhos_expl_rej": [],
        "all_features_in_exps": [], 
        "adicoes_fase1_h": [], "remocoes_fase2_h": [],
        "adicoes_fase1_f": [], "remocoes_fase2_f": []
    }

    print(f"Processando {len(X_test)} instâncias de teste para gerar o relatório...")
    
    for i in range(len(X_test)):
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
            #log_instancia_atual.append("\n\n<<<<<<<<<<<<<<<< MÉTODO 1: INICIAÇÃO HEURÍSTICA >>>>>>>>>>>>>>>>")
            expl_inicial_h1, expl_inicial_h2 = gerar_explicacao_inicial_rejeitada_heuristica(modelo, inst_df, X_train, log_instancia_atual)
            expl_final_h, adicoes_h, remocoes_h, _ = encontrar_explicacao_otimizada_para_rejeitada(modelo, inst_df, expl_inicial_h1, expl_inicial_h2, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            stats['adicoes_fase1_h'].append(adicoes_h)
            stats['remocoes_fase2_h'].append(remocoes_h)

            log_instancia_atual.append("\n\n<<<<<<<<<<<<<<<< MÉTODO 2: INICIAÇÃO FORMAL >>>>>>>>>>>>>>>>")
            expl_inicial_f1, expl_inicial_f2 = gerar_explicacao_inicial_rejeitada_formal(modelo, inst_df, X_train, t_plus, t_minus, log_instancia_atual)
            expl_final_f, adicoes_f, remocoes_f, premissa_vencedora_f = encontrar_explicacao_otimizada_para_rejeitada(modelo, inst_df, expl_inicial_f1, expl_inicial_f2, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            stats['adicoes_fase1_f'].append(adicoes_f)
            stats['remocoes_fase2_f'].append(remocoes_f)
            
            expl_final = expl_final_f
            premissa_final = premissa_vencedora_f
            
            stats['tamanhos_expl_rej'].append(len(expl_final))

            log_instancia_atual.append("\n------------------- RESUMO DA COMPARAÇÃO DE EFICIÊNCIA -------------------")
            log_instancia_atual.append(f"  - Iniciação Heurística: {adicoes_h} adições, {remocoes_h} remoções -> Explicação final com {len(expl_final_h)} features.")
            log_instancia_atual.append(f"  - Iniciação Formal:     {adicoes_f} adições, {remocoes_f} remoções -> Explicação final com {len(expl_final_f)} features.")
            if adicoes_f < adicoes_h:
                log_instancia_atual.append("  - Veredito: O método Formal foi MAIS EFICIENTE (menos passos de reforço).")
            elif adicoes_f == adicoes_h:
                 log_instancia_atual.append("  - Veredito: Ambos os métodos tiveram a mesma eficiência.")
            else:
                log_instancia_atual.append("  - Veredito: O método Heurístico foi mais eficiente neste caso.")
            log_instancia_atual.append("--------------------------------------------------------------------------")

        else: # Instância Classificada
            expl_inicial = one_explanation_formal(modelo, inst_df, X_train, t_plus, t_minus, pred_class, log_instancia_atual)
            expl_robusta, adicoes_classif = executar_fase_1_reforco_unidirecional(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            expl_final, remocoes_classif = executar_fase_2_minimizacao_unidirecional(modelo, inst_df, expl_robusta, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            
            if pred_class == 1:
                stats['tamanhos_expl_pos'].append(len(expl_final))
            else:
                stats['tamanhos_expl_neg'].append(len(expl_final))
            
            stats['adicoes_fase1_f'].append(adicoes_classif)
            stats['remocoes_fase2_f'].append(remocoes_classif)
            
        log_instancia_atual.append(f"\n  >> RESULTADO FINAL DA INSTÂNCIA #{i} (Baseado no Método Formal):")
        
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
        
        if rejeitada:
            logs_rejeitadas.append("\n".join(log_instancia_atual))
        else:
            logs_classificadas.append("\n".join(log_instancia_atual))
        
        stats['all_features_in_exps'].extend([feat.split(' = ')[0] for feat in expl_final])

    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("        RELATÓRIO DE ANÁLISE DE PI-EXPLICAÇÕES (MÉTODO BIDIRECIONAL)\n")
        f.write("================================================================================\n\n")

        f.write("[CONFIGURAÇÕES GERAIS E DO MODELO]\n\n")
        f.write(f"  - Dataset: {nome_dataset}\n")
        f.write(f"  - Total de Instâncias de Teste: {len(X_test)}\n")
        f.write(f"  - Número de Features do Modelo: {len(X_train.columns)}\n")
        acuracia_geral = modelo.score(X_test, y_test)
        f.write(f"  - Acurácia do Modelo (teste, sem rejeição): {acuracia_geral:.2%}\n")
        f.write(f"  - Thresholds de Rejeição: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n")
        f.write(f"  - Intercepto do Modelo (w0): {intercepto:.4f}\n\n")

        f.write("================================================================================\n")
        f.write("                    ANÁLISE DETALHADA POR INSTÂNCIA\n")
        f.write("================================================================================\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("                         SEÇÃO A: INSTÂNCIAS REJEITADAS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("\n\n".join(logs_rejeitadas))

        f.write("\n\n--------------------------------------------------------------------------------\n")
        f.write("                       SEÇÃO B: INSTÂNCIAS CLASSIFICADAS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("\n\n".join(logs_classificadas))

        f.write("\n\n================================================================================\n")
        f.write("                         RESUMO ESTATÍSTICO GERAL\n")
        f.write("================================================================================\n\n")

        aceitas_mask = ~((scores_teste >= t_minus) & (scores_teste <= t_plus))
        taxa_rejeicao = 1 - np.mean(aceitas_mask)
        acuracia_com_rejeicao = modelo.score(X_test[aceitas_mask], y_test[aceitas_mask]) if np.sum(aceitas_mask) > 0 else "N/A"
        
        f.write("[Métricas de Desempenho do Modelo]\n")
        f.write(f"  - Acurácia Geral (sem rejeição): {acuracia_geral:.2%}\n")
        f.write(f"  - Taxa de Rejeição no Teste: {taxa_rejeicao:.2%} ({len(logs_rejeitadas)} de {len(X_test)} instâncias)\n")
        f.write(f"  - Acurácia com Opção de Rejeição (nas {np.sum(aceitas_mask)} instâncias aceitas): {acuracia_com_rejeicao if isinstance(acuracia_com_rejeicao, str) else f'{acuracia_com_rejeicao:.2%}'}\n\n")

        f.write("[Estatísticas do Tamanho das Explicações]\n")
        total_exps = len(X_test)
        nomes_para_stats = {1: "Positiva", 0: "Negativa"}
        for tipo_pred, lista, nome_classe in [(1, stats['tamanhos_expl_pos'], nomes_para_stats.get(1)), (0, stats['tamanhos_expl_neg'], nomes_para_stats.get(0)), ("Rejeitada", stats['tamanhos_expl_rej'], "REJEITADA")]:
            n = len(lista)
            perc = (n / total_exps * 100) if total_exps > 0 else 0
            f.write(f"  - Classe {nome_classe} ({n} instâncias - {perc:.1f}% do total):\n")
            if n > 0:
                f.write(f"    - Tamanho Explicação (Min / Média / Max): {np.min(lista)} / {np.mean(lista):.2f} / {np.max(lista)}\n")
        f.write("\n")

        f.write("[Análise de Importância de Features]\n")
        f.write("  - Top 10 Features Mais Frequentes em Todas as Explicações:\n")
        if stats['all_features_in_exps']:
            for feat, count in Counter(stats['all_features_in_exps']).most_common(10):
                f.write(f"    - {feat}: {count} vezes\n")
        
        f.write("\n  - Top 10 Pesos (Coeficientes) do Modelo (por valor absoluto):\n")
        pesos_df = pd.DataFrame({'feature': X_train.columns, 'peso': pesos, 'abs_peso': np.abs(pesos)}).sort_values(by='abs_peso', ascending=False)
        for _, row in pesos_df.head(10).iterrows():
            f.write(f"    - {row['feature']:<25}: {row['peso']:.4f}\n")
        f.write("\n")
        
        f.write("[Análise do Processo de Geração da Explicação (Método Formal)]\n")
        total_exps_f = len(stats['adicoes_fase1_f'])
        if total_exps_f > 0:
            instancias_com_adicao = sum(1 for x in stats['adicoes_fase1_f'] if x > 0)
            f.write(f"  - Instâncias que precisaram de reforço na Fase 1: {instancias_com_adicao} ({instancias_com_adicao/total_exps_f*100:.2f}%)\n")
            if instancias_com_adicao > 0:
                media_adicoes = np.mean([x for x in stats['adicoes_fase1_f'] if x > 0])
                f.write(f"    - Média de features adicionadas (quando houve reforço): {media_adicoes:.2f}\n")
            
            instancias_com_remocao = sum(1 for x in stats['remocoes_fase2_f'] if x > 0)
            f.write(f"  - Instâncias com remoção efetiva de features na Fase 2: {instancias_com_remocao} ({instancias_com_remocao/total_exps_f*100:.2f}%)\n")
            if instancias_com_remocao > 0:
                media_remocoes = np.mean([x for x in stats['remocoes_fase2_f'] if x > 0])
                f.write(f"    - Média de features removidas (quando houve remoção): {media_remocoes:.2f}\n")
        
        f.write("\n[Análise de Eficiência dos Métodos de Iniciação para Rejeitadas]\n")
        if len(stats['adicoes_fase1_h']) > 0:
            media_adicoes_h = np.mean(stats['adicoes_fase1_h'])
            adicoes_f_rej = [stats['adicoes_fase1_f'][i] for i, r in enumerate(X_test.index) if t_minus <= scores_teste[i] <= t_plus]
            
            if adicoes_f_rej:
                media_adicoes_f_rej = np.mean(adicoes_f_rej)
                f.write(f"  - Média de Adições (Reforço) com Iniciação Heurística: {media_adicoes_h:.2f}\n")
                f.write(f"  - Média de Adições (Reforço) com Iniciação Formal: {media_adicoes_f_rej:.2f}\n")
                if media_adicoes_f_rej < media_adicoes_h:
                    f.write("  - Conclusão Geral: O método de iniciação Formal demonstrou ser mais eficiente.\n\n")
                else:
                    f.write("  - Conclusão Geral: Não houve ganho de eficiência significativo com o método Formal.\n\n")
            else:
                 f.write("  - Não foi possível calcular a média para o método Formal em instâncias rejeitadas.\n\n")
        else:
            f.write("  - Nenhuma instância rejeitada para comparar a eficiência dos métodos.\n\n")

    print(f"\nRelatório final salvo em: {caminho_relatorio}")


def main():
    """
    Função principal que orquestra o carregamento de dados, treinamento do modelo
    e geração do relatório de PI-Explicação usando uma Pipeline.
    """
    todos_hiperparametros = carregar_hiperparametros()

    selecao_result = selecionar_dataset_e_classe()
    if not selecao_result:
        print("Nenhum dataset selecionado. Encerrando.")
        return
    
    nome_dataset, _, X_data, y_data, nomes_classes = selecao_result

    config_do_dataset = todos_hiperparametros.get(nome_dataset)
    parametros_para_modelo = DEFAULT_LOGREG_PARAMS

    if config_do_dataset and 'params' in config_do_dataset:
        parametros_para_modelo = config_do_dataset['params']
        print(f"\nUsando hiperparâmetros otimizados para '{nome_dataset}':")
        print(f"  Parâmetros: {parametros_para_modelo}")
    else:
        print(f"\nAVISO: Parâmetros para '{nome_dataset}' não encontrados. Usando modelo padrão.")
        print(f"  Parâmetros: {parametros_para_modelo}")

    pipeline_modelo = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', LogisticRegression(**parametros_para_modelo, random_state=RANDOM_STATE))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_data
    )
    
    pipeline_modelo.fit(X_train, y_train)
    
    print(f"\nModelo treinado. Acurácia no teste (sem rejeição): {pipeline_modelo.score(X_test, y_test):.2%}")
    
    print("Calculando thresholds de rejeição...")
    t_plus, t_minus = calcular_thresholds(pipeline_modelo, X_train, y_train)
    print(f"Thresholds definidos: t+ = {t_plus:.4f}, t- = {t_minus:.4f}")
    
    gerar_relatorio_consolidado(
        pipeline_modelo, X_test, y_test, X_train, y_train, nomes_classes, nome_dataset, t_plus, t_minus
    )
    
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()
