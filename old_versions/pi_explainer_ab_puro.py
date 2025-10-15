import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

"""
Diferente do pi_explainer_additive_bidirecional, este código importa a função do dataset 
sem o tratamento no pima, onde remove as features impossíveis (com valores zxerados)
e ainda tem o conjundo inicial de datasets como iris e etc.
Ele também não implemanta o pipeline do SKLEARN que ajuda na robustez e organização evitando erros.
"""
# Importa a função de seleção do módulo de datasets
from datasets.datasets_basic import selecionar_dataset_e_classe

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

def calcular_thresholds(modelo: LogisticRegression, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[float, float]:
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

def calculate_deltas(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame, premis_class: int) -> np.ndarray:
    coefs = modelo.coef_[0]
    instance_vals = instance_df.iloc[0].values
    deltas = np.zeros_like(coefs)
    X_train_min, X_train_max = X_train.min().values, X_train.max().values
    for i, (coef, val) in enumerate(zip(coefs, instance_vals)):
        if premis_class == 1: # Pior caso para classe 1 é virar classe 0
            pior_valor = X_train_min[i] if coef > 0 else X_train_max[i]
        else: # premis_class == 0. Pior caso para classe 0 é virar classe 1
            pior_valor = X_train_max[i] if coef > 0 else X_train_min[i]
        deltas[i] = (val - pior_valor) * coef
    return deltas

def one_explanation(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, log_collector: List[str]):
    score = modelo.decision_function(instance_df)[0]
    pred_class = modelo.predict(instance_df)[0]
    # Para one_explanation, o cálculo de deltas pode usar a predição original como premissa
    deltas = calculate_deltas(modelo, instance_df, X_train, pred_class)
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []
    
    is_rejected = t_minus <= score <= t_plus

    log_collector.append(f"   [Fase 0] Calculando explicação inicial:")
    log_collector.append(f"     - Score da Instância: {score:.4f}, Predição: {'Rejeitada' if is_rejected else 'Classe ' + str(pred_class)}")
    
    if is_rejected:
        soma_deltas_cumulativa = 0.0
        target_delta_sum = abs(score) # Heurística simples para rejeitadas
        log_collector.append(f"     - Objetivo (Rejeitada): Soma de |deltas| >= |score| ({target_delta_sum:.4f})")
        
        for i in indices_ordenados:
            delta_atual = abs(deltas[i])
            if soma_deltas_cumulativa > target_delta_sum and explicacao:
                log_collector.append(f"       - Parando. A soma cumulativa ({soma_deltas_cumulativa:.4f}) já atingiu a meta.")
                break
            feature_nome = X_train.columns[i]
            soma_deltas_cumulativa += delta_atual
            explicacao.append(f"{feature_nome} = {instance_df.iloc[0, i]:.4f}")
            log_collector.append(f"       - Adicionando '{feature_nome}'. |Delta| = {delta_atual:.4f}. Soma cumulativa = {soma_deltas_cumulativa:.4f}.")
    else: # Lógica para CLASSIFICADOS
        score_base = score - np.sum(deltas)
        soma_deltas_cumulativa = score_base
        target_score = t_plus if pred_class == 1 else t_minus
        log_collector.append(f"     - Objetivo (Classe {pred_class}): Score cumulativo {' >=' if pred_class == 1 else ' <='} t{'+' if pred_class == 1 else '-'} ({target_score:.4f})")
        log_collector.append(f"     - Começando com Score Base: {score_base:.4f}")

        for i in indices_ordenados:
            if pred_class == 1 and soma_deltas_cumulativa > target_score and explicacao:
                log_collector.append(f"       - Parando. Score cumulativo ({soma_deltas_cumulativa:.4f}) atingiu a meta.")
                break
            if pred_class == 0 and soma_deltas_cumulativa < target_score and explicacao:
                log_collector.append(f"       - Parando. Score cumulativo ({soma_deltas_cumulativa:.4f}) atingiu a meta.")
                break

            feature_nome = X_train.columns[i]
            delta_atual_com_sinal = deltas[i]
            soma_deltas_cumulativa += delta_atual_com_sinal
            explicacao.append(f"{feature_nome} = {instance_df.iloc[0, i]:.4f}")
            log_collector.append(f"       - Adicionando '{feature_nome}'. Delta = {delta_atual_com_sinal:.4f}. Score cumulativo = {soma_deltas_cumulativa:.4f}.")

    if not explicacao and len(deltas) > 0:
        idx = indices_ordenados[0]
        explicacao.append(f"{X_train.columns[idx]} = {instance_df.iloc[0, idx]:.4f}")

    log_collector.append(f"   -> Explicação Inicial Gerada ({len(explicacao)} feats): {explicacao}")
    return explicacao

def formatar_calculo_score(modelo: LogisticRegression, instance_df: pd.DataFrame, feature_names: List[str]) -> str:
    pesos = modelo.coef_[0]
    intercepto = modelo.intercept_[0]
    valores = instance_df.iloc[0].values
    termos_str = [f"({pesos[i]:.4f} * {valores[i]:.4f})" for i, nome in enumerate(feature_names)]
    calculo_str = " + ".join(termos_str)
    score_final = np.dot(pesos, valores) + intercepto
    return f"      Cálculo do Score: {calculo_str} + ({intercepto:.4f}) = {score_final:.4f}"

def perturbar_e_validar_com_log(modelo: LogisticRegression, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int) -> Tuple[str, str, List[str]]:
    calc_log = [f"          (Log Perturbação: Explicação atual com {len(explicacao)} features fixas)"]
    inst_pert = instance_df.copy()
    features_explicacao = {f.split(' = ')[0] for f in explicacao}
    
    perturbar_para_diminuir_score = (direcao_override == 1)
    
    for feat_idx, feat_nome in enumerate(X_train.columns):
        if feat_nome in features_explicacao:
            continue
        
        coef = modelo.coef_[0][feat_idx]
        train_min = X_train[feat_nome].min()
        train_max = X_train[feat_nome].max()
        
        if perturbar_para_diminuir_score:
            valor_pert = train_min if coef > 0 else train_max
        else: # perturbar para aumentar
            valor_pert = train_max if coef > 0 else train_min
        
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
    
    # Gerar log detalhado da instância perturbada
    # calc_log.append(formatar_calculo_score(modelo, inst_pert, X_train.columns))
    
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
        
    pred_str = f"REJEITADA (Score: {score_pert:.4f})" if pert_rejeitada else f"{class_names[pred_pert]} (Score: {score_pert:.4f})"
    return status, pred_str, calc_log

#==============================================================================
# FUNÇÕES DE FASE (ADITIVA -> SUBTRATIVA)
#==============================================================================

def executar_fase_1_reforco_bidirecional(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo Bi-Direcional Otimizado")
    expl_robusta = list(expl_inicial)
    adicoes = 0
    
    # Deltas para ordenar a adição de features. Usamos premissa 1 (evitar classe 0) como padrão.
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=1)
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
        
        if not adicionou_feature: # Segurança
             break
    
    log_collector.append(f"\n   -> Fim da Fase 1. Explicação robusta final tem {len(expl_robusta)} features.")
    return expl_robusta, adicoes

def executar_fase_2_minimizacao_bidirecional(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 2] Início da Minimização Subtrativa Bi-Direcional Otimizada")
    expl_minima = list(expl_robusta)
    remocoes = 0

    # Deltas para ordenar a remoção. Usamos premissa 1 (evitar classe 0) como padrão.
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=1)
    
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

# Funções uniderecionais para instâncias classificadas
def executar_fase_1_reforco_unidirecional(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo Uni-Direcional")
    expl_robusta = list(expl_inicial)
    adicoes = 0
    pred_class = modelo.predict(instance_df)[0]
    direcao = 1 if pred_class == 1 else 0 # 1 para diminuir score, 0 para aumentar

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

#==============================================================================
# GERAÇÃO DE RELATÓRIO (Função Principal)
#==============================================================================

def gerar_relatorio_consolidado(modelo: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float):
    # --- MUDANÇA 1: Nome do arquivo ---
    caminho_relatorio = os.path.join("report",f"pi_explainer_ab_puro_{nome_dataset}.txt")
    
    scores_teste = modelo.decision_function(X_test)
    intercepto = modelo.intercept_[0]
    pesos = modelo.coef_[0]

    # --- MUDANÇA 2: Buffers para armazenar logs e reordenar a escrita ---
    logs_rejeitadas = []
    logs_classificadas = []
    stats = {
        "tamanhos_expl_neg": [], "tamanhos_expl_pos": [], "tamanhos_expl_rej": [],
        "all_features_in_exps": [], "adicoes_fase1": [], "remocoes_fase2": []
    }

    print(f"Processando {len(X_test)} instâncias de teste para gerar o relatório...")
    
    # --- MUDANÇA 3: Loop para coletar dados, sem escrever no arquivo ---
    for i in range(len(X_test)):
        log_instancia_atual = []
        inst_df = X_test.iloc[[i]]
        score = scores_teste[i]
        pred_class = modelo.predict(inst_df)[0]
        rejeitada = t_minus <= score <= t_plus
        
        pred_str = f"REJEITADA (Score: {score:.4f})" if rejeitada else f"CLASSE {pred_class} (Score: {score:.4f})"
        log_instancia_atual.append(f"--- INSTÂNCIA #{i} | Predição Original: {pred_str} | Classe Real: {class_names[y_test.iloc[i]]} ---")

        expl_inicial = one_explanation(modelo, inst_df, X_train, t_plus, t_minus, log_collector=log_instancia_atual)
        
        adicoes, remocoes = 0, 0
        
        if rejeitada:
            expl_robusta, adicoes = executar_fase_1_reforco_bidirecional(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            expl_final, remocoes = executar_fase_2_minimizacao_bidirecional(modelo, inst_df, expl_robusta, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            stats['tamanhos_expl_rej'].append(len(expl_final))
        else: # Instância Classificada
            expl_robusta, adicoes = executar_fase_1_reforco_unidirecional(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            expl_final, remocoes = executar_fase_2_minimizacao_unidirecional(modelo, inst_df, expl_robusta, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            if pred_class == 1:
                stats['tamanhos_expl_pos'].append(len(expl_final))
            else:
                stats['tamanhos_expl_neg'].append(len(expl_final))

        # --- MUDANÇA 4: Adiciona o bloco de resumo da instância no log ---
        resumo_processo = f"Explicação inicial com {len(expl_inicial)} features. "
        if adicoes > 0:
            resumo_processo += f"{adicoes} feature(s) adicionada(s) na Fase 1. "
        else:
            resumo_processo += "Explicação inicial já era robusta. "
        if remocoes > 0:
            resumo_processo += f"{remocoes} feature(s) removida(s) na Fase 2."
        else:
             resumo_processo += "Nenhuma feature pôde ser removida."
            
        log_instancia_atual.append(f"\n  >> RESULTADO FINAL DA INSTÂNCIA #{i}:")
        log_instancia_atual.append(f"     - PI-EXPLICAÇÃO FINAL: {expl_final}")
        log_instancia_atual.append(f"     - Tamanho da Explicação: {len(expl_final)}")
        log_instancia_atual.append(f"     - Resumo do Processo: {resumo_processo}")

        # Armazena o log completo no buffer apropriado
        if rejeitada:
            logs_rejeitadas.append("\n".join(log_instancia_atual))
        else:
            logs_classificadas.append("\n".join(log_instancia_atual))
        
        # Coleta estatísticas
        stats['all_features_in_exps'].extend([feat.split(' = ')[0] for feat in expl_final])
        stats['adicoes_fase1'].append(adicoes)
        stats['remocoes_fase2'].append(remocoes)

    # --- MUDANÇA 5: Escrita final no arquivo, usando os buffers e o formato projetado ---
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
        # Ajuste para usar nomes de classes do array `class_names`
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
        
        f.write("[Análise do Processo de Geração da Explicação]\n")
        total_exps = len(stats['adicoes_fase1'])
        instancias_com_adicao = sum(1 for x in stats['adicoes_fase1'] if x > 0)
        f.write(f"  - Instâncias que precisaram de reforço na Fase 1: {instancias_com_adicao} ({instancias_com_adicao/total_exps*100:.2f}%)\n")
        if instancias_com_adicao > 0:
            ### LINHA CORRIGIDA AQUI ###
            media_adicoes = np.mean([x for x in stats['adicoes_fase1'] if x > 0])
            f.write(f"    - Média de features adicionadas (quando houve reforço): {media_adicoes:.2f}\n")
        
        instancias_com_remocao = sum(1 for x in stats['remocoes_fase2'] if x > 0)
        f.write(f"  - Instâncias com remoção efetiva de features na Fase 2: {instancias_com_remocao} ({instancias_com_remocao/total_exps*100:.2f}%)\n")
        if instancias_com_remocao > 0:
            media_remocoes = np.mean([x for x in stats['remocoes_fase2'] if x > 0])
            f.write(f"    - Média de features removidas (quando houve remoção): {media_remocoes:.2f}\n")

    print(f"\nRelatório final salvo em: {caminho_relatorio}")

def main():
    selecao_result = selecionar_dataset_e_classe() 
    if not selecao_result:
        print("Nenhum dataset selecionado. Encerrando.")
        return
    
    nome_dataset, _, X_data, y_data, nomes_classes = selecao_result
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_data)
    
    modelo = LogisticRegression(**DEFAULT_LOGREG_PARAMS)
    modelo.fit(X_train, y_train)
    
    print(f"\nModelo treinado. Acurácia no teste (sem rejeição): {modelo.score(X_test, y_test):.2%}")
    
    print("Calculando thresholds de rejeição...")
    t_plus, t_minus = calcular_thresholds(modelo, X_train, y_train)
    print(f"Thresholds definidos: t+ = {t_plus:.4f}, t- = {t_minus:.4f}")
    
    gerar_relatorio_consolidado(modelo, X_test, y_test, X_train, y_train, nomes_classes, nome_dataset, t_plus, t_minus)
    
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()