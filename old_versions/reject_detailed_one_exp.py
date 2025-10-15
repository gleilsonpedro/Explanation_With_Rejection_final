"""
 ==============================================================================
                     DESCRIÇÃO DO ALGORITMO

 Este script implementa um método para gerar "PI-Explicações" para as predições
 de um classificador de Regressão Logística com opção de rejeição.
 Uma PI-Explicação (Explicação por Implicante Primo) é um conjunto de features
 que é, ao mesmo tempo, SUFICIENTE para garantir a predição (robusto) e
 MINIMAL (irredutível, ou seja, nenhum subconjunto dele é suficiente).

 O processo geral para cada instância segue três fases:
 1. Geração Inicial (`one_explanation`): Cria uma explicação heurística inicial.
 2. Reforço de Robustez (`executar_fase_1_reforco_aditivo`): Adiciona features
    iterativamente à explicação até que ela se torne robusta contra perturbações
    nos piores cenários.
 3. Minimização (`executar_fase_2_minimizacao_subtrativa`): Remove features da
    explicação robusta, uma a uma (da menos importante para a mais), para
    garantir que o resultado final seja minimal (irredutível).

 A lógica de análise difere entre instâncias classificadas e rejeitadas.

 ------------------------------------------------------------------------------
                 ANÁLISE PARA CLASSES POSITIVA E NEGATIVA
 ------------------------------------------------------------------------------
 Para instâncias com uma predição clara (positiva ou negativa), o script executa
 uma única "corrida" completa do processo de 3 fases.

 - Função de Validação: `perturbar_e_validar_com_log`.
 - Teste de Robustez: A robustez é avaliada de forma UNIDIRECIONAL.
   - Para a classe positiva (+1): O teste garante que a explicação impede que o
     score da instância caia abaixo do limiar de classificação `t_plus` quando
     as features livres são perturbadas para o pior caso.
   - Para a classe negativa (-1): O teste garante que a explicação impede que o
     score suba acima do limiar `t_minus`.

 O resultado é uma PI-Explicação que garante a classificação naquela direção específica.

 ------------------------------------------------------------------------------
                    ANÁLISE PARA A CLASSE DE REJEIÇÃO
 ------------------------------------------------------------------------------
 Para instâncias na zona de rejeição, a robustez precisa ser bidirecional,
 ou seja, a explicação deve garantir que o score não saia do intervalo
 [`t_minus`, `t_plus`].

 Este script utiliza uma HEURÍSTICA chamada "Duas Corridas Unidirecionais"
 para simplificar este problema:

 - Estratégia: Em vez de um teste bidirecional, o script executa duas análises
   completas e independentes, cada uma testando uma das fronteiras da rejeição.

   1. Corrida 1 (Teste vs. Classe 0): Encontra uma explicação minimal e robusta
      que impede o score de cair abaixo de `t_minus`.

   2. Corrida 2 (Teste vs. Classe 1): Encontra outra explicação minimal e robusta
      que impede o score de subir acima de `t_plus`.

 - Decisão Final: Ao final, o script compara a cardinalidade (o número de
   features) das duas explicações resultantes e define a MENOR delas como a
   explicação final para a instância rejeitada.

 - Sobre Minimalidade e Confiabilidade:
   - O processo de minimização na Fase 2 garante que a explicação de CADA CORRIDA
     seja **minimal** (irredutível). Nenhum item pode ser removido daquele conjunto
     sem que ele perca sua robustez (naquela direção testada).
   - O algoritmo não garante encontrar a explicação de **mínima cardinalidade**
     (a menor possível em termos absolutos), mas sim uma que é minimal.
   - **Ponto Crítico**: A explicação final escolhida para a classe de rejeição
     (a menor entre as duas corridas) só tem sua robustez garantida em UMA
     direção. Ela representa uma aproximação que favorece explicações mais
     curtas, mas não é formalmente uma PI-Explicação para o caso de rejeição,
     pois pode ser frágil à perturbação na direção "não vencedora".
 ==============================================================================
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

# Importa a função de seleção do módulo de datasets
#from datasets.datasets_novo import selecionar_dataset_e_classe
from datasets.datasets_tratado import selecionar_dataset_e_classe

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
# LÓGICA CENTRAL DE EXPLICAÇÃO
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
    """
    Calcula os deltas com base em uma premissa de classe explícita.
    """
    coefs = modelo.coef_[0]
    instance_vals = instance_df.iloc[0].values
    deltas = np.zeros_like(coefs)
    X_train_min, X_train_max = X_train.min().values, X_train.max().values
    for i, (coef, val) in enumerate(zip(coefs, instance_vals)):
        # O pior valor é calculado com base na premissa de classe fornecida
        if premis_class == 1: 
            pior_valor = X_train_min[i] if coef > 0 else X_train_max[i]
        else:
            pior_valor = X_train_max[i] if coef > 0 else X_train_min[i]
        deltas[i] = (val - pior_valor) * coef
    return deltas

def one_explanation(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, log_collector: List[str]):
    """
    Gera a explicação inicial usando a lógica baseada no score.
    Esta função agora chama a versão correta de calculate_deltas.
    """
    score = modelo.decision_function(instance_df)[0]
    pred_class = modelo.predict(instance_df)[0]
    
    # Chama a calculate_deltas com 4 argumentos, incluindo a premissa de classe
    deltas = calculate_deltas(modelo, instance_df, X_train, pred_class)
    
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []
    
    is_rejected = t_minus <= score <= t_plus

    log_collector.append(f"   [Fase 0] Calculando explicação inicial:")
    log_collector.append(f"     - Score da Instância: {score:.4f}, Predição: {'Rejeitada' if is_rejected else 'Classe ' + str(pred_class)}")
    
    if is_rejected:
        soma_deltas_cumulativa = 0.0
        target_delta_sum = abs(score)
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

def formatar_calculo_score(modelo: LogisticRegression, instance_df: pd.DataFrame, feature_names: List[str], deltas_originais: np.ndarray) -> str:
    pesos = modelo.coef_[0]
    intercepto = modelo.intercept_[0]
    valores = instance_df.iloc[0].values
    termos_com_delta = []
    for i, nome in enumerate(feature_names):
        termo_str = f"({pesos[i]:.4f} * {valores[i]:.4f})"
        abs_delta = abs(deltas_originais[i])
        termos_com_delta.append((termo_str, abs_delta))
    termos_com_delta.sort(key=lambda x: x[1], reverse=True)
    termos_ordenados_str = [item[0] for item in termos_com_delta]
    calculo_str = " + ".join(termos_ordenados_str)
    score_final = np.dot(pesos, valores) + intercepto
    return f"      Cálculo do Score: {calculo_str} + ({intercepto:.4f}) = {score_final:.4f}"

def perturbar_e_validar_com_log(modelo: LogisticRegression, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int, deltas_originais: np.ndarray) -> Tuple[str, str, List[str]]:
    calc_log = [f"      -> Detalhes da Perturbação (Explicação atual com {len(explicacao)} features fixas):"]
    inst_pert = instance_df.copy()
    features_explicacao = {f.split(' = ')[0] for f in explicacao}
    perturbar_para_diminuir_score = (direcao_override == 1)
    indices_ordenados_por_delta = np.argsort(-np.abs(deltas_originais))

    for feat_idx in indices_ordenados_por_delta:
        feat_nome = X_train.columns[feat_idx]
        if feat_nome in features_explicacao:
            valor_original = instance_df.iloc[0, feat_idx]
            calc_log.append(f"        - '{feat_nome:<18}': Feature FIXA na explicação. Valor mantido: {valor_original:.4f}")
            continue
        coef = modelo.coef_[0][feat_idx]
        train_min = X_train[feat_nome].min()
        train_max = X_train[feat_nome].max()
        log_line = f"        - '{feat_nome:<18}': Coef ({coef:>7.4f}). "
        if perturbar_para_diminuir_score:
            valor_pert = train_min if coef > 0 else train_max
            log_line += f"Coef {' >' if coef > 0 else '<='} 0 -> Perturbar com {'Min' if coef > 0 else 'Max'} ({valor_pert:.4f})."
        else:
            valor_pert = train_max if coef > 0 else train_min
            log_line += f"Coef {' >' if coef > 0 else '<='} 0 -> Perturbar com {'Max' if coef > 0 else 'Min'} ({valor_pert:.4f})."
        calc_log.append(log_line)
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert

    calculo_score_str = formatar_calculo_score(modelo, inst_pert, X_train.columns, deltas_originais)
    calc_log.append(calculo_score_str)
    score_pert = modelo.decision_function(inst_pert)[0]
    pred_pert = modelo.predict(inst_pert)[0]
    pert_rejeitada = t_minus <= score_pert <= t_plus
    calc_log.append(f"      -> Score da Instância Perturbada: {score_pert:.4f} {'Sim' if pert_rejeitada else 'Não'} ∈ [t- ({t_minus:.4f}), t+ ({t_plus:.4f})]")
    
    pred_original_class = modelo.predict(instance_df)[0]
    is_original_rejected = t_minus <= modelo.decision_function(instance_df)[0] <= t_plus
    
    if is_original_rejected:
        status = "VÁLIDA (Rejeição Manteve-se)" if pert_rejeitada else "INVÁLIDA (Saiu da Rejeição)"
    else:
        status = "VÁLIDA (Classificação Manteve-se)" if pred_pert == pred_original_class and not pert_rejeitada else "INVÁLIDA"
        
    pred_str = f"REJEITADA (Score: {score_pert:.4f})" if pert_rejeitada else f"{class_names[pred_pert]} (Score: {score_pert:.4f})"
    return status, pred_str, calc_log

def executar_fase_1_reforco_aditivo(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int, deltas_originais: np.ndarray, log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo")
    expl_robusta = list(expl_inicial)
    adicoes = 0  # <--- Inicializa o contador
    while True:
        log_collector.append(f"     - Testando robustez da explicação atual com {len(expl_robusta)} features...")
        status, pred_str_pert, calc_log = perturbar_e_validar_com_log(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, class_names, direcao_override, deltas_originais)
        log_collector.extend(calc_log)
        if status.startswith("VÁLIDA"):
            log_collector.append(f"     -> SUCESSO: Explicação é robusta. (Resultado: {pred_str_pert})")
            break
        log_collector.append(f"     -> FALHA: Explicação é fraca. (Resultado: {pred_str_pert})")
        if len(expl_robusta) == X_train.shape[1]:
            log_collector.append("     -> ATENÇÃO: Todas as features já foram adicionadas. Impossível reforçar mais.")
            break
        indices_ordenados = np.argsort(-np.abs(deltas_originais))
        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta}
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set:
                log_collector.append(f"     -> REFORÇANDO: Adicionando a próxima feature de maior impacto: '{feat_nome}'.\n")
                expl_robusta.append(f"{feat_nome} = {instance_df.iloc[0, idx]:.4f}")
                adicoes += 1  # <--- Incrementa o contador
                break
    log_collector.append(f"   -> Fim da Fase 1. Explicação robusta final tem {len(expl_robusta)} features.")
    return expl_robusta, adicoes  # <--- Retorna dois valores

def executar_fase_2_minimizacao_subtrativa(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int, deltas_originais: np.ndarray, log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 2] Início da Minimização Subtrativa")
    expl_minima = list(expl_robusta)
    remocoes = 0  # <--- Inicializa o contador
    status_inicial, _, _ = perturbar_e_validar_com_log(modelo, instance_df, expl_minima, X_train, t_plus, t_minus, class_names, direcao_override, deltas_originais)
    if not status_inicial.startswith("VÁLIDA"):
        log_collector.append("     -> PULANDO MINIMIZAÇÃO: A explicação de entrada já não é robusta.")
        return expl_minima, remocoes # <--- Retorna dois valores
        
    features_para_remover = sorted([f.split(' = ')[0] for f in expl_minima], key=lambda nome: abs(deltas_originais[X_train.columns.get_loc(nome)]))
    log_collector.append(f"     - Ordem de tentativa de remoção (do menor |delta| para o maior): {features_para_remover}")
    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1:
            log_collector.append("     -> Parando minimização para manter ao menos uma feature.")
            break
        log_collector.append(f"\n     - TENTANDO REMOVER: '{feat_nome}'...")
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        status_temp, pred_str_pert, calc_log = perturbar_e_validar_com_log(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, class_names, direcao_override, deltas_originais)
        log_collector.extend(calc_log)
        if status_temp.startswith("VÁLIDA"):
            log_collector.append(f"       -> SUCESSO: A remoção de '{feat_nome}' manteve a robustez. Explicação agora com {len(expl_temp)} features.")
            expl_minima = expl_temp
            remocoes += 1  # <--- Incrementa o contador
        else:
            log_collector.append(f"       -> FALHA: A remoção de '{feat_nome}' quebrou a robustez. Mantendo a feature.")
            
    log_collector.append(f"\n   -> Fim da Fase 2. Explicação mínima final tem {len(expl_minima)} features.")
    return expl_minima, remocoes  # <--- Retorna dois valores

def formatar_tabela_explicacao(modelo: LogisticRegression, instance_df: pd.DataFrame, deltas_originais: np.ndarray, explicacao: List[str], X_train: pd.DataFrame) -> str:
    if not explicacao:
        return "      - Nenhuma feature na explicação final desta corrida."
    linhas = [f"      - Explicação final desta corrida ({len(explicacao)} features):",
              f"        {'Feature':<25} | {'Inst. Valor':>12} | {'Peso (Coef.)':>12} | {'Delta':>12}"]
    linhas.append(f"        {'-'*25} | {'-'*12} | {'-'*12} | {'-'*12}")
    feature_nomes = [f.split(' = ')[0] for f in explicacao]
    dados_tabela = []
    for nome in feature_nomes:
        idx = X_train.columns.get_loc(nome)
        dados_tabela.append({
            "nome": nome,
            "valor": instance_df.iloc[0, idx],
            "peso": modelo.coef_[0][idx],
            "delta": deltas_originais[idx],
            "abs_delta": abs(deltas_originais[idx])
        })
    dados_tabela.sort(key=lambda x: x['abs_delta'], reverse=True)
    for item in dados_tabela:
        linhas.append(f"        {item['nome']:<25} | {item['valor']:>12.4f} | {item['peso']:>12.4f} | {item['delta']:>12.4f}")
    return "\n".join(linhas)

def gerar_relatorio_consolidado(modelo: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float):
    caminho_relatorio = f"9_detalhado_{nome_dataset}.txt"
    scores_teste = modelo.decision_function(X_test)
    intercepto = modelo.intercept_[0]
    pesos = modelo.coef_[0]
    stats = {
        "tamanhos_expl_neg": [], "tamanhos_expl_pos": [], "tamanhos_expl_rej": [],
        "vencedor_rejeicao": Counter(), "all_features_in_exps": []
    }

    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        # ... (cabeçalho do relatório mantido) ...
        print(f"Gerando relatório para {len(X_test)} instâncias de teste...")
        for i in range(len(X_test)):
            inst_df = X_test.iloc[[i]]
            score = scores_teste[i]
            pred_class = modelo.predict(inst_df)[0]
            rejeitada = t_minus <= score <= t_plus
            pred_str = f"REJEITADA (Score: {score:.4f})" if rejeitada else f"{class_names[pred_class]} (Score: {score:.4f})"
            f.write(f"\n--- INSTÂNCIA #{i} | Predição Original: {pred_str} | Classe Real: {class_names[y_test.iloc[i]]} ---\n")

            # ##########################################################################
            # LINHA CORRIGIDA AQUI
            # Agora a chamada principal também passa a premissa de classe
            deltas_originais = calculate_deltas(modelo, inst_df, X_train, pred_class)
            # ##########################################################################
            
            log_fase0 = []
            expl_inicial = one_explanation(modelo, inst_df, X_train, t_plus, t_minus, log_collector=log_fase0)

            if rejeitada:
                # O resto da lógica de duas corridas do 'detalhado' permanece
                # Ela usará `deltas_originais` que agora são calculados corretamente
                f.write("\n" + "\n".join(log_fase0) + "\n")
                f.write("\n" + "="*15 + " [INÍCIO CORRIDA 1: Teste de robustez contra uma virada para a CLASSE 0 (Diminuindo o Score)] " + "="*15 + "\n")
                log_corrida1 = []
                expl_robusta1, _ = executar_fase_1_reforco_aditivo(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, 1, deltas_originais, log_corrida1)
                expl_final1, _ = executar_fase_2_minimizacao_subtrativa(modelo, inst_df, expl_robusta1, X_train, t_plus, t_minus, class_names, 1, deltas_originais, log_corrida1)
                f.write("\n".join(log_corrida1) + "\n")
                
                f.write("\n" + "="*15 + " [INÍCIO CORRIDA 2: Teste de robustez contra uma virada para a CLASSE 1 (Aumentando o Score)] " + "="*15 + "\n")
                log_corrida2 = []
                expl_robusta2, _ = executar_fase_1_reforco_aditivo(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, 0, deltas_originais, log_corrida2)
                expl_final2, _ = executar_fase_2_minimizacao_subtrativa(modelo, inst_df, expl_robusta2, X_train, t_plus, t_minus, class_names, 0, deltas_originais, log_corrida2)
                f.write("\n".join(log_corrida2) + "\n")
                
                f.write("\n  >> DECISÃO FINAL DA INSTÂNCIA REJEITADA:\n")
                if len(expl_final1) <= len(expl_final2):
                    vencedor, expl_final = "Corrida 1 (Teste vs Classe 0)", expl_final1
                else:
                    vencedor, expl_final = "Corrida 2 (Teste vs Classe 1)", expl_final2
                stats['vencedor_rejeicao'][vencedor] += 1
                stats['tamanhos_expl_rej'].append(len(expl_final))
                f.write(f"  - Vencedor: {vencedor}. A explicação mais curta e robusta foi encontrada com {len(expl_final)} features.\n")
                f.write(f"  - PI-EXPLICAÇÃO FINAL: {expl_final}\n")
            else:
                log_classificado = []
                direcao = 1 if pred_class == 1 else 0
                expl_robusta, _ = executar_fase_1_reforco_aditivo(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, direcao, deltas_originais, log_classificado)
                expl_final, _ = executar_fase_2_minimizacao_subtrativa(modelo, inst_df, expl_robusta, X_train, t_plus, t_minus, class_names, direcao, deltas_originais, log_classificado)
                f.write("\n".join(log_classificado) + "\n")
                f.write(f"\n  >> PI-EXPLICAÇÃO FINAL (CLASSIFICADA): {expl_final}\n")
                if pred_class == 1:
                    stats['tamanhos_expl_pos'].append(len(expl_final))
                else:
                    stats['tamanhos_expl_neg'].append(len(expl_final))
            
            stats['all_features_in_exps'].extend([feat.split(' = ')[0] for feat in expl_final])


        f.write("\n\n" + "="*105 + "\nRESUMO DAS MÉTRICAS GERAIS\n" + "="*105 + "\n\n")
        acuracia_geral = modelo.score(X_test, y_test)
        aceitas_mask = ~((scores_teste >= t_minus) & (scores_teste <= t_plus))
        taxa_rejeicao = 1 - np.mean(aceitas_mask)
        acuracia_com_rejeicao = modelo.score(X_test[aceitas_mask], y_test[aceitas_mask]) if np.sum(aceitas_mask) > 0 else "N/A"
        f.write(f"\n[Métricas de Desempenho do Modelo]\n")
        f.write(f"  - Acurácia Geral (sem rejeição): {acuracia_geral:.2%}\n")
        f.write(f"  - Taxa de Rejeição no Teste: {taxa_rejeicao:.2%}\n")
        f.write(f"  - Acurácia com Opção de Rejeição (nas instâncias aceitas): {acuracia_com_rejeicao if isinstance(acuracia_com_rejeicao, str) else f'{acuracia_com_rejeicao:.2%}'}\n")

        f.write(f"\n[Métricas das Explicações Geradas]\n")
        total_exps = len(stats['tamanhos_expl_neg']) + len(stats['tamanhos_expl_pos']) + len(stats['tamanhos_expl_rej'])
        for tipo, lista in [("Negativas", stats['tamanhos_expl_neg']), ("Positivas", stats['tamanhos_expl_pos']), ("Rejeitadas", stats['tamanhos_expl_rej'])]:
            n = len(lista)
            perc = (n / total_exps * 100) if total_exps > 0 else 0
            f.write(f"  - Classe {tipo}: {n} instâncias ({perc:.1f}%)\n")
            if n > 0:
                f.write(f"    - Tamanho Explicação (Min/Média/Max): {np.min(lista)} / {np.mean(lista):.2f} / {np.max(lista)}\n")
        
        f.write("\n[Estatísticas das Features (no Conjunto de Treino)]\n")
        std_c0 = X_train[y_train == 0].std()
        std_c1 = X_train[y_train == 1].std()
        f.write(f"  {'Feature':<25} | {'Desv. Padrão (Classe 0)':>25} | {'Desv. Padrão (Classe 1)':>25}\n")
        f.write(f"  {'-'*25} | {'-'*25} | {'-'*25}\n")
        for feat in X_train.columns:
            f.write(f"  {feat:<25} | {std_c0.get(feat, 0):>25.4f} | {std_c1.get(feat, 0):>25.4f}\n")

        f.write("\n[Análise de Importância de Features]\n")
        f.write("  - Top 10 Features Mais Frequentes em Explicações:\n")
        for feat, count in Counter(stats['all_features_in_exps']).most_common(10):
            f.write(f"    - {feat}: {count} vezes\n")
        
        f.write("\n  - Top 10 Pesos (Coeficientes) do Modelo (por valor absoluto):\n")
        pesos_df = pd.DataFrame({'feature': X_train.columns, 'peso': pesos, 'abs_peso': np.abs(pesos)}).sort_values(by='abs_peso', ascending=False)
        for _, row in pesos_df.head(10).iterrows():
            f.write(f"    - {row['feature']:<25}: {row['peso']:.4f}\n")

    print(f"\nRelatório consolidado salvo em: {caminho_relatorio}")

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
    gerar_relatorio_consolidado(modelo, X_test, y_test, X_train, y_train, nomes_classes, nome_dataset, t_plus, t_minus) # <<<< CORREÇÃO AQUI 2
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()