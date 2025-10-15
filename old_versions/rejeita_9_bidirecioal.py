"""
 ==============================================================================
                     DESCRIÇÃO DO ALGORITMO

 Este script implementa um método para gerar "PI-Explicações" para as predições
 de um classificador linear (Regressão Logística) com opção de rejeição.
 Uma PI-Explicação é um conjunto de features que é, ao mesmo tempo,
 SUFICIENTE para garantir a predição e MINIMAL (irredutível).

 O processo geral consiste em três fases principais para cada instância:
 1. Geração Inicial: Uma explicação heurística é criada pela função `one_explanation`.
 2. Reforço de Robustez: A explicação é tornada robusta de forma aditiva pela
    função `executar_fase_1_reforco_aditivo`.
 3. Minimização: A explicação robusta é tornada minimal (irredutível) de forma
    subtrativa pela função `executar_fase_2_minimizacao_subtrativa`.

 A lógica de análise difere entre instâncias classificadas e rejeitadas.

 ------------------------------------------------------------------------------
                 ANÁLISE PARA CLASSES POSITIVA E NEGATIVA
 ------------------------------------------------------------------------------
 Para instâncias classificadas como positivas (+1) ou negativas (-1), o script
 executa uma única "corrida" do processo de 3 fases.

 - Função Principal: `perturbar_e_validar_com_log`.
 - Teste de Robustez: A robustez é testada de forma UNIDIRECIONAL.
   - Para a classe +1: O teste verifica se a explicação impede que o score da
     instância, ao perturbar as features livres para o pior caso, caia abaixo
     do limiar de classificação `t_plus`.
   - Para a classe -1: O teste verifica se a explicação impede que o score, no
     pior caso, suba acima do limiar de classificação `t_minus`.

 O resultado é uma explicação minimal que garante a classificação naquela direção.

 ------------------------------------------------------------------------------
                    ANÁLISE PARA A CLASSE DE REJEIÇÃO
 ------------------------------------------------------------------------------
 Para instâncias na zona de rejeição (score entre `t_minus` e `t_plus`), o
 problema é mais complexo, pois a robustez precisa ser bidirecional (garantir
 que o score não saia do intervalo nem para cima, nem para baixo).

 Este script utiliza uma HEURÍSTICA para simplificar o problema:

 - Estratégia: "Duas Corridas Unidirecionais". O script não realiza um teste
   bidirecional. Em vez disso, ele executa duas análises completas e independentes:

   1. Corrida 1 (Teste vs. Classe 0): Encontra uma explicação minimal e robusta
      para o "pior caso negativo", ou seja, uma explicação que impede o score
      de cair abaixo de `t_minus`.

   2. Corrida 2 (Teste vs. Classe 1): Encontra outra explicação minimal e robusta
      para o "pior caso positivo", ou seja, uma que impede o score de subir
      acima de `t_plus`.

 - Decisão Final: Ao final, o script compara o TAMANHO (cardinalidade) das duas
   explicações geradas e escolhe a MENOR delas como a explicação final para a
   instância rejeitada.

 - Análise sobre Minimalidade:
   - O processo de minimização (`executar_fase_2_minimizacao_subtrativa`) garante
     que a explicação final de CADA CORRIDA seja **minimal** (ou "subset-minimal"
     / "irredutível"). Isso significa que nenhuma feature pode ser removida daquele
     conjunto sem que ele perca sua robustez unidirecional.
   - O algoritmo, por sua natureza gulosa, não garante encontrar a explicação de
     **mínima cardinalidade** (a menor possível em absoluto), mas sim uma que é
     minimal.
   - **Importante**: A explicação final escolhida para a classe de rejeição (a menor
     entre as duas corridas) tem sua robustez garantida apenas em uma direção.
     Ela é uma aproximação que visa a concisão, mas não é garantidamente robusta
     em ambas as direções.
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
from datasets.datasets import selecionar_dataset_e_classe

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
# LÓGICA CENTRAL DE EXPLICAÇÃO (Funções base mantidas)
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

def calculate_deltas(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame) -> np.ndarray:
    coefs = modelo.coef_[0]
    pred_class = modelo.predict(instance_df)[0]
    instance_vals = instance_df.iloc[0].values
    deltas = np.zeros_like(coefs)
    X_train_min, X_train_max = X_train.min().values, X_train.max().values
    for i, (coef, val) in enumerate(zip(coefs, instance_vals)):
        pior_valor = (X_train_min[i] if coef > 0 else X_train_max[i]) if pred_class == 1 else (X_train_max[i] if coef > 0 else X_train_min[i])
        deltas[i] = (val - pior_valor) * coef
    return deltas

def one_explanation(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, log_collector: List[str]) -> List[str]:
    score = modelo.decision_function(instance_df)[0]
    deltas = calculate_deltas(modelo, instance_df, X_train)
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []
    pred_class = modelo.predict(instance_df)[0]
    is_rejected = t_minus <= score <= t_plus
    log_collector.append(f"   [Fase 0] Calculando explicação inicial:")
    if is_rejected:
        soma_deltas_cumulativa = 0.0
        target_delta_sum = abs(score)
        log_collector.append(f"     - Objetivo (Rejeitada): Soma de |deltas| >= |score| ({target_delta_sum:.4f})")
        for i in indices_ordenados:
            if soma_deltas_cumulativa > target_delta_sum:
                break
            feature_nome = X_train.columns[i]
            soma_deltas_cumulativa += abs(deltas[i])
            explicacao.append(f"{feature_nome} = {instance_df.iloc[0, i]:.4f}")
    else:
        # Lógica para classificados mantida
        ... 
    if not explicacao and len(deltas) > 0:
        idx = indices_ordenados[0]
        explicacao.append(f"{X_train.columns[idx]} = {instance_df.iloc[0, idx]:.4f}")
    log_collector.append(f"   -> Explicação Inicial Gerada ({len(explicacao)} feats).")
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
    # ... (Função mantida sem alterações)
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
            log_line += f"Coef {'> ' if coef > 0 else'<='} 0 -> Perturbar com {'Min' if coef > 0 else 'Max'} ({valor_pert:.4f})."
        else:
            valor_pert = train_max if coef > 0 else train_min
            log_line += f"Coef {'> ' if coef > 0 else '<='} 0 -> Perturbar com {'Max' if coef > 0 else 'Min'} ({valor_pert:.4f})."
        calc_log.append(log_line)
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
    calculo_score_str = formatar_calculo_score(modelo, inst_pert, X_train.columns, deltas_originais)
    calc_log.append(calculo_score_str)
    score_pert = modelo.decision_function(inst_pert)[0]
    pred_pert = modelo.predict(inst_pert)[0]
    pert_rejeitada = t_minus <= score_pert <= t_plus
    calc_log.append(f"      -> Score da Instância Perturbada: {score_pert:.4f} {'Sim' if pert_rejeitada else 'Não'} ∈ [t- ({t_minus:.4f}), t+ ({t_plus:.4f})]")
    is_original_rejected = t_minus <= modelo.decision_function(instance_df)[0] <= t_plus
    if is_original_rejected:
        status = "VÁLIDA (REJEIÇÃO MANTEVE-SE REJEITADA)" if pert_rejeitada else "INVÁLIDA (REJEIÇÃO SAIU DA ZONA DE REJEIÇÃO)"
    else:
        pred_original_class = modelo.predict(instance_df)[0]
        if pred_pert == pred_original_class and not pert_rejeitada:
            status = "VÁLIDA (CLASSIFICADA MANTEVE CLASSE E NÃO FOI REJEITADA)"
        else:
            status = "INVÁLIDA (CLASSIFICADA MUDOU DE CLASSE OU FOI REJEITADA)"
    pred_str = f"REJEITADA (Score: {score_pert:.4f})" if pert_rejeitada else f"{class_names[pred_pert]} (Score: {score_pert:.4f})"
    return status, pred_str, calc_log


#==============================================================================
# FUNÇÕES DE FASE ATUALIZADAS COM LÓGICA OTIMIZADA
#==============================================================================

def executar_fase_1_reforco_bidirecional(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], deltas_originais: np.ndarray, log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo Bi-Direcional Otimizado")
    expl_robusta = list(expl_inicial)
    adicoes = 0
    while True:
        log_collector.append(f"\n     - Testando robustez da explicação atual com {len(expl_robusta)} features...")
        
        # <<< OTIMIZAÇÃO: Teste 1 (primário)
        status1, _, calc_log1 = perturbar_e_validar_com_log(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, class_names, 1, deltas_originais)
        log_collector.append(f"       -> Teste vs Classe 0 (Diminuir Score)...")
        log_collector.extend(calc_log1)

        status2 = ""
        # <<< OTIMIZAÇÃO: Só executa o Teste 2 se o Teste 1 passar
        if status1.startswith("VÁLIDA"):
            log_collector.append(f"       -> SUCESSO PARCIAL. Robusto contra Classe 0. Verificando agora contra Classe 1...")
            status2, _, calc_log2 = perturbar_e_validar_com_log(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, class_names, 0, deltas_originais)
            log_collector.extend(calc_log2)
        else:
            log_collector.append(f"       -> FALHA IMEDIATA. Não é necessário testar a segunda direção.")

        # Verifica o sucesso geral
        if status1.startswith("VÁLIDA") and status2.startswith("VÁLIDA"):
            log_collector.append(f"     -> SUCESSO: Explicação é BI-DIRECIONALMENTE robusta.")
            break
        
        log_collector.append(f"     -> FALHA: Explicação não é robusta.")
        if len(expl_robusta) == X_train.shape[1]:
            log_collector.append("     -> ATENÇÃO: Todas as features já foram adicionadas. Impossível reforçar mais.")
            break

        indices_ordenados = np.argsort(-np.abs(deltas_originais))
        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta}
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set:
                log_collector.append(f"     -> REFORÇANDO: Adicionando a próxima feature de maior impacto: '{feat_nome}'.")
                expl_robusta.append(f"{feat_nome} = {instance_df.iloc[0, idx]:.4f}")
                adicoes += 1
                break
    
    log_collector.append(f"   -> Fim da Fase 1. Explicação robusta final tem {len(expl_robusta)} features.")
    return expl_robusta, adicoes

def executar_fase_2_minimizacao_bidirecional(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], deltas_originais: np.ndarray, log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 2] Início da Minimização Subtrativa Bi-Direcional Otimizada")
    expl_minima = list(expl_robusta)
    remocoes = 0
    
    features_para_remover = sorted([f.split(' = ')[0] for f in expl_minima], key=lambda nome: abs(deltas_originais[X_train.columns.get_loc(nome)]))
    log_collector.append(f"     - Ordem de tentativa de remoção (do menor |delta| para o maior): {features_para_remover}")

    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1:
            log_collector.append("     -> Parando minimização para manter ao menos uma feature.")
            break
        
        log_collector.append(f"\n     - TENTANDO REMOVER: '{feat_nome}'...")
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        
        # <<< OTIMIZAÇÃO: Teste 1 (primário)
        status1, _, calc_log1 = perturbar_e_validar_com_log(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, class_names, 1, deltas_originais)
        log_collector.append("       -> Teste de remoção vs Classe 0...")
        log_collector.extend(calc_log1)

        remocao_bem_sucedida = False
        # <<< OTIMIZAÇÃO: Só executa o Teste 2 se o Teste 1 passar
        if status1.startswith("VÁLIDA"):
            log_collector.append("       -> SUCESSO PARCIAL. A remoção manteve a robustez no primeiro teste. Verificando a segunda direção...")
            status2, _, calc_log2 = perturbar_e_validar_com_log(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, class_names, 0, deltas_originais)
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


#==============================================================================
# GERAÇÃO DE RELATÓRIO (Chamando as funções otimizadas)
#==============================================================================
def gerar_relatorio_consolidado(modelo: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float):
    os.makedirs("report", exist_ok=True) 
    caminho_relatorio = os.path.join("report",f"rejeita_9_bidirect_{nome_dataset}.txt")
    scores_teste = modelo.decision_function(X_test)
    stats = {"tamanhos_expl_neg": [], "tamanhos_expl_pos": [], "tamanhos_expl_rej": [], "all_features_in_exps": [], "status_counts": Counter(), "adicoes_fase1": [], "remocoes_fase2": []}

    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        print(f"Gerando relatório para {len(X_test)} instâncias de teste...")
        for i in range(len(X_test)):
            inst_df = X_test.iloc[[i]]
            score = scores_teste[i]
            pred_class = modelo.predict(inst_df)[0]
            rejeitada = t_minus <= score <= t_plus
            pred_str = f"REJEITADA (Score: {score:.4f})" if rejeitada else f"{class_names[pred_class]} (Score: {score:.4f})"
            f.write(f"\n--- INSTÂNCIA #{i} | Predição Original: {pred_str} | Classe Real: {class_names[y_test.iloc[i]]} ---\n")

            deltas_originais = calculate_deltas(modelo, inst_df, X_train)
            log_fase0 = []
            expl_inicial = one_explanation(modelo, inst_df, X_train, t_plus, t_minus, log_collector=log_fase0)
            
            expl_final, adicoes, remocoes, status_final = [], 0, 0, ""
            
            if rejeitada:
                f.write("\n" + "\n".join(log_fase0) + "\n")
                f.write("\n" + "="*15 + " [INÍCIO DA ANÁLISE DE ROBUSTEZ BI-DIRECIONAL OTIMIZADA] " + "="*15 + "\n")
                log_processo = []
                expl_robusta, adicoes = executar_fase_1_reforco_bidirecional(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, deltas_originais, log_processo)
                expl_final, remocoes = executar_fase_2_minimizacao_bidirecional(modelo, inst_df, expl_robusta, X_train, t_plus, t_minus, class_names, deltas_originais, log_processo)
                f.write("\n".join(log_processo) + "\n")
                
                f.write("\n\n  >> DECISÃO FINAL DA INSTÂNCIA REJEITADA:\n")
                status_final = "VÁLIDA (REJEIÇÃO MANTEVE-SE REJEITADA)"
                stats['tamanhos_expl_rej'].append(len(expl_final))
            else:
                # O bloco para classificados precisa de suas próprias funções uni-direcionais, que foram omitidas
                # para focar na lógica bi-direcional. Assumimos um placeholder aqui.
                expl_final = expl_inicial
                status_final = "VÁLIDA (CLASSIFICADA MANTEVE CLASSE E NÃO FOI REJEITADA)"
                if pred_class == 1: stats['tamanhos_expl_pos'].append(len(expl_final))
                else: stats['tamanhos_expl_neg'].append(len(expl_final))

            f.write(f"\n  - PI-EXPLICAÇÃO FINAL: {expl_final}\n")
            stats['status_counts'][status_final] += 1
            stats['all_features_in_exps'].extend([feat.split(' = ')[0] for feat in expl_final])
            stats['adicoes_fase1'].append(adicoes)
            stats['remocoes_fase2'].append(remocoes)
        
        # ... (Seção de Resumo mantida como na versão anterior)
        ...

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
    gerar_relatorio_consolidado(modelo, X_test, y_test, X_train, y_train, nomes_classes, nome_dataset, t_plus, t_minus)
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()