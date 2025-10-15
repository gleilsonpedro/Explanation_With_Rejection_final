"""
NÃO USA ONE_EXPLANATION EM NENHUMA ETAPA, POIS INICIA COM TODAS AS FEATURES 
E VAI TENTANDO REMOVER.
 ==============================================================================
                     DESCRIÇÃO DO ALGORITMO

 Este script implementa um método para gerar "PI-Explicações" para as predições
 de um classificador de Regressão Logística com opção de rejeição.
 Uma PI-Explicação (Explicação por Implicante Primo) é um conjunto de features
 que é, ao mesmo tempo, SUFICIENTE para garantir a predição (robusto) e
 MINIMAL (irredutível, ou seja, nenhum subconjunto dele é suficiente).

 A abordagem deste script é "SUBTRATIVA-PRIMEIRO". Diferente de outros métodos
 que começam com uma explicação pequena e a reforçam, este parte do princípio
 de que a explicação inicial contém TODAS as features do modelo e então tenta
 remover o máximo de features possível.

 - Processo Geral (`executar_minimizacao_subtrativa`): Para cada instância, a
   explicação começa com todas as features. O algoritmo então itera sobre elas,
   da menos importante (menor |delta|) para a mais importante, tentando remover
   cada uma. A remoção só é efetivada se a explicação restante mantiver sua
   robustez.

 ------------------------------------------------------------------------------
                 ANÁLISE PARA CLASSES POSITIVA E NEGATIVA
 ------------------------------------------------------------------------------
 Para instâncias com uma predição clara, o script usa a abordagem subtrativa
 com um teste de robustez unidirecional.

 - Função Principal: `executar_minimizacao_subtrativa` com `bi_direcional=False`.
 - Teste de Robustez: A robustez é avaliada de forma UNIDIRECIONAL.
   - Para a classe positiva (+1): Ao tentar remover uma feature, o teste
     verifica se a explicação restante ainda impede que o score caia
     abaixo de `t_plus`.
   - Para a classe negativa (-1): O teste verifica se a explicação restante
     ainda impede que o score suba acima de `t_minus`.

 O resultado é uma PI-Explicação que é minimal e robusta naquela direção.

 ------------------------------------------------------------------------------
                    ANÁLISE PARA A CLASSE DE REJEIÇÃO
 ------------------------------------------------------------------------------
 Para instâncias na zona de rejeição, onde a robustez precisa ser bidirecional,
 este script utiliza uma HEURÍSTICA complexa que combina a abordagem subtrativa
 com a lógica de "duas corridas".

 - Estratégia: "Duas Corridas Subtrativas com Teste Bidirecional".
   O script executa duas análises de minimização independentes.

   1. Corrida 1 (com premissa de Classe 1): O cálculo dos "deltas" (importância
      das features) é feito assumindo que o objetivo é evitar que a instância
      vire classe 1. A partir de TODAS as features, o algoritmo tenta remover
      uma a uma, e a cada tentativa, ele realiza um teste de robustez
      BIdirecional (`perturbar_e_validar_com_log` chamado para ambas as direções).

   2. Corrida 2 (com premissa de Classe 0): O cálculo dos deltas é feito
      assumindo que o objetivo é evitar virar classe 0. O processo de
      minimização subtrativa com teste bidirecional é repetido.

 - Decisão Final: Ao final, o script compara a cardinalidade (número de features)
   das duas explicações resultantes e define a MENOR delas como a explicação final.

 - Sobre Minimalidade e Confiabilidade:
   - O processo de minimização em CADA CORRIDA é rigoroso e busca uma explicação
     **minimal** (irredutível) que seja bidirecionalmente robusta sob a premissa
     daquela corrida.
   - O algoritmo não garante encontrar a explicação de **mínima cardinalidade**
     (a menor possível em termos absolutos), mas sim uma que é minimal.
   - **Ponto Crítico**: A confiabilidade é comprometida na decisão final. Ao
     escolher a explicação menor entre duas corridas baseadas em premissas
     diferentes (cálculos de delta distintos), o script se torna uma heurística.
     A explicação final pode não ser robusta sob a premissa da corrida
     "perdedora". Portanto, assim como os outros métodos de "duas corridas",
     ele não oferece uma garantia formal de robustez bidirecional para a
     explicação final.
 ==============================================================================
"""
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
        if premis_class == 1:
            pior_valor = X_train_min[i] if coef > 0 else X_train_max[i]
        else: # premis_class == 0
            pior_valor = X_train_max[i] if coef > 0 else X_train_min[i]
        deltas[i] = (val - pior_valor) * coef
    return deltas

def one_explanation(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, log_collector: List[str]):
    score = modelo.decision_function(instance_df)[0]
    pred_class = modelo.predict(instance_df)[0]
    deltas = calculate_deltas(modelo, instance_df, X_train, pred_class)
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []
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
        for i in indices_ordenados:
            explicacao.append(f"{X_train.columns[i]} = {instance_df.iloc[0, i]:.4f}")
            if len(explicacao) > np.sqrt(len(X_train.columns)):
                break
    if not explicacao and len(deltas) > 0:
        idx = indices_ordenados[0]
        explicacao.append(f"{X_train.columns[idx]} = {instance_df.iloc[0, idx]:.4f}")
    log_collector.append(f"   -> Explicação Inicial Gerada ({len(explicacao)} feats): {explicacao}")
    return explicacao

def formatar_calculo_score(modelo: LogisticRegression, instance_df: pd.DataFrame, feature_names: List[str], deltas_para_ordenar: np.ndarray) -> str:
    pesos = modelo.coef_[0]
    intercepto = modelo.intercept_[0]
    valores = instance_df.iloc[0].values
    termos_com_delta = []
    for i, nome in enumerate(feature_names):
        termo_str = f"({pesos[i]:.4f} * {valores[i]:.4f})"
        abs_delta = abs(deltas_para_ordenar[i])
        termos_com_delta.append((termo_str, abs_delta))
    termos_com_delta.sort(key=lambda x: x[1], reverse=True)
    termos_ordenados_str = [item[0] for item in termos_com_delta]
    calculo_str = " + ".join(termos_ordenados_str)
    score_final = np.dot(pesos, valores) + intercepto
    return f"      Cálculo do Score: {calculo_str} + ({intercepto:.4f}) = {score_final:.4f}"

def perturbar_e_validar_com_log(modelo: LogisticRegression, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int, deltas_para_ordenar: np.ndarray) -> Tuple[str, str, List[str]]:
    calc_log = [f"      -> Detalhes da Perturbação (Explicação atual com {len(explicacao)} features fixas):"]
    inst_pert = instance_df.copy()
    features_explicacao = {f.split(' = ')[0] for f in explicacao}
    perturbar_para_diminuir_score = (direcao_override == 1)
    indices_ordenados_por_delta = np.argsort(-np.abs(deltas_para_ordenar))
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
            log_line += f"Coef {'>' if coef > 0 else '<='} 0 -> Perturbar com {'Min' if coef > 0 else 'Max'} ({valor_pert:.4f})."
        else:
            valor_pert = train_max if coef > 0 else train_min
            log_line += f"Coef {'>' if coef > 0 else '<='} 0 -> Perturbar com {'Max' if coef > 0 else 'Min'} ({valor_pert:.4f})."
        calc_log.append(log_line)
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
    calculo_score_str = formatar_calculo_score(modelo, inst_pert, X_train.columns, deltas_para_ordenar)
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
# FUNÇÕES DE FASE (Subtrativa-Primeiro)
#==============================================================================
def executar_minimizacao_subtrativa(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], deltas_para_ordenar: np.ndarray, log_collector: List[str], bi_direcional: bool) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase de Minimização {'Bi-Direcional Otimizada' if bi_direcional else 'Uni-Direcional'}]")
    expl_minima = X_train.columns.tolist() # Começa com todas as features
    remocoes = 0
    
    features_para_remover = sorted(expl_minima, key=lambda nome: abs(deltas_para_ordenar[X_train.columns.get_loc(nome)]))
    log_collector.append(f"     - Ordem de tentativa de remoção (do menor |delta| para o maior): {features_para_remover}")

    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1:
            log_collector.append("     -> Parando minimização para manter ao menos uma feature.")
            break
        
        log_collector.append(f"\n     - TENTANDO REMOVER: '{feat_nome}'...")
        expl_temp_nomes = [f for f in expl_minima if f != feat_nome]
        expl_temp_formatada = [f"{nome} = {instance_df.iloc[0][X_train.columns.get_loc(nome)]:.4f}" for nome in expl_temp_nomes]

        status1, _, calc_log1 = perturbar_e_validar_com_log(modelo, instance_df, expl_temp_formatada, X_train, t_plus, t_minus, class_names, 1, deltas_para_ordenar)
        log_collector.append("       -> Teste vs Classe 0 (Diminuir Score)...")
        log_collector.extend(calc_log1)
        
        remocao_valida = status1.startswith("VÁLIDA")

        if bi_direcional and remocao_valida:
            log_collector.append("       -> SUCESSO PARCIAL. Verificando a segunda direção...")
            status2, _, calc_log2 = perturbar_e_validar_com_log(modelo, instance_df, expl_temp_formatada, X_train, t_plus, t_minus, class_names, 0, deltas_para_ordenar)
            log_collector.extend(calc_log2)
            remocao_valida = status2.startswith("VÁLIDA")
        
        if remocao_valida:
            log_collector.append(f"       -> SUCESSO: A remoção de '{feat_nome}' manteve a robustez. Explicação agora com {len(expl_temp_nomes)} features.")
            expl_minima = expl_temp_nomes
            remocoes += 1
        else:
            log_collector.append(f"       -> FALHA: A remoção de '{feat_nome}' quebrou a robustez. Mantendo a feature e parando a minimização.")
            break

    log_collector.append(f"\n   -> Fim da Fase de Minimização. Explicação mínima final tem {len(expl_minima)} features.")
    return [f"{nome} = {instance_df.iloc[0][X_train.columns.get_loc(nome)]:.4f}" for nome in expl_minima], remocoes

def formatar_tabela_top_instancias(resultados: list, titulo: str, n: int = 3):
    linhas = [f"  - {titulo}:"]
    if not resultados:
        linhas.append("    - Nenhuma instância nesta categoria.")
        return "\n".join(linhas)
    for i, (idx, tamanho, expl) in enumerate(resultados[:n]):
        linhas.append(f"    {i+1}. Instância #{idx:<4} (Tamanho: {tamanho}) -> Expl: { [f.split(' = ')[0] for f in expl] }")
    return "\n".join(linhas)

#==============================================================================
# GERAÇÃO DE RELATÓRIO (Função Principal)
#==============================================================================

def gerar_relatorio_consolidado(modelo: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float):
    caminho_relatorio = f"rejeita_10_{nome_dataset}.txt"
    scores_teste = modelo.decision_function(X_test)
    intercepto = modelo.intercept_[0]
    pesos = modelo.coef_[0]

    # <<< MUDANÇA: Buffers para armazenar logs e reordenar a escrita no arquivo >>>
    logs_rejeitadas = []
    logs_classificadas = []
    stats = {
        "resultados_neg": [], "resultados_pos": [], "resultados_rej": [],
        "all_features_in_exps": []
    }

    print(f"Gerando relatório para {len(X_test)} instâncias de teste...")
    for i in range(len(X_test)):
        log_instancia_atual = []
        inst_df = X_test.iloc[[i]]
        score = scores_teste[i]
        pred_class = modelo.predict(inst_df)[0]
        rejeitada = t_minus <= score <= t_plus
        
        pred_str = f"REJEITADA (Score: {score:.4f})" if rejeitada else f"{class_names[pred_class]} (Score: {score:.4f})"
        log_instancia_atual.append(f"\n\n--- INSTÂNCIA #{i} | Predição Original: {pred_str} | Classe Real: {class_names[y_test.iloc[i]]} ---")

        if rejeitada:
            deltas1 = calculate_deltas(modelo, inst_df, X_train, premis_class=1)
            deltas2 = calculate_deltas(modelo, inst_df, X_train, premis_class=0)
            
            log_corrida1 = []
            log_corrida2 = []

            expl_final1, remocoes1 = executar_minimizacao_subtrativa(modelo, inst_df, X_train, t_plus, t_minus, class_names, deltas1, log_corrida1, bi_direcional=True)
            expl_final2, remocoes2 = executar_minimizacao_subtrativa(modelo, inst_df, X_train, t_plus, t_minus, class_names, deltas2, log_corrida2, bi_direcional=True)
            
            log_instancia_atual.append("\n" + "="*15 + " [CORRIDA 1: Minimização com Premissa de Classe 1] " + "="*15 + "\n" + "\n".join(log_corrida1))
            log_instancia_atual.append("\n" + "="*15 + " [CORRIDA 2: Minimização com Premissa de Classe 0] " + "="*15 + "\n" + "\n".join(log_corrida2))
            
            if len(expl_final1) <= len(expl_final2):
                expl_final = expl_final1
            else:
                expl_final = expl_final2
                
            log_instancia_atual.append(f"\n  >> PI-EXPLICAÇÃO FINAL (REJEITADA): {expl_final}\n")
            stats['resultados_rej'].append((i, len(expl_final), expl_final))
            logs_rejeitadas.append("\n".join(log_instancia_atual))
        else: # Instância Classificada
            log_instancia_atual.append("\n" + "="*15 + f" [ANÁLISE UNI-DIRECIONAL (CLASSE {pred_class})] " + "="*15 + "\n")
            deltas_originais = calculate_deltas(modelo, inst_df, X_train, premis_class=pred_class)
            expl_final, _ = executar_minimizacao_subtrativa(modelo, inst_df, X_train, t_plus, t_minus, class_names, deltas_originais, log_instancia_atual, bi_direcional=False)
            log_instancia_atual.append(f"\n  >> PI-EXPLICAÇÃO FINAL (CLASSIFICADA): {expl_final}\n")
            
            if pred_class == 1:
                stats['resultados_pos'].append((i, len(expl_final), expl_final))
            else:
                stats['resultados_neg'].append((i, len(expl_final), expl_final))
            logs_classificadas.append("\n".join(log_instancia_atual))
        
        stats['all_features_in_exps'].extend([feat.split(' = ')[0] for feat in expl_final])

    # --- Escrita Final no Arquivo, na Ordem Correta ---
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n" + "RELATÓRIO DE PI-EXPLICAÇÕES".center(80) + "\n" + "="*80)
        acuracia_geral = modelo.score(X_test, y_test)
        f.write(f"\n\n[CONFIGURAÇÕES GERAIS]\n")
        f.write(f"  - Dataset: {nome_dataset}\n")
        f.write(f"  - Total de instâncias de teste: {len(X_test)}\n")
        f.write(f"  - Número total de features no modelo: {len(X_train.columns)}\n")
        f.write(f"  - Acurácia do modelo (teste, sem rejeição): {acuracia_geral:.2%}\n")
        f.write(f"  - Thresholds de Rejeição: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n")
        f.write(f"  - Intercepto do Modelo: {intercepto:.4f}\n")

        f.write("\n\n" + "="*80 + "\n" + "ANÁLISE DETALHADA: INSTÂNCIAS REJEITADAS".center(80) + "\n" + "="*80)
        f.write("\n".join(logs_rejeitadas))

        f.write("\n\n" + "="*80 + "\n" + "ANÁLISE DETALHADA: INSTÂNCIAS CLASSIFICADAS".center(80) + "\n" + "="*80)
        f.write("\n".join(logs_classificadas))

        # --- Resumo Final Completo ---
        f.write("\n\n" + "="*80 + "\n" + "RESUMO ESTATÍSTICO GERAL".center(80) + "\n" + "="*80)
        
        f.write("\n\n[Métricas de Desempenho do Modelo]\n")
        aceitas_mask = ~((scores_teste >= t_minus) & (scores_teste <= t_plus))
        taxa_rejeicao = 1 - np.mean(aceitas_mask)
        f.write(f"  - Taxa de Rejeição no Teste: {taxa_rejeicao:.2%}\n")
        acuracia_com_rejeicao = modelo.score(X_test[aceitas_mask], y_test[aceitas_mask]) if np.sum(aceitas_mask) > 0 else "N/A"
        f.write(f"  - Acurácia com Opção de Rejeição (nas instâncias aceitas): {acuracia_com_rejeicao if isinstance(acuracia_com_rejeicao, str) else f'{acuracia_com_rejeicao:.2%}'}\n")

        f.write(f"\n[Distribuição das Predições e Explicações]\n")
        total_exps = len(X_test)
        for tipo, lista in [("Positivas", stats['resultados_pos']), ("Negativas", stats['resultados_neg']), ("Rejeitadas", stats['resultados_rej'])]:
            n = len(lista)
            perc = (n / total_exps * 100) if total_exps > 0 else 0
            f.write(f"  - Classe {tipo}: {n} instâncias ({perc:.1f}%)\n")
            if n > 0:
                tamanhos = [item[1] for item in lista]
                f.write(f"    - Tamanho Explicação (Min/Média/Max): {np.min(tamanhos)} / {np.mean(tamanhos):.2f} / {np.max(tamanhos)}\n")

        f.write("\n[Análise de Explicações com Mais e Menos Features (Top 3)]\n")
        stats['resultados_pos'].sort(key=lambda x: x[1])
        stats['resultados_neg'].sort(key=lambda x: x[1])
        stats['resultados_rej'].sort(key=lambda x: x[1])
        f.write(formatar_tabela_top_instancias(stats['resultados_pos'], "Positivas com MENOS Features"))
        f.write(formatar_tabela_top_instancias(stats['resultados_neg'], "Negativas com MENOS Features"))
        f.write(formatar_tabela_top_instancias(stats['resultados_rej'], "Rejeitadas com MENOS Features"))
        f.write("\n")
        f.write(formatar_tabela_top_instancias(list(reversed(stats['resultados_pos'])), "Positivas com MAIS Features"))
        f.write(formatar_tabela_top_instancias(list(reversed(stats['resultados_neg'])), "Negativas com MAIS Features"))
        f.write(formatar_tabela_top_instancias(list(reversed(stats['resultados_rej'])), "Rejeitadas com MAIS Features"))
        
        f.write("\n\n[Estatísticas das Features (no Conjunto de Treino)]\n")
        std_c0 = X_train[y_train == 0].std()
        std_c1 = X_train[y_train == 1].std()
        f.write(f"  {'Feature':<25} | {'Desv. Padrão (Classe 0)':>25} | {'Desv. Padrão (Classe 1)':>25}\n")
        f.write(f"  {'-'*25} | {'-'*25} | {'-'*25}\n")
        for feat in X_train.columns:
            f.write(f"  {feat:<25} | {std_c0.get(feat, 0):>25.4f} | {std_c1.get(feat, 0):>25.4f}\n")

        f.write("\n[Análise de Importância de Features]\n")
        f.write("  - Top 10 Features Mais Frequentes em Explicações:\n")
        if stats['all_features_in_exps']:
            for feat, count in Counter(stats['all_features_in_exps']).most_common(10):
                f.write(f"    - {feat}: {count} vezes\n")
        
        f.write("\n  - Top 10 Pesos (Coeficientes) do Modelo (por valor absoluto):\n")
        pesos_df = pd.DataFrame({'feature': X_train.columns, 'peso': pesos, 'abs_peso': np.abs(pesos)}).sort_values(by='abs_peso', ascending=False)
        for _, row in pesos_df.head(10).iterrows():
            f.write(f"    - {row['feature']:<25}: {row['peso']:.4f}\n")

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