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
# As funções base (calcular_thresholds, calculate_deltas, one_explanation, 
# perturbar_e_validar) são mantidas como na versão anterior, pois sua lógica
# interna está correta. As mudanças ocorrerão em como elas são orquestradas
# e logadas no relatório.

def calcular_thresholds(modelo: LogisticRegression, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[float, float]:
    decision_scores = modelo.decision_function(X_train)
    min_custo = float('inf')
    melhor_t_plus, melhor_t_minus = 0.1, -0.1
    score_max = np.max(decision_scores) if len(decision_scores) > 0 else 0.1
    score_min = np.min(decision_scores) if len(decision_scores) > 0 else -0.1
    t_plus_candidatos = np.linspace(0.01, max(score_max, 0.1), 50)
    t_minus_candidatos = np.linspace(min(score_min, -0.1), -0.01, 50)
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

def one_explanation(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float) -> List[str]:
    score = modelo.decision_function(instance_df)[0]
    pred_class = modelo.predict(instance_df)[0]
    foi_rejeitada = t_minus <= score <= t_plus
    deltas = calculate_deltas(modelo, instance_df, X_train)
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao, soma_deltas_cumulativa = [], 0.0
    score_base = score - np.sum(deltas)
    if foi_rejeitada:
        target_delta_sum = abs(score)
        for i in indices_ordenados:
            if soma_deltas_cumulativa <= target_delta_sum + EPSILON:
                soma_deltas_cumulativa += abs(deltas[i])
                explicacao.append(f"{X_train.columns[i]} = {instance_df.iloc[0, i]:.4f}")
            else: break
    elif pred_class == 1:
        target_delta_sum = t_plus - score_base + EPSILON
        for i in indices_ordenados:
            if soma_deltas_cumulativa <= target_delta_sum:
                soma_deltas_cumulativa += deltas[i]
                explicacao.append(f"{X_train.columns[i]} = {instance_df.iloc[0, i]:.4f}")
            else: break
    else:
        target_delta_sum = score_base - t_minus + EPSILON
        for i in indices_ordenados:
            if soma_deltas_cumulativa <= target_delta_sum:
                soma_deltas_cumulativa += abs(deltas[i])
                explicacao.append(f"{X_train.columns[i]} = {instance_df.iloc[0, i]:.4f}")
            else: break
    if not explicacao and len(deltas) > 0:
        idx = indices_ordenados[0]
        explicacao.append(f"{X_train.columns[idx]} = {instance_df.iloc[0, idx]:.4f}")
    return explicacao

def perturbar_e_validar(modelo: LogisticRegression, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int) -> Tuple[str, str]:
    inst_pert = instance_df.copy()
    features_explicacao = {f.split(' = ')[0] for f in explicacao}
    score_original = modelo.decision_function(instance_df)[0]
    original_rejeitada = t_minus <= score_original <= t_plus
    perturbar_para_diminuir_score = (direcao_override == 1)
    for feat_idx, feat_nome in enumerate(X_train.columns):
        if feat_nome not in features_explicacao:
            coef = modelo.coef_[0][feat_idx]
            valor_pert = (X_train[feat_nome].min() if coef > 0 else X_train[feat_nome].max()) if perturbar_para_diminuir_score else (X_train[feat_nome].max() if coef > 0 else X_train[feat_nome].min())
            inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
    score_pert = modelo.decision_function(inst_pert)[0]
    pred_pert = modelo.predict(inst_pert)[0]
    pert_rejeitada = t_minus <= score_pert <= t_plus
    status = ""
    if original_rejeitada:
        status = "VÁLIDA (Rejeição Manteve-se)" if pert_rejeitada else "INVÁLIDA (Saiu da Rejeição)"
    else: # Lógica para instâncias classificadas, não usada no fluxo de rejeição
        pred_original = modelo.predict(instance_df)[0]
        if pert_rejeitada: status = "INVÁLIDA (Caiu na Rejeição)"
        elif pred_pert != pred_original: status = f"INVÁLIDA (Mudou de Classe para {class_names[pred_pert]})"
        else: status = "VÁLIDA (Classificação Manteve-se)"
    return status, f"REJEITADA (Score: {score_pert:.4f})" if pert_rejeitada else f"{class_names[pred_pert]} (Score: {score_pert:.4f})"

#==============================================================================
# NOVAS FUNÇÕES DE FASE PARA LOGGING DETALHADO
#==============================================================================

def executar_fase_1_reforco_aditivo(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int) -> Tuple[List[str], List[str]]:
    """Executa a fase de reforço e retorna a explicação robusta e um log detalhado."""
    log = []
    expl_robusta = list(expl_inicial)
    
    while True:
        status, pred_str_pert = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, class_names, direcao_override)
        
        if status.startswith("VÁLIDA"):
            log.append(f"   -> SUCESSO: Explicação com {len(expl_robusta)} features é robusta nesta direção.")
            break
        
        log.append(f"   -> FALHA: Explicação com {len(expl_robusta)} features é fraca. (Resultado da Perturbação: {pred_str_pert})")

        if len(expl_robusta) == X_train.shape[1]:
            log.append("   -> ATENÇÃO: Todas as features já foram adicionadas. Impossível reforçar mais.")
            break

        deltas = calculate_deltas(modelo, instance_df, X_train)
        indices_ordenados = np.argsort(-np.abs(deltas))
        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta}
        
        adicionou = False
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set:
                log.append(f"   -> REFORÇANDO: Adicionando a feature de maior impacto: '{feat_nome}'.")
                expl_robusta.append(f"{feat_nome} = {instance_df.iloc[0, idx]:.4f}")
                adicionou = True
                break
        if not adicionou: # Caso improvável, mas seguro
             break

    return expl_robusta, log

def executar_fase_2_minimizacao_subtrativa(modelo: LogisticRegression, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], direcao_override: int) -> Tuple[List[str], List[str]]:
    """Executa a fase de minimização e retorna a explicação final e um log detalhado."""
    log = []
    expl_minima = list(expl_robusta)

    status_inicial, _ = perturbar_e_validar(modelo, instance_df, expl_minima, X_train, t_plus, t_minus, class_names, direcao_override)
    if not status_inicial.startswith("VÁLIDA"):
        log.append("   -> PULANDO MINIMIZAÇÃO: A explicação de entrada já não é robusta.")
        return expl_minima, log

    deltas = calculate_deltas(modelo, instance_df, X_train)
    # Tenta remover da menos importante para a mais importante
    features_para_remover = sorted(
        [f.split(' = ')[0] for f in expl_minima],
        key=lambda nome: abs(deltas[X_train.columns.get_loc(nome)])
    )

    for feat_nome in features_para_remover:
        if len(expl_minima) <= 1:
            log.append("   -> Parando minimização para manter ao menos uma feature.")
            break
        
        log.append(f"   -> TENTANDO REMOVER: '{feat_nome}'...")
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        
        status_temp, pred_str_pert = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, class_names, direcao_override)
        
        if status_temp.startswith("VÁLIDA"):
            log.append(f"     - SUCESSO: A remoção de '{feat_nome}' manteve a robustez. Explicação agora com {len(expl_temp)} features.")
            expl_minima = expl_temp
        else:
            log.append(f"     - FALHA: A remoção de '{feat_nome}' quebrou a robustez. (Resultado: {pred_str_pert}). Mantendo a feature.")

    return expl_minima, log


#==============================================================================
# GERAÇÃO DE RELATÓRIO
#==============================================================================

def gerar_relatorio_consolidado(modelo: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float):
    """Gera um único relatório com detalhes completos das fases para instâncias rejeitadas."""
    caminho_relatorio = f"relatorio_consolidado_{nome_dataset}.txt"
    scores_teste = modelo.decision_function(X_test)
    intercepto = modelo.intercept_[0]
    pesos = modelo.coef_[0]
    X_train_min, X_train_max = X_train.min(), X_train.max()
    
    # Dicionário para estatísticas do resumo final
    stats = {"total_instancias": len(X_test), "status_counts": Counter(), "vencedor_rejeicao": Counter(), "tamanhos_expl_validas": []}

    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("="*105 + f"\nRELATÓRIO CONSOLIDADO DE PI-EXPLICAÇÕES - {nome_dataset.upper()}\n" + "="*105 + "\n\n")
        f.write(f"[CONFIGURAÇÃO]\n - Dataset: {nome_dataset}\n - Instâncias de Teste: {len(X_test)}\n")
        f.write(f" - Thresholds: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n - Intercepto do Modelo: {intercepto:.4f}\n\n")
        f.write("-" * 105 + "\nDETALHAMENTO POR INSTÂNCIA\n" + "-" * 105 + "\n")

        print(f"Gerando relatório para {len(X_test)} instâncias de teste...")
        for i in range(len(X_test)):
            inst_df = X_test.iloc[[i]]
            score = scores_teste[i]
            rejeitada = t_minus <= score <= t_plus
            pred_str = f"REJEITADA (Score: {score:.4f})" if rejeitada else f"{class_names[modelo.predict(inst_df)[0]]} (Score: {score:.4f})"
            f.write(f"\n--- INSTÂNCIA #{i} | Predição Original: {pred_str} | Classe Real: {class_names[y_test.iloc[i]]} ---\n")

            if rejeitada:
                deltas = calculate_deltas(modelo, inst_df, X_train)
                f.write(f"  Score da Instância: {score:.4f}\n")
                f.write(f"  {'Feature':<25} | {'Inst. Valor':>12} | {'Peso (Coef.)':>12} | {'Delta':>12} | {'Train Min':>12} | {'Train Max':>12}\n")
                f.write(f"  {'-'*25} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}\n")
                for feat_idx, feat_nome in enumerate(X_train.columns):
                    f.write(f"  {feat_nome:<25} | {inst_df.iloc[0, feat_idx]:>12.4f} | {pesos[feat_idx]:>12.4f} | {deltas[feat_idx]:>12.4f} | {X_train_min.iloc[feat_idx]:>12.4f} | {X_train_max.iloc[feat_idx]:>12.4f}\n")
                
                f.write("\n  -> ANÁLISE BI-DIRECIONAL INICIADA <-\n")
                
                expl_inicial = one_explanation(modelo, inst_df, X_train, t_plus, t_minus)
                
                # --- Corrida 1: Perturbar para AUMENTAR o score (em direção à Classe 1) ---
                f.write("\n  --- [Corrida 1: Teste de robustez contra perturbação para CLASSE 1 (Aumentar Score)] ---\n")
                f.write(f"  [Fase 0] Explicação Inicial ({len(expl_inicial)} feats): {expl_inicial}\n")
                f.write(f"  [Fase 1] Início do Reforço Aditivo:\n")
                expl_robusta1, log_fase1_1 = executar_fase_1_reforco_aditivo(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, direcao_override=0)
                f.write("\n".join(log_fase1_1) + "\n")
                f.write(f"  [Fase 2] Início da Minimização Subtrativa:\n")
                expl_final1, log_fase2_1 = executar_fase_2_minimizacao_subtrativa(modelo, inst_df, expl_robusta1, X_train, t_plus, t_minus, class_names, direcao_override=0)
                f.write("\n".join(log_fase2_1) + "\n")
                f.write(f"  --- Fim da Corrida 1. Resultado: {len(expl_final1)} features. ---\n")
                len1 = len(expl_final1)

                # --- Corrida 2: Perturbar para DIMINUIR o score (em direção à Classe 0) ---
                f.write("\n  --- [Corrida 2: Teste de robustez contra perturbação para CLASSE 0 (Diminuir Score)] ---\n")
                f.write(f"  [Fase 0] Explicação Inicial ({len(expl_inicial)} feats): {expl_inicial}\n")
                f.write(f"  [Fase 1] Início do Reforço Aditivo:\n")
                expl_robusta2, log_fase1_2 = executar_fase_1_reforco_aditivo(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, direcao_override=1)
                f.write("\n".join(log_fase1_2) + "\n")
                f.write(f"  [Fase 2] Início da Minimização Subtrativa:\n")
                expl_final2, log_fase2_2 = executar_fase_2_minimizacao_subtrativa(modelo, inst_df, expl_robusta2, X_train, t_plus, t_minus, class_names, direcao_override=1)
                f.write("\n".join(log_fase2_2) + "\n")
                f.write(f"  --- Fim da Corrida 2. Resultado: {len(expl_final2)} features. ---\n")
                len2 = len(expl_final2)

                # --- Decisão Final ---
                f.write("\n  >> DECISÃO FINAL DA INSTÂNCIA:\n")
                if len1 <= len2:
                    vencedor = "Corrida 1 (Aumentar Score)" + (" (empate)" if len1 == len2 else "")
                    expl_final = expl_final1
                    stats['vencedor_rejeicao'][vencedor] += 1
                else:
                    vencedor = "Corrida 2 (Diminuir Score)"
                    expl_final = expl_final2
                    stats['vencedor_rejeicao'][vencedor] += 1
                
                status_final_check, _ = perturbar_e_validar(modelo, inst_df, expl_final, X_train, t_plus, t_minus, class_names, direcao_override=0) # Checa com uma direção
                stats['status_counts'][status_final_check] += 1
                stats['tamanhos_expl_validas'].append(len(expl_final))
                f.write(f"  Vencedor: {vencedor}. A explicação mais curta foi encontrada com {len(expl_final)} features.\n")
                f.write(f"  PI-EXPLICAÇÃO FINAL: {expl_final}\n")

            else:
                f.write("  -> Instância CLASSIFICADA. Análise não detalhada neste relatório.\n")
                stats['status_counts']["CLASSIFICADA"] += 1
        
        # --- Resumo Final ---
        f.write("\n\n" + "="*105 + "\nRESUMO DAS MÉTRICAS GERAIS\n" + "="*105 + "\n\n")
        f.write(f"Total de Instâncias Processadas: {stats['total_instancias']}\n\nContagem de Status das Explicações Finais:\n")
        for status, count in stats["status_counts"].most_common(): f.write(f"  - {status}: {count} ({(count / stats['total_instancias']) * 100:.2f}%)\n")
        f.write("\nEstratégia Vencedora para Instâncias Rejeitadas:\n")
        total_rej = sum(stats["vencedor_rejeicao"].values())
        if total_rej > 0:
            for tipo, count in stats["vencedor_rejeicao"].items(): f.write(f"  - {tipo}: {count} ({(count / total_rej) * 100:.2f}%)\n")
        else: f.write("  - Nenhuma instância rejeitada.\n")

    print(f"\nRelatório consolidado salvo em: {caminho_relatorio}")

#==============================================================================
# FUNÇÃO PRINCIPAL
#==============================================================================

def main():
    """Função principal que orquestra a execução do script."""
    selecao_result = selecionar_dataset_e_classe()
    if not selecao_result:
        print("Nenhum dataset selecionado. Encerrando.")
        return
    
    nome_dataset, _, X_data, y_data, nomes_classes = selecao_result

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_data
    )
    
    modelo = LogisticRegression(**DEFAULT_LOGREG_PARAMS)
    modelo.fit(X_train, y_train)
    print(f"\nModelo treinado. Acurácia no teste (sem rejeição): {modelo.score(X_test, y_test):.2%}")

    print("Calculando thresholds de rejeição...")
    t_plus, t_minus = calcular_thresholds(modelo, X_train, y_train)
    print(f"Thresholds definidos: t+ = {t_plus:.4f}, t- = {t_minus:.4f}")
    
    gerar_relatorio_consolidado(
        modelo, X_test, y_test, X_train, nomes_classes, nome_dataset, t_plus, t_minus
    )
    
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()