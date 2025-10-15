import os
import json
import time
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any
from datasets.datasets_Mateus_comparation_complete import selecionar_dataset_e_classe
from auxiliary_files.results_handler import update_method_results


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
    "pima_indians_diabetes":{'test_size': 0.3, 'rejection_cost': 0.24},
    "sonar":                {'test_size': 0.3, 'rejection_cost': 0.24},
    "vertebral_column":     {'test_size': 0.3, 'rejection_cost': 0.24},
    "breast_cancer":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "spambase":             {'test_size': 0.1, 'rejection_cost': 0.24},
    "banknote_auth":        {'test_size': 0.2, 'rejection_cost': 0.24},
    "heart_disease":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine_quality":         {'test_size': 0.2, 'rejection_cost': 0.24},
    "creditcard":           {'subsample_size': 0.1, 'test_size': 0.3, 'rejection_cost': 0.24}
}

# Constante global que a sua função 'calcular_thresholds' espera encontrar
# O valor dela será atualizado dinamicamente pela função main
WR_REJECTION_COST: float = 0.0

# outras constantes
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
    """Carrega o arquivo JSON com os hiperparâmetros otimizados."""
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
    """Calcula os limiares da zona de rejeição (t+ e t-)."""
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
        if premis_class == 1:
            pior_valor_escalonado = X_train_scaled_min[i] if coef > 0 else X_train_scaled_max[i]
        else:
            pior_valor_escalonado = X_train_scaled_max[i] if coef > 0 else X_train_scaled_min[i]
        deltas[i] = (scaled_val - pior_valor_escalonado) * coef
    return deltas

# [CLASSE REJEITADA] 01: MÉTODO DE INICIAÇÃO FORMAL (ARTIGO)
def one_explanation_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, premis_class: int, log_collector: List[str]):
    score = modelo.decision_function(instance_df)[0]
    deltas = calculate_deltas(modelo, instance_df, X_train, premis_class)
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []
    score_base = score - np.sum(deltas)
    soma_deltas_cumulativa = score_base
    target_score = t_plus if premis_class == 1 else t_minus

    for i in indices_ordenados:
        if (premis_class == 1 and soma_deltas_cumulativa > target_score and explicacao) or \
           (premis_class == 0 and soma_deltas_cumulativa < target_score and explicacao):
            break

        feature_nome = X_train.columns[i]
        soma_deltas_cumulativa += deltas[i]
        explicacao.append(f"{feature_nome} = {instance_df.iloc[0, i]:.4f}")

    if not explicacao and len(deltas) > 0:
        explicacao.append(f"{X_train.columns[indices_ordenados[0]]} = {instance_df.iloc[0, indices_ordenados[0]]:.4f}")
    
    log_collector.append(f"   [Passo 1] Geração da Explicação Inicial:")
    log_collector.append(f"     - Explicação Inicial Gerada ({len(explicacao)} feats): {explicacao}")

    return explicacao

# [CLASSE REJEITADA] 03: GERAÇÃO DAS DUAS INICIAÇÕES FORMAIS
def gerar_explicacao_inicial_rejeitada_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, log_collector: List[str]):
    log_collector.append("\n==================== ESTRATÉGIA DE BUSCA 1 (Objetivo: Evitar Classe 0) ====================")
    expl_inicial_path1 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 1, log_collector)

    log_collector.append("\n==================== ESTRATÉGIA DE BUSCA 2 (Objetivo: Evitar Classe 1) ====================")
    expl_inicial_path2 = one_explanation_formal(modelo, instance_df, X_train, t_plus, t_minus, 0, log_collector)

    return expl_inicial_path1, expl_inicial_path2

# [TÓPICO GERAL] 02: VALIDAÇÃO DE ROBUSTEZ (O JUIZ)
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
        
        valor_pert = (train_min if coef > 0 else train_max) if perturbar_para_diminuir_score else (train_max if coef > 0 else train_min)
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
def executar_fase_1_reforco_bidirecional(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str], premissa_ordenacao: int) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo (Premissa de ordenação: evitar Classe {1 - premissa_ordenacao})")
    expl_robusta = list(expl_inicial)
    adicoes = 0
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
        
        if not adicionou_feature: break
    
    log_collector.append(f"\n   -> Fim da Fase 1. Explicação robusta final tem {len(expl_robusta)} features.")
    return expl_robusta, adicoes

# [CLASSE REJEITADA] 05: MINIMIZAÇÃO BIDIRECIONAL (REMOVENDO FEATURES)
def executar_fase_2_minimizacao_bidirecional(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str], premissa_ordenacao: int) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 2] Início da Minimização Subtrativa (Premissa de ordenação: evitar Classe {1-premissa_ordenacao})")
    expl_minima = list(expl_robusta)
    remocoes = 0
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
def executar_fase_1_reforco_unidirecional(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]) -> Tuple[List[str], int]:
    log_collector.append(f"\n   [Fase 1] Início do Reforço Aditivo Uni-Direcional")
    expl_robusta = list(expl_inicial)
    adicoes = 0
    pred_class = modelo.predict(instance_df)[0]
    direcao = 1 if pred_class == 1 else 0
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
        if len(expl_robusta) == X_train.shape[1]: break

        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta}
        adicionou_feature = False
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set:
                expl_robusta.append(f"{feat_nome} = {instance_df.iloc[0, idx]:.4f}")
                adicoes += 1
                adicionou_feature = True
                break
        if not adicionou_feature: break

    log_collector.append(f"\n   -> Fim da Fase 1. Explicação robusta final tem {len(expl_robusta)} features.")
    return expl_robusta, adicoes

# [CLASSE REJEITADA] 06: BUSCA OTIMIZADA (A GENERAL)
def encontrar_explicacao_otimizada_para_rejeitada(
    modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial_path1: List[str], expl_inicial_path2: List[str], 
    X_train: pd.DataFrame, t_plus: float, t_minus: float, class_names: List[str], log_collector: List[str]
) -> Tuple[List[str], int, int, int, Dict, Dict]:
    
    path1_results = {}
    expl_robusta_1, adicoes_1 = executar_fase_1_reforco_bidirecional(modelo, instance_df, expl_inicial_path1, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=1)
    expl_final_1, remocoes_1 = executar_fase_2_minimizacao_bidirecional(modelo, instance_df, expl_robusta_1, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=1)
    path1_results['adicoes'] = adicoes_1
    path1_results['remocoes'] = remocoes_1
    path1_results['expl_final'] = expl_final_1
    path1_results['expl_robusta_len'] = len(expl_robusta_1)
    
    path2_results = {}
    expl_robusta_2, adicoes_2 = executar_fase_1_reforco_bidirecional(modelo, instance_df, expl_inicial_path2, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=0)
    expl_final_2, remocoes_2 = executar_fase_2_minimizacao_bidirecional(modelo, instance_df, expl_robusta_2, X_train, t_plus, t_minus, class_names, log_collector, premissa_ordenacao=0)
    path2_results['adicoes'] = adicoes_2
    path2_results['remocoes'] = remocoes_2
    path2_results['expl_final'] = expl_final_2
    path2_results['expl_robusta_len'] = len(expl_robusta_2)

    if len(expl_final_1) <= len(expl_final_2):
        return expl_final_1, adicoes_1, remocoes_1, 1, path1_results, path2_results
    else:
        return expl_final_2, adicoes_2, remocoes_2, 0, path1_results, path2_results

#==============================================================================
# GERAÇÃO DE RELATÓRIO (Função Principal)
#==============================================================================
def gerar_relatorio_aprimorado(modelo: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, 
                               class_names: List[str], nome_dataset: str, t_plus: float, t_minus: float, 
                               rejection_cost: float, test_size_atual: float):
    
    caminho_relatorio = os.path.join("report", f"peab_comp_mat_{nome_dataset}.txt")
    if not os.path.exists("report"):
        os.makedirs("report")

    stats = {
        "tamanhos_expl_pos": [], "tamanhos_expl_neg": [], "tamanhos_expl_rej": [],
        "all_features_in_exps": [], "adicoes_fase1": [], "remocoes_fase2": [],
        "tempos_execucao": [], "resultados_completos": []
    }

    start_time = time.perf_counter()
    print(f"Processando {len(X_test)} instâncias de teste para gerar o relatório...")
    scores_teste = modelo.decision_function(X_test)

    for i in range(len(X_test)):
        inst_start_time = time.perf_counter()
        log_instancia_atual = []
        inst_df = X_test.iloc[[i]]
        score = scores_teste[i]
        pred_class = modelo.predict(inst_df)[0]
        rejeitada = t_minus <= score <= t_plus
        
        resultado_instancia = {'id_instancia': i, 'score': score, 'classe_real': class_names[y_test.iloc[i]]}
        
        if rejeitada:
            resultado_instancia['status'] = "REJEITADA"
            expl_inicial_f1, expl_inicial_f2 = gerar_explicacao_inicial_rejeitada_formal(modelo, inst_df, X_train, t_plus, t_minus, log_instancia_atual)
            expl_final, adicoes, remocoes, premissa, path1, path2 = encontrar_explicacao_otimizada_para_rejeitada(
                modelo, inst_df, expl_inicial_f1, expl_inicial_f2, X_train, t_plus, t_minus, class_names, log_instancia_atual
            )
            stats['tamanhos_expl_rej'].append(len(expl_final))
            resultado_instancia['caminho1'] = {'expl_inicial': expl_inicial_f1, **path1}
            resultado_instancia['caminho2'] = {'expl_inicial': expl_inicial_f2, **path2}
            resultado_instancia['caminho_vencedor'] = 1 if premissa == 1 else 2
        else: # Instância Classificada
            status_str = f"CLASSE {pred_class}"
            resultado_instancia['status'] = status_str
            expl_inicial = one_explanation_formal(modelo, inst_df, X_train, t_plus, t_minus, pred_class, log_instancia_atual)
            expl_final, adicoes = executar_fase_1_reforco_unidirecional(modelo, inst_df, expl_inicial, X_train, t_plus, t_minus, class_names, log_instancia_atual)
            remocoes = 0
            resultado_instancia['expl_inicial'] = expl_inicial
            if pred_class == 1:
                stats['tamanhos_expl_pos'].append(len(expl_final))
            else:
                stats['tamanhos_expl_neg'].append(len(expl_final))
        
        log_instancia_atual.append(f"\n  >> RESULTADO FINAL DA INSTÂNCIA #{i}:")
        log_instancia_atual.append(f"     - PI-EXPLICAÇÃO FINAL (Tamanho: {len(expl_final)}):")
        if expl_final:
            premissa_final = premissa if rejeitada else pred_class
            deltas_finais = calculate_deltas(modelo, inst_df, X_train, premis_class=premissa_final)
            for feat_explicacao in expl_final:
                nome_feat = feat_explicacao.split(' = ')[0]
                idx_feat = X_train.columns.get_loc(nome_feat)
                delta_val = deltas_finais[idx_feat]
                log_instancia_atual.append(f"       - {feat_explicacao}  (Delta: {delta_val:.4f})")

        stats['all_features_in_exps'].extend([feat.split(' = ')[0] for feat in expl_final])
        stats['adicoes_fase1'].append(adicoes)
        stats['remocoes_fase2'].append(remocoes)
        
        resultado_instancia.update({
            'explicacao_final': expl_final, 'tamanho_explicacao': len(expl_final),
            'adicoes_fase1': adicoes, 'remocoes_fase2': remocoes,
            'log_detalhado': "\n".join(log_instancia_atual)
        })
        stats['resultados_completos'].append(resultado_instancia)
        stats['tempos_execucao'].append(time.perf_counter() - inst_start_time)

    total_duration = time.perf_counter() - start_time
    
    # Escrevendo o arquivo de log aprimorado
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("       RELATÓRIO DE ANÁLISE DE EXPLICAÇÕES ABDUTIVAS (MÉTODO PEAB)\n")
        f.write("================================================================================\n\n")

        f.write("[CONFIGURAÇÕES GERAIS E DO MODELO]\n\n")
        f.write(f"  - Dataset: {nome_dataset}\n")
        f.write(f"  - Total de Instâncias de Teste: {len(X_test)}\n")
        f.write(f"  - Tamanho do Conjunto de Teste: {test_size_atual:.0%}\n")
        f.write(f"  - Acurácia do Modelo (teste, sem rejeição): {modelo.score(X_test, y_test):.2%}\n")
        f.write(f"  - Thresholds de Rejeição: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n")
        f.write(f"  - Custo de Rejeição (WR): {rejection_cost:.2f}\n")
        f.write(f"  - Tempo Total de Geração das Explicações: {total_duration:.2f} segundos\n")
        f.write(f"  - Tempo Médio por Instância: {np.mean(stats['tempos_execucao']):.4f} segundos\n\n")

        f.write("================================================================================\n")
        f.write("                    ANÁLISE DETALHADA POR INSTÂNCIA\n")
        f.write("================================================================================\n\n")

        rejeitadas = [r for r in stats['resultados_completos'] if r['status'] == 'REJEITADA']
        classificadas = [r for r in stats['resultados_completos'] if r['status'] != 'REJEITADA']
        
        f.write(f"--------------------------------------------------------------------------------\n")
        f.write(f"                         SEÇÃO A: INSTÂNCIAS REJEITADAS ({len(rejeitadas)})\n")
        f.write(f"--------------------------------------------------------------------------------\n\n")
        
        for r in rejeitadas:
            f.write(f"--- INSTÂNCIA #{r['id_instancia']} | Status: {r['status']} (Score: {r['score']:.4f}) | Classe Real: {r['classe_real']} ---\n")
            f.write("   --> Objetivo: Encontrar a menor explicação que mantenha a instância na zona de rejeição, testando dois caminhos de otimização.\n\n")
            
            for i, caminho_data in enumerate([r['caminho1'], r['caminho2']]):
                caminho_num = i + 1
                objetivo = "evitar a Classe 0" if i == 0 else "evitar a Classe 1"
                f.write(f"   [Caminho {caminho_num}: Otimizando para {objetivo}]\n")
                f.write(f"     - Explicação Inicial: {len(caminho_data['expl_inicial'])} features\n")
                
                if caminho_data['adicoes'] == 0:
                    f.write("     - Validação e Reforço (Fase 1): A explicação inicial já se mostrou robusta. Nenhuma feature adicionada.\n")
                else:
                    f.write(f"     - Validação e Reforço (Fase 1): {caminho_data['adicoes']} feature(s) adicionada(s) para garantir robustez.\n")
                
                remocoes_count = caminho_data['remocoes']
                total_para_remover = caminho_data['expl_robusta_len']
                f.write(f"     - Minimização (Fase 2): Tentando remover {total_para_remover} features de menor impacto...\n")
                
                caminho_log_str = f"ESTRATÉGIA DE BUSCA {caminho_num}"
                secao_log_completa = r['log_detalhado'].split(caminho_log_str)
                if len(secao_log_completa) > 2:
                    secao_log_fase2 = secao_log_completa[2].split('[Fase 2]')[1]
                    sucessos = re.findall(r"SUCESSO TOTAL: A remoção de '(.+?)'.+?agora com (\d+) features\.", secao_log_fase2)
                    if sucessos:
                        for idx, sucesso in enumerate(sucessos):
                            scores_match = re.findall(r"Score da Instância Perturbada: ([\-0-9\.]+)", secao_log_fase2.split(sucesso[0])[idx])[-2:]
                            f.write(f"       - SUCESSO: Remoção de '{sucesso[0]}' manteve a robustez. (Scores: {scores_match[0]}, {scores_match[1]})\n")
                
                f.write(f"       - Resumo: {remocoes_count} de {total_para_remover} features removidas com sucesso.\n")
                f.write(f"     --> Resultado do Caminho {caminho_num}: Explicação com {len(caminho_data['expl_final'])} features.\n\n")

            f.write("   [Decisão Final]\n")
            f.write(f"   -> O Caminho {r['caminho_vencedor']} foi escolhido.\n\n")
            f.write(f"  >> RESULTADO FINAL DA INSTÂNCIA #{r['id_instancia']}:\n")
            f.write(f"     - PI-EXPLICAÇÃO FINAL (Tamanho: {r['tamanho_explicacao']}):\n")
            final_log_section = r['log_detalhado'].split(f">> RESULTADO FINAL DA INSTÂNCIA #{r['id_instancia']}")[1]
            delta_lines = re.findall(r"-\s(.+?\(Delta:.+?\))", final_log_section)
            for line in delta_lines:
                f.write(f"       - {line}\n")
            f.write("\n\n")
        
        f.write(f"--------------------------------------------------------------------------------\n")
        f.write(f"                       SEÇÃO B: INSTÂNCIAS CLASSIFICADAS ({len(classificadas)})\n")
        f.write(f"--------------------------------------------------------------------------------\n\n")
        
        for r in classificadas:
            f.write(f"--- INSTÂNCIA #{r['id_instancia']} | Status: {r['status']} (Score: {r['score']:.4f}) | Classe Real: {r['classe_real']} ---\n")
            f.write(f"   --> Objetivo: Encontrar a menor explicação que garanta a classificação como {r['status']}.\n\n")
            f.write(f"     - Explicação Inicial (Candidata): {len(r['expl_inicial'])} features\n")
            
            score_perturbado_match = re.search(r"Score da Instância Perturbada: ([\-0-9\.]+)", r['log_detalhado'])
            if r['adicoes_fase1'] == 0 and score_perturbado_match:
                score_str = score_perturbado_match.group(1).strip()
                f.write(f"     - Validação (Fase 1): A explicação candidata se mostrou robusta. (Score após perturbação: {score_str})\n\n")
            else:
                 f.write(f"     - Validação e Reforço (Fase 1): {r['adicoes_fase1']} feature(s) foram adicionadas para garantir robustez.\n\n")
            
            f.write(f"  >> RESULTADO FINAL DA INSTÂNCIA #{r['id_instancia']}:\n")
            f.write(f"     - PI-EXPLICAÇÃO FINAL (Tamanho: {r['tamanho_explicacao']}):\n")
            final_log_section = r['log_detalhado'].split(f">> RESULTADO FINAL DA INSTÂNCIA #{r['id_instancia']}")[1]
            delta_lines = re.findall(r"-\s(.+?\(Delta:.+?\))", final_log_section)
            for line in delta_lines:
                f.write(f"       - {line}\n")
            f.write("\n\n")

        # Seção de Resumo Estatístico
        f.write("================================================================================\n")
        f.write("                         RESUMO ESTATÍSTICO GERAL\n")
        f.write("================================================================================\n\n")
        
        f.write("Métricas de Desempenho do Classificador\n")
        f.write("-----------------------------------------\n")
        rejeitadas_mask = np.array([r['status'] == 'REJEITADA' for r in stats['resultados_completos']])
        aceitas_mask = ~rejeitadas_mask
        taxa_rejeicao = np.mean(rejeitadas_mask)
        acuracia_com_rejeicao = modelo.score(X_test[aceitas_mask], y_test[aceitas_mask]) if np.sum(aceitas_mask) > 0 else "N/A"
        f.write(f"  - Taxa de Rejeição no Teste: {taxa_rejeicao:.2%} ({sum(rejeitadas_mask)} de {len(X_test)} instâncias)\n")
        f.write(f"  - Acurácia com Opção de Rejeição (nas {sum(aceitas_mask)} instâncias aceitas): {acuracia_com_rejeicao if isinstance(acuracia_com_rejeicao, str) else f'{acuracia_com_rejeicao:.2%}'}\n\n")

        f.write("Estatísticas sobre o Tamanho das Explicações\n")
        f.write("----------------------------------------------\n")
        for nome_classe, lista in [("Positiva", stats['tamanhos_expl_pos']), ("Negativa", stats['tamanhos_expl_neg']), ("Rejeitada", stats['tamanhos_expl_rej'])]:
            n = len(lista)
            perc = (n / len(X_test) * 100) if X_test.shape[0] > 0 else 0
            f.write(f"  - Classe {nome_classe} ({n} instâncias - {perc:.1f}% do total):\n")
            if n > 0:
                f.write(f"    - Tamanho Médio: {np.mean(lista):.2f} ± {np.std(lista):.2f} (Min: {np.min(lista)}, Max: {np.max(lista)})\n")
        f.write("\n")

        f.write("Análise de Relevância das Features\n")
        f.write("------------------------------------\n")
        if stats['all_features_in_exps']:
            f.write("  - Top 10 Features Mais Frequentes nas Explicações:\n")
            for feat, count in Counter(stats['all_features_in_exps']).most_common(10):
                f.write(f"    - {feat}: {count} vezes\n")
        f.write("\n")
        
        f.write("Análise do Processo de Geração\n")
        f.write("--------------------------------\n")
        f.write("  - Para Instâncias Classificadas (Positivas/Negativas):\n")
        if len(classificadas) > 0:
            adicoes_c = sum(1 for r in classificadas if r['adicoes_fase1'] > 0)
            f.write(f"    - Precisaram de reforço (Fase 1): {adicoes_c} ({adicoes_c/len(classificadas)*100:.2f}%)\n")
        
        f.write("  - Para Instâncias Rejeitadas:\n")
        if len(rejeitadas) > 0:
            adicoes_r = sum(1 for r in rejeitadas if r['caminho1']['adicoes'] > 0 or r['caminho2']['adicoes'] > 0)
            f.write(f"    - Precisaram de reforço (Fase 1): {adicoes_r} ({adicoes_r/len(rejeitadas)*100:.2f}%)\n")
            remocoes_r = sum(1 for r in rejeitadas if r['remocoes_fase2'] > 0)
            f.write(f"    - Tiveram remoção de features (Fase 2): {remocoes_r} ({remocoes_r/len(rejeitadas)*100:.2f}%)\n")
            if remocoes_r > 0:
                media_remocoes_r = np.mean([r['remocoes_fase2'] for r in rejeitadas if r['remocoes_fase2'] > 0])
                f.write(f"      - Média de features removidas (quando houve): {media_remocoes_r:.2f}\n")
        f.write("\n")

    print(f"\nRelatório final salvo em: {caminho_relatorio}")
    return stats


#==============================================================================
# FUNÇÃO PRINCIPAL (MAIN)
#==============================================================================
def main():
    """
    Orquestra o carregamento, treinamento, análise e geração de relatórios.
    """
    todos_hiperparametros = carregar_hiperparametros()
    selecao_result = selecionar_dataset_e_classe()
    if not selecao_result or selecao_result[0] is None:
        print("Nenhum dataset selecionado. Encerrando.")
        return
    
    nome_dataset, _, X_data, y_data, nomes_classes = selecao_result
    config_experimento = DATASET_CONFIG.get(nome_dataset, {'test_size': 0.3, 'rejection_cost': 0.25})
    test_size_atual = config_experimento['test_size']
    rejection_cost_atual = config_experimento['rejection_cost']

    global WR_REJECTION_COST
    WR_REJECTION_COST = rejection_cost_atual

    parametros_para_modelo = todos_hiperparametros.get(nome_dataset, {}).get('params', DEFAULT_LOGREG_PARAMS)
    
    pipeline_modelo = Pipeline([
        ('scaler', MinMaxScaler()),
        ('modelo', LogisticRegression(**parametros_para_modelo, random_state=RANDOM_STATE))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y_data
    )
    
    pipeline_modelo.fit(X_train, y_train)
    
    print(f"\nModelo treinado. Acurácia no teste (sem rejeição): {pipeline_modelo.score(X_test, y_test):.2%}")
    print(f"Custo de Rejeição (WR) definido como: {WR_REJECTION_COST}")
    
    print("Calculando thresholds de rejeição...")
    t_plus, t_minus = calcular_thresholds(pipeline_modelo, X_train, y_train)
    print(f"Thresholds definidos: t+ = {t_plus:.4f}, t- = {t_minus:.4f}")

    # --- NOVO FLUXO UNIFICADO ---
    stats = gerar_relatorio_aprimorado(
        pipeline_modelo, X_test, y_test, X_train, y_train, nomes_classes, nome_dataset, 
        t_plus, t_minus, WR_REJECTION_COST, test_size_atual
    )

    # 3. Prepara os dados e atualiza o JSON centralizado
    def get_stats_from_list(data_list):
        if not data_list: return {'count': 0, 'min_length': 0, 'mean_length': 0, 'max_length': 0, 'std_length': 0}
        return {
            'count': len(data_list),
            'min_length': int(np.min(data_list)),
            'mean_length': float(np.mean(data_list)),
            'max_length': int(np.max(data_list)),
            'std_length': float(np.std(data_list))
        }

    rejeitadas_mask = np.array([r['status'] == 'REJEITADA' for r in stats['resultados_completos']])
    aceitas_mask = ~rejeitadas_mask
    acc_com_rej = pipeline_modelo.score(X_test[aceitas_mask], y_test[aceitas_mask]) if sum(aceitas_mask) > 0 else 0.0

    results_data = {
        "config": {"dataset_name": nome_dataset, "test_size": test_size_atual, "rejection_cost": WR_REJECTION_COST},
        "thresholds": {"t_plus": float(t_plus), "t_minus": float(t_minus)},
        "performance": {
            "accuracy_without_rejection": pipeline_modelo.score(X_test, y_test) * 100,
            "accuracy_with_rejection": acc_com_rej * 100,
            "rejection_rate": np.mean(rejeitadas_mask) * 100
        },
        "explanation_stats": {
            "positive": get_stats_from_list(stats['tamanhos_expl_pos']),
            "negative": get_stats_from_list(stats['tamanhos_expl_neg']),
            "rejected": get_stats_from_list(stats['tamanhos_expl_rej'])
        },
        "computation_time": {
            "total": sum(stats['tempos_execucao']),
            "mean_per_instance": np.mean(stats['tempos_execucao'])
        },
        "top_features": [{"feature": f, "count": c} for f, c in Counter(stats['all_features_in_exps']).most_common(10)]
    }
    
    update_method_results("peab", nome_dataset, results_data)
    
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()