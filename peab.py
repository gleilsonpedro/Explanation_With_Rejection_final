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

# [MODIFICAÇÃO IMPORTANTE] Mantendo suas importações originais
from data.datasets import selecionar_dataset_e_classe, carregar_dataset
from utils.results_handler import update_method_results

#==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
#==============================================================================
RANDOM_STATE: int = 42

# Configurações específicas de MNIST (aplicadas automaticamente quando mnist é selecionado)
MNIST_CONFIG = {
    'feature_mode': 'raw',           # 'raw' (784 features) ou 'pool2x2' (196 features)
    'digit_pair': (3, 8),            # Par de dígitos para comparação (classe A vs classe B)
    'top_k_features': None,          # Número de features mais importantes (None = usar todas) ou  o numero com a quantidade de features mais importantes, ex 200
    'test_size': 0.3,                # Proporção do conjunto de teste
    'rejection_cost': 0.24,          # Custo de rejeição
    'subsample_size': 1.0           # Proporção de subamostragem do dataset completo
}

DATASET_CONFIG = {
    # "iris":                 {'test_size': 0.3, 'rejection_cost': 0.24},  # substituído por MNIST no menu
    "mnist":                MNIST_CONFIG,  # Configuração automática do MNIST
    "wine":                 {'test_size': 0.3, 'rejection_cost': 0.24},
    "breast_cancer":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "pima_indians_diabetes":{'test_size': 0.3, 'rejection_cost': 0.24},
    "vertebral_column":     {'test_size': 0.3, 'rejection_cost': 0.24},
    "sonar":                {'test_size': 0.3, 'rejection_cost': 0.24},
    "spambase":             {'test_size': 0.1, 'rejection_cost': 0.24},
    "banknote_auth":        {'test_size': 0.2, 'rejection_cost': 0.24},
    "heart_disease":        {'test_size': 0.3, 'rejection_cost': 0.24},
    "wine_quality":         {'test_size': 0.2, 'rejection_cost': 0.24},
    "creditcard":           {'subsample_size': 0.1, 'test_size': 0.3, 'rejection_cost': 0.24}
}
OUTPUT_BASE_DIR: str = 'results/report/peab'
HIPERPARAMETROS_FILE: str = 'json/hiperparametros.json'
DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2', 'C': 0.01, 'solver': 'liblinear', 'max_iter': 1000
}

#==============================================================================
# [NOVA SEÇÃO] CONTROLES E TEMPLATES DE LOG TÉCNICO
#==============================================================================

# Ativa o formato técnico conciso por instância (mantém JSON inalterado)
TECHNICAL_LOGS: bool = True
MAX_LOG_FEATURES: int = 200  # acima disso, o log técnico por instância é suprimido automaticamente
MAX_LOG_STEPS: int = 60      # no máximo quantas linhas de passos detalhados serão emitidas por instância

SYMBOL_LEGEND = [
    "LEGENDA DOS SÍMBOLOS",
    "   δ = Contribuição individual (w_i × x_i)",
    "   Σδ = Soma acumulada (intercepto + Σδ_i)",
    "   ● = Feature mantida (essencial)",
    "   ○ = Feature removida (não essencial)",
    "   ↑ = Feature aumenta score (favorável à classe 1)",
    "   ↓ = Feature diminui score (favorável à classe 0)",
    "   s'= Valor da feature no pior cenário(wrost case)"
]

LOG_TEMPLATES = {
    # Cabeçalho de processamento por instância (para relatório)
    'processamento_header': (
        "**********  PROCESSAMENTO POR INSTÂNCIA  **********\n"
    ),

    # Instâncias CLASSIFICADAS
    'classificada_analise': "├── Análise: Score está {posicao}. Buscando o menor conjunto que garante a classificação.",
    'classificada_min_inicio': "├── Iniciando processo de minimização com {num_features} features.",
    'classificada_ordem': "├── Tentativas de desafixação (ordem de maior impacto |δ|): {lista}",
    'classificada_step_sucesso': "├─ ○ {feat} (δ: {delta:+.3f}): s' = {score:.3f} ({cond}) → SUCESSO. DESAFIXADA.",
    'classificada_step_falha': "├─ ● {feat} (δ: {delta:+.3f}): s' = {score:.3f} ({cond}) → FALHA. ESSENCIAL.",

    # Instâncias REJEITADAS
    'rejeitada_analise': "├── Zona de Rejeição: [{t_minus:.4f}, {t_plus:.4f}]",
    'rejeitada_prova_header': "├── Prova de Minimalidade (partindo de conjunto robusto após heurística; verificação bidirecional):",
    'rejeitada_feat_header_sucesso': "├─ ○ {feat} (δ: {delta:+.3f}):",
    'rejeitada_feat_header_falha': "├─ ● {feat} (δ: {delta:+.3f}):",
    'rejeitada_subteste_neg': "│   ├─ Teste vs Lado Negativo: s' = {score:.3f} ({cmp}) {ok}",
    'rejeitada_subteste_pos': "│   └─ Teste vs Lado Positivo: s' = {score:.3f} ({cmp}) {ok}",
    'rejeitada_feat_footer_sucesso': "│   └─> SUCESSO. Feature DESAFIXADA.",
    'rejeitada_feat_footer_falha': "│   └─> FALHA. Feature ESSENCIAL (precisa ser fixada).",
}
# Flag para permitir desativar a fase 1 de reforço para INSTÂNCIAS CLASSIFICADAS (positivas/negativas).
# Quando True, para classificadas usa diretamente a explicação inicial (one_explanation_formal)
# e segue para a fase de minimização; rejeitadas continuam com reforço bidirecional.
DISABLE_REFORCO_CLASSIFICADAS: bool = True
# Largura mínima opcional da zona de rejeição. Se > 0 força t_plus - t_minus >= MIN_REJECTION_WIDTH.
# Ajuste se desejar garantir rejeição em pares ambíguos. Coloque 0 para comportamento antigo.
MIN_REJECTION_WIDTH: float = 0.0
# coisas do antigo com o novo _ remover depois de testes de funcionamento
def _get_lr(modelo: Pipeline):
    """Retorna a etapa de Regressão Logística independente do nome do passo ('model' ou 'modelo')."""
    if 'model' in modelo.named_steps:
        return modelo.named_steps['model']
    if 'modelo' in modelo.named_steps:
        return modelo.named_steps['modelo']
    raise KeyError("Nenhum passo de regressão logística encontrado no Pipeline ('model' ou 'modelo')")


#==============================================================================
#  LÓGICA FORMAL DE EXPLICAÇÃO 
#==============================================================================
def carregar_hiperparametros(caminho_arquivo: str = HIPERPARAMETROS_FILE) -> dict:
    try:
        with open(caminho_arquivo, 'r') as f:
            params = json.load(f)
        print(f"\n[INFO] Arquivo de hiperparâmetros '{caminho_arquivo}' carregado com sucesso.")
        return params
    except FileNotFoundError:
        print(f"\n[AVISO] Arquivo '{caminho_arquivo}' não encontrado. Usando parâmetros padrão.")
        return {}
    except json.JSONDecodeError:
        print(f"\n[ERRO] Arquivo '{caminho_arquivo}' corrompido. Usando parâmetros padrão.")
        return {}

def calculate_deltas(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, premis_class: int) -> np.ndarray:
    scaler = modelo.named_steps['scaler'] # normalizador
    logreg = _get_lr(modelo) # modelo de reg logist
    coefs = logreg.coef_[0] # pesos
    instance_df_ordered = instance_df[X_train.columns] # ordena colunas
    scaled_instance_vals = scaler.transform(instance_df_ordered)[0] # instancia escalonada
    X_train_scaled = scaler.transform(X_train) 
    X_train_scaled_min = X_train_scaled.min(axis=0) 
    X_train_scaled_max = X_train_scaled.max(axis=0)
    deltas = np.zeros_like(coefs) # craindo um arrai de zeros com o mesmo tamanho dos coeficientes
    for i, (coef, scaled_val) in enumerate(zip(coefs, scaled_instance_vals)):
        if premis_class == 1:
            pior_valor_escalonado = X_train_scaled_min[i] if coef > 0 else X_train_scaled_max[i]
        else:
            pior_valor_escalonado = X_train_scaled_max[i] if coef > 0 else X_train_scaled_min[i]
        # delta = (x_i - s') * w_i
        deltas[i] = (scaled_val - pior_valor_escalonado) * coef
    return deltas

def one_explanation_formal(modelo: Pipeline, instance_df: pd.DataFrame, X_train: pd.DataFrame, t_plus: float, t_minus: float, premis_class: int) -> List[str]:
    score = modelo.decision_function(instance_df)[0]
    explicacao = []
    
    # Calcula os deltas (contribuição de cada feature no pior caso)
    deltas = calculate_deltas(modelo, instance_df, X_train, premis_class)
    
    # Ordena pelo maior impacto absoluto (Lógica Original do PEAB)
    indices_ordenados = np.argsort(-np.abs(deltas))
    
    score_base = score - np.sum(deltas)
    soma_deltas_cumulativa = score_base 
    target_score = t_plus if premis_class == 1 else t_minus
    
    # [NOVO] Margem de tolerância para erros numéricos (float)
    # Isso impede que o algoritmo adicione uma feature extra se o score for exatamente igual ao limiar.
    EPSILON = 1e-6 

    for i in indices_ordenados:
        feature_nome = X_train.columns[i]
        valor_original_feature = instance_df.iloc[0, X_train.columns.get_loc(feature_nome)]
        
        # Só adiciona se o delta for relevante (evita sujeira numérica)
        if abs(deltas[i]) > 1e-9:
             soma_deltas_cumulativa += deltas[i]
             explicacao.append(f"{feature_nome} = {valor_original_feature:.4f}")
        
        # [ALTERAÇÃO AQUI] melhorando as classificadas
        # Antes: > (maior estrito). 
        # Agora: >= (maior ou igual) COM tolerância EPSILON.
        
        if premis_class == 1:
            # Se já ultrapassou t_plus OU está "colado" nele (diferença menor que epsilon)
            if soma_deltas_cumulativa >= (target_score - EPSILON) and explicacao:
                break
        else: # premis_class == 0
            # Se já desceu abaixo de t_minus OU está "colado" nele
            if soma_deltas_cumulativa <= (target_score + EPSILON) and explicacao:
                break
                
    # Fallback de segurança (caso lista vazia)
    if not explicacao and len(X_train.columns) > 0:
         logreg = _get_lr(modelo)
         idx_max = np.argmax(np.abs(logreg.coef_[0]))
         feat_nome = X_train.columns[idx_max]
         valor_feat = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
         explicacao.append(f"{feat_nome} = {valor_feat:.4f}")

    return explicacao

def perturbar_e_validar(modelo: Pipeline, instance_df: pd.DataFrame, explicacao: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, direcao_override: int) -> Tuple[bool, float]:
    if not explicacao:
        return False, 0.0
    inst_pert = instance_df.copy() #criando uma cópia para perturação
    features_explicacao = {f.split(' = ')[0] for f in explicacao} # extrai o nome das feat para a explic.
    #define a direção da perturbação:
        # direcao_override = 1 → perturbar para diminuir score
        # direcao_override = 0 → perturbar para aumentar score
    perturbar_para_diminuir_score = (direcao_override == 1)
    modelo_interno = _get_lr(modelo) # modelo de reg logist
    X_train_min = X_train.min(axis=0) 
    X_train_max = X_train.max(axis=0)
    for feat_idx, feat_nome in enumerate(X_train.columns):
        if feat_nome in features_explicacao:
            continue # pular features da explicação (nãso alterar)
        coef = modelo_interno.coef_[0][feat_idx] 
        # define o valor perrtiurbado a feature que nao esta fixa
        valor_pert = (X_train_min[feat_nome] if coef > 0 else X_train_max[feat_nome]) if perturbar_para_diminuir_score else (X_train_max[feat_nome] if coef > 0 else X_train_min[feat_nome])
        inst_pert.loc[inst_pert.index[0], feat_nome] = valor_pert
    # calcula o score da instancia perturbada
    score_pert = modelo.decision_function(inst_pert)[0]
    # verificando se o s'(o score perturbasdo) caiu na rejeição
    pert_rejeitada = t_minus <= score_pert <= t_plus
   
    score_original = modelo.decision_function(instance_df)[0]
    # verificando se a instancia original é rejeitada (retorna boelano)
    is_original_rejected = t_minus <= score_original <= t_plus
    if is_original_rejected: # se a boleana for True
        return pert_rejeitada, score_pert
    else:
        # Caso original CLASSIFICADA: exigir explicitamente contra t+ / t-
        pred_original_class = int(modelo.predict(instance_df)[0])
        if pred_original_class == 1:
            # Para classe positiva, robusto se s' >= t+
            return (score_pert >= t_plus), score_pert
        else:
            # Para classe negativa, robusto se s' <= t-
            return (score_pert <= t_minus), score_pert

def fase_1_reforco(modelo: Pipeline, instance_df: pd.DataFrame, expl_inicial: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int) -> Tuple[List[str], int]:
    expl_robusta = list(expl_inicial) # copia da explicação inicial
    adicoes = 0 # Contador de features adicionadas
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class = premisa_ordenacao) # Cálculo dos deltas para ordenação
    # Ordenar índices das features por |δ| decrescente ( mais importnate primeiro)
    indices_ordenados = np.argsort(-np.abs(deltas_para_ordenar))
    while True: #para as rejeitadas testa as duas direções
        if is_rejected:
            valido1, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, 0) # direção para diminuir score, o nuemro 1 so indica a direçao do override
            valido2, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, 1) # direção para aumentar score
            if valido1 and valido2: break # se for valido nas duas direções sai do loop
        else: # para as classificadas testa a direção da predição
            direcao = 1 if modelo.predict(instance_df)[0] == 1 else 0 # direção baseada na predição
            valido, _ = perturbar_e_validar(modelo, instance_df, expl_robusta, X_train, t_plus, t_minus, direcao) 
            if valido: break # se for valido sai do loop
        
        if len(expl_robusta) == X_train.shape[1]: break # se todas as features já estão na explicação pare
        features_explicacao_set = {f.split(' = ')[0] for f in expl_robusta} # extrai nomes das features na explicação atual para facilitar verificação
        adicionou_feature = False 
        for idx in indices_ordenados:
            feat_nome = X_train.columns[idx]
            if feat_nome not in features_explicacao_set: # se a feature não está na explicação atual
                # Adiciona a feature à explicação robusta
                valor_feat = instance_df.iloc[0, X_train.columns.get_loc(feat_nome)]
                expl_robusta.append(f"{feat_nome} = {valor_feat:.4f}")
                adicoes += 1
                adicionou_feature = True 
                break
        if not adicionou_feature: break # se não conseguiu adicionar mais features, sai do loop
    return expl_robusta, adicoes 

def fase_2_minimizacao(modelo: Pipeline, instance_df: pd.DataFrame, expl_robusta: List[str], X_train: pd.DataFrame, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int, log_passos: List[Dict]) -> Tuple[List[str], int]:
    expl_minima = list(expl_robusta) # copia da explicação robusta
    remocoes = 0
    deltas_para_ordenar = calculate_deltas(modelo, instance_df, X_train, premis_class=premisa_ordenacao)
    # Remover da mais importante para a menos importante (maior |δ| primeiro)
    features_para_remover = sorted(
        [f.split(' = ')[0] for f in expl_minima],
        key=lambda nome: abs(deltas_para_ordenar[X_train.columns.get_loc(nome)]),
        reverse=True
    )
    for feat_nome in features_para_remover: # tenta remover cada feature
        if len(expl_minima) <= 1: break
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)] # explicação temporária sem a feature atual 
        
        remocao_bem_sucedida = False
        score_pert_final = None
        delta_feat = float(deltas_para_ordenar[X_train.columns.get_loc(feat_nome)])

        if is_rejected:
            valido1, score_p1 = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, 1)
            valido2, score_p2 = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, 0)
            ok_neg = bool(valido1)
            ok_pos = bool(valido2)
            if valido1 and valido2:
                remocao_bem_sucedida = True
            # Log detalhado bidirecional para rejeitadas
            log_passos.append({
                'feat_nome': feat_nome,
                'valor': instance_df.iloc[0, X_train.columns.get_loc(feat_nome)],
                'delta': delta_feat,
                'score_neg': score_p1,
                'ok_neg': ok_neg,
                'score_pos': score_p2,
                'ok_pos': ok_pos,
                'sucesso': remocao_bem_sucedida
            })
        else:
            direcao = 1 if modelo.predict(instance_df)[0] == 1 else 0
            remocao_bem_sucedida, score_pert_final = perturbar_e_validar(modelo, instance_df, expl_temp, X_train, t_plus, t_minus, direcao)
            # Log unidirecional para classificadas
            log_passos.append({
                'feat_nome': feat_nome,
                'valor': instance_df.iloc[0, X_train.columns.get_loc(feat_nome)],
                'delta': delta_feat,
                'score_perturbado': score_pert_final,
                'sucesso': remocao_bem_sucedida
            })

        if remocao_bem_sucedida:
            expl_minima = expl_temp
            remocoes += 1
            
    return expl_minima, remocoes

#==============================================================================
# FUNÇÕES DE GERAÇÃO E FORMATAÇÃO DE LOG
#==============================================================================

def gerar_explicacao_instancia(instancia_df: pd.DataFrame, modelo: Pipeline, X_train: pd.DataFrame, t_plus: float, t_minus: float) -> Tuple[List[str], List[str], int, int]:
    """Gera explicação da instância e monta log técnico conciso (sem alterar JSON)."""
        # REJEITADAS 01
    # Verificando se a instância é rejeitada
    is_rejected = t_minus <= modelo.decision_function(instancia_df)[0] <= t_plus
    log_formatado: List[str] = []

    # Suprime logs técnicos automaticamente em alta dimensionalidade
    emit_tech_logs = TECHNICAL_LOGS and (X_train.shape[1] <= MAX_LOG_FEATURES)

    if is_rejected:
        # Rejeitadas: prova de minimalidade com verificação bidirecional
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['rejeitada_analise'].format(t_minus=t_minus, t_plus=t_plus))
            log_formatado.append(LOG_TEMPLATES['rejeitada_prova_header'])

        # Dois caminhos (premissas opostas). Coletar passos para o melhor caminho.
        # Caminho 1 (premisa 1) chama one explanations para o calculo inicial usando premiss_cvlasse 1
        expl_inicial_p1 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 1)
        expl_robusta_p1, adicoes1 = fase_1_reforco(modelo, instancia_df, expl_inicial_p1, X_train, t_plus, t_minus, True, 1)
        passos_p1: List[Dict[str, Any]] = []
        expl_final_p1, remocoes1 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p1, X_train, t_plus, t_minus, True, 1, passos_p1)

        # Caminho 2 (premisa 0) usada no expl_robusta_p2 e expl_final_p2
        expl_inicial_p2 = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, 0)
        expl_robusta_p2, adicoes2 = fase_1_reforco(modelo, instancia_df, expl_inicial_p2, X_train, t_plus, t_minus, True, 0)
        passos_p2: List[Dict[str, Any]] = []
        expl_final_p2, remocoes2 = fase_2_minimizacao(modelo, instancia_df, expl_robusta_p2, X_train, t_plus, t_minus, True, 0, passos_p2)

        # Seleção do melhor
        if len(expl_final_p1) <= len(expl_final_p2):
            expl_final, adicoes, remocoes = expl_final_p1, adicoes1, remocoes1
            passos_escolhidos = passos_p1
            expl_inicial_robusta_escolhida = expl_robusta_p1
        else:
            expl_final, adicoes, remocoes = expl_final_p2, adicoes2, remocoes2
            passos_escolhidos = passos_p2
            expl_inicial_robusta_escolhida = expl_robusta_p2

        # Informar conjunto inicial robusto e formatar passos
        if emit_tech_logs:
            feats_iniciais = sorted([f.split(' = ')[0] for f in expl_inicial_robusta_escolhida])
            log_formatado.append(
                f"├── Conjunto inicial (heurística): {len(feats_iniciais)} features {feats_iniciais}"
            )
            # Limita a quantidade de passos logados
            for passo in passos_escolhidos[:MAX_LOG_STEPS]:
                key_header = 'rejeitada_feat_header_sucesso' if passo.get('sucesso', False) else 'rejeitada_feat_header_falha'
                log_formatado.append(LOG_TEMPLATES[key_header].format(feat=passo['feat_nome'], delta=passo.get('delta', 0.0)))
                cmp_neg = f"> t- ({t_minus:.4f})" if passo.get('score_neg', 0.0) > t_minus else f"< t- ({t_minus:.4f})"
                ok_neg = "OK." if passo.get('ok_neg', False) else "FALHA."
                log_formatado.append(LOG_TEMPLATES['rejeitada_subteste_neg'].format(score=passo.get('score_neg', 0.0), cmp=cmp_neg, ok=ok_neg))
                cmp_pos = f"< t+ ({t_plus:.4f})" if passo.get('score_pos', 0.0) < t_plus else f"\u2265 t+ ({t_plus:.4f})"
                ok_pos = "OK." if passo.get('ok_pos', False) else "FALHA."
                log_formatado.append(LOG_TEMPLATES['rejeitada_subteste_pos'].format(score=passo.get('score_pos', 0.0), cmp=cmp_pos, ok=ok_pos))
                footer_key = 'rejeitada_feat_footer_sucesso' if passo.get('sucesso', False) else 'rejeitada_feat_footer_falha'
                log_formatado.append(LOG_TEMPLATES[footer_key])
            # Indica truncamento do log
            if len(passos_escolhidos) > MAX_LOG_STEPS:
                log_formatado.append(f"│   ... {len(passos_escolhidos) - MAX_LOG_STEPS} passos omitidos por limite de log ...")

    else:
        # Classificadas: verificação unidirecional
        pred_class = int(modelo.predict(instancia_df)[0])
        posicao = 'acima de t+' if pred_class == 1 else 'abaixo de t-'
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['classificada_analise'].format(posicao=posicao))

        expl_inicial = one_explanation_formal(modelo, instancia_df, X_train, t_plus, t_minus, pred_class)
        if DISABLE_REFORCO_CLASSIFICADAS:
            # Pula fase de reforço: usa explicação inicial diretamente
            expl_robusta = expl_inicial
            adicoes = 0
            if emit_tech_logs:
                log_formatado.append(f"├── Fase de reforço DESATIVADA (DISABLE_REFORCO_CLASSIFICADAS=True). Usando explicação inicial com {len(expl_robusta)} features.")
        else:
            expl_robusta, adicoes = fase_1_reforco(modelo, instancia_df, expl_inicial, X_train, t_plus, t_minus, False, pred_class)

        # Ordem por maior |δ| (nas features da explicação robusta)
        deltas = calculate_deltas(modelo, instancia_df, X_train, premis_class=pred_class)
        feats_em_robusta = [f.split(' = ')[0] for f in expl_robusta]
        ordem = sorted(
            feats_em_robusta,
            key=lambda nome: abs(deltas[X_train.columns.get_loc(nome)]),
            reverse=True
        )
        if emit_tech_logs:
            log_formatado.append(LOG_TEMPLATES['classificada_min_inicio'].format(num_features=len(expl_robusta)))
            log_formatado.append(LOG_TEMPLATES['classificada_ordem'].format(lista=str(ordem)))

        # Minimizacao e coleta
        passos: List[Dict[str, Any]] = []
        expl_final, remocoes = fase_2_minimizacao(modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, False, pred_class, passos)

        # Formatar passos
        if emit_tech_logs:
            limiar = t_plus if pred_class == 1 else t_minus
            for p in passos[:MAX_LOG_STEPS]:
                cond = ("> t+ (" + f"{limiar:.4f})") if pred_class == 1 else ("< t- (" + f"{limiar:.4f})")
                key = 'classificada_step_sucesso' if p.get('sucesso', False) else 'classificada_step_falha'
                score_show = p.get('score_perturbado', np.nan)
                log_formatado.append(LOG_TEMPLATES[key].format(feat=p['feat_nome'], delta=p.get('delta', 0.0), score=score_show, cond=cond))
            if len(passos) > MAX_LOG_STEPS:
                log_formatado.append(f"├─ ... {len(passos) - MAX_LOG_STEPS} passos omitidos por limite de log ...")

    return [f.split(' = ')[0] for f in expl_final], log_formatado, adicoes, remocoes

#==============================================================================
# EXECUÇÃO PRINCIPAL 1 a ser chamada
#==============================================================================
def executar_experimento_para_dataset(dataset_name: str):
    print(f"\n==================== EXECUTANDO PARA DATASET: {dataset_name.upper()} ====================")
    
    # 1. Carregar Configurações
    todos_hiperparametros = carregar_hiperparametros()
    X, y, nomes_classes, rejection_cost_atual, test_size_atual = configurar_experimento(dataset_name)

    parametros_para_modelo = DEFAULT_LOGREG_PARAMS.copy()
    config_do_modelo = todos_hiperparametros.get(dataset_name)
    if config_do_modelo and 'params' in config_do_modelo:
        valid_keys = LogisticRegression().get_params().keys()
        parametros_carregados = {k: v for k, v in config_do_modelo['params'].items() if k in valid_keys}
        parametros_para_modelo.update(parametros_carregados)
        print(f"[INFO] Usando hiperparâmetros otimizados para '{dataset_name}': {parametros_para_modelo}")
    else:
        print(f"[AVISO] Parâmetros para '{dataset_name}' não encontrados. Usando modelo padrão: {parametros_para_modelo}")

    # 2. Aplicar redução de features (top-k) ANTES do treino, se configurado
    cfg = DATASET_CONFIG.get(dataset_name, {})
    top_k = cfg.get('top_k_features', None)
    
    if top_k and top_k > 0 and top_k < X.shape[1]:
        # Treinar modelo temporário para obter importâncias
        print(f"\n[INFO] Treinando modelo temporário para seleção de features...")
        modelo_temp, _, _, _ = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
        X_train_temp, X_test_temp, _, _ = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
        
        # Selecionar top-k features
        X_train_temp, X_test_temp, selected_features = aplicar_selecao_top_k_features(X_train_temp, X_test_temp, modelo_temp, top_k)
        
        # Reduzir X completo para apenas as features selecionadas
        X = X[selected_features]
        print(f"[INFO] Dataset reduzido de {len(X.columns)} para {top_k} features.")
    
    # 3. Treinar Modelo FINAL (com features reduzidas ou completas)
    modelo, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(X, y, test_size_atual, rejection_cost_atual, parametros_para_modelo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_atual, random_state=RANDOM_STATE, stratify=y)
    
    print(f"\n[INFO] Modelo treinado. Acurácia no teste (sem rejeição): {modelo.score(X_test, y_test):.2%}")
    print(f"[INFO] Custo de Rejeição (WR) definido como: {rejection_cost_atual:.2f}")
    print(f"[INFO] Thresholds de rejeição calculados: t+ = {t_plus:.4f}, t- = {t_minus:.4f}")
    
    # Aviso se zona de rejeição muito pequena testando MNIST
    zona_rejeicao = abs(t_plus - t_minus)
    if zona_rejeicao < 0.001:
        print(f"\n⚠️  AVISO: Zona de rejeição MÍNIMA (|t+ - t-| = {zona_rejeicao:.6f})")
        print(f"   Isso significa que o modelo está rejeitando muito pouco.")
        print(f"   Causas possíveis:")
        print(f"   • Modelo muito confiante (acurácia alta: {modelo.score(X_test, y_test):.2%})")
        print(f"   • Custo de rejeição alto demais (WR = {rejection_cost_atual:.2f})")
        print(f"   • Classes bem separadas (pouca ambiguidade)")
        print(f"   Recomendações:")
        print(f"   • Diminuir rejection_cost: {rejection_cost_atual:.2f} → {rejection_cost_atual/2:.2f}")
        print(f"   • Aumentar subsample_size para ter mais dados")
        print(f"   • Escolher par de classes mais ambíguo\n")

    # 3. Gerar Predições
    decision_scores_test = modelo.decision_function(X_test)
    y_pred_test = np.full(y_test.shape, -1, dtype=int)
    y_pred_test[decision_scores_test >= t_plus] = 1
    y_pred_test[decision_scores_test <= t_minus] = 0
    rejected_mask = (y_pred_test == -1)
    y_pred_test_final = y_pred_test.copy()
    y_pred_test_final[rejected_mask] = 2

    # 4. Gerar Explicações e Coletar Dados
    print(f"\n[INFO] Gerando explicações para {len(X_test)} instâncias de teste...")
    
    # Importar barra de progresso
    from utils.progress_bar import ProgressBar
    
    start_time_total = time.perf_counter()
    resultados_instancias = []
    times_pos, times_neg, times_rej = [], [], []
    adicoes_pos, adicoes_neg, adicoes_rej = [], [], []
    remocoes_pos, remocoes_neg, remocoes_rej = [], [], []

    # Criar barra de progresso
    with ProgressBar(total=len(X_test), description=f"PEAB Explicando {dataset_name}") as pbar:
        for i in range(len(X_test)):
            inst_start_time = time.perf_counter() # contagem do tempo por instância
            
            instancia_df = X_test.iloc[[i]]
            pred_class_code = y_pred_test_final[i]
            
            expl_final_nomes, log_formatado, adicoes, remocoes = gerar_explicacao_instancia(instancia_df, modelo, X_train, t_plus, t_minus)
            
            inst_end_time = time.perf_counter()
            inst_duration = inst_end_time - inst_start_time

            # [MODIFICAÇÃO] Adiciona o cabeçalho ao log formatado
            header = f"--- INSTÂNCIA #{i} | CLASSE REAL: {nomes_classes[y_test.iloc[i]]} | PREDIÇÃO: {'REJEITADA' if pred_class_code == 2 else 'CLASSE ' + str(pred_class_code)} | SCORE: {decision_scores_test[i]:.4f} ---"
            log_final_com_header = [header] + log_formatado

            resultados_instancias.append({
                'id': i, # [MODIFICAÇÃO] ID sequencial
                'classe_real': nomes_classes[y_test.iloc[i]],
                'predicao': 'REJEITADA' if pred_class_code == 2 else nomes_classes[pred_class_code],
                'pred_code': int(pred_class_code),
                'score': decision_scores_test[i],
                'explicacao': sorted(expl_final_nomes),
                'tamanho_explicacao': len(expl_final_nomes),
                'log_detalhado': log_final_com_header
            })

            if pred_class_code == 2:
                times_rej.append(inst_duration)
                adicoes_rej.append(adicoes)
                remocoes_rej.append(remocoes)
            elif pred_class_code == 1:
                times_pos.append(inst_duration)
                adicoes_pos.append(adicoes)
                remocoes_pos.append(remocoes)
            else:
                times_neg.append(inst_duration)
                adicoes_neg.append(adicoes)
                remocoes_neg.append(remocoes)
            
            # Atualizar barra de progresso
            pbar.update()
    
    # Barra de progresso já foi fechada automaticamente pelo context manager
    tempo_total_explicacoes = time.perf_counter() - start_time_total
    print(f"\n[INFO] Geração de explicações concluída em {tempo_total_explicacoes:.2f} segundos.")

    # 5. Coletar Métricas
    metricas_dict = coletar_metricas(
        resultados_instancias, y_test, y_pred_test_final, rejected_mask,
        tempo_total_explicacoes, model_params, modelo, X_test, X_train.columns,
        times_pos, times_neg, times_rej,
        adicoes_pos, adicoes_neg, adicoes_rej,
        remocoes_pos, remocoes_neg, remocoes_rej
    )

    # 6. Persistir no JSON cumulativo
    # Para MNIST, usar chave única por par de dígitos, ex: mnist_8_vs_3
    dataset_json_key = dataset_name
    if dataset_name == 'mnist':
        cfg_mnist = DATASET_CONFIG.get('mnist', {})
        digit_pair = cfg_mnist.get('digit_pair')
        if digit_pair and len(digit_pair) == 2:
            dataset_json_key = f"mnist_{digit_pair[0]}_vs_{digit_pair[1]}"
            print(f"\n[INFO] Salvando resultados MNIST sob chave única: '{dataset_json_key}' (evita sobrescrever outros pares).")
        else:
            print("\n[AVISO] Par de dígitos MNIST não definido corretamente. Usando chave padrão 'mnist'.")

    dataset_cache_para_json = montar_dataset_cache(
        dataset_name=dataset_name,  # Mantém nome base interno para metadados
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        nomes_classes=nomes_classes,
        t_plus=t_plus,
        t_minus=t_minus,
        WR_REJECTION_COST=rejection_cost_atual,
        test_size_atual=test_size_atual,
        model_params=model_params,
        metricas_dict=metricas_dict,
        y_pred_test=y_pred_test_final,
        decision_scores_test=decision_scores_test,
        rejected_mask=rejected_mask,
        resultados_instancias=resultados_instancias
    )
    update_method_results('peab', dataset_json_key, dataset_cache_para_json)

    # 7. Gerar Relatório
    gerar_relatorio_texto(dataset_name, test_size_atual, rejection_cost_atual, modelo, t_plus, t_minus, len(X_test), metricas_dict, resultados_instancias)
    
    print(f"\n==================== EXECUÇÃO PARA {dataset_name.upper()} CONCLUÍDA ====================")


def coletar_metricas(resultados_instancias, y_test, y_pred_test_final, rejected_mask,
                     tempo_total, model_params, modelo: Pipeline, X_test: pd.DataFrame, feature_names: List[str],
                     times_pos, times_neg, times_rej, adicoes_pos, adicoes_neg, adicoes_rej, remocoes_pos, remocoes_neg, remocoes_rej):
    """Coleta e calcula todas as métricas do experimento, incluindo as novas seções."""
    # Stats de tamanho
    stats_pos_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 1]
    stats_neg_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 0]
    stats_rej_list = [r['tamanho_explicacao'] for r in resultados_instancias if r['pred_code'] == 2]

    # Métricas de desempenho
    acc_sem_rej = float(np.mean(modelo.predict(X_test) == y_test) * 100)
    acc_com_rej = float(np.mean(y_pred_test_final[~rejected_mask] == y_test.iloc[~rejected_mask]) * 100) if np.any(~rejected_mask) else 100.0
    taxa_rej = float(np.mean(rejected_mask) * 100)

    # Métricas de tempo
    avg_time_pos = float(np.mean(times_pos)) if times_pos else 0.0
    avg_time_neg = float(np.mean(times_neg)) if times_neg else 0.0
    avg_time_rej = float(np.mean(times_rej)) if times_rej else 0.0

    # Função auxiliar para stats do processo
    def get_proc_stats(adicoes, remocoes):
        inst_com_adicao = sum(1 for x in adicoes if x > 0)
        media_adicoes = float(np.mean([x for x in adicoes if x > 0])) if inst_com_adicao > 0 else 0.0
        inst_com_remocao = sum(1 for x in remocoes if x > 0)
        media_remocoes = float(np.mean([x for x in remocoes if x > 0])) if inst_com_remocao > 0 else 0.0
        return {
            'inst_com_adicao': int(inst_com_adicao),
            'perc_adicao': float((inst_com_adicao / len(adicoes) * 100) if adicoes else 0.0),
            'media_adicoes': media_adicoes,
            'inst_com_remocao': int(inst_com_remocao),
            'perc_remocao': float((inst_com_remocao / len(remocoes) * 100) if remocoes else 0.0),
            'media_remocoes': media_remocoes
        }

    return {
        'acuracia_sem_rejeicao': acc_sem_rej,
        'acuracia_com_rejeicao': acc_com_rej,
        'taxa_rejeicao': taxa_rej,
        'stats_explicacao_positiva': {
            'instancias': len(stats_pos_list),
            'media': float(np.mean(stats_pos_list)) if stats_pos_list else 0.0,
            'std_dev': float(np.std(stats_pos_list)) if stats_pos_list else 0.0,
            'min': int(np.min(stats_pos_list)) if stats_pos_list else 0,
            'max': int(np.max(stats_pos_list)) if stats_pos_list else 0
        },
        'stats_explicacao_negativa': {
            'instancias': len(stats_neg_list),
            'media': float(np.mean(stats_neg_list)) if stats_neg_list else 0.0,
            'std_dev': float(np.std(stats_neg_list)) if stats_neg_list else 0.0,
            'min': int(np.min(stats_neg_list)) if stats_neg_list else 0,
            'max': int(np.max(stats_neg_list)) if stats_neg_list else 0
        },
        'stats_explicacao_rejeitada': {
            'instancias': len(stats_rej_list),
            'media': float(np.mean(stats_rej_list)) if stats_rej_list else 0.0,
            'std_dev': float(np.std(stats_rej_list)) if stats_rej_list else 0.0,
            'min': int(np.min(stats_rej_list)) if stats_rej_list else 0,
            'max': int(np.max(stats_rej_list)) if stats_rej_list else 0
        },
        'tempo_total': float(tempo_total),
        'tempo_medio_instancia': float(tempo_total / len(y_test) if len(y_test) > 0 else 0.0),
        'tempo_medio_positivas': avg_time_pos,
        'tempo_medio_negativas': avg_time_neg,
        'tempo_medio_rejeitadas': avg_time_rej,
        'features_frequentes': Counter([feat for r in resultados_instancias for feat in r['explicacao']]).most_common(),
        'pesos_modelo': sorted(((name, float(model_params['coefs'][name])) for name in feature_names), key=lambda item: abs(item[1]), reverse=True),
        'intercepto': float(model_params['intercepto']),
        'processo_stats_pos': get_proc_stats(adicoes_pos, remocoes_pos),
        'processo_stats_neg': get_proc_stats(adicoes_neg, remocoes_neg),
        'processo_stats_rej': get_proc_stats(adicoes_rej, remocoes_rej)
    }

#==============================================================================
# FUNÇÕES AUXILIARES
#==============================================================================

def configurar_experimento(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], float, float]:
    """Carrega o dataset e as configurações específicas do experimento.

    Se 'subsample_size' existir na configuração do dataset, aplica uma
    subamostragem estratificada (útil para datasets grandes, como 'creditcard').
    
    Para MNIST, aplica configurações automáticas de par de dígitos e modo de features.
    """
    # Aplicar configurações específicas de MNIST ANTES de carregar
    if dataset_name == 'mnist':
        from data import datasets as ds_module
        cfg = DATASET_CONFIG.get(dataset_name, {})
        
        # Definir modo de features e par de dígitos
        feature_mode = cfg.get('feature_mode', 'raw')
        digit_pair = cfg.get('digit_pair', None)
        
        # Configurar opções globais de MNIST
        ds_module.set_mnist_options(feature_mode, digit_pair)
        
        # Log informativo
        print(f"\n{'='*80}")
        print(f"  CONFIGURAÇÃO AUTOMÁTICA DO MNIST")
        print(f"{'='*80}")
        print(f"  • Modo de Features: {feature_mode.upper()} ({'784 features (28x28)' if feature_mode == 'raw' else '196 features (14x14 pooled)'})")
        print(f"  • Par de Dígitos: {digit_pair[0]} vs {digit_pair[1]}")
        print(f"  • Subsample Size: {cfg.get('subsample_size', 'N/A'):.1%}" if cfg.get('subsample_size') else f"  • Subsample Size: Dataset completo")
        print(f"  • Test Size: {cfg.get('test_size', 0.3):.1%}")
        print(f"  • Rejection Cost: {cfg.get('rejection_cost', 0.24):.2f}")
        
        # Top-K Features
        top_k = cfg.get('top_k_features', None)
        if top_k and top_k > 0:
            print(f"  • Top-K Features: SIM (usando as {top_k} features mais importantes)")
        else:
            print(f"  • Top-K Features: NÃO (usando todas as features)")
        print(f"{'='*80}\n")
    
    X, y, nomes_classes = carregar_dataset(dataset_name)
    if X is None or y is None or nomes_classes is None:
        raise ValueError(f"Falha ao carregar dataset '{dataset_name}'")
    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        print(f"[AVISO] Configuração não encontrada para '{dataset_name}'. Usando test_size=0.3 e rejection_cost=0.24.")
        cfg = {'test_size': 0.3, 'rejection_cost': 0.24}

    # Subamostragem estratificada opcional
    if 'subsample_size' in cfg and cfg['subsample_size'] is not None:
        frac = cfg['subsample_size']
        if not (0 < frac <= 1):
            raise ValueError(f"subsample_size inválido ({frac}) para dataset '{dataset_name}'. Esperado 0 < frac <= 1.")
        idx = np.arange(len(y))
        sample_idx, _ = train_test_split(idx, test_size=(1 - frac), random_state=RANDOM_STATE, stratify=y)
        # Aplicar seleção
        X = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]
        y = y.iloc[sample_idx] if isinstance(y, pd.Series) else y[sample_idx]

    return X, y, nomes_classes, cfg['rejection_cost'], cfg['test_size']


def aplicar_selecao_top_k_features(X_train: pd.DataFrame, X_test: pd.DataFrame, pipeline: Pipeline, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Seleciona as top-k features mais importantes baseado nos pesos absolutos do modelo treinado.
    
    Args:
        X_train: Conjunto de treino completo
        X_test: Conjunto de teste completo
        pipeline: Pipeline treinado com todas as features
        top_k: Número de features a manter
    
    Returns:
        X_train_reduced, X_test_reduced, selected_features
    """
    # Obter coeficientes do modelo
    logreg = _get_lr(pipeline)
    coefs = logreg.coef_[0]
    feature_names = list(X_train.columns)
    
    # Ordenar features por importância (valor absoluto do peso)
    importances = [(name, abs(coefs[i])) for i, name in enumerate(feature_names)]
    importances_sorted = sorted(importances, key=lambda x: x[1], reverse=True)
    
    # Selecionar top-k
    selected_features = [name for name, _ in importances_sorted[:top_k]]
    
    print(f"\n[INFO] Redução de Features Aplicada:")
    print(f"  • Features originais: {len(feature_names)}")
    print(f"  • Features selecionadas (top-{top_k}): {len(selected_features)}")
    print(f"  • Features mais importantes: {selected_features[:10]}...")
    
    # Retornar subconjuntos
    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]
    
    return X_train_reduced, X_test_reduced, selected_features


def treinar_e_avaliar_modelo(X: pd.DataFrame, y: pd.Series, test_size: float, rejection_cost: float, logreg_params: Dict[str, Any]) -> Tuple[Pipeline, float, float, Dict[str, Any]]:
    """Treina o modelo com os hiperparâmetros fornecidos, otimiza os limiares e retorna (modelo, t+, t-, params)."""
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    # Pipeline com hiperparâmetros injetados
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, **logreg_params)),
    ])
    pipeline.fit(X_train, y_train)

    # Otimização de thresholds em cima do conjunto de treino
    decision_scores = pipeline.decision_function(X_train)
    qs = np.linspace(0, 1, 100)
    search_space = np.unique(np.quantile(decision_scores, qs))
    best_risk, best_t_plus, best_t_minus = float('inf'), 0.0, 0.0
    for i in range(len(search_space)):
        for j in range(i, len(search_space)):
            t_minus, t_plus = float(search_space[i]), float(search_space[j])
            # Skip se largura mínima for exigida e não atendida
            if MIN_REJECTION_WIDTH > 0.0 and (t_plus - t_minus) < MIN_REJECTION_WIDTH:
                continue
            y_pred = np.full(y_train.shape, -1)
            accepted = (decision_scores >= t_plus) | (decision_scores <= t_minus)
            y_pred[decision_scores >= t_plus] = 1
            y_pred[decision_scores <= t_minus] = 0
            error_rate = np.mean(y_pred[accepted] != y_train[accepted]) if np.any(accepted) else 0.0
            rejection_rate = 1.0 - np.mean(accepted)
            risk = float(error_rate + rejection_cost * rejection_rate)
            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, t_plus, t_minus

    # Fallback: se não encontrou (caso MIN_REJECTION_WIDTH muito grande) refaz sem restrição
    if best_risk == float('inf'):
        for i in range(len(search_space)):
            for j in range(i, len(search_space)):
                t_minus, t_plus = float(search_space[i]), float(search_space[j])
                y_pred = np.full(y_train.shape, -1)
                accepted = (decision_scores >= t_plus) | (decision_scores <= t_minus)
                y_pred[decision_scores >= t_plus] = 1
                y_pred[decision_scores <= t_minus] = 0
                error_rate = np.mean(y_pred[accepted] != y_train[accepted]) if np.any(accepted) else 0.0
                rejection_rate = 1.0 - np.mean(accepted)
                risk = float(error_rate + rejection_cost * rejection_rate)
                if risk < best_risk:
                    best_risk, best_t_plus, best_t_minus = risk, t_plus, t_minus

    # Parâmetros do modelo para logs/JSON
    coefs = pipeline.named_steps['model'].coef_[0]
    feature_names = list(X.columns)
    model_params = {
        'coefs': {name: float(w) for name, w in zip(feature_names, coefs)},
        'intercepto': float(pipeline.named_steps['model'].intercept_[0]),
        'scaler_params': {
            'min': [float(v) for v in pipeline.named_steps['scaler'].min_],
            'scale': [float(v) for v in pipeline.named_steps['scaler'].scale_],
        },
        **logreg_params
    }

    return pipeline, float(best_t_plus), float(best_t_minus), model_params

def gerar_relatorio_texto(dataset_name, test_size, wr, modelo, t_plus, t_minus, num_test, metricas, resultados_instancias):
    output_path = os.path.join(OUTPUT_BASE_DIR, f"peab_{dataset_name}.txt")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"\n[INFO] Salvando relatório detalhado em: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n" + "          RELATÓRIO DE ANÁLISE - MÉTODO PEAB (EXPLAINABLE AI)\n" + "="*80 + "\n\n")
        # [NOVO] Legenda dos símbolos (concisa)
        f.write("\n".join(SYMBOL_LEGEND) + "\n\n")
        f.write("[ PARÂMETROS DO EXPERIMENTO ]\n\n")
        f.write(f"  - Dataset: {dataset_name}\n")
        f.write(f"  - Proporção de Teste: {int(test_size*100)}%\n")
        f.write(f"  - Total de Instâncias de Teste: {num_test}\n")
        f.write(f"  - Custo de Rejeição (wr): {wr:.2f}\n\n")
        f.write("[ PARÂMETROS DO MODELO DE REGRESSÃO LOGÍSTICA ]\n\n")
        f.write(f"  - Número de Features: {len(metricas['pesos_modelo'])}\n")
        f.write(f"  - Acurácia (sem rejeição): {metricas['acuracia_sem_rejeicao']:.2f}%\n")
        f.write(f"  - Limiar Superior (t+): {t_plus:.4f}\n")
        f.write(f"  - Limiar Inferior (t-): {t_minus:.4f}\n")
        f.write(f"  - Intercepto (w0): {metricas['intercepto']:.4f}\n\n")
        # [NOVO] Métricas de Desempenho
        f.write("[ MÉTRICAS DE DESEMPENHO DO MODELO ]\n\n")
        f.write(f"  - Acurácia Geral (sem rejeição): {metricas['acuracia_sem_rejeicao']:.2f}%\n")
        f.write(f"  - Taxa de Rejeição no Teste: {metricas['taxa_rejeicao']:.2f}% ({metricas['stats_explicacao_rejeitada']['instancias']} de {num_test} instâncias)\n")
        f.write(f"  - Acurácia com Opção de Rejeição (nas {num_test - metricas['stats_explicacao_rejeitada']['instancias']} instâncias aceitas): {metricas['acuracia_com_rejeicao']:.2f}%\n\n")
        # [NOVO] Custo Computacional
        f.write("[ DESEMPENHO COMPUTACIONAL ]\n\n")
        f.write(f"  - Tempo Total de Geração das Explicações: {metricas['tempo_total']:.2f} segundos\n")
        f.write(f"  - Tempo Médio por Instância: {metricas['tempo_medio_instancia']:.4f} segundos\n\n")
        f.write("  - Custo Computacional por Classe:\n")
        f.write(f"    - Tempo Médio (Positivas): {metricas['tempo_medio_positivas']:.4f} segundos\n")
        f.write(f"    - Tempo Médio (Negativas): {metricas['tempo_medio_negativas']:.4f} segundos\n")
        f.write(f"    - Tempo Médio (Rejeitadas): {metricas['tempo_medio_rejeitadas']:.4f} segundos\n\n")
        
        f.write(LOG_TEMPLATES['processamento_header'] + "\n")
        rejeitadas = [r for r in resultados_instancias if r['predicao'] == 'REJEITADA']
        classificadas = [r for r in resultados_instancias if r['predicao'] != 'REJEITADA']
        if rejeitadas:
            f.write("-" * 80 + "\n" + "                         SEÇÃO A: INSTÂNCIAS REJEITADAS\n" + "-" * 80 + "\n")
            for r in rejeitadas:
                for log_line in r['log_detalhado']:
                    f.write(f"{log_line}\n")
                f.write(f"\n   --> RESULTADO FINAL (Instância #{r['id']}):\n")
                f.write(f"       - EXPLICAÇÃO ABDUTIVA (Tamanho: {r['tamanho_explicacao']}): {sorted(r['explicacao'])}\n\n")
        if classificadas:
            f.write("-" * 80 + "\n" + "                       SEÇÃO B: INSTÂNCIAS CLASSIFICADAS\n" + "-" * 80 + "\n")
            for r in classificadas:
                for log_line in r['log_detalhado']:
                    f.write(f"{log_line}\n")
                f.write(f"\n   --> RESULTADO FINAL (Instância #{r['id']}):\n")
                f.write(f"       - EXPLICAÇÃO ABDUTIVA (Tamanho: {r['tamanho_explicacao']}): {sorted(r['explicacao'])}\n\n")
        
        f.write("="*80 + "\n" + "                    SUMÁRIO ESTATÍSTICO DAS EXPLICAÇÕES\n" + "="*80 + "\n\n")
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
        
        def print_proc_stats(label: str, stats: Dict[str, Any]):
            f.write(f"[ ESTATÍSTICAS DO PROCESSO - {label} ]\n")
            f.write(f"  - Instâncias com adições na Fase 1: {stats['inst_com_adicao']} ({stats['perc_adicao']:.2f}%)\n")
            if stats['inst_com_adicao'] > 0:
                f.write(f"    - Média de features adicionadas: {stats['media_adicoes']:.2f}\n")
            f.write(f"  - Instâncias com remoção efetiva na Fase 2: {stats['inst_com_remocao']} ({stats['perc_remocao']:.2f}%)\n")
            if stats['inst_com_remocao'] > 0:
                f.write(f"    - Média de features removidas: {stats['media_remocoes']:.2f}\n\n")

        print_proc_stats("Positiva", metricas['processo_stats_pos'])
        print_proc_stats("Negativa", metricas['processo_stats_neg'])
        print_proc_stats("Rejeitada", metricas['processo_stats_rej'])

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
    # Extrai parâmetros do scaler para JSON
    scaler_params = {
        'min': [float(v) for v in model_params['scaler_params']['min']],
        'scale': [float(v) for v in model_params['scaler_params']['scale']]
    }

    feature_names = list(X_train.columns)
    coefs_ordered = [float(model_params['coefs'][col]) for col in feature_names]
    intercepto = float(model_params['intercepto'])

    # Converte X_test e y_test
    X_test_dict = {str(col): [float(x) for x in X_test[col].tolist()] for col in X_test.columns}
    y_test_list = [int(v) for v in y_test.tolist()]

    # Per-instance
    per_instance = []
    for i, rid in enumerate(X_test.index):
        per_instance.append({
            'id': str(rid),
            'y_true': int(y_test.iloc[i]),
            'y_pred': int(y_pred_test[i]) if int(y_pred_test[i]) in (0, 1) else -1,
            'rejected': bool(rejected_mask[i]),
            'decision_score': float(decision_scores_test[i]),
            'explanation': list(resultados_instancias[i]['explicacao']),
            'explanation_size': int(resultados_instancias[i]['tamanho_explicacao'])
        })

    # Metadados extras para MNIST (modo de features e par selecionado)
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

    dataset_cache = {
        'config': {
            'dataset_name': dataset_name,
            'test_size': float(test_size_atual),
            'random_state': RANDOM_STATE,
            'rejection_cost': float(WR_REJECTION_COST),
            'subsample_size': float(DATASET_CONFIG.get(dataset_name, {}).get('subsample_size', 0.0)) if DATASET_CONFIG.get(dataset_name, {}).get('subsample_size') else None,
            **mnist_meta
        },
        'model': {
            'params': {k: v for k, v in model_params.items() if k not in ['coefs', 'intercepto', 'scaler_params']},
        },
        'thresholds': {
            't_plus': float(t_plus),
            't_minus': float(t_minus)
        },
        'performance': {
            'accuracy_without_rejection': float(metricas_dict['acuracia_sem_rejeicao']),
            'accuracy_with_rejection': float(metricas_dict['acuracia_com_rejeicao']),
            'rejection_rate': float(metricas_dict['taxa_rejeicao'])
        },
        'explanation_stats': {
            'positive': {
                'count': int(metricas_dict['stats_explicacao_positiva']['instancias']),
                'min_length': int(metricas_dict['stats_explicacao_positiva']['min']),
                'mean_length': float(metricas_dict['stats_explicacao_positiva']['media']),
                'max_length': int(metricas_dict['stats_explicacao_positiva']['max']),
                'std_length': float(metricas_dict['stats_explicacao_positiva']['std_dev'])
            },
            'negative': {
                'count': int(metricas_dict['stats_explicacao_negativa']['instancias']),
                'min_length': int(metricas_dict['stats_explicacao_negativa']['min']),
                'mean_length': float(metricas_dict['stats_explicacao_negativa']['media']),
                'max_length': int(metricas_dict['stats_explicacao_negativa']['max']),
                'std_length': float(metricas_dict['stats_explicacao_negativa']['std_dev'])
            },
            'rejected': {
                'count': int(metricas_dict['stats_explicacao_rejeitada']['instancias']),
                'min_length': int(metricas_dict['stats_explicacao_rejeitada']['min']),
                'mean_length': float(metricas_dict['stats_explicacao_rejeitada']['media']),
                'max_length': int(metricas_dict['stats_explicacao_rejeitada']['max']),
                'std_length': float(metricas_dict['stats_explicacao_rejeitada']['std_dev'])
            }
        },
        'computation_time': {
            'total': float(metricas_dict.get('tempo_total', 0.0)),
            'mean_per_instance': float(metricas_dict.get('tempo_medio_instancia', 0.0)),
            'positive': float(metricas_dict.get('tempo_medio_positivas', 0.0)),
            'negative': float(metricas_dict.get('tempo_medio_negativas', 0.0)),
            'rejected': float(metricas_dict.get('tempo_medio_rejeitadas', 0.0))
        },
        'top_features': [
            {"feature": feat, "count": int(count)}
            for feat, count in metricas_dict['features_frequentes']
        ],
        'data': {
            'feature_names': feature_names,
            'class_names': list(nomes_classes),
            'X_test': X_test_dict,
            'y_test': y_test_list
        },
        'model': {
            'params': {k: v for k, v in model_params.items() if k not in ['coefs', 'intercepto', 'scaler_params']},
            'coefs': coefs_ordered,
            'intercept': intercepto,
            'scaler_params': scaler_params
        },
        'per_instance': per_instance
    }
    return dataset_cache

if __name__ == '__main__':
    nome_dataset, _, _, _, _ = selecionar_dataset_e_classe()
    if not nome_dataset:
        print("Operação cancelada pelo usuário.")
    else:
        try:
            executar_experimento_para_dataset(nome_dataset)
            print("\n[INFO] Processo concluído com sucesso.")
        except Exception as e:
            print(f"\n[ERRO FATAL] Ocorreu um erro durante a execução para o dataset '{nome_dataset}': {e}")
            import traceback
            traceback.print_exc()