"""
Script para análise detalhada de instâncias específicas usando cache cumulativo(JSON).
"""

import os
import sys
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Any

# Fontes de dados
PKL_CACHE_FILE = Path("cache/cache_cumulativo.pkl")
RESULTS_JSON_FILE = Path("json/comparative_results.json")

def _load_from_json() -> dict:
    with open(RESULTS_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'peab' not in data or not isinstance(data['peab'], dict):
        raise ValueError("Arquivo JSON não contém chave 'peab' válida")

    compat = {}
    for dataset, info in data['peab'].items():
        try:
            feature_names = info['data']['feature_names']
            class_names = info['data']['class_names']
            X_test = info['data']['X_test']
            y_test = info['data']['y_test']
            t_plus = info['thresholds']['t_plus']
            t_minus = info['thresholds']['t_minus']
            coefs = info['model']['coefs']
            intercept = info['model']['intercept']
            scaler_params = info['model']['scaler_params']
            per_inst = info.get('per_instance', [])

            per_instance = {
                'y_true': [pi.get('y_true') for pi in per_inst],
                'y_pred': [pi.get('y_pred') if not pi.get('rejected') else -1 for pi in per_inst],
                'decision_scores': [pi.get('decision_score') for pi in per_inst],
                'rejected': [pi.get('rejected') for pi in per_inst]
            }

            compat[dataset] = {
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': feature_names,
                'nomes_classes': class_names,
                't_plus': t_plus,
                't_minus': t_minus,
                'model_coefs': coefs,
                'model_intercept': intercept,
                'scaler_params': scaler_params,
                'stats': {
                    'acuracia_geral': info.get('performance', {}).get('accuracy_with_rejection', 0.0)
                },
                'per_instance': per_instance
            }
        except Exception as e:
            print(f"Aviso: dataset '{dataset}' no JSON está incompleto: {e}")
    return compat

def load_cache() -> dict:
    """Carrega dados das explicações a partir do JSON (prioritário) ou PKL (legado)."""
    if RESULTS_JSON_FILE.exists():
        try:
            compat = _load_from_json()
            if not compat:
                raise ValueError("JSON encontrado, mas sem datasets completos no formato novo")
            return compat
        except Exception as e:
            print(f"Falha ao carregar JSON: {e}. Tentando PKL legado...")
    if PKL_CACHE_FILE.exists():
        cache = joblib.load(PKL_CACHE_FILE)
        return cache if isinstance(cache, dict) else {}
    raise FileNotFoundError("Nenhuma fonte de dados encontrada (JSON nem PKL)")

def list_available_datasets(cache: dict) -> None:
    """Lista os datasets disponíveis no cache."""
    print("\nDatasets disponiveis no cache:")
    for i, dataset in enumerate(sorted(cache.keys()), 1):
        total = 0
        block = cache[dataset]
        if isinstance(block.get('X_test'), dict):
            first_feature = next(iter(block['X_test'].values()))
            total = len(first_feature) if isinstance(first_feature, list) else 0
        print(f"{i}. {dataset:<25} ({total} instancias)")
    print()

def format_line():
    return "=" * 72

# ============================
# Formatação e Legendas
# ============================
SYMBOL_LEGEND = [
    "LEGENDA DOS SÍMBOLOS",
    "   δ (Delta): Contribuição real da feature nesta instância (w_i × x_i)",
    "   Σδ: Soma acumulada das contribuições",
    "   ● Feature Essencial: precisa ser fixada para garantir o resultado",
    "   ○ Feature Desafixada: pode variar sem alterar o resultado",
    "   s': Score após desafixar/perturbar sob pior cenário",
]

def salvar_relatorio_unico(dataset_nome: str, idx: int, texto: str, dados_json: dict):
    """Salva um único relatório textual e o JSON auxiliar para a instância.

    Estrutura: results/explanations/<dataset>/instancia_<idx>/
    """
    pasta = os.path.join("results", "explanations", dataset_nome, f"instancia_{idx}")
    os.makedirs(pasta, exist_ok=True)
    arq_txt = os.path.join(pasta, "relatorio_detalhado.txt")
    with open(arq_txt, "w", encoding="utf-8") as f:
        f.write(texto)
    with open(os.path.join(pasta, "dados.json"), "w", encoding="utf-8") as f:
        json.dump(dados_json, f, indent=2, ensure_ascii=False)
    print(f"\nRelatório salvo em: {arq_txt}\n")

# ============================
# Apoio Matemático (linear, sem sklearn)
# ============================
def compute_score(b: float, w: np.ndarray, x_scaled: np.ndarray) -> float:
    return float(b + np.dot(w, x_scaled))

def worst_value_for_direction(coef: float, direction: str) -> float:
    # Em espaço escalonado [0,1]
    # direction: 'min' -> minimizar s, 'max' -> maximizar s
    if direction == 'min':
        return 0.0 if coef > 0 else 1.0
    else:  # 'max'
        return 1.0 if coef > 0 else 0.0

def s_prime_with_fixed(b: float, w: np.ndarray, x_scaled: np.ndarray, fixed_idx: set, direction: str) -> float:
    x_mod = x_scaled.copy()
    for i, wi in enumerate(w):
        if i in fixed_idx:
            continue
        x_mod[i] = worst_value_for_direction(wi, direction)
    return compute_score(b, w, x_mod)

# ============================
# Funções espelhadas do peab_2 (paridade exata)
# ============================
def _orig_min_max_from_scaler(scaler_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstrói min/max originais a partir de scaler_params do MinMaxScaler."""
    scale = np.array(scaler_params['scale'], dtype=float)
    min_ = np.array(scaler_params['min'], dtype=float)
    # Para feature_range=(0,1): x_scaled = x*scale + min_;
    # x_min = -min_/scale ; x_max = x_min + 1/scale
    x_min = -min_ / scale
    x_max = x_min + 1.0 / scale
    return x_min, x_max

def pe2_calculate_deltas(w: np.ndarray, x_scaled: np.ndarray, premis_class: int) -> np.ndarray:
    """Equivalente ao calculate_deltas do peab_2 usando espaço escalonado (0/1)."""
    # Em peab_2 usa min/max do X_train escalonado, que são 0 e 1 no MinMaxScaler.
    worst = np.zeros_like(x_scaled)
    for i, wi in enumerate(w):
        if premis_class == 1:
            worst[i] = 0.0 if wi > 0 else 1.0
        else:
            worst[i] = 1.0 if wi > 0 else 0.0
    return (x_scaled - worst) * w

def pe2_one_explanation_formal(b: float, w: np.ndarray, x_scaled: np.ndarray, t_plus: float, t_minus: float, premis_class: int, feature_values: np.ndarray, feature_names: list) -> list:
    score = compute_score(b, w, x_scaled)
    deltas = pe2_calculate_deltas(w, x_scaled, premis_class)
    indices_ordenados = np.argsort(-np.abs(deltas))
    explicacao = []  # lista de strings "feat = valor"
    score_base = score - float(np.sum(deltas))
    soma_deltas = score_base
    target = t_plus if premis_class == 1 else t_minus
    for i in indices_ordenados:
        if abs(deltas[i]) > 1e-9:
            soma_deltas += deltas[i]
            explicacao.append(f"{feature_names[i]} = {feature_values[i]:.4f}")
        if (premis_class == 1 and soma_deltas > target and explicacao) or (premis_class == 0 and soma_deltas < target and explicacao):
            break
    if not explicacao and len(deltas) > 0:
        idx0 = int(indices_ordenados[0])
        explicacao.append(f"{feature_names[idx0]} = {feature_values[idx0]:.4f}")
    return explicacao

def _decision_from_original(b: float, w: np.ndarray, orig_vals: np.ndarray, scaler_params: dict) -> float:
    scale = np.array(scaler_params['scale'], dtype=float)
    min_ = np.array(scaler_params['min'], dtype=float)
    x_scaled = orig_vals * scale + min_
    return compute_score(b, w, x_scaled)

def pe2_perturbar_e_validar(explicacao: list, feature_names: list, b: float, w: np.ndarray, instance_orig: np.ndarray, scaler_params: dict, t_plus: float, t_minus: float, direcao_override: int) -> Tuple[bool, float]:
    if not explicacao:
        return False, 0.0
    feats_expl = {f.split(' = ')[0] for f in explicacao}
    x_min, x_max = _orig_min_max_from_scaler(scaler_params)
    inst_mod = instance_orig.copy()
    diminuir = (direcao_override == 1)
    for i, fname in enumerate(feature_names):
        if fname in feats_expl:
            continue
        coef = w[i]
        if diminuir:
            inst_mod[i] = x_min[i] if coef > 0 else x_max[i]
        else:
            inst_mod[i] = x_max[i] if coef > 0 else x_min[i]
    score_pert = _decision_from_original(b, w, inst_mod, scaler_params)
    score_original = _decision_from_original(b, w, instance_orig, scaler_params)
    pert_rejeitada = (t_minus <= score_pert <= t_plus)
    is_original_rej = (t_minus <= score_original <= t_plus)
    if is_original_rej:
        return pert_rejeitada, score_pert
    else:
        pred_original = 1 if score_original >= 0 else 0
        pred_pert = 1 if score_pert >= 0 else 0
        return (pred_pert == pred_original) and (not pert_rejeitada), score_pert

def pe2_fase_1_reforco(b: float, w: np.ndarray, x_scaled: np.ndarray, instance_orig: np.ndarray, feature_names: list, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int, scaler_params: dict) -> Tuple[list, int]:
    expl_robusta = pe2_one_explanation_formal(b, w, x_scaled, t_plus, t_minus, premisa_ordenacao, instance_orig, feature_names)
    adicoes = 0
    deltas = pe2_calculate_deltas(w, x_scaled, premisa_ordenacao)
    indices_ordenados = np.argsort(-np.abs(deltas))
    while True:
        if is_rejected:
            valido1, _ = pe2_perturbar_e_validar(expl_robusta, feature_names, b, w, instance_orig, scaler_params, t_plus, t_minus, 1)
            valido2, _ = pe2_perturbar_e_validar(expl_robusta, feature_names, b, w, instance_orig, scaler_params, t_plus, t_minus, 0)
            if valido1 and valido2:
                break
        else:
            direcao = 1 if (_decision_from_original(b, w, instance_orig, scaler_params) >= 0) else 0
            valido, _ = pe2_perturbar_e_validar(expl_robusta, feature_names, b, w, instance_orig, scaler_params, t_plus, t_minus, direcao)
            if valido:
                break
        if len(expl_robusta) == len(feature_names):
            break
        feats_set = {f.split(' = ')[0] for f in expl_robusta}
        adicionou = False
        for idx in indices_ordenados:
            nome = feature_names[idx]
            if nome not in feats_set:
                expl_robusta.append(f"{nome} = {instance_orig[idx]:.4f}")
                adicoes += 1
                adicionou = True
                break
        if not adicionou:
            break
    return expl_robusta, adicoes

def pe2_fase_2_minimizacao(b: float, w: np.ndarray, x_scaled: np.ndarray, instance_orig: np.ndarray, feature_names: list, t_plus: float, t_minus: float, is_rejected: bool, premisa_ordenacao: int, scaler_params: dict) -> Tuple[list, int, list]:
    expl_minima = pe2_one_explanation_formal(b, w, x_scaled, t_plus, t_minus, premisa_ordenacao, instance_orig, feature_names)
    # reforça primeiro para ficar robusta
    expl_minima, _ = pe2_fase_1_reforco(b, w, x_scaled, instance_orig, feature_names, t_plus, t_minus, is_rejected, premisa_ordenacao, scaler_params)
    remocoes = 0
    passos = []
    deltas = pe2_calculate_deltas(w, x_scaled, premisa_ordenacao)
    feats_para_remover = sorted(
        [f.split(' = ')[0] for f in expl_minima],
        key=lambda nome: abs(deltas[feature_names.index(nome)]),
        reverse=True
    )
    for feat_nome in feats_para_remover:
        if len(expl_minima) <= 1:
            break
        expl_temp = [f for f in expl_minima if not f.startswith(feat_nome)]
        delta_feat = float(deltas[feature_names.index(feat_nome)])
        if is_rejected:
            ok1, s1 = pe2_perturbar_e_validar(expl_temp, feature_names, b, w, instance_orig, scaler_params, t_plus, t_minus, 1)
            ok2, s2 = pe2_perturbar_e_validar(expl_temp, feature_names, b, w, instance_orig, scaler_params, t_plus, t_minus, 0)
            sucesso = ok1 and ok2
            passos.append({'feat_nome': feat_nome, 'delta': delta_feat, 'score_neg': s1, 'ok_neg': bool(ok1), 'score_pos': s2, 'ok_pos': bool(ok2), 'sucesso': sucesso})
        else:
            direcao = 1 if (_decision_from_original(b, w, instance_orig, scaler_params) >= 0) else 0
            sucesso, s_ = pe2_perturbar_e_validar(expl_temp, feature_names, b, w, instance_orig, scaler_params, t_plus, t_minus, direcao)
            passos.append({'feat_nome': feat_nome, 'delta': delta_feat, 'score_perturbado': s_, 'sucesso': sucesso})
        if sucesso:
            expl_minima = expl_temp
            remocoes += 1
    return expl_minima, remocoes, passos

def deltas_for_premise(w: np.ndarray, x_scaled: np.ndarray, premis_class: int) -> np.ndarray:
    # Delta_i = (x_i - worst_i) * w_i, onde worst_i depende da classe-premissa
    worst = np.zeros_like(x_scaled)
    for i, wi in enumerate(w):
        if premis_class == 1:
            worst[i] = 0.0 if wi > 0 else 1.0
        else:
            worst[i] = 1.0 if wi > 0 else 0.0
    return (x_scaled - worst) * w

def gerar_secao4_estabilidade(feature_names, valores_escalonados, coefs, contribuicoes,
                              explicacao_minimal, intercepto, score, t_plus, t_minus,
                              pred_class, is_rejected, scaler_params=None):
    """Gera a secao 4: testes de estabilidade."""
    
    secao4 = f"""
{'='*80}
SECAO 4: VERIFICACAO DE ESTABILIDADE
{'='*80}

Testando robustez da explicacao com perturbacoes:

"""
    
    epsilon = 0.01  # Perturbação de 1%
    
    for i in explicacao_minimal[:5]:  # Testa as 5 features principais
        valor_original = valores_escalonados[i]
        
        # Perturbação positiva
        valor_pos = valor_original * (1 + epsilon)
        contrib_pos = coefs[i] * valor_pos
        delta_pos = contrib_pos - contribuicoes[i]
        score_pos = score + delta_pos
        
        # Perturbação negativa
        valor_neg = valor_original * (1 - epsilon)
        contrib_neg = coefs[i] * valor_neg
        delta_neg = contrib_neg - contribuicoes[i]
        score_neg = score + delta_neg
        
        secao4 += f"\nFeature: {feature_names[i]}\n"
        secao4 += f"  Valor original: {valor_original:.6f}\n"
        secao4 += f"  Perturbacao +{epsilon*100}%: x' = {valor_pos:.6f} -> Delta_s = {delta_pos:+.6f} -> s' = {score_pos:.6f}\n"
        secao4 += f"  Perturbacao -{epsilon*100}%: x' = {valor_neg:.6f} -> Delta_s = {delta_neg:+.6f} -> s' = {score_neg:.6f}\n"
        
        # Verifica se mantém a classificação
        if not is_rejected:
            if pred_class == 1:
                manteve_pos = score_pos >= t_plus
                manteve_neg = score_neg >= t_plus
            else:
                manteve_pos = score_pos <= t_minus
                manteve_neg = score_neg <= t_minus
            
            secao4 += f"  Estavel: {'+' if manteve_pos else 'X'} (positiva), {'+' if manteve_neg else 'X'} (negativa)\n"
    
    return secao4


def gerar_secao5_minimizacao(feature_names, contribuicoes, explicacao_minimal,
                             intercepto, score, t_plus, t_minus, pred_class, is_rejected):
    """Gera a seção 5: prova de minimalidade."""
    
    secao5 = f"""
{'='*80}
SECAO 5: PROVA DE MINIMALIDADE
{'='*80}

Teorema: A explicacao E = {{{', '.join([feature_names[i] for i in explicacao_minimal])}}} eh minimal.

Prova por contradicao:
Suponha que existe E' subset E tal que E' tambem satisfaz a condicao de classificacao.
Testando remocao de cada feature:

"""
    
    for i in explicacao_minimal[:5]:  # Testa remoção das 5 principais
        score_sem_feature = score - contribuicoes[i]
        
        secao5 += f"\nTentativa: Remover {feature_names[i]}\n"
        secao5 += f"  Score sem esta feature: s' = {score:.6f} - {contribuicoes[i]:.6f} = {score_sem_feature:.6f}\n"
        
        if not is_rejected:
            if pred_class == 1:
                manteve = score_sem_feature >= t_plus
                secao5 += f"  Condicao: s' >= t+ -> {score_sem_feature:.6f} >= {t_plus:.6f} -> {manteve}\n"
            else:
                manteve = score_sem_feature <= t_minus
                secao5 += f"  Condicao: s' <= t- -> {score_sem_feature:.6f} <= {t_minus:.6f} -> {manteve}\n"
            
            if manteve:
                secao5 += f"  AVISO: A feature poderia ser removida!\n"
            else:
                secao5 += f"  Feature eh NECESSARIA (remocao viola a condicao)\n"
    
    secao5 += f"\nPortanto: Nenhuma feature em E pode ser removida sem violar a condicao.\n"
    secao5 += f"Logo, E eh MINIMAL.\n"
    
    return secao5


def gerar_secao6_interpretacao(dataset_nome, idx, nomes_classes, true_class, pred_class,
                               score, t_plus, t_minus, feature_names, valores_originais,
                               explicacao_minimal, contribuicoes, tamanho_explicacao,
                               is_rejected, classe_texto):
    """Gera a seção 6: interpretação final."""
    
    secao6 = f"""
{'='*80}
SECAO 6: INTERPRETACAO E RACIOCINIO ABDUTIVO
{'='*80}

SUMARIO DA EXPLICACAO:
----------------------
Dataset: {dataset_nome}
Instancia: {idx}
Classe Real: {nomes_classes[true_class]}
Predicao: {classe_texto}
Score Final: {score:.6f}
Status: {'REJEITADA' if is_rejected else 'ACEITA'}

EXPLICACAO MINIMAL:
-------------------
A classificacao eh determinada por {tamanho_explicacao} feature(s):

"""
    
    for pos, i in enumerate(explicacao_minimal, 1):
        secao6 += f"{pos}. {feature_names[i]}\n"
        secao6 += f"   Valor: {valores_originais[i]:.6f}\n"
        secao6 += f"   Contribuicao: {contribuicoes[i]:+.6f}\n"
    
    secao6 += f"\nRACIOCINIO ABDUTIVO:\n"
    secao6 += f"--------------------\n"
    
    if is_rejected:
        secao6 += f"O modelo REJEITOU esta instancia porque:\n"
        secao6 += f"  - Score s(x) = {score:.6f}\n"
        secao6 += f"  - Intervalo de rejeicao: {t_minus:.6f} < s(x) < {t_plus:.6f}\n"
        secao6 += f"  - Interpretacao: Incerteza alta, decisao nao confiavel\n"
    elif pred_class == 1:
        secao6 += f"O modelo classificou como {nomes_classes[1]} porque:\n"
        secao6 += f"  - Score s(x) = {score:.6f} >= t+ = {t_plus:.6f}\n"
        secao6 += f"  - As features listadas contribuem positivamente\n"
        secao6 += f"  - Esta eh a explicacao MINIMAL suficiente\n"
    else:
        secao6 += f"O modelo classificou como {nomes_classes[0]} porque:\n"
        secao6 += f"  - Score s(x) = {score:.6f} <= t- = {t_minus:.6f}\n"
        secao6 += f"  - As features listadas contribuem para score baixo\n"
        secao6 += f"  - Esta eh a explicacao MINIMAL suficiente\n"
    
    # Análise de correção
    if true_class == pred_class and not is_rejected:
        secao6 += f"\nPREDICAO CORRETA: Classe real = Classe predita\n"
    elif is_rejected:
        secao6 += f"\nINSTANCIA REJEITADA: Modelo optou por nao decidir\n"
    else:
        secao6 += f"\nPREDICAO INCORRETA: Classe real ({nomes_classes[true_class]}) != Classe predita ({classe_texto})\n"
    
    secao6 += f"\n{'='*80}\n"
    secao6 += f"FIM DA EXPLICACAO DETALHADA\n"
    secao6 += f"{'='*80}\n"
    
    return secao6


def gerar_relatorio_matematico_completo(dataset_nome, idx, true_class, pred_class, score, t_plus, t_minus,
                                        intercepto, acuracia_geral, nomes_classes, feature_names,
                                        valores_originais, valores_escalonados, coefs, contribuicoes,
                                        indices_ordenados, is_rejected, status_texto, classe_texto, scaler_params=None):
    """Gera um relatório único e técnico, agora espelhando exatamente o cálculo do peab_2."""

    w = np.array(coefs, dtype=float)
    x_scaled = np.array(valores_escalonados, dtype=float)
    b = float(intercepto)
    x_orig = np.array(valores_originais, dtype=float)

    # Cabeçalho e legenda
    header = (
        f"{'='*80}\n"
        f"          PROVA MATEMÁTICA DA EXPLICAÇÃO MINIMAL - INSTÂNCIA #{idx}\n"
        f"{'='*80}\n"
        f"Dataset: {dataset_nome}\n"
        f"Classe Real: {nomes_classes[true_class]}\n"
        f"Predição: {('REJEITADA' if is_rejected else ('CLASSE ' + str(pred_class) + ' (' + classe_texto + ')'))}\n"
        f"Score Final: {score:.4f}\n\n"
        + "\n".join(SYMBOL_LEGEND) + "\n\n"
    )

    # Seção 1: Dados iniciais e tabela de contribuições
    sec1 = (
        f"{'='*80}\nSEÇÃO 1: DADOS INICIAIS (O PONTO DE PARTIDA)\n{'='*80}\n"
        f"- Intercepto do Modelo (Viés Inicial): {b:.4f}\n"
        f"- Limiares de Decisão: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n"
        + (f"- Condição para Classe 1: Score >= {t_plus:.4f}\n" if not is_rejected and pred_class == 1 else (f"- Condição para Classe 0: Score <= {t_minus:.4f}\n" if not is_rejected else f"- Condição de Rejeição: {t_minus:.4f} < Score < {t_plus:.4f}\n"))
        + "\nTabela de Contribuições (δ):\n"
        + "-"*80 + "\n"
        + f"{'Feature':<30} | {'Original':>10} | {'Normalizado':>12} | {'Peso':>10} | {'Contribuição (δ)':>18}\n"
        + "-"*80 + "\n"
    )
    for i, fname in enumerate(feature_names):
        arrow = '↑' if w[i]*x_scaled[i] >= 0 else '↓'
        sec1 += f"{fname:<30} | {valores_originais[i]:>10.4f} | {x_scaled[i]:>12.4f} | {w[i]:>10.4f} | {w[i]*x_scaled[i]:>+18.4f} {arrow}\n"
    soma = float(np.sum(w * x_scaled))
    sec1 += ("-"*80 + "\n" + f"Soma das Contribuições: {soma:+.4f}\n" + f"Score Verificado: {b:.4f} + {soma:+.4f} = {b+soma:.4f}\n\n")

    # Seção 2 e 3: Cálculo exato do peab_2 (heurística + reforço + minimização)
    if scaler_params is None:
        scaler_params = {'scale': [1.0]*len(feature_names), 'min': [0.0]*len(feature_names)}

    if is_rejected:
        expl1, _, passos1 = pe2_fase_2_minimizacao(b, w, x_scaled, x_orig, feature_names, t_plus, t_minus, True, 1, scaler_params)
        expl2, _, passos2 = pe2_fase_2_minimizacao(b, w, x_scaled, x_orig, feature_names, t_plus, t_minus, True, 0, scaler_params)
        if len(expl1) <= len(expl2):
            expl_min = expl1
            passos = passos1
        else:
            expl_min = expl2
            passos = passos2
        tipo = "REJEITADA (bidirecional)"
    else:
        prem = int(pred_class)
        expl_min, _, passos = pe2_fase_2_minimizacao(b, w, x_scaled, x_orig, feature_names, t_plus, t_minus, False, prem, scaler_params)
        tipo = f"CLASSIFICADA (classe {prem})"

    feats_min = [f.split(' = ')[0] for f in expl_min]
    sec2 = f"{'='*80}\nSEÇÃO 2: EXPLICAÇÃO MINIMAL (PARIDADE COM PEAB_2)\n{'='*80}\n"
    sec2 += f"Tipo: {tipo}\n"
    sec2 += f"Conjunto minimal ({len(feats_min)} features): {sorted(feats_min)}\n\n"

    sec3 = f"{'='*80}\nSEÇÃO 3: PASSOS DE MINIMIZAÇÃO (RESUMO)\n{'='*80}\n"
    if is_rejected:
        for p in passos:
            cmp_neg = f"> t- ({t_minus:.4f})" if p.get('score_neg', 0.0) > t_minus else f"<= t- ({t_minus:.4f})"
            cmp_pos = f"< t+ ({t_plus:.4f})" if p.get('score_pos', 0.0) < t_plus else f">= t+ ({t_plus:.4f})"
            sec3 += (
                f"- {p['feat_nome']} (δ {p.get('delta', 0.0):+.3f}): s'_neg={p.get('score_neg', np.nan):.3f} ({cmp_neg}), "
                f"s'_pos={p.get('score_pos', np.nan):.3f} ({cmp_pos}) -> {'OK' if p.get('sucesso', False) else 'FALHA'}\n"
            )
    else:
        limiar_val = t_plus if pred_class == 1 else t_minus
        cond = "> t+" if pred_class == 1 else "< t-"
        for p in passos:
            sec3 += (
                f"- {p['feat_nome']} (δ {p.get('delta', 0.0):+.3f}): s'={p.get('score_perturbado', np.nan):.3f} ({cond} ({limiar_val:.4f})) -> "
                f"{'OK' if p.get('sucesso', False) else 'FALHA'}\n"
            )

    # Seção final: interpretação e fechamento (como no formato anterior)
    explicacao_minimal_idx = [feature_names.index(n) for n in feats_min]
    tamanho_explicacao = len(explicacao_minimal_idx)
    secao_final = gerar_secao6_interpretacao(
        dataset_nome, idx, nomes_classes, true_class, pred_class,
        score, t_plus, t_minus, feature_names, valores_originais,
        explicacao_minimal_idx, contribuicoes, tamanho_explicacao,
        is_rejected, classe_texto
    )

    return header + sec1 + sec2 + sec3 + secao_final


def processar_instancia(dataset_nome, cache_dataset, X_test, y_test, t_plus, t_minus, nomes_classes, feature_names, idx):
    """Processa uma instância específica e gera explicações matemáticas rigorosas."""
    try:
        # Extrai dados do cache
        per_instance_data = cache_dataset.get('per_instance', {})
        coefs = cache_dataset.get('model_coefs', None)
        intercepto = cache_dataset.get('model_intercept', None)
        scaler_params = cache_dataset.get('scaler_params', None)
        acuracia_geral = cache_dataset.get('stats', {}).get('acuracia_geral', 0.0)
        
        # Verifica se o cache tem os dados necessários
        if coefs is None or intercepto is None or scaler_params is None:
            print(f"AVISO: Cache antigo detectado para {dataset_nome}!")
            print(f"Execute peab_comparation.py novamente para gerar cache completo.")
            print(f"Dados faltando: coefs={coefs is None}, intercept={intercepto is None}, scaler={scaler_params is None}")
            return False
        
        # Dados da instância
        instance = X_test.iloc[[idx]]
        true_class = int(y_test.iloc[idx])
        pred_class = int(per_instance_data['y_pred'][idx])
        score = float(per_instance_data['decision_scores'][idx])
        is_rejected = bool(per_instance_data['rejected'][idx])
        
        # Reconstrói valores escalonados
        valores_originais = [float(instance.iloc[0, i]) for i in range(len(feature_names))]
        valores_escalonados = []
        
        # Verifica se scaler_params tem estrutura correta
        if scaler_params and 'scale' in scaler_params and 'min' in scaler_params:
            for i, val in enumerate(valores_originais):
                # MinMaxScaler: X_scaled = (X - min) / (max - min) = X * scale + min_
                val_scaled = val * scaler_params['scale'][i] + scaler_params['min'][i]
                valores_escalonados.append(float(val_scaled))
        else:
            # Se não houver scaler_params, assume que já está escalonado
            valores_escalonados = valores_originais.copy()
        
        # Calcula contribuições
        contribuicoes = [coefs[i] * valores_escalonados[i] for i in range(len(feature_names))]
        
        # Ordena features por contribuição absoluta
        indices_ordenados = sorted(range(len(contribuicoes)), key=lambda i: abs(contribuicoes[i]), reverse=True)
        
        # Determina status e limiar relevante
        if is_rejected:
            status_texto = "REJEITADA"
            classe_texto = "Rejeitada"
        elif pred_class == 1:
            status_texto = "ACEITA - CLASSE 1 (POSITIVA)"
            classe_texto = nomes_classes[1] if len(nomes_classes) > 1 else "Classe 1"
        else:
            status_texto = "ACEITA - CLASSE 0 (NEGATIVA)"
            classe_texto = nomes_classes[0] if len(nomes_classes) > 0 else "Classe 0"
        
        # Gera relatório completo
        relatorio = gerar_relatorio_matematico_completo(
            dataset_nome, idx, true_class, pred_class, score, t_plus, t_minus,
            intercepto, acuracia_geral, nomes_classes, feature_names,
            valores_originais, valores_escalonados, coefs, contribuicoes,
            indices_ordenados, is_rejected, status_texto, classe_texto, scaler_params
        )
        
        # Prepara dados JSON
        dados_json = {
            "dataset": dataset_nome,
            "instancia": int(idx),
            "classe_real": str(nomes_classes[true_class]),
            "classe_predita": classe_texto,
            "score": float(score),
            "status": status_texto,
            "thresholds": {"t_plus": float(t_plus), "t_minus": float(t_minus)},
            "intercepto": float(intercepto),
            "acuracia_global": float(acuracia_geral),
            "features": {
                str(feature_names[i]): {
                    "valor_original": float(valores_originais[i]),
                    "valor_escalonado": float(valores_escalonados[i]),
                    "peso": float(coefs[i]),
                    "contribuicao": float(contribuicoes[i])
                }
                for i in range(len(feature_names))
            }
        }
        
        # Salva relatório único + JSON
        salvar_relatorio_unico(dataset_nome, idx, relatorio, dados_json)
        
    except Exception as e:
        print(f"ERRO ao processar instancia {idx}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_prova_detalhada():
    """Função principal para análise detalhada de instâncias."""
    print(format_line())
    print("PROVA MATEMATICA DETALHADA DE UMA INSTANCIA")
    
    try:
        # Carrega o cache cumulativo
        cache_completo = load_cache()
        
        # Lista datasets disponíveis
        list_available_datasets(cache_completo)
        
        # Seleção do dataset
        while True:
            try:
                escolha = input("Digite o numero do dataset desejado: ")
                datasets = sorted(cache_completo.keys())
                dataset_nome = datasets[int(escolha) - 1]
                break
            except (ValueError, IndexError):
                print("ERRO: Escolha invalida. Tente novamente.")
        
        try:
            # Obtém dados do cache para o dataset selecionado
            cache_dataset = cache_completo[dataset_nome]
            
            # Verifica componentes necessários
            required_keys = ['X_test', 'y_test', 't_plus', 't_minus', 'nomes_classes', 'feature_names']
            missing_keys = [k for k in required_keys if k not in cache_dataset]
            if missing_keys:
                raise KeyError(f"Dados ausentes: {', '.join(missing_keys)}")
            
            # Reconstrói os DataFrames e Series
            feature_names = cache_dataset['feature_names']
            
            # Reconstrói X_test e y_test
            try:
                X_test = pd.DataFrame(cache_dataset['X_test'])
                X_test = X_test[feature_names]  # Garante ordem correta das colunas
                y_test = pd.Series(cache_dataset['y_test'])
            except Exception as e:
                print("Conteúdo bruto de X_test:", type(cache_dataset['X_test']))
                print("Primeiras chaves:", list(cache_dataset['X_test'].keys())[:5] if isinstance(cache_dataset['X_test'], dict) else "N/A")
                raise ValueError(f"Erro ao reconstruir dados de teste: {e}")

            # Reconstrói X_train e y_train (opcional no JSON)
            if 'X_train' in cache_dataset and 'y_train' in cache_dataset:
                try:
                    X_train = pd.DataFrame(cache_dataset['X_train'])
                    X_train = X_train[feature_names]
                    y_train = pd.Series(cache_dataset['y_train'])
                except Exception:
                    # Dados de treino ausentes ou não necessários para esta análise
                    X_train, y_train = None, None
            
            # Carrega outros parâmetros
            t_plus = float(cache_dataset['t_plus'])
            t_minus = float(cache_dataset['t_minus'])
            nomes_classes = cache_dataset['nomes_classes']
            
            # Mostra informações sobre o dataset
            print(f"\nDataset: {dataset_nome}")
            print(f"Total de instancias de teste: {len(X_test)}")
            print(f"Classes disponiveis: {', '.join([str(nc) for nc in nomes_classes])}")
            
            # Seleção da instância
            while True:
                try:
                    idx = int(input("\nDigite o numero da instancia para analisar (0 ate {}): ".format(len(X_test)-1)))
                    if 0 <= idx < len(X_test):
                        break
                    print("ERRO: Indice fora do intervalo valido.")
                except ValueError:
                    print("ERRO: Por favor, digite um numero valido.")
            
            # Processa a instância selecionada
            processar_instancia(dataset_nome, cache_dataset, X_test, y_test,
                              t_plus, t_minus, nomes_classes, feature_names, idx)
            
        except Exception as e:
            print(f"ERRO ao processar dataset {dataset_nome}: {e}")
            return
        
    except Exception as e:
        print(f"ERRO geral: {e}")
        return

if __name__ == "__main__":
    try:
        run_prova_detalhada()
    except KeyboardInterrupt:
        print("\n\nOperacao cancelada pelo usuario.")
    except Exception as e:
        print(f"\n\nErro: {str(e)}")
    finally:
        print("\n" + format_line())