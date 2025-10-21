# -*- coding: utf-8 -*-
"""
Script para análise detalhada de instâncias específicas usando cache cumulativo.
"""

import os
import sys
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from datetime import datetime

# Constante para o cache cumulativo
CACHE_FILE = Path("cache/cache_cumulativo.pkl")

def load_cache() -> dict:
    """Carrega o cache cumulativo."""
    try:
        if not CACHE_FILE.exists():
            print(f"❌ Erro: Cache não encontrado em {CACHE_FILE}")
            print("Execute primeiro o script peab_comparation.py para gerar o cache.")
            sys.exit(1)
        
        cache = joblib.load(CACHE_FILE)
        if not isinstance(cache, dict) or not cache:
            raise ValueError("Cache inválido ou vazio")
            
        return cache
        
    except Exception as e:
        print(f"❌ Erro ao carregar cache: {e}")
        print("Execute novamente o script peab_comparation.py para gerar o cache.")
        sys.exit(1)

def list_available_datasets(cache: dict) -> None:
    """Lista os datasets disponíveis no cache."""
    print("\n📊 Datasets disponíveis no cache:")
    for i, dataset in enumerate(sorted(cache.keys()), 1):
        # Conta o número de instâncias corretamente (tamanho da lista de uma feature)
        if 'X_test' in cache[dataset] and isinstance(cache[dataset]['X_test'], dict):
            first_feature = list(cache[dataset]['X_test'].values())[0]
            total = len(first_feature) if isinstance(first_feature, list) else 0
        else:
            total = 0
        print(f"{i}. {dataset:<25} ({total} instâncias)")
    print()

def format_line():
    return "=" * 72

def salvar_relatorios(dataset_nome, idx, explicacao_tecnica, explicacao_academica, dados_json):
    """Salva relatórios técnico, acadêmico e JSON para uma instância."""
    pasta = os.path.join("results", "explanations", dataset_nome, f"instancia_{idx}")
    os.makedirs(pasta, exist_ok=True)
    
    with open(os.path.join(pasta, "relatorio_completo.txt"), "w", encoding="utf-8") as f:
        f.write(explicacao_tecnica)
    with open(os.path.join(pasta, "prova_matematica.txt"), "w", encoding="utf-8") as f:
        f.write(explicacao_academica)
    with open(os.path.join(pasta, "dados.json"), "w", encoding="utf-8") as f:
        json.dump(dados_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Relatórios salvos em: {pasta}\n")

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
                                        indices_ordenados, is_rejected, status_texto, classe_texto):
    """
    Gera relatório matemático rigoroso seguindo o template de prova formal.
    """
    
    # ============================================================================
    # SEÇÃO 1: DEFINIÇÕES DO MODELO
    # ============================================================================
    secao1 = f"""
{'='*80}
EXPLICACAO DETALHADA - INSTANCIA {idx}
Dataset: {dataset_nome}
{'='*80}

CLASSE REAL: {nomes_classes[true_class]}
PREDICAO: {classe_texto}
STATUS: {status_texto}
SCORE: {score:.6f}
THRESHOLDS: t+ = {t_plus:.6f}, t- = {t_minus:.6f}
ACURACIA GLOBAL: {acuracia_geral:.4f}

{'='*80}
SECAO 1: DEFINICAO DO MODELO
{'='*80}

O modelo de classificacao utiliza uma funcao linear de decisao:

    s(x) = w_0 + sum_i w_i * x_i

Onde:
- s(x) eh o score de decisao
- w_0 = {intercepto:.6f} (intercepto)
- w_i sao os pesos de cada feature
- x_i sao os valores escalonados das features

Regra de classificacao:
- Se s(x) >= t+ = {t_plus:.6f} -> Classe 1 ({nomes_classes[1] if len(nomes_classes) > 1 else 'Positiva'})
- Se s(x) <= t- = {t_minus:.6f} -> Classe 0 ({nomes_classes[0]})
- Se t- < s(x) < t+ -> REJEITADA (incerteza)

"""
    
    # ============================================================================
    # SECAO 2: TABELA DE FEATURES
    # ============================================================================
    secao2 = f"""
{'='*80}
SECAO 2: VALORES E CONTRIBUICOES DAS FEATURES
{'='*80}

"""
    secao2 += f"{'Feature':<30} | {'Original':>12} | {'Escalonado':>12} | {'Peso':>12} | {'Contribuicao':>14}\n"
    secao2 += f"{'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}\n"
    
    for i in range(len(feature_names)):
        secao2 += f"{feature_names[i]:<30} | {valores_originais[i]:>12.6f} | {valores_escalonados[i]:>12.6f} | {coefs[i]:>12.6f} | {contribuicoes[i]:>14.6f}\n"
    
    soma_contribuicoes = sum(contribuicoes)
    score_calculado = intercepto + soma_contribuicoes
    
    secao2 += f"\n{'Intercepto (w_0)':<30} | {'':<12} | {'':<12} | {'':<12} | {intercepto:>14.6f}\n"
    secao2 += f"{'='*30}=+={'='*12}=+={'='*12}=+={'='*12}=+={'='*14}\n"
    secao2 += f"{'SCORE TOTAL':<30} | {'':<12} | {'':<12} | {'':<12} | {score_calculado:>14.6f}\n"
    secao2 += f"\nVerificacao: s(x) = {intercepto:.6f} + {soma_contribuicoes:.6f} = {score_calculado:.6f}\n"
    
    # ============================================================================
    # SECAO 3: CONSTRUCAO DA EXPLICACAO
    # ============================================================================
    secao3 = f"""
{'='*80}
SECAO 3: CONSTRUCAO PROGRESSIVA DA EXPLICACAO
{'='*80}

Algoritmo de construcao da explicacao minimal E = {{f_1, f_2, ..., f_k}}:

Passo 0 (baseline): s_0 = w_0 = {intercepto:.6f}

"""
    
    score_acumulado = intercepto
    explicacao_minimal = []
    
    for pos, i in enumerate(indices_ordenados, 1):
        score_anterior = score_acumulado
        score_acumulado += contribuicoes[i]
        explicacao_minimal.append(i)
        
        secao3 += f"Passo {pos}: Adicionando {feature_names[i]}\n"
        secao3 += f"         s_({pos}) = s_({pos-1}) + w_({i}) * x_({i})\n"
        secao3 += f"         s_({pos}) = {score_anterior:.6f} + ({coefs[i]:.6f} x {valores_escalonados[i]:.6f})\n"
        secao3 += f"         s_({pos}) = {score_anterior:.6f} + {contribuicoes[i]:.6f}\n"
        secao3 += f"         s_({pos}) = {score_acumulado:.6f}\n"
        
        # Verifica se já atingiu o threshold
        if not is_rejected:
            if pred_class == 1 and score_acumulado >= t_plus:
                secao3 += f"\nCONDICAO ATINGIDA: s_({pos}) = {score_acumulado:.6f} >= t+ = {t_plus:.6f}\n"
                secao3 += f"  Explicacao minimal encontrada com {pos} features.\n"
                break
            elif pred_class == 0 and score_acumulado <= t_minus:
                secao3 += f"\nCONDICAO ATINGIDA: s_({pos}) = {score_acumulado:.6f} <= t- = {t_minus:.6f}\n"
                secao3 += f"  Explicacao minimal encontrada com {pos} features.\n"
                break
        
        secao3 += "\n"
    
    tamanho_explicacao = len(explicacao_minimal)
    
    return secao1 + secao2 + secao3 + gerar_secao4_estabilidade(
        feature_names, valores_escalonados, coefs, contribuicoes, explicacao_minimal,
        intercepto, score, t_plus, t_minus, pred_class, is_rejected, scaler_params=None
    ) + gerar_secao5_minimizacao(
        feature_names, contribuicoes, explicacao_minimal, intercepto, score,
        t_plus, t_minus, pred_class, is_rejected
    ) + gerar_secao6_interpretacao(
        dataset_nome, idx, nomes_classes, true_class, pred_class, score,
        t_plus, t_minus, feature_names, valores_originais, explicacao_minimal,
        contribuicoes, tamanho_explicacao, is_rejected, classe_texto
    )


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
            indices_ordenados, is_rejected, status_texto, classe_texto
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
        
        # Salva relatórios
        salvar_relatorios(dataset_nome, idx, relatorio, relatorio, dados_json)
        
    except Exception as e:
        print(f"ERRO ao processar instancia {idx}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_prova_detalhada():
    """Função principal para análise detalhada de instâncias."""
    print(format_line())
    print("🧩 PROVA MATEMÁTICA DETALHADA DE UMA INSTÂNCIA")
    
    try:
        # Carrega o cache cumulativo
        cache_completo = load_cache()
        
        # Lista datasets disponíveis
        list_available_datasets(cache_completo)
        
        # Seleção do dataset
        while True:
            try:
                escolha = input("Digite o número do dataset desejado: ")
                datasets = sorted(cache_completo.keys())
                dataset_nome = datasets[int(escolha) - 1]
                break
            except (ValueError, IndexError):
                print("❌ Escolha inválida. Tente novamente.")
        
        try:
            # Obtém dados do cache para o dataset selecionado
            cache_dataset = cache_completo[dataset_nome]
            
            # Verifica componentes necessários
            required_keys = ['X_test', 'y_test', 't_plus', 't_minus', 'nomes_classes', 'feature_names']
            missing_keys = [k for k in required_keys if k not in cache_dataset]
            if missing_keys:
                raise KeyError(f"Dados ausentes no cache: {', '.join(missing_keys)}")
            
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

            # Reconstrói X_train e y_train
            try:
                X_train = pd.DataFrame(cache_dataset['X_train'])
                X_train = X_train[feature_names]
                y_train = pd.Series(cache_dataset['y_train'])
            except Exception as e:
                print("Conteúdo bruto de X_train:", type(cache_dataset['X_train']))
                raise ValueError(f"Erro ao reconstruir dados de treino: {e}")
            
            # Carrega outros parâmetros
            t_plus = float(cache_dataset['t_plus'])
            t_minus = float(cache_dataset['t_minus'])
            nomes_classes = cache_dataset['nomes_classes']
            
            # Mostra informações sobre o dataset
            print(f"\n📊 Dataset: {dataset_nome}")
            print(f"Total de instâncias de teste: {len(X_test)}")
            print(f"Classes disponíveis: {', '.join([str(nc) for nc in nomes_classes])}")
            
            # Seleção da instância
            while True:
                try:
                    idx = int(input("\nDigite o número da instância para analisar (0 até {}): ".format(len(X_test)-1)))
                    if 0 <= idx < len(X_test):
                        break
                    print("❌ Índice fora do intervalo válido.")
                except ValueError:
                    print("❌ Por favor, digite um número válido.")
            
            # Processa a instância selecionada
            processar_instancia(dataset_nome, cache_dataset, X_test, y_test,
                              t_plus, t_minus, nomes_classes, feature_names, idx)
            
        except Exception as e:
            print(f"❌ Erro ao processar dataset {dataset_nome}: {e}")
            return
        
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        return

if __name__ == "__main__":
    try:
        run_prova_detalhada()
    except KeyboardInterrupt:
        print("\n\n❌ Operação cancelada pelo usuário.")
    except Exception as e:
        print(f"\n\n❌ Erro: {str(e)}")
    finally:
        print("\n" + format_line())