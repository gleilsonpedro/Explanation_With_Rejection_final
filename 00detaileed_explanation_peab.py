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
CACHE_FILE = Path("auxiliary_files/cache_cumulativo.pkl")

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
        stats = cache[dataset].get('stats', {})
        total = len(cache[dataset]['X_test'])
        print(f"{i}. {dataset:<25} ({total} instâncias)")
    print()

def format_line():
    return "=" * 72

def salvar_relatorios(dataset_nome, idx, explicacao_tecnica, explicacao_academica, dados_json):
    """Salva relatórios técnico, acadêmico e JSON para uma instância."""
    pasta = os.path.join("explicacoes_detalhadas", dataset_nome, f"instancia_{idx}")
    os.makedirs(pasta, exist_ok=True)
    
    with open(os.path.join(pasta, "explicacao_tecnica.txt"), "w", encoding="utf-8") as f:
        f.write(explicacao_tecnica)
    with open(os.path.join(pasta, "explicacao_academica.txt"), "w", encoding="utf-8") as f:
        f.write(explicacao_academica)
    with open(os.path.join(pasta, "explicacao_dados.json"), "w", encoding="utf-8") as f:
        json.dump(dados_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Relatórios salvos em: {pasta}\n")

def processar_instancia(dataset_nome, cache_dataset, pipeline, X_test, y_test, t_plus, t_minus, nomes_classes, feature_names, idx):
    """Processa uma instância específica e gera explicações."""
    try:
        # Prepara a instância para análise
        instance = X_test.iloc[[idx]]
        true_class = y_test.iloc[idx]
        
        # Obtém a predição e o score
        pred_class = pipeline.predict(instance)[0]
        score = pipeline.decision_function(instance)[0]
        
        # Determina o status da instância (aceita/rejeitada)
        rejected = (score > t_minus and score < t_plus)
        status = "REJEITADA" if rejected else "ACEITA"
        
        # Extrai componentes do pipeline e verifica se são válidos
        try:
            scaler = pipeline.named_steps['scaler']
            classifier = pipeline.named_steps['modelo']
        except KeyError as e:
            raise ValueError(f"Erro ao acessar componentes do pipeline: {e}")
        
        # Coleta dados para explicação com verificações
        scaled_instance = scaler.transform(instance)
        coefs = classifier.coef_[0]
        
        # Calcula contribuições
        contributions = coefs * scaled_instance[0]
        sorted_idx = np.argsort(np.abs(contributions))[::-1]
        
        # Gera explicação técnica
        explicacao_tecnica = f"""
=== ANÁLISE TÉCNICA DA INSTÂNCIA {idx} ===
Dataset: {dataset_nome}
Status: {status}
Classe Real: {nomes_classes[true_class]}
{'Classe Predita: ' + str(nomes_classes[pred_class]) if not rejected else 'Predição: REJEITADA'}
Score: {score:.4f}
Thresholds: t+ = {t_plus:.4f}, t- = {t_minus:.4f}

Top Contribuições:
"""
        for i in sorted_idx[:5]:
            explicacao_tecnica += f"{feature_names[i]}: {contributions[i]:.4f}\n"
        
        # Gera explicação acadêmica
        explicacao_academica = f"""
Análise Matemática Detalhada - Instância {idx}
================================================
1. Informações Básicas:
   - Dataset: {dataset_nome}
   - Classe Verdadeira: {nomes_classes[true_class]}
   - Score de Decisão: {score:.4f}
   
2. Análise de Rejeição:
   - Threshold Superior (t+): {t_plus:.4f}
   - Threshold Inferior (t-): {t_minus:.4f}
   - Decisão: {status}
   
3. Decomposição das Contribuições:
"""
        for i in sorted_idx[:5]:
            explicacao_academica += f"   {feature_names[i]}: {contributions[i]:.4f}\n"
        
        # Prepara dados para JSON
        dados_json = {
            "dataset": dataset_nome,
            "instancia": int(idx),
            "classe_real": str(nomes_classes[true_class]),
            "classe_predita": str(nomes_classes[pred_class]) if not rejected else "REJEITADA",
            "score": float(score),
            "thresholds": {
                "t_plus": float(t_plus),
                "t_minus": float(t_minus)
            },
            "status": status,
            "contribuicoes": {
                str(feature_names[i]): float(contributions[i])
                for i in sorted_idx[:5]
            }
        }
        
        # Salva os relatórios
        salvar_relatorios(dataset_nome, idx, explicacao_tecnica, explicacao_academica, dados_json)
        
    except Exception as e:
        print(f"❌ Erro ao processar instância {idx}: {e}")
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
            required_keys = ['pipeline_modelo', 'X_test', 'y_test', 't_plus', 't_minus', 'nomes_classes', 'feature_names']
            missing_keys = [k for k in required_keys if k not in cache_dataset]
            if missing_keys:
                raise KeyError(f"Dados ausentes no cache: {', '.join(missing_keys)}")
            
            # Carrega o pipeline
            pipeline = cache_dataset['pipeline_modelo']
            
            # Reconstrói os DataFrames e Series
            feature_names = cache_dataset['feature_names']
            
            # Reconstrói X_test e y_test
            try:
                X_test = pd.DataFrame.from_dict(cache_dataset['X_test'])
                X_test.columns = feature_names
                y_test = pd.Series(cache_dataset['y_test'])
            except Exception as e:
                raise ValueError(f"Erro ao reconstruir dados de teste: {e}")
            
            # Reconstrói X_train e y_train
            try:
                X_train = pd.DataFrame.from_dict(cache_dataset['X_train'])
                X_train.columns = feature_names
                y_train = pd.Series(cache_dataset['y_train'])
            except Exception as e:
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
            processar_instancia(dataset_nome, cache_dataset, pipeline, X_test, y_test,
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