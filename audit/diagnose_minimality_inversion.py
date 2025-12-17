#!/usr/bin/env python3
"""
Diagnóstico do padrão invertido de minimalidade entre positivas e negativas.

Este script investiga por que:
- Banknote: Positivas 94.05%, Negativas 4.94%  
- Vertebral: Positivas 0.98%, Negativas 92.75%

Hipótese: A perturbação uniforme está gerando scores que caem 
predominantemente de um lado da fronteira de decisão (score=0).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from utils.shared_training import get_shared_pipeline


def diagnose_dataset(dataset_name: str, n_samples: int = 100):
    """
    Diagnóstico de um dataset para entender o padrão de perturbações.
    """
    print(f"\n{'='*70}")
    print(f"  DIAGNÓSTICO: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Carregar pipeline
    pipeline_data = get_shared_pipeline(dataset_name)
    if pipeline_data is None:
        print(f"❌ Erro ao carregar pipeline para {dataset_name}")
        return
    
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = pipeline_data
    feature_names = meta['feature_names']
    
    print(f"📊 Features: {len(feature_names)}")
    print(f"📊 Amostras teste: {len(X_test)}")
    print(f"📊 t_plus: {t_plus:.4f}, t_minus: {t_minus:.4f}")
    
    # Obter scores das perturbações uniformes
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Gerar n_samples perturbações uniformes
    n_features = X_train_arr.shape[1]
    perturbacoes = np.zeros((n_samples, n_features))
    
    for feat_idx in range(n_features):
        feat_min = X_train_arr[:, feat_idx].min()
        feat_max = X_train_arr[:, feat_idx].max()
        perturbacoes[:, feat_idx] = np.random.uniform(feat_min, feat_max, n_samples)
    
    # Obter predições e scores
    predicoes = pipeline.predict(perturbacoes)
    scores = pipeline.decision_function(perturbacoes)
    
    # Analisar distribuição
    print(f"\n📈 DISTRIBUIÇÃO DOS SCORES (perturbações uniformes)")
    print(f"   Min: {scores.min():.4f}")
    print(f"   Max: {scores.max():.4f}")
    print(f"   Mean: {scores.mean():.4f}")
    print(f"   Median: {np.median(scores):.4f}")
    print(f"   Std: {scores.std():.4f}")
    
    # Contar predições
    n_positivas = np.sum(predicoes == 1)
    n_negativas = np.sum(predicoes == 0)
    n_zona_rejeicao = np.sum((scores >= t_minus) & (scores <= t_plus))
    
    print(f"\n📊 DISTRIBUIÇÃO DAS PREDIÇÕES:")
    print(f"   Positivas (score > 0): {n_positivas} ({100*n_positivas/n_samples:.1f}%)")
    print(f"   Negativas (score < 0): {n_negativas} ({100*n_negativas/n_samples:.1f}%)")
    print(f"   Zona rejeição [{t_minus:.4f}, {t_plus:.4f}]: {n_zona_rejeicao} ({100*n_zona_rejeicao/n_samples:.1f}%)")
    
    # Score bias - percentis
    print(f"\n📊 PERCENTIS DOS SCORES:")
    for p in [10, 25, 50, 75, 90]:
        print(f"   P{p}: {np.percentile(scores, p):.4f}")
    
    # Verificar se há bias
    bias_ratio = n_positivas / max(n_negativas, 1)
    print(f"\n⚠️  RATIO positivas/negativas: {bias_ratio:.2f}")
    if bias_ratio > 2:
        print(f"   → BIAS FORTE para POSITIVAS - perturbações uniformes tendem a score > 0")
    elif bias_ratio < 0.5:
        print(f"   → BIAS FORTE para NEGATIVAS - perturbações uniformes tendem a score < 0")
    else:
        print(f"   → Distribuição razoavelmente balanceada")
    
    # Análise por feature: quais features "empurram" para qual lado
    print(f"\n📊 ANÁLISE DE COEFICIENTES DO MODELO:")
    try:
        if hasattr(pipeline, 'named_steps'):
            logreg = pipeline.named_steps.get('classifier') or pipeline.named_steps.get('logisticregression')
        else:
            logreg = pipeline
        
        if logreg and hasattr(logreg, 'coef_'):
            coefs = logreg.coef_[0]
            intercept = logreg.intercept_[0]
            
            print(f"   Intercept: {intercept:.4f}")
            print(f"   Soma coefs: {coefs.sum():.4f}")
            print(f"   Coefs positivos: {np.sum(coefs > 0)}")
            print(f"   Coefs negativos: {np.sum(coefs < 0)}")
            
            # Top 5 features que empurram para cada lado
            sorted_idx = np.argsort(coefs)
            print(f"\n   Top 5 → NEGATIVAS (coefs mais negativos):")
            for i in sorted_idx[:5]:
                print(f"      {feature_names[i]}: {coefs[i]:.4f}")
            
            print(f"\n   Top 5 → POSITIVAS (coefs mais positivos):")
            for i in sorted_idx[-5:][::-1]:
                print(f"      {feature_names[i]}: {coefs[i]:.4f}")
    except Exception as e:
        print(f"   Erro ao analisar coeficientes: {e}")
    
    # CONCLUSÃO
    print(f"\n{'='*70}")
    print(f"  DIAGNÓSTICO PARA MINIMALIDADE:")
    print(f"{'='*70}")
    print(f"""
Para teste de minimalidade:
- Removemos 1 feature da explicação e fixamos as outras
- Perturbamos todas as features NÃO explicativas uniformemente
- Se >95% das perturbações mantêm a predição → feature redundante

PROBLEMA IDENTIFICADO:
Se perturbações uniformes geram {100*n_positivas/n_samples:.0f}% positivas / {100*n_negativas/n_samples:.0f}% negativas:

→ Para instâncias POSITIVAS: 
   Mesmo removendo feature, ~{100*n_positivas/n_samples:.0f}% das perturbações serão positivas
   Se {100*n_positivas/n_samples:.0f}% > 95% → feature parece "redundante" (CORRETO ou VIÉS?)
   
→ Para instâncias NEGATIVAS:
   Mesmo removendo feature, ~{100*n_negativas/n_samples:.0f}% das perturbações serão negativas
   Se {100*n_negativas/n_samples:.0f}% < 95% → feature parece "necessária" (CORRETO ou VIÉS?)
""")
    
    return {
        'dataset': dataset_name,
        'bias_ratio': bias_ratio,
        'pct_positivas': 100*n_positivas/n_samples,
        'pct_negativas': 100*n_negativas/n_samples,
        'mean_score': scores.mean(),
        't_plus': t_plus,
        't_minus': t_minus
    }


def main():
    datasets = ['pima_indians_diabetes', 'breast_cancer', 'banknote', 'vertebral_column', 'sonar']
    
    print("="*70)
    print("  DIAGNÓSTICO DE PADRÃO INVERTIDO DE MINIMALIDADE")
    print("="*70)
    
    np.random.seed(42)
    resultados = []
    
    for dataset in datasets:
        try:
            resultado = diagnose_dataset(dataset, n_samples=1000)
            if resultado:
                resultados.append(resultado)
        except Exception as e:
            print(f"❌ Erro em {dataset}: {e}")
    
    # Resumo
    print("\n" + "="*70)
    print("  RESUMO DO DIAGNÓSTICO")
    print("="*70)
    print(f"\n{'Dataset':<25} {'Bias Ratio':>12} {'%Positivas':>12} {'%Negativas':>12} {'Score Médio':>12}")
    print("-"*70)
    for r in resultados:
        print(f"{r['dataset']:<25} {r['bias_ratio']:>12.2f} {r['pct_positivas']:>12.1f} {r['pct_negativas']:>12.1f} {r['mean_score']:>12.4f}")
    
    print(f"""
INTERPRETAÇÃO:
- Se Bias Ratio > 2: O espaço de perturbação favorece POSITIVAS
  → Positivas terão alta minimalidade (falsamente redundante)
  → Negativas terão baixa minimalidade (corretamente necessário)
  
- Se Bias Ratio < 0.5: O espaço de perturbação favorece NEGATIVAS
  → Negativas terão alta minimalidade (falsamente redundante)
  → Positivas terão baixa minimalidade (corretamente necessário)

SOLUÇÃO PROPOSTA:
1. Normalizar o teste por baseline: Subtrair a % base do espaço
2. Usar perturbações estratificadas: 50% empurrando para cada lado
3. Comparar com baseline de perturbação total (sem nenhuma feature fixa)
""")


if __name__ == "__main__":
    main()
