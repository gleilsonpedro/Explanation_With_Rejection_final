#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Investigação: Por que explicações de rejeitadas têm baixa fidelidade?
"""

import numpy as np
import pandas as pd
import json
from utils.shared_training import get_shared_pipeline

# Carregar dados
dataset = 'pima_indians_diabetes'
pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset)

# Carregar explicações do PEAB
with open(f'json/peab/{dataset}.json', 'r') as f:
    resultados = json.load(f)

explicacoes = resultados.get('explicacoes', resultados.get('per_instance', []))
feature_names = meta['feature_names']

print("="*80)
print("INVESTIGAÇÃO: Fidelidade de Predições Rejeitadas")
print("="*80)
print(f"\nDataset: {dataset}")
print(f"Zona de rejeição: [{t_minus:.4f}, {t_plus:.4f}]")
print(f"Largura da zona: {t_plus - t_minus:.4f}")

# Filtrar apenas rejeitadas
rejeitadas = [exp for exp in explicacoes if exp.get('rejected', exp.get('rejeitada', False))]
print(f"\nTotal de instâncias rejeitadas: {len(rejeitadas)}")

# Analisar algumas instâncias
print("\n" + "="*80)
print("ANÁLISE DETALHADA DE 5 INSTÂNCIAS REJEITADAS")
print("="*80)

for i, exp in enumerate(rejeitadas[:5]):
    idx = int(exp.get('id', exp.get('indice', 0)))
    explicacao = exp.get('explanation', exp.get('explicacao', []))
    
    # Pegar instância original
    try:
        instancia = X_test.loc[idx].values
    except:
        instancia = X_test.iloc[idx].values
    score_original = pipeline.decision_function([instancia])[0]
    
    print(f"\n{'─'*80}")
    print(f"INSTÂNCIA {i+1} - Índice {idx}")
    print(f"{'─'*80}")
    print(f"Score original: {score_original:.4f}")
    print(f"Explicação ({len(explicacao)} features): {explicacao[:3]}...")
    
    # Features fixadas e perturbadas
    features_fixas_idx = [feature_names.index(feat) for feat in explicacao if feat in feature_names]
    features_perturbar_idx = [i for i in range(len(feature_names)) if i not in features_fixas_idx]
    
    print(f"Features fixadas: {len(features_fixas_idx)}")
    print(f"Features perturbadas: {len(features_perturbar_idx)}")
    
    # Fazer algumas perturbações
    n_perturb = 100
    perturbacoes = np.tile(instancia, (n_perturb, 1))
    
    for feat_idx in features_perturbar_idx:
        feat_min = X_train.iloc[:, feat_idx].min()
        feat_max = X_train.iloc[:, feat_idx].max()
        perturbacoes[:, feat_idx] = np.random.uniform(feat_min, feat_max, n_perturb)
    
    # Calcular scores
    scores = pipeline.decision_function(perturbacoes)
    
    # Analisar distribuição
    na_zona = np.sum((scores >= t_minus) & (scores <= t_plus))
    abaixo = np.sum(scores < t_minus)
    acima = np.sum(scores > t_plus)
    
    print(f"\nDISTRIBUIÇÃO DE {n_perturb} PERTURBAÇÕES:")
    print(f"  Na zona de rejeição: {na_zona} ({na_zona/n_perturb*100:.1f}%)")
    print(f"  Abaixo da zona:      {abaixo} ({abaixo/n_perturb*100:.1f}%)")
    print(f"  Acima da zona:       {acima} ({acima/n_perturb*100:.1f}%)")
    
    # Estatísticas dos scores
    print(f"\nESTATÍSTICAS DOS SCORES:")
    print(f"  Média:    {scores.mean():.4f}")
    print(f"  Mediana:  {np.median(scores):.4f}")
    print(f"  Min:      {scores.min():.4f}")
    print(f"  Max:      {scores.max():.4f}")
    print(f"  Std:      {scores.std():.4f}")
    
    # Verificar se há drift sistemático
    if scores.mean() < t_minus:
        print(f"\n⚠️  PROBLEMA: Perturbações puxam sistematicamente para BAIXO da zona")
    elif scores.mean() > t_plus:
        print(f"\n⚠️  PROBLEMA: Perturbações puxam sistematicamente para CIMA da zona")
    else:
        print(f"\n✓ Perturbações ficam próximas da zona (média dentro)")

print("\n" + "="*80)
print("ANÁLISE GERAL")
print("="*80)

# Estatísticas gerais
scores_originais = []
tamanhos_explicacao = []

for exp in rejeitadas:
    idx = int(exp.get('id', exp.get('indice', 0)))
    try:
        instancia = X_test.loc[idx].values
    except:
        instancia = X_test.iloc[idx].values
    score = pipeline.decision_function([instancia])[0]
    scores_originais.append(score)
    explicacao = exp.get('explanation', exp.get('explicacao', []))
    tamanhos_explicacao.append(len(explicacao))

print(f"\nScores originais das rejeitadas:")
print(f"  Média: {np.mean(scores_originais):.4f}")
print(f"  Std:   {np.std(scores_originais):.4f}")
print(f"  Min:   {np.min(scores_originais):.4f}")
print(f"  Max:   {np.max(scores_originais):.4f}")

print(f"\nTamanho das explicações:")
print(f"  Média: {np.mean(tamanhos_explicacao):.2f}")
print(f"  Min:   {np.min(tamanhos_explicacao)}")
print(f"  Max:   {np.max(tamanhos_explicacao)}")

print("\n" + "="*80)
print("HIPÓTESES POSSÍVEIS:")
print("="*80)
print("""
1. EXPLICAÇÕES INCOMPLETAS
   → PEAB não identifica todas as features necessárias para manter rejeição
   → Precisa ajustar algoritmo PEAB para rejeitadas

2. PERTURBAÇÃO MUITO DRÁSTICA
   → Perturbar 40-50% das features é muito agressivo
   → Zona de rejeição é estreita, perturbações grandes saem facilmente

3. ZONA DE REJEIÇÃO MUITO PEQUENA
   → Zona [{:.4f}, {:.4f}] pode ser estreita demais
   → Instâncias naturalmente "escapam" com pequenas mudanças

4. INSTÂNCIAS GENUINAMENTE AMBÍGUAS
   → Rejeitadas SÃO difíceis por natureza
   → Talvez 23% seja o máximo possível para essas instâncias
""".format(t_minus, t_plus))

print("="*80)
