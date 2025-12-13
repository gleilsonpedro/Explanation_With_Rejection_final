#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verificar se instâncias rejeitadas estão realmente na zona (usando normalização)
"""

import json
import numpy as np
from utils.shared_training import get_shared_pipeline

# Carregar dados
dataset = 'pima_indians_diabetes'
pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset)

# Carregar JSON
with open(f'json/peab/{dataset}.json', 'r') as f:
    data = json.load(f)

# Pegar parâmetros de normalização
norm_info = data['thresholds']['normalization']
max_abs = norm_info['max_abs']

print("="*80)
print("VERIFICAÇÃO DE NORMALIZAÇÃO")
print("="*80)
print(f"\nThresholds salvos no JSON:")
print(f"  t_minus: {t_minus:.4f}")
print(f"  t_plus:  {t_plus:.4f}")
print(f"  max_abs: {max_abs:.4f}")

# Calcular scores reais
decision_scores = pipeline.decision_function(X_test)
print(f"\nScores do modelo:")
print(f"  Min: {decision_scores.min():.4f}")
print(f"  Max: {decision_scores.max():.4f}")
print(f"  Mean: {decision_scores.mean():.4f}")

# Normalizar como o PEAB faz
mean_score = np.mean(decision_scores)
std_score = np.std(decision_scores)
scores_z = (decision_scores - mean_score) / std_score
scores_norm = scores_z / max_abs

print(f"\nApós normalização:")
print(f"  Min: {scores_norm.min():.4f}")
print(f"  Max: {scores_norm.max():.4f}")
print(f"  Mean: {scores_norm.mean():.4f}")

# Verificar instâncias rejeitadas
explicacoes = data['per_instance']
rejeitadas = [exp for exp in explicacoes if exp.get('rejected', False)]

print(f"\n{'='*80}")
print(f"VERIFICAÇÃO DE {len(rejeitadas)} INSTÂNCIAS REJEITADAS")
print(f"{'='*80}")

erros = 0
for exp in rejeitadas:
    idx = int(exp['id'])
    
    # Usar o score normalizado do JSON (campo correto)
    score_norm_salvo = exp.get('decision_score_normalized', exp['decision_score'])
    
    # Verificar se está na zona
    na_zona = t_minus <= score_norm_salvo <= t_plus
    
    if not na_zona:
        erros += 1
        print(f"\n❌ Instância {idx}:")
        print(f"   Score normalizado salvo: {score_norm_salvo:.4f}")
        print(f"   Na zona? {na_zona} (zona: [{t_minus:.4f}, {t_plus:.4f}])")

if erros == 0:
    print("\n✓ Todas as instâncias rejeitadas estão corretamente na zona!")
else:
    print(f"\n❌ {erros}/{len(rejeitadas)} instâncias NÃO estão na zona de rejeição!")
    print("   PROBLEMA: Há inconsistência entre scores salvos e critério de rejeição")

print(f"\n{'='*80}")
