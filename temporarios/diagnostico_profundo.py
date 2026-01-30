"""
DIAGNÓSTICO PROFUNDO: Por que PULP não gera positivas no Breast Cancer?
=========================================================================
"""
import json
import numpy as np
from pathlib import Path

print("="*80)
print("DIAGNÓSTICO: Por que PULP não gera POSITIVAS no Breast Cancer?")
print("="*80)

# Carregar JSONs
with open("json/peab/breast_cancer.json") as f:
    peab = json.load(f)

with open("json/pulp/breast_cancer.json") as f:
    pulp = json.load(f)

print("\n1️⃣ THRESHOLDS:")
peab_t_plus = peab['thresholds']['t_plus']
peab_t_minus = peab['thresholds']['t_minus']
pulp_t_plus = pulp['t_plus']
pulp_t_minus = pulp['t_minus']

print(f"   PEAB: t+ = {peab_t_plus:.8f}, t- = {peab_t_minus:.8f}")
print(f"   PULP: t+ = {pulp_t_plus:.8f}, t- = {pulp_t_minus:.8f}")
print(f"   Diferença t+: {abs(peab_t_plus - pulp_t_plus):.10f}")
print(f"   Diferença t-: {abs(peab_t_minus - pulp_t_minus):.10f}")

print("\n2️⃣ NORMALIZAÇÃO (max_abs):")
peab_max_abs = peab['model']['params']['norm_params']['max_abs']
pulp_params = pulp.get('params', {})
print(f"   PEAB max_abs: {peab_max_abs}")
print(f"   PULP params: {pulp_params}")

print("\n3️⃣ ANÁLISE DE SCORES:")
print("\n   Vamos verificar os scores de algumas instâncias...")

# Pegar explicações do PEAB que são POSITIVAS
peab_explicacoes = peab.get('explicacoes', [])
if not peab_explicacoes:
    print("   ⚠️  JSON do PEAB não tem lista 'explicacoes'")
    print("   Isso é normal se o JSON só tem estatísticas agregadas")
else:
    positivas_peab = [e for e in peab_explicacoes if e.get('tipo_predicao') == 'POSITIVA']
    print(f"   PEAB tem {len(positivas_peab)} explicações positivas")

print("\n4️⃣ DISTRIBUIÇÕES:")
print("\n   PEAB:")
print(f"      Positivas: {peab['explanation_stats']['positive']['count']}")
print(f"      Negativas: {peab['explanation_stats']['negative']['count']}")
print(f"      Rejeitadas: {peab['explanation_stats']['rejected']['count']}")

print("\n   PULP:")
pulp_stats = pulp.get('estatisticas_por_tipo', {})
print(f"      Positivas: {pulp_stats.get('positiva', {}).get('instancias', 0)}")
print(f"      Negativas: {pulp_stats.get('negativa', {}).get('instancias', 0)}")
print(f"      Rejeitadas: {pulp_stats.get('rejeitada', {}).get('instancias', 0)}")

print("\n5️⃣ PERFORMANCE DO MODELO:")
print("\n   PEAB:")
peab_perf = peab['performance']
print(f"      Acurácia sem rejeição: {peab_perf['accuracy_without_rejection']:.2f}%")
print(f"      Acurácia com rejeição: {peab_perf['accuracy_with_rejection']:.2f}%")
print(f"      Taxa de rejeição: {peab_perf['rejection_rate']:.2f}%")

print("\n   PULP:")
pulp_metr = pulp['metricas_modelo']
print(f"      Acurácia sem rejeição: {pulp_metr['acuracia_sem_rejeicao']*100:.2f}%")
print(f"      Acurácia com rejeição: {pulp_metr['acuracia_com_rejeicao']*100:.2f}%")
print(f"      Taxa de rejeição: {pulp_metr['taxa_rejeicao']*100:.2f}%")

print("\n" + "="*80)
print("🔍 HIPÓTESES:")
print("="*80)

# Comparar taxas de rejeição
peab_rej_rate = peab_perf['rejection_rate']
pulp_rej_rate = pulp_metr['taxa_rejeicao'] * 100

print(f"\n1. Taxa de rejeição:")
print(f"   PEAB: {peab_rej_rate:.2f}%")
print(f"   PULP: {pulp_rej_rate:.2f}%")

if abs(pulp_rej_rate - peab_rej_rate) > 50:
    print("   ❌ MUITO DIFERENTE! PULP está rejeitando MUITO MAIS!")
    print("   → Possível causa: Thresholds diferentes ou problema na classificação")
elif pulp_rej_rate > peab_rej_rate + 10:
    print("   ⚠️  PULP rejeita mais que PEAB")
    print("   → Isso pode explicar por que não tem positivas")

# Verificar se o threshold está muito alto
print(f"\n2. Threshold t+ muito alto?")
print(f"   t+ = {pulp_t_plus:.6f}")
if pulp_t_plus > 0.95:
    print("   ⚠️  t+ muito próximo de 1.0!")
    print("   → Isso dificulta ter instâncias positivas")
    print("   → A maioria dos scores pode estar < t+")

print("\n" + "="*80)
print("💡 POSSÍVEIS SOLUÇÕES (SEM MUDAR TUDO):")
print("="*80)
print("""
HIPÓTESE 1: Thresholds levemente diferentes causam grande impacto
→ SOLUÇÃO: Garantir que PULP use EXATAMENTE os mesmos thresholds do PEAB
   (copiar direto do JSON do PEAB, não recalcular)

HIPÓTESE 2: A lógica de classificação no PULP está errada
→ SOLUÇÃO: Verificar a linha onde determina o tipo de predição:
   - Se score >= t_plus: POSITIVA
   - Conferir se está usando score normalizado corretamente

HIPÓTESE 3: Problema na normalização do score
→ SOLUÇÃO: Verificar se max_abs está sendo usado corretamente
   - score_norm = score_raw / max_abs

PRÓXIMO PASSO: Vou criar um script para verificar qual dessas é o problema!
""")
print("="*80)
