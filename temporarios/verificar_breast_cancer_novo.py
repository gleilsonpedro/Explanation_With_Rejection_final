"""
Verificar se o JSON do breast_cancer gerado tem POSITIVAS
"""
import json

print("Verificando JSON do PULP recém-gerado:")
print("="*70)

with open("json/pulp/breast_cancer.json") as f:
    pulp = json.load(f)

print(f"\nThresholds usados pelo PULP:")
print(f"  t+ = {pulp['t_plus']:.8f}")
print(f"  t- = {pulp['t_minus']:.8f}")

print(f"\nDistribuição:")
stats = pulp.get('estatisticas_por_tipo', {})
print(f"  Positivas: {stats.get('positiva', {}).get('instancias', 0)}")
print(f"  Negativas: {stats.get('negativa', {}).get('instancias', 0)}")
print(f"  Rejeitadas: {stats.get('rejeitada', {}).get('instancias', 0)}")

print("\n" + "="*70)
print("Comparando com PEAB:")
print("="*70)

with open("json/peab/breast_cancer.json") as f:
    peab = json.load(f)

print(f"\nThresholds do PEAB:")
print(f"  t+ = {peab['thresholds']['t_plus']:.8f}")
print(f"  t- = {peab['thresholds']['t_minus']:.8f}")

peab_stats = peab['explanation_stats']
print(f"\nDistribuição:")
print(f"  Positivas: {peab_stats['positive']['count']}")
print(f"  Negativas: {peab_stats['negative']['count']}")
print(f"  Rejeitadas: {peab_stats['rejected']['count']}")

print("\n" + "="*70)
print("ANÁLISE:")
print("="*70)

if abs(pulp['t_plus'] - peab['thresholds']['t_plus']) < 0.000001:
    print("✅ Thresholds SÃO IDÊNTICOS!")
    print("\n⚠️  MAS ainda assim não tem positivas no PULP!")
    print("\nPOSSÍVEIS CAUSAS:")
    print("1. Bug na lógica de classificação do PULP")
    print("2. Problema na normalização do score")
    print("3. max_abs diferente entre PEAB e PULP")
else:
    print(f"❌ Thresholds DIFERENTES!")
    print(f"   Diferença: {abs(pulp['t_plus'] - peab['thresholds']['t_plus']):.10f}")
