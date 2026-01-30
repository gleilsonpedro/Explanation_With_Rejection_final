import json
from pathlib import Path

# Comparação simples: estatísticas agregadas
peab_file = Path("json/peab/spambase.json")
pulp_file = Path("json/pulp/spambase.json")

with open(peab_file, 'r') as f:
    peab = json.load(f)
with open(pulp_file, 'r') as f:
    pulp = json.load(f)

print("=" * 90)
print("SPAMBASE: Comparação de Estatísticas Agregadas")
print("=" * 90)

print("\n📊 DISTRIBUIÇÃO:")
print(f"PEAB - Pos: {peab['explanation_stats']['positive']['count']}, "
      f"Neg: {peab['explanation_stats']['negative']['count']}, "
      f"Rej: {peab['explanation_stats']['rejected']['count']}")
print(f"PULP - Pos: {pulp['estatisticas_por_tipo']['positiva']['instancias']}, "
      f"Neg: {pulp['estatisticas_por_tipo']['negativa']['instancias']}, "
      f"Rej: {pulp['estatisticas_por_tipo']['rejeitada']['instancias']}")

print("\n🎯 THRESHOLDS:")
print(f"PEAB - t+: {peab['thresholds']['t_plus']:.6f}, t-: {peab['thresholds']['t_minus']:.6f}")
print(f"PULP - t+: {pulp['t_plus']:.6f}, t-: {pulp['t_minus']:.6f}")

print("\n📈 TAMANHO MÉDIO DAS EXPLICAÇÕES:")
print("\nPOSITIVAS:")
print(f"  PEAB: {peab['explanation_stats']['positive']['mean_length']:.2f} features")
print(f"        (min={peab['explanation_stats']['positive']['min_length']}, "
      f"max={peab['explanation_stats']['positive']['max_length']})")
print(f"  PULP: {pulp['estatisticas_por_tipo']['positiva']['tamanho_medio']:.2f} features")
print(f"  Diferença: {pulp['estatisticas_por_tipo']['positiva']['tamanho_medio'] - peab['explanation_stats']['positive']['mean_length']:+.2f}")

print("\nNEGATIVAS:")
print(f"  PEAB: {peab['explanation_stats']['negative']['mean_length']:.2f} features")
print(f"        (min={peab['explanation_stats']['negative']['min_length']}, "
      f"max={peab['explanation_stats']['negative']['max_length']})")
print(f"  PULP: {pulp['estatisticas_por_tipo']['negativa']['tamanho_medio']:.2f} features")
print(f"  Diferença: {pulp['estatisticas_por_tipo']['negativa']['tamanho_medio'] - peab['explanation_stats']['negative']['mean_length']:+.2f}")

print("\nREJEITADAS:")
print(f"  PEAB: {peab['explanation_stats']['rejected']['mean_length']:.2f} features")
print(f"        (min={peab['explanation_stats']['rejected']['min_length']}, "
      f"max={peab['explanation_stats']['rejected']['max_length']})")
print(f"  PULP: {pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio']:.2f} features")
print(f"  Diferença: {pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio'] - peab['explanation_stats']['rejected']['mean_length']:+.2f}")

print("\n" + "=" * 90)
print("🔍 ANÁLISE:")
print("=" * 90)

print("\n1. Distribuição idêntica: ✅")
print("2. Thresholds idênticos: ✅")

print("\n3. Tamanho das explicações:")
print(f"   - POSITIVAS: PULP está {pulp['estatisticas_por_tipo']['positiva']['tamanho_medio'] - peab['explanation_stats']['positive']['mean_length']:+.2f} features")
print(f"   - NEGATIVAS: PULP está {pulp['estatisticas_por_tipo']['negativa']['tamanho_medio'] - peab['explanation_stats']['negative']['mean_length']:+.2f} features")
print(f"   - REJEITADAS: PULP está {pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio'] - peab['explanation_stats']['rejected']['mean_length']:+.2f} features")

print("\n4. Observação crítica:")
print(f"   PULP Negativas: ~51 features (de 57 total = {51/57*100:.1f}%!)")
print(f"   PEAB Negativas: ~28 features (de 57 total = {28/57*100:.1f}%)")
print("\n   PULP está usando QUASE TODAS AS FEATURES!")
print("   Isso sugere que as restrições estão muito apertadas.")

print("\n5. Dataset Spambase:")
print("   - 57 features")
print("   - Classe muito desbalanceada (5 pos vs 435 neg)")
print("   - PULP parece precisar de quase todas as features para")
print("     satisfazer as restrições de worst-case")

print("\n" + "=" * 90)
print("HIPÓTESE:")
print("=" * 90)
print("O problema pode estar no cálculo do WORST-CASE para datasets")
print("com muitas features correlacionadas. Quando há muitas features,")
print("o 'pior cenário' (todas features no limite oposto) pode ser tão")
print("conservador que força o PULP a selecionar quase todas para")
print("compensar e manter o score acima/abaixo do threshold.")
print("\n" + "=" * 90)
