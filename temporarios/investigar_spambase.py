import json
from pathlib import Path

# Investiga spambase
peab_file = Path("json/peab/spambase.json")
pulp_file = Path("json/pulp/spambase.json")

with open(peab_file, 'r') as f:
    peab = json.load(f)
with open(pulp_file, 'r') as f:
    pulp = json.load(f)

print("=" * 80)
print("INVESTIGAÇÃO PROFUNDA: SPAMBASE")
print("=" * 80)

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

print("\n📈 DETALHES POR TIPO:")
print("\nPOSITIVAS:")
print(f"  PEAB: {peab['explanation_stats']['positive']['count']} instâncias, "
      f"média {peab['explanation_stats']['positive']['mean_length']:.2f} features")
print(f"  PULP: {pulp['estatisticas_por_tipo']['positiva']['instancias']} instâncias, "
      f"média {pulp['estatisticas_por_tipo']['positiva']['tamanho_medio']:.2f} features")
print(f"  Diferença: {pulp['estatisticas_por_tipo']['positiva']['tamanho_medio'] - peab['explanation_stats']['positive']['mean_length']:.2f}")

print("\nNEGATIVAS:")
print(f"  PEAB: {peab['explanation_stats']['negative']['count']} instâncias, "
      f"média {peab['explanation_stats']['negative']['mean_length']:.2f} features")
print(f"  PULP: {pulp['estatisticas_por_tipo']['negativa']['instancias']} instâncias, "
      f"média {pulp['estatisticas_por_tipo']['negativa']['tamanho_medio']:.2f} features")
print(f"  Diferença: {pulp['estatisticas_por_tipo']['negativa']['tamanho_medio'] - peab['explanation_stats']['negative']['mean_length']:.2f}")

# Calcula ponderado
peab_class = (peab['explanation_stats']['positive']['mean_length'] * peab['explanation_stats']['positive']['count'] +
              peab['explanation_stats']['negative']['mean_length'] * peab['explanation_stats']['negative']['count']) / \
             (peab['explanation_stats']['positive']['count'] + peab['explanation_stats']['negative']['count'])

pulp_class = (pulp['estatisticas_por_tipo']['positiva']['tamanho_medio'] * pulp['estatisticas_por_tipo']['positiva']['instancias'] +
              pulp['estatisticas_por_tipo']['negativa']['tamanho_medio'] * pulp['estatisticas_por_tipo']['negativa']['instancias']) / \
             (pulp['estatisticas_por_tipo']['positiva']['instancias'] + pulp['estatisticas_por_tipo']['negativa']['instancias'])

print("\n💡 ANÁLISE:")
print(f"Média ponderada CLASSIFICADAS:")
print(f"  PEAB: {peab_class:.2f}")
print(f"  PULP: {pulp_class:.2f}")
print(f"  Diferença: {pulp_class - peab_class:.2f} ({((pulp_class - peab_class)/peab_class*100):.1f}%)")

print("\n" + "=" * 80)
print("OBSERVAÇÃO:")
print("=" * 80)
print("Spambase tem APENAS 5 positivas e 435 negativas (muito desbalanceado).")
print("A diferença está principalmente nas POSITIVAS:")
peab_pos = peab['explanation_stats']['positive']['mean_length']
pulp_pos = pulp['estatisticas_por_tipo']['positiva']['tamanho_medio']
print(f"  Positivas: PULP {pulp_pos:.2f} vs PEAB {peab_pos:.2f} (+{pulp_pos-peab_pos:.2f} features)")
print("\nCom apenas 5 instâncias positivas, uma pequena variação individual")
print("causa grande impacto na média ponderada.")
print("\nIsso pode ser:")
print("1. Variação numérica tolerável no solver (PULP está usando EPSILON=1e-5)")
print("2. PEAB encontrando soluções levemente melhores que o ótimo por sorte")
print("3. Bug ainda presente no PULP para casos extremamente desbalanceados")
