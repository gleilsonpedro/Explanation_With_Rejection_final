import json
from pathlib import Path

# Verifica o resultado do Breast Cancer após o conserto
peab_file = Path("json/peab/breast_cancer.json")
pulp_file = Path("json/pulp/breast_cancer.json")

with open(peab_file, 'r') as f:
    peab = json.load(f)
with open(pulp_file, 'r') as f:
    pulp = json.load(f)

print("=" * 80)
print("VERIFICAÇÃO: Breast Cancer após correção do bug de normalização")
print("=" * 80)

# Distribuição
print("\nDISTRIBUIÇÃO:")
print(f"PEAB  - Pos: {peab['explanation_stats']['positive']['count']}, Neg: {peab['explanation_stats']['negative']['count']}, Rej: {peab['explanation_stats']['rejected']['count']}")
print(f"PULP  - Pos: {pulp['estatisticas_por_tipo']['positiva']['instancias']}, Neg: {pulp['estatisticas_por_tipo']['negativa']['instancias']}, Rej: {pulp['estatisticas_por_tipo']['rejeitada']['instancias']}")

# Thresholds
print("\nTHRESHOLDS:")
print(f"PEAB  - t+: {peab['thresholds']['t_plus']:.6f}, t-: {peab['thresholds']['t_minus']:.6f}")
print(f"PULP  - t+: {pulp['t_plus']:.6f}, t-: {pulp['t_minus']:.6f}")

# Tamanho médio das explicações
print("\nTAMANHO MÉDIO DAS EXPLICAÇÕES:")
peab_class = (peab['explanation_stats']['positive']['mean_length'] * peab['explanation_stats']['positive']['count'] +
              peab['explanation_stats']['negative']['mean_length'] * peab['explanation_stats']['negative']['count']) / \
             (peab['explanation_stats']['positive']['count'] + peab['explanation_stats']['negative']['count'])
pulp_class = (pulp['estatisticas_por_tipo']['positiva']['tamanho_medio'] * pulp['estatisticas_por_tipo']['positiva']['instancias'] +
              pulp['estatisticas_por_tipo']['negativa']['tamanho_medio'] * pulp['estatisticas_por_tipo']['negativa']['instancias']) / \
             (pulp['estatisticas_por_tipo']['positiva']['instancias'] + pulp['estatisticas_por_tipo']['negativa']['instancias'])
print(f"PEAB  - Classificadas: {peab_class:.2f}, Rejeitadas: {peab['explanation_stats']['rejected']['mean_length']:.2f}")
print(f"PULP  - Classificadas: {pulp_class:.2f}, Rejeitadas: {pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio']:.2f}")

# Comparação crítica
print("\n" + "=" * 80)
if pulp_class <= peab_class:
    print("✅ PULP agora está MENOR ou IGUAL ao PEAB nas classificadas!")
    print(f"   Redução de {peab_class:.2f} → {pulp_class:.2f}")
else:
    print("❌ PULP ainda está MAIOR que PEAB nas classificadas")
    print(f"   {pulp_class:.2f} > {peab_class:.2f}")
print("=" * 80)
