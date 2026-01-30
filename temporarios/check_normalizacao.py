import json
from pathlib import Path

# Verifica rapidamente os resultados
datasets = ['breast_cancer', 'spambase']

print("=" * 80)
print("VERIFICAÇÃO RÁPIDA: Normalização reativada")
print("=" * 80)

for dataset in datasets:
    peab_file = Path(f"json/peab/{dataset}.json")
    pulp_file = Path(f"json/pulp/{dataset}.json")
    
    if not pulp_file.exists():
        print(f"\n❌ {dataset}: PULP JSON não encontrado")
        continue
    
    with open(peab_file, 'r') as f:
        peab = json.load(f)
    with open(pulp_file, 'r') as f:
        pulp = json.load(f)
    
    # Calcula médias
    peab_class = (peab['explanation_stats']['positive']['mean_length'] * peab['explanation_stats']['positive']['count'] +
                  peab['explanation_stats']['negative']['mean_length'] * peab['explanation_stats']['negative']['count']) / \
                 (peab['explanation_stats']['positive']['count'] + peab['explanation_stats']['negative']['count'])
    
    pulp_class = (pulp['estatisticas_por_tipo']['positiva']['tamanho_medio'] * pulp['estatisticas_por_tipo']['positiva']['instancias'] +
                  pulp['estatisticas_por_tipo']['negativa']['tamanho_medio'] * pulp['estatisticas_por_tipo']['negativa']['instancias']) / \
                 (pulp['estatisticas_por_tipo']['positiva']['instancias'] + pulp['estatisticas_por_tipo']['negativa']['instancias'])
    
    peab_rej = peab['explanation_stats']['rejected']['mean_length']
    pulp_rej = pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio']
    
    print(f"\n{dataset.upper()}:")
    print(f"  Classificadas: PEAB={peab_class:.2f}, PULP={pulp_class:.2f}, Diff={pulp_class-peab_class:+.2f}")
    print(f"  Rejeitadas:    PEAB={peab_rej:.2f}, PULP={pulp_rej:.2f}, Diff={pulp_rej-peab_rej:+.2f}")
    
    if abs(pulp_class - peab_class) < 0.5:
        print(f"  ✅ Classificadas OK")
    else:
        print(f"  ❌ Classificadas com diferença de {abs(pulp_class - peab_class):.2f}")

print("\n" + "=" * 80)
