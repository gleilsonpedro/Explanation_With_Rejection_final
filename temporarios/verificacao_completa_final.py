import json
from pathlib import Path

datasets = ['breast_cancer', 'pima_indians_diabetes', 'sonar', 'vertebral_column', 'banknote', 'heart_disease', 'spambase']

print("=" * 90)
print("VERIFICAÇÃO COMPLETA: PEAB vs PULP após correção do bug de normalização")
print("=" * 90)

problemas = []
sucessos = []

for dataset in datasets:
    peab_file = Path(f"json/peab/{dataset}.json")
    pulp_file = Path(f"json/pulp/{dataset}.json")
    
    if not peab_file.exists() or not pulp_file.exists():
        print(f"\n❌ {dataset}: Arquivos não encontrados")
        continue
    
    with open(peab_file, 'r') as f:
        peab = json.load(f)
    with open(pulp_file, 'r') as f:
        pulp = json.load(f)
    
    # Extrai dados
    peab_pos = peab['explanation_stats']['positive']['count']
    peab_neg = peab['explanation_stats']['negative']['count']
    peab_rej = peab['explanation_stats']['rejected']['count']
    
    pulp_pos = pulp['estatisticas_por_tipo']['positiva']['instancias']
    pulp_neg = pulp['estatisticas_por_tipo']['negativa']['instancias']
    pulp_rej = pulp['estatisticas_por_tipo']['rejeitada']['instancias']
    
    # Calcula tamanho médio classificadas
    peab_class = (peab['explanation_stats']['positive']['mean_length'] * peab_pos +
                  peab['explanation_stats']['negative']['mean_length'] * peab_neg) / (peab_pos + peab_neg)
    
    pulp_class = (pulp['estatisticas_por_tipo']['positiva']['tamanho_medio'] * pulp_pos +
                  pulp['estatisticas_por_tipo']['negativa']['tamanho_medio'] * pulp_neg) / (pulp_pos + pulp_neg)
    
    peab_rej_size = peab['explanation_stats']['rejected']['mean_length']
    pulp_rej_size = pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio']
    
    # Thresholds
    peab_tplus = peab['thresholds']['t_plus']
    peab_tminus = peab['thresholds']['t_minus']
    pulp_tplus = pulp['t_plus']
    pulp_tminus = pulp['t_minus']
    
    print(f"\n{'='*90}")
    print(f"Dataset: {dataset.upper()}")
    print(f"{'='*90}")
    
    # Verifica distribuição
    dist_ok = (peab_pos == pulp_pos) and (peab_neg == pulp_neg) and (peab_rej == pulp_rej)
    if dist_ok:
        print(f"✅ Distribuição: Pos={peab_pos}, Neg={peab_neg}, Rej={peab_rej}")
    else:
        print(f"❌ Distribuição diferente!")
        print(f"   PEAB: Pos={peab_pos}, Neg={peab_neg}, Rej={peab_rej}")
        print(f"   PULP: Pos={pulp_pos}, Neg={pulp_neg}, Rej={pulp_rej}")
    
    # Verifica thresholds
    thresh_ok = abs(peab_tplus - pulp_tplus) < 1e-6 and abs(peab_tminus - pulp_tminus) < 1e-6
    if thresh_ok:
        print(f"✅ Thresholds: t+={peab_tplus:.6f}, t-={peab_tminus:.6f}")
    else:
        print(f"❌ Thresholds diferentes!")
        print(f"   PEAB: t+={peab_tplus:.6f}, t-={peab_tminus:.6f}")
        print(f"   PULP: t+={pulp_tplus:.6f}, t-={pulp_tminus:.6f}")
    
    # Verifica tamanho explicações CLASSIFICADAS
    print(f"\n📊 Tamanho Médio CLASSIFICADAS:")
    print(f"   PEAB: {peab_class:.2f} features")
    print(f"   PULP: {pulp_class:.2f} features")
    
    if pulp_class > peab_class + 0.01:  # Tolerância de 0.01
        diff = pulp_class - peab_class
        percent = (diff / peab_class) * 100
        print(f"   ❌ PULP maior: +{diff:.2f} features (+{percent:.1f}%)")
        problemas.append({
            'dataset': dataset,
            'tipo': 'Classificadas',
            'peab': peab_class,
            'pulp': pulp_class,
            'diff': diff,
            'percent': percent
        })
    elif pulp_class < peab_class - 0.01:
        diff = peab_class - pulp_class
        percent = (diff / peab_class) * 100
        print(f"   ✅ PULP melhor: -{diff:.2f} features (-{percent:.1f}%)")
        sucessos.append(f"{dataset} Classificadas: PULP {pulp_class:.2f} < PEAB {peab_class:.2f}")
    else:
        print(f"   ✅ Praticamente idênticos")
        sucessos.append(f"{dataset} Classificadas: PULP ≈ PEAB ({pulp_class:.2f})")
    
    # Verifica tamanho explicações REJEITADAS
    print(f"\n📊 Tamanho Médio REJEITADAS:")
    print(f"   PEAB: {peab_rej_size:.2f} features")
    print(f"   PULP: {pulp_rej_size:.2f} features")
    
    if pulp_rej_size > peab_rej_size + 0.01:
        diff = pulp_rej_size - peab_rej_size
        percent = (diff / peab_rej_size) * 100
        print(f"   ❌ PULP maior: +{diff:.2f} features (+{percent:.1f}%)")
        problemas.append({
            'dataset': dataset,
            'tipo': 'Rejeitadas',
            'peab': peab_rej_size,
            'pulp': pulp_rej_size,
            'diff': diff,
            'percent': percent
        })
    elif pulp_rej_size < peab_rej_size - 0.01:
        diff = peab_rej_size - pulp_rej_size
        percent = (diff / peab_rej_size) * 100
        print(f"   ✅ PULP melhor: -{diff:.2f} features (-{percent:.1f}%)")
        sucessos.append(f"{dataset} Rejeitadas: PULP {pulp_rej_size:.2f} < PEAB {peab_rej_size:.2f}")
    else:
        print(f"   ✅ Praticamente idênticos")
        sucessos.append(f"{dataset} Rejeitadas: PULP ≈ PEAB ({pulp_rej_size:.2f})")

print("\n" + "=" * 90)
print("RESUMO FINAL")
print("=" * 90)

if problemas:
    print(f"\n❌ {len(problemas)} PROBLEMA(S) DETECTADO(S):")
    print("-" * 90)
    for i, p in enumerate(problemas, 1):
        print(f"{i}. {p['dataset']} ({p['tipo']}): PULP {p['pulp']:.2f} > PEAB {p['peab']:.2f}")
        print(f"   Diferença: +{p['diff']:.2f} features (+{p['percent']:.1f}%)")
else:
    print("\n🎉 NENHUM PROBLEMA DETECTADO!")
    print("✅ PULP está gerando explicações menores ou iguais ao PEAB em todos os casos!")

if sucessos:
    print(f"\n✅ {len(sucessos)} CASO(S) DE SUCESSO:")
    print("-" * 90)
    for s in sucessos:
        print(f"   • {s}")

print("\n" + "=" * 90)
