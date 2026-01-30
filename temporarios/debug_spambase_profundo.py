import json
from pathlib import Path
import numpy as np

# Carrega resultados
peab_file = Path("json/peab/spambase.json")
pulp_file = Path("json/pulp/spambase.json")

with open(peab_file, 'r') as f:
    peab = json.load(f)
with open(pulp_file, 'r') as f:
    pulp = json.load(f)

print("=" * 90)
print("ANÁLISE PROFUNDA: SPAMBASE - Comparação PEAB vs PULP")
print("=" * 90)

# Coleta explicações de ambos
peab_exps = peab['explanations']
pulp_exps = pulp['explicacoes']

print(f"\nTotal de instâncias: {len(peab_exps)}")
print(f"PEAB: {len(peab_exps)} explicações")
print(f"PULP: {len(pulp_exps)} explicações")

# Agrupa por tipo de predição
peab_pos = [exp for exp in peab_exps if exp['prediction_type'] == 'positive']
peab_neg = [exp for exp in peab_exps if exp['prediction_type'] == 'negative']
peab_rej = [exp for exp in peab_exps if exp['prediction_type'] == 'rejected']

pulp_pos = [exp for exp in pulp_exps if exp['tipo'] == 'positiva']
pulp_neg = [exp for exp in pulp_exps if exp['tipo'] == 'negativa']
pulp_rej = [exp for exp in pulp_exps if exp['tipo'] == 'rejeitada']

print("\n" + "=" * 90)
print("COMPARAÇÃO POR TIPO DE PREDIÇÃO")
print("=" * 90)

# Analisa POSITIVAS
print("\n📊 POSITIVAS (5 instâncias):")
print("-" * 90)
if len(peab_pos) == len(pulp_pos):
    for i in range(len(peab_pos)):
        peab_size = len(peab_pos[i]['explanation'])
        pulp_size = len(pulp_pos[i]['features'])
        diff = pulp_size - peab_size
        print(f"  Instância {i+1}: PEAB={peab_size:2d} features, PULP={pulp_size:2d} features, Diff={diff:+3d}")
else:
    print(f"  ⚠️ Quantidade diferente! PEAB={len(peab_pos)}, PULP={len(pulp_pos)}")

# Analisa NEGATIVAS (amostra das primeiras 10)
print("\n📊 NEGATIVAS (primeiras 10 de 435):")
print("-" * 90)
if len(peab_neg) == len(pulp_neg):
    for i in range(min(10, len(peab_neg))):
        peab_size = len(peab_neg[i]['explanation'])
        pulp_size = len(pulp_neg[i]['features'])
        diff = pulp_size - peab_size
        print(f"  Instância {i+1}: PEAB={peab_size:2d} features, PULP={pulp_size:2d} features, Diff={diff:+3d}")
else:
    print(f"  ⚠️ Quantidade diferente! PEAB={len(peab_neg)}, PULP={len(pulp_neg)}")

# Estatísticas detalhadas
print("\n" + "=" * 90)
print("ESTATÍSTICAS DETALHADAS")
print("=" * 90)

def calc_stats(exps, size_key):
    sizes = [len(exp[size_key]) for exp in exps]
    return {
        'min': min(sizes),
        'max': max(sizes),
        'mean': np.mean(sizes),
        'median': np.median(sizes),
        'std': np.std(sizes)
    }

peab_pos_stats = calc_stats(peab_pos, 'explanation')
pulp_pos_stats = calc_stats(pulp_pos, 'features')
peab_neg_stats = calc_stats(peab_neg, 'explanation')
pulp_neg_stats = calc_stats(pulp_neg, 'features')

print("\nPOSITIVAS:")
print(f"  PEAB: min={peab_pos_stats['min']}, max={peab_pos_stats['max']}, "
      f"média={peab_pos_stats['mean']:.2f}, mediana={peab_pos_stats['median']:.0f}, "
      f"std={peab_pos_stats['std']:.2f}")
print(f"  PULP: min={pulp_pos_stats['min']}, max={pulp_pos_stats['max']}, "
      f"média={pulp_pos_stats['mean']:.2f}, mediana={pulp_pos_stats['median']:.0f}, "
      f"std={pulp_pos_stats['std']:.2f}")

print("\nNEGATIVAS:")
print(f"  PEAB: min={peab_neg_stats['min']}, max={peab_neg_stats['max']}, "
      f"média={peab_neg_stats['mean']:.2f}, mediana={peab_neg_stats['median']:.0f}, "
      f"std={peab_neg_stats['std']:.2f}")
print(f"  PULP: min={pulp_neg_stats['min']}, max={pulp_neg_stats['max']}, "
      f"média={pulp_neg_stats['mean']:.2f}, mediana={pulp_neg_stats['median']:.0f}, "
      f"std={pulp_neg_stats['std']:.2f}")

# Analisa distribuição de tamanhos
print("\n" + "=" * 90)
print("DISTRIBUIÇÃO DE TAMANHOS (NEGATIVAS)")
print("=" * 90)

peab_neg_sizes = [len(exp['explanation']) for exp in peab_neg]
pulp_neg_sizes = [len(exp['features']) for exp in pulp_neg]

# Conta quantas explicações em cada faixa
ranges = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]
print("\nFaixa de features | PEAB | PULP")
print("-" * 40)
for r_min, r_max in ranges:
    peab_count = sum(1 for s in peab_neg_sizes if r_min <= s <= r_max)
    pulp_count = sum(1 for s in pulp_neg_sizes if r_min <= s <= r_max)
    print(f"  {r_min:2d} - {r_max:2d}         | {peab_count:4d} | {pulp_count:4d}")

# Verifica se há padrão sistemático
print("\n" + "=" * 90)
print("ANÁLISE DE PADRÃO")
print("=" * 90)

pulp_maiores = sum(1 for i in range(len(peab_neg)) if len(pulp_neg[i]['features']) > len(peab_neg[i]['explanation']))
pulp_menores = sum(1 for i in range(len(peab_neg)) if len(pulp_neg[i]['features']) < len(peab_neg[i]['explanation']))
pulp_iguais = sum(1 for i in range(len(peab_neg)) if len(pulp_neg[i]['features']) == len(peab_neg[i]['explanation']))

print(f"\nNEGATIVAS ({len(peab_neg)} instâncias):")
print(f"  PULP > PEAB: {pulp_maiores} casos ({pulp_maiores/len(peab_neg)*100:.1f}%)")
print(f"  PULP < PEAB: {pulp_menores} casos ({pulp_menores/len(peab_neg)*100:.1f}%)")
print(f"  PULP = PEAB: {pulp_iguais} casos ({pulp_iguais/len(peab_neg)*100:.1f}%)")

# Diferenças médias
diffs = [len(pulp_neg[i]['features']) - len(peab_neg[i]['explanation']) for i in range(len(peab_neg))]
print(f"\nDiferença média: {np.mean(diffs):.2f} features")
print(f"Diferença mediana: {np.median(diffs):.0f} features")
print(f"Diferença máxima: {max(diffs)} features")
print(f"Diferença mínima: {min(diffs)} features")

# Casos extremos
print("\n" + "=" * 90)
print("CASOS EXTREMOS (5 maiores diferenças)")
print("=" * 90)

diffs_indexed = [(i, len(pulp_neg[i]['features']) - len(peab_neg[i]['explanation'])) 
                 for i in range(len(peab_neg))]
diffs_indexed.sort(key=lambda x: x[1], reverse=True)

print("\nÍndice | PEAB | PULP | Diferença")
print("-" * 45)
for idx, diff in diffs_indexed[:5]:
    peab_size = len(peab_neg[idx]['explanation'])
    pulp_size = len(pulp_neg[idx]['features'])
    print(f"  {idx:4d} | {peab_size:4d} | {pulp_size:4d} | {diff:+4d}")

print("\n" + "=" * 90)
print("CONCLUSÃO")
print("=" * 90)
print(f"\nPULP está gerando explicações sistematicamente MAIORES.")
print(f"Isso ocorre em {pulp_maiores}/{len(peab_neg)} = {pulp_maiores/len(peab_neg)*100:.1f}% dos casos.")
print(f"\nDiferença média: +{np.mean(diffs):.2f} features por explicação")
print(f"\nIsso sugere que há um problema na formulação do PULP para este dataset")
print("ou que as restrições estão muito relaxadas, permitindo soluções subótimas.")
