"""
Verificar se MinExp MNIST melhorou após reexperimento
"""
import json
import numpy as np

print("=" * 80)
print("COMPARAÇÃO MNIST - TODOS OS MÉTODOS")
print("=" * 80)

# PEAB
with open('json/peab/mnist_3_vs_8.json') as f:
    peab = json.load(f)
peab_c = [p['computation_time'] for p in peab['per_instance'] if not p.get('rejected')]
peab_mean = np.mean(peab_c) * 1000

print(f"\nPEAB MNIST:")
print(f"  Classificadas: {peab_mean:10.2f} ms ({peab_mean/1000:.3f} segundos)")

# MinExp
with open('json/minexp/mnist.json') as f:
    minexp = json.load(f)
minexp_c = [p['computation_time'] for p in minexp['per_instance'] if not p.get('rejected')]
minexp_r = [p['computation_time'] for p in minexp['per_instance'] if p.get('rejected')]
minexp_mean_c = np.mean(minexp_c) * 1000
minexp_mean_r = np.mean(minexp_r) * 1000 if minexp_r else 0

print(f"\nMinExp MNIST (NOVO - após reexperimento):")
print(f"  Classificadas: {minexp_mean_c:10.2f} ms ({minexp_mean_c/1000:.3f} segundos)")
if minexp_r:
    print(f"  Rejeitadas:    {minexp_mean_r:10.2f} ms ({minexp_mean_r/1000:.3f} segundos)")

# Comparação
prop = minexp_mean_c / peab_mean
print(f"\n{'=' * 80}")
print(f"PROPORÇÃO: MinExp é {prop:.0f}x mais lento que PEAB no MNIST")
print(f"{'=' * 80}")

# Histórico
print(f"\nVALORES HISTÓRICOS:")
print(f"  MinExp ANTES do reexperimento: 67,574 ms (67.6 segundos)")
print(f"  MinExp DEPOIS do reexperimento: {minexp_mean_c:,.0f} ms ({minexp_mean_c/1000:.1f} segundos)")
print(f"  Diferença: {(minexp_mean_c - 67574):+,.0f} ms ({((minexp_mean_c - 67574)/67574)*100:+.1f}%)")

# Conclusão
print(f"\n{'=' * 80}")
print("CONCLUSÃO")
print(f"{'=' * 80}")

if minexp_mean_c > 60000:  # > 60 segundos
    print("\n🔴 MinExp MNIST continua EXTREMAMENTE LENTO (>60s/instância)")
    print("\nNÃO era erro experimental! MinExp é REALMENTE muito lento em MNIST.")
    print("\nPROVÁVEIS CAUSAS:")
    print("  1. MNIST tem 784 features (28x28 pixels)")
    print("  2. MinExp busca explicação MÍNIMA → processo combinatorial")
    print("  3. Com 784 features, espaço de busca é IMENSO")
    print("  4. PEAB é rápido (23ms) porque usa aproximação gulosa")
    print("  5. MinExp é exato mas MUITO custoso em alta dimensionalidade")
    print("\nRECOMENDAÇÃO:")
    print("  → Aceitar que MinExp é lento em MNIST (é característica do método)")
    print("  → Manter o valor na tabela (está correto)")
    print("  → Mencionar no artigo que MinExp não escala bem para muitas features")
elif minexp_mean_c < 10000:  # < 10 segundos  
    print("\n✓ VALOR CORRIGIDO! Agora está em valores razoáveis.")
    print("\nEu ESTAVA CERTO: era OUTLIER por erro experimental")
    print("Os ~67 segundos eram anormais, valor correto é ~", minexp_mean_c/1000, "segundos")
else:
    print("\n⚠ Valor ainda alto mas melhor que antes")
    print("Pode ter tido melhoria mas ainda está acima do esperado")
