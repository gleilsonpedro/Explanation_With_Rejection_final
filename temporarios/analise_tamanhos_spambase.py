"""
Debug spambase: Vamos ver algumas instâncias PULP que estão muito grandes
"""
import json
from pathlib import Path

# Carrega resultados do PULP e PEAB para comparar os tamanhos
pulp_file = Path("json/pulp/spambase.json")

with open(pulp_file, 'r') as f:
    pulp = json.load(f)

print("=" * 90)
print("DEBUG SPAMBASE: Análise das instâncias PULP")
print("=" * 90)

# Pega explicações
explicacoes = pulp['explicacoes']

# Separa por tipo
negativas = [exp for exp in explicacoes if exp['tipo'] == 'negativa']

print(f"\nTotal de negativas: {len(negativas)}")

# Analisa distribuição de tamanhos
tamanhos = [len(exp['features']) for exp in negativas]
print(f"\nEstatísticas de tamanho:")
print(f"  Mínimo: {min(tamanhos)}")
print(f"  Máximo: {max(tamanhos)}")
print(f"  Média: {sum(tamanhos)/len(tamanhos):.2f}")
print(f"  Mediana: {sorted(tamanhos)[len(tamanhos)//2]}")

# Conta por faixas
print(f"\nDistribuição por faixa:")
import numpy as np
faixas = [(0, 30), (31, 40), (41, 50), (51, 60)]
for r_min, r_max in faixas:
    count = sum(1 for t in tamanhos if r_min <= t <= r_max)
    print(f"  {r_min:2d}-{r_max:2d}: {count:4d} instâncias ({count/len(tamanhos)*100:.1f}%)")

# Pega algumas instâncias com MUITAS features (>50)
grandes = [exp for exp in negativas if len(exp['features']) > 50]
print(f"\n🔍 Instâncias com >50 features: {len(grandes)}")

if len(grandes) > 0:
    print("\nExemplos (primeiras 5):")
    for i, exp in enumerate(grandes[:5]):
        print(f"  {i+1}. {len(exp['features'])} features: {sorted(exp['features'])[:10]}...")

# Pega algumas instâncias pequenas
pequenas = [exp for exp in negativas if len(exp['features']) < 30]
print(f"\n✅ Instâncias com <30 features: {len(pequenas)}")

if len(pequenas) > 0:
    print("\nExemplos (primeiras 5):")
    for i, exp in enumerate(pequenas[:5]):
        print(f"  {i+1}. {len(exp['features'])} features: {sorted(exp['features'])[:10]}...")

# Análise: Parece que a MAIORIA das instâncias tem ~51 features (quase todas as 57!)
print("\n" + "=" * 90)
print("OBSERVAÇÃO IMPORTANTE")
print("=" * 90)
print(f"\nA maioria das negativas ({sum(1 for t in tamanhos if t > 50)}/{len(tamanhos)}) ")
print("tem MAIS DE 50 features (de 57 disponíveis).")
print("\nIsso sugere que o PULP está selecionando quase todas as features!")
print("Isso é muito estranho para um problema de MINIMIZAÇÃO.")
print("\nPossíveis causas:")
print("1. As restrições estão muito apertadas (quase impossíveis de satisfazer)")
print("2. O dataset tem features muito correlacionadas/redundantes")
print("3. Bug na formulação do worst-case")
