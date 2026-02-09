"""
Análise: MinExp MNIST está com tempo MUITO alto (67 segundos/instância)
Vamos comparar com outros datasets para ver se é erro experimental
"""
import json
import numpy as np

print("=" * 100)
print("ANÁLISE: Tempos do MinExp em todos os datasets")
print("=" * 100)

datasets = [
    ("Banknote", "banknote.json"),
    ("Vertebral", "vertebral_column.json"),
    ("Pima", "pima_indians_diabetes.json"),
    ("Heart", "heart_disease.json"),
    ("Credit", "creditcard.json"),
    ("Breast", "breast_cancer.json"),
    ("Covertype", "covertype.json"),
    ("Spambase", "spambase.json"),
    ("Sonar", "sonar.json"),
    ("MNIST", "mnist.json"),
]

tempos_classif = []
tempos_rej = []

for nome, arquivo in datasets:
    try:
        with open(f"json/minexp/{arquivo}") as f:
            data = json.load(f)
        
        per_instance = data.get("per_instance", [])
        classif = [p["computation_time"] for p in per_instance if not p.get("rejected")]
        rej = [p["computation_time"] for p in per_instance if p.get("rejected")]
        
        if classif:
            mean_c = np.mean(classif) * 1000
            tempos_classif.append((nome, mean_c))
        
        if rej:
            mean_r = np.mean(rej) * 1000
            tempos_rej.append((nome, mean_r))
        
        n_features = data.get("n_features", "?")
        
        print(f"\n{nome:15} (features={n_features}):")
        print(f"  Classif: {mean_c:10.2f} ms ({len(classif):3} inst)")
        if rej:
            print(f"  Rejeita: {mean_r:10.2f} ms ({len(rej):3} inst)")
        else:
            print(f"  Rejeita: (sem rejeitadas)")
            
    except Exception as e:
        print(f"\n{nome:15}: ERRO - {e}")

print("\n" + "=" * 100)
print("ANÁLISE COMPARATIVA")
print("=" * 100)

# Ordenar por tempo
tempos_classif_sorted = sorted(tempos_classif, key=lambda x: x[1])

print("\nRanking de tempo (classificadas):")
for i, (nome, tempo) in enumerate(tempos_classif_sorted, 1):
    print(f"{i:2}. {nome:15} {tempo:10.2f} ms")

# Estatísticas
tempos_apenas = [t for _, t in tempos_classif]
media_geral = np.mean(tempos_apenas)
mediana = np.median(tempos_apenas)
std = np.std(tempos_apenas)

print(f"\nESTATÍSTICAS:")
print(f"  Média:   {media_geral:10.2f} ms")
print(f"  Mediana: {mediana:10.2f} ms")
print(f"  Desvio:  {std:10.2f} ms")
print(f"  Mínimo:  {min(tempos_apenas):10.2f} ms")
print(f"  Máximo:  {max(tempos_apenas):10.2f} ms")

# Verificar se MNIST é outlier
mnist_tempo = [t for n, t in tempos_classif if n == "MNIST"][0]
z_score = (mnist_tempo - media_geral) / std

print(f"\n{'!' * 100}")
print(f"MNIST ANÁLISE:")
print(f"{'!' * 100}")
print(f"  Tempo MNIST: {mnist_tempo:10.2f} ms = {mnist_tempo/1000:.2f} segundos")
print(f"  Z-score: {z_score:.2f} (número de desvios padrão da média)")
print()

if z_score > 3:
    print(f"  ⚠️  ALERTA: MNIST está {z_score:.1f} desvios padrão acima da média!")
    print(f"      Isso indica OUTLIER EXTREMO - provável ERRO EXPERIMENTAL")
    print(f"      Recomendação: REFAZER o experimento MNIST com MinExp")
    print()
    print(f"      Comparação:")
    print(f"      - Segundo maior: {tempos_classif_sorted[-2][1]:.2f} ms ({tempos_classif_sorted[-2][0]})")
    print(f"      - MNIST: {mnist_tempo:.2f} ms")
    print(f"      - Diferença: {(mnist_tempo / tempos_classif_sorted[-2][1]):.1f}x mais lento")
elif z_score > 2:
    print(f"  ⚠️  MNIST é significativamente mais lento que outros datasets")
    print(f"      Pode ser normal (MNIST tem mais features) ou erro experimental")
    print(f"      Recomendação: Revisar os dados")
else:
    print(f"  ✓ Tempo MNIST está dentro do esperado para a variação dos datasets")

print("\n" + "=" * 100)
print("CONCLUSÃO")
print("=" * 100)
print("""
Se MNIST for outlier extremo (z-score > 3):
  → Valor está TECNICAMENTE correto (é o que está no JSON)
  → Mas indica ERRO EXPERIMENTAL (algo deu errado no experimento)
  → Recomendação: REFAZER experimento MinExp para MNIST

Se MNIST estiver dentro da variação normal:
  → Valor está correto
  → MNIST é naturalmente mais lento (mais features, mais complexo)
""")
