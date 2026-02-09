"""
Comparação: Proporção MinExp/PEAB em todos os datasets
Para verificar se MNIST está com valor anômalo
"""
import json
import numpy as np

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
    ("MNIST", "mnist_3_vs_8.json"),
]

print("=" * 110)
print("PROPORÇÃO MinExp/PEAB: Quanto MinExp é mais lento que PEAB?")
print("=" * 110)
print(f"\n{'Dataset':<15} {'PEAB (ms)':<12} {'MinExp (ms)':<12} {'Proporção':<12} {'Status'}")
print("-" * 110)

proporcoes = []

for nome, arquivo in datasets:
    try:
        # PEAB
        with open(f"json/peab/{arquivo}") as f:
            peab = json.load(f)
        peab_per = peab.get("per_instance", [])
        peab_c = [p["computation_time"] for p in peab_per if not p.get("rejected")]
        peab_mean = np.mean(peab_c) * 1000 if peab_c else 0
        
        # MinExp
        minexp_file = arquivo.replace("mnist_3_vs_8", "mnist")
        with open(f"json/minexp/{minexp_file}") as f:
            minexp = json.load(f)
        minexp_per = minexp.get("per_instance", [])
        minexp_c = [p["computation_time"] for p in minexp_per if not p.get("rejected")]
        minexp_mean = np.mean(minexp_c) * 1000 if minexp_c else 0
        
        # Proporção
        if peab_mean > 0 and minexp_mean > 0:
            prop = minexp_mean / peab_mean
            proporcoes.append((nome, prop))
            
            # Status
            if prop > 1000:
                status = "🔴 ANORMAL!"
            elif prop > 500:
                status = "⚠️  ALTO"
            elif prop > 200:
                status = "⚡ OK (alto)"
            else:
                status = "✓ OK"
            
            print(f"{nome:<15} {peab_mean:>10.2f}   {minexp_mean:>10.2f}   {prop:>9.0f}x    {status}")
        
    except FileNotFoundError as e:
        print(f"{nome:<15} {'ARQUIVO NÃO ENCONTRADO':<50}")
    except Exception as e:
        print(f"{nome:<15} ERRO: {e}")

print("\n" + "=" * 110)
print("ANÁLISE ESTATÍSTICA DAS PROPORÇÕES")
print("=" * 110)

if proporcoes:
    props = [p for _, p in proporcoes]
    
    print(f"\nMédia:     {np.mean(props):>8.1f}x")
    print(f"Mediana:   {np.median(props):>8.1f}x")
    print(f"Mínimo:    {min(props):>8.1f}x")
    print(f"Máximo:    {max(props):>8.1f}x")
    print(f"Desvio:    {np.std(props):>8.1f}x")
    
    # Identificar outliers
    media = np.mean(props)
    std = np.std(props)
    
    print(f"\n{'!' * 110}")
    print("OUTLIERS (> 2 desvios padrão):")
    print(f"{'!' * 110}")
    
    outliers_encontrados = False
    for nome, prop in proporcoes:
        z = (prop - media) / std
        if abs(z) > 2:
            outliers_encontrados = True
            print(f"  {nome:<15} {prop:>8.1f}x  (z-score: {z:>6.2f})")
    
    if not outliers_encontrados:
        print("  Nenhum outlier encontrado")
    
    print("\n" + "=" * 110)
    print("CONCLUSÃO SOBRE MinExp MNIST")
    print("=" * 110)
    
    mnist_prop = [p for n, p in proporcoes if n == "MNIST"]
    if mnist_prop:
        mnist_prop = mnist_prop[0]
        z_score = (mnist_prop - media) / std
        
        print(f"\nProporção MNIST: {mnist_prop:.0f}x")
        print(f"Z-score: {z_score:.2f}")
        
        if z_score > 3:
            print("\n🔴 VEREDITO: ERRO EXPERIMENTAL CONFIRMADO!")
            print("   → MinExp MNIST está com valor ABSOLUTAMENTE ANORMAL")
            print("   → É um OUTLIER EXTREMO (>3 desvios padrão)")
            print("   → Recomendação: REFAZER experimento MinExp para MNIST")
            print("   → O valor está no JSON, mas claramente algo deu ERRADO")
        elif z_score > 2:
            print("\n⚠️  VEREDITO: VALOR SUSPEITO")
            print("   → MinExp MNIST está mais lento que o esperado")
            print("   → Pode ser normal ou erro experimental")
            print("   → Recomendação: Revisar experimento")
        else:
            print("\n✓ VEREDITO: VALOR NORMAL")
            print("   → MinExp MNIST está dentro do esperado")
            print("   → MNIST é naturalmente mais complexo")
