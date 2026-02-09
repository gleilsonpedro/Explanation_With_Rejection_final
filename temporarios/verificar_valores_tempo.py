"""
Verifica valores de tempo calculados a partir dos JSONs para todos os métodos.
"""
import json
import numpy as np

datasets = [
    ("banknote", "banknote", "banknote", "banknote", "Banknote"),
    ("vertebral_column", "vertebral_column", "vertebral_column", "vertebral_column", "Vertebral Column"),
    ("pima_indians_diabetes", "pima_indians_diabetes", "pima_indians_diabetes", "pima_indians_diabetes", "Pima Indians"),
    ("heart_disease", "heart_disease", "heart_disease", "heart_disease", "Heart Disease"),
    ("creditcard", "creditcard", "creditcard", "creditcard", "Credit Card"),
    ("breast_cancer", "breast_cancer", "breast_cancer", "breast_cancer", "Breast Cancer"),
    ("covertype", "covertype", "covertype", "covertype", "Covertype"),
    ("spambase", "spambase", "spambase", "spambase", "Spambase"),
    ("sonar", "sonar", "sonar", "sonar", "Sonar"),
    ("mnist_3_vs_8", "mnist", "mnist", "mnist", "MNIST (3 vs 8)"),
]

print("=" * 100)
print("VERIFICAÇÃO DE VALORES DE TEMPO - Classificadas e Rejeitadas")
print("=" * 100)

for peab_name, pulp_name, anchor_name, minexp_name, display_name in datasets:
    print(f"\n{display_name}")
    print("-" * 100)
    
    # PEAB (MINABRO)
    try:
        with open(f"json/peab/{peab_name}.json") as f:
            dados = json.load(f)
        per_instance = dados.get("per_instance", [])
        
        classif_times = [p.get("computation_time", 0) for p in per_instance if not p.get("rejected", False)]
        rej_times = [p.get("computation_time", 0) for p in per_instance if p.get("rejected", False)]
        
        if classif_times:
            classif_mean = np.mean(classif_times) * 1000
            classif_std = np.std(classif_times, ddof=1) * 1000 if len(classif_times) > 1 else 0.0
            print(f"  MINABRO Classificadas: {classif_mean:.2f} ± {classif_std:.2f} ms ({len(classif_times)} inst)")
        
        if rej_times:
            rej_mean = np.mean(rej_times) * 1000
            rej_std = np.std(rej_times, ddof=1) * 1000 if len(rej_times) > 1 else 0.0
            print(f"  MINABRO Rejeitadas:   {rej_mean:.2f} ± {rej_std:.2f} ms ({len(rej_times)} inst)")
    except Exception as e:
        print(f"  MINABRO: Erro - {e}")
    
    # MinExp (AbLinRO)
    try:
        with open(f"json/minexp/{minexp_name}.json") as f:
            dados = json.load(f)
        per_instance = dados.get("per_instance", [])
        
        classif_times = [p.get("computation_time", 0) for p in per_instance if not p.get("rejected", False)]
        rej_times = [p.get("computation_time", 0) for p in per_instance if p.get("rejected", False)]
        
        if classif_times:
            classif_mean = np.mean(classif_times) * 1000
            classif_std = np.std(classif_times, ddof=1) * 1000 if len(classif_times) > 1 else 0.0
            print(f"  AbLinRO Classificadas: {classif_mean:.2f} ± {classif_std:.2f} ms ({len(classif_times)} inst)")
            # mostrar min e max para identificar outliers
            if classif_times:
                print(f"           (min: {min(classif_times)*1000:.2f} ms, max: {max(classif_times)*1000:.2f} ms)")
        
        if rej_times:
            rej_mean = np.mean(rej_times) * 1000
            rej_std = np.std(rej_times, ddof=1) * 1000 if len(rej_times) > 1 else 0.0
            print(f"  AbLinRO Rejeitadas:   {rej_mean:.2f} ± {rej_std:.2f} ms ({len(rej_times)} inst)")
            if rej_times:
                print(f"           (min: {min(rej_times)*1000:.2f} ms, max: {max(rej_times)*1000:.2f} ms)")
    except Exception as e:
        print(f"  AbLinRO: Erro - {e}")
