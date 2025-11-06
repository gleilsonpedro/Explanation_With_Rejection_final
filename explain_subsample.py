"""
Mostra o tamanho de todos os datasets e explica por que só alguns usam subsample
"""
import sys
sys.path.insert(0, '.')

from data.datasets import carregar_dataset, set_mnist_options

datasets_info = {
    'breast_cancer': 'Câncer de mama',
    'wine': 'Vinho',
    'pima_indians_diabetes': 'Diabetes Pima',
    'vertebral_column': 'Coluna vertebral',
    'sonar': 'Sonar',
    'spambase': 'Spam',
    'banknote_auth': 'Autenticação de notas',
    'heart_disease': 'Doenças cardíacas',
    'wine_quality': 'Qualidade do vinho',
}

print("="*80)
print("TAMANHO DOS DATASETS E USO DE SUBSAMPLE")
print("="*80)

print(f"\n{'Dataset':<30} {'Instâncias':>12} {'Features':>10} {'Subsample?':>12}")
print("-"*80)

for dataset_name, descricao in datasets_info.items():
    try:
        X, y, _ = carregar_dataset(dataset_name)
        n_instances = X.shape[0]
        n_features = X.shape[1]
        print(f"{descricao:<30} {n_instances:>12,} {n_features:>10} {'Não':>12}")
    except Exception as e:
        print(f"{descricao:<30} {'ERRO':>12} {'-':>10} {'-':>12}")

# MNIST
try:
    set_mnist_options('raw', (7, 9))
    X, y, _ = carregar_dataset('mnist')
    print(f"{'MNIST (7 vs 9)':<30} {X.shape[0]:>12,} {X.shape[1]:>10} {'SIM (2%)':>12}")
except:
    print(f"{'MNIST':<30} {'ERRO':>12} {'-':>10} {'-':>12}")

print("\n" + "="*80)
print("POR QUE SÓ MNIST USA SUBSAMPLE?")
print("="*80)
print("""
1. DATASETS PEQUENOS/MÉDIOS (sem subsample):
   - Breast Cancer: 569 instâncias, 30 features
   - Wine: 178 instâncias, 13 features
   - Pima: 768 instâncias, 8 features
   - Sonar: 208 instâncias, 60 features
   - Vertebral: 310 instâncias, 6 features
   
   → Tempo de execução: RÁPIDO (poucos segundos por instância)
   → Não precisa de subsample!

2. MNIST (COM subsample de 2%):
   - Original: 13,966 instâncias (classe 7 vs 9)
   - Com subsample: ~280 instâncias
   - Features: 784 pixels
   
   → Sem subsample: ~3-5 horas de execução! ❌
   → Com subsample 2%: ~20-30 minutos ✅
   
   → MOTIVO: 784 features tornam cada explicação MUITO lenta
   → Subsample mantém fidelidade estatística com tempo viável

3. CreditCard (COM subsample de 10%):
   - Original: ~284,807 instâncias
   - Com subsample: ~28,481 instâncias
   
   → Sem subsample: IMPRATICÁVEL (dias de execução) ❌
   → Com subsample 10%: ~2-3 horas ✅
""")

print("="*80)
print("CONCLUSÃO:")
print("="*80)
print("""
✅ Datasets pequenos (<1000 instâncias): SEM subsample
✅ MNIST (784 features): subsample 2% (fidelidade + tempo viável)
✅ CreditCard (285k instâncias): subsample 10% (necessário para viabilidade)

O subsample é aplicado ANTES do split treino/teste, mantendo:
- Distribuição estratificada das classes
- Fidelidade estatística dos dados
- Viabilidade computacional para trabalhos acadêmicos
""")
print("="*80)
