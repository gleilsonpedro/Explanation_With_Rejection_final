"""
ANÁLISE DETALHADA: Verificar discrepâncias PULP vs PEAB
=========================================================
"""
import json
from pathlib import Path

DATASETS = [
    "banknote",
    "breast_cancer", 
    "heart_disease",
    "pima_indians_diabetes",
    "sonar",
    "spambase",
    "vertebral_column"
]

DATASET_NAMES = {
    "banknote": "Banknote",
    "breast_cancer": "Breast Cancer",
    "heart_disease": "Heart Disease",
    "pima_indians_diabetes": "Pima Indians",
    "sonar": "Sonar",
    "spambase": "Spambase",
    "vertebral_column": "Vertebral Column"
}

print("="*80)
print("ANÁLISE: PULP vs PEAB - Tamanhos de Explicações")
print("="*80)

problemas = []

for dataset in DATASETS:
    print(f"\n{'='*80}")
    print(f"📊 {DATASET_NAMES[dataset].upper()}")
    print(f"{'='*80}")
    
    peab_path = Path(f"json/peab/{dataset}.json")
    pulp_path = Path(f"json/pulp/{dataset}.json")
    
    if not peab_path.exists() or not pulp_path.exists():
        print("⚠️  Arquivos não encontrados!")
        continue
    
    with open(peab_path) as f:
        peab = json.load(f)
    with open(pulp_path) as f:
        pulp = json.load(f)
    
    # Thresholds
    peab_t_plus = peab['thresholds']['t_plus']
    peab_t_minus = peab['thresholds']['t_minus']
    pulp_t_plus = pulp['t_plus']
    pulp_t_minus = pulp['t_minus']
    
    print(f"\n1️⃣ THRESHOLDS:")
    print(f"   PEAB: t+ = {peab_t_plus:.8f}, t- = {peab_t_minus:.8f}")
    print(f"   PULP: t+ = {pulp_t_plus:.8f}, t- = {pulp_t_minus:.8f}")
    
    if abs(peab_t_plus - pulp_t_plus) > 0.000001:
        print(f"   ⚠️  DIFERENTES! Diff = {abs(peab_t_plus - pulp_t_plus):.10f}")
    else:
        print(f"   ✅ Idênticos")
    
    # Distribuição
    peab_stats = peab['explanation_stats']
    pulp_stats = pulp.get('estatisticas_por_tipo', {})
    
    peab_pos = peab_stats.get('positive', {}).get('count', 0)
    peab_neg = peab_stats.get('negative', {}).get('count', 0)
    peab_rej = peab_stats.get('rejected', {}).get('count', 0)
    
    pulp_pos = pulp_stats.get('positiva', {}).get('instancias', 0)
    pulp_neg = pulp_stats.get('negativa', {}).get('instancias', 0)
    pulp_rej = pulp_stats.get('rejeitada', {}).get('instancias', 0)
    
    print(f"\n2️⃣ DISTRIBUIÇÃO:")
    print(f"   PEAB: Pos={peab_pos}, Neg={peab_neg}, Rej={peab_rej}")
    print(f"   PULP: Pos={pulp_pos}, Neg={pulp_neg}, Rej={pulp_rej}")
    
    if peab_pos != pulp_pos or peab_neg != pulp_neg or peab_rej != pulp_rej:
        print(f"   ❌ DISTRIBUIÇÕES DIFERENTES!")
        problemas.append(f"{dataset}: Distribuições diferentes")
    else:
        print(f"   ✅ Distribuições idênticas")
    
    # Tamanhos
    print(f"\n3️⃣ TAMANHOS MÉDIOS:")
    
    # PEAB
    if (peab_pos + peab_neg) > 0:
        peab_classif = (
            peab_stats['positive']['mean_length'] * peab_pos +
            peab_stats['negative']['mean_length'] * peab_neg
        ) / (peab_pos + peab_neg)
    else:
        peab_classif = 0
    peab_rej_size = peab_stats['rejected']['mean_length']
    
    # PULP
    if (pulp_pos + pulp_neg) > 0:
        pos_size = pulp_stats.get('positiva', {}).get('tamanho_medio', 0)
        neg_size = pulp_stats.get('negativa', {}).get('tamanho_medio', 0)
        pulp_classif = (pos_size * pulp_pos + neg_size * pulp_neg) / (pulp_pos + pulp_neg)
    else:
        pulp_classif = 0
    pulp_rej_size = pulp_stats.get('rejeitada', {}).get('tamanho_medio', 0)
    
    print(f"\n   Classificadas:")
    print(f"      PEAB: {peab_classif:.2f} features")
    print(f"      PULP: {pulp_classif:.2f} features")
    
    diff_classif = pulp_classif - peab_classif
    if pulp_classif > peab_classif + 0.1:
        pct = ((pulp_classif / peab_classif) - 1) * 100 if peab_classif > 0 else 999
        print(f"      ❌ PULP MAIOR (+{diff_classif:.2f}, +{pct:.1f}%)")
        problemas.append(f"{dataset} Classificadas: PULP {pulp_classif:.2f} > PEAB {peab_classif:.2f}")
    elif pulp_classif < peab_classif - 0.1:
        print(f"      ✅ PULP menor ({diff_classif:.2f})")
    else:
        print(f"      ✅ Praticamente iguais")
    
    print(f"\n   Rejeitadas:")
    print(f"      PEAB: {peab_rej_size:.2f} features")
    print(f"      PULP: {pulp_rej_size:.2f} features")
    
    diff_rej = pulp_rej_size - peab_rej_size
    if pulp_rej_size > peab_rej_size + 0.1:
        pct = ((pulp_rej_size / peab_rej_size) - 1) * 100 if peab_rej_size > 0 else 999
        print(f"      ❌ PULP MAIOR (+{diff_rej:.2f}, +{pct:.1f}%)")
        problemas.append(f"{dataset} Rejeitadas: PULP {pulp_rej_size:.2f} > PEAB {peab_rej_size:.2f}")
    elif pulp_rej_size < peab_rej_size - 0.1:
        print(f"      ✅ PULP menor ({diff_rej:.2f})")
    else:
        print(f"      ✅ Praticamente iguais")

print("\n" + "="*80)
print("📋 RESUMO DOS PROBLEMAS")
print("="*80)

if problemas:
    print(f"\n⚠️  {len(problemas)} problema(s) detectado(s):\n")
    for i, p in enumerate(problemas, 1):
        print(f"{i}. {p}")
    
    print("\n" + "="*80)
    print("🔍 ANÁLISE:")
    print("="*80)
    print("""
PROBLEMAS CRÍTICOS IDENTIFICADOS:

1. Breast Cancer: PULP 19.18 vs PEAB 1.69 (11x maior!)
   → Possível causa: PULP está incluindo features desnecessárias
   
2. Spambase: PULP 51.02 vs PEAB 28.10 (quase 2x maior)
   → Possível causa: PULP não está convergindo para o ótimo

3. Distribuições diferentes em alguns datasets
   → Isso indica que a lógica de classificação ainda está diferente

HIPÓTESE PRINCIPAL:
O problema da normalização foi corrigido, MAS pode haver outro bug na
formulação do problema de otimização do PULP. Especificamente:

- As restrições podem estar muito FROUXAS (EPSILON muito grande?)
- O cálculo de worst-case pode estar errado
- O solver pode não estar convergindo para o ótimo global

AÇÃO RECOMENDADA:
Verificar a formulação do problema de otimização no PULP (linhas 110-160)
especialmente o cálculo dos worst-case scores e as restrições.
""")
else:
    print("\n✅ Nenhum problema detectado! Todos os resultados corretos.")

print("="*80)
