"""
DIAGNÓSTICO: Comparação PULP vs PEAB
====================================
Analisa os JSONs existentes para detectar anomalias.
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

print("="*80)
print("DIAGNÓSTICO: PULP vs PEAB")
print("="*80)

problemas = []

for dataset in DATASETS:
    print(f"\n{'='*80}")
    print(f"📊 Dataset: {dataset.upper()}")
    print(f"{'='*80}")
    
    # Carregar JSONs
    peab_path = Path(f"json/peab/{dataset}.json")
    pulp_path = Path(f"json/pulp/{dataset}.json")
    
    if not peab_path.exists() or not pulp_path.exists():
        print("⚠️  Arquivos não encontrados!")
        continue
    
    with open(peab_path) as f:
        peab_data = json.load(f)
    with open(pulp_path) as f:
        pulp_data = json.load(f)
    
    # 1. VERIFICAR THRESHOLDS
    print("\n1️⃣ THRESHOLDS:")
    peab_t_plus = peab_data['thresholds']['t_plus']
    peab_t_minus = peab_data['thresholds']['t_minus']
    pulp_t_plus = pulp_data['t_plus']
    pulp_t_minus = pulp_data['t_minus']
    
    print(f"   PEAB: t+ = {peab_t_plus:.6f}, t- = {peab_t_minus:.6f}")
    print(f"   PULP: t+ = {pulp_t_plus:.6f}, t- = {pulp_t_minus:.6f}")
    
    if abs(peab_t_plus - pulp_t_plus) > 0.001 or abs(peab_t_minus - pulp_t_minus) > 0.001:
        print("   ⚠️  THRESHOLDS DIFERENTES!")
        problemas.append(f"{dataset}: Thresholds diferentes")
    else:
        print("   ✅ Thresholds iguais")
    
    # 2. VERIFICAR TIPOS DE PREDIÇÃO
    print("\n2️⃣ DISTRIBUIÇÃO POR TIPO:")
    
    peab_stats = peab_data['explanation_stats']
    pulp_stats = pulp_data.get('estatisticas_por_tipo', {})
    
    print("\n   PEAB:")
    peab_pos = peab_stats.get('positive', {}).get('count', 0)
    peab_neg = peab_stats.get('negative', {}).get('count', 0)
    peab_rej = peab_stats.get('rejected', {}).get('count', 0)
    print(f"      Positivas: {peab_pos}")
    print(f"      Negativas: {peab_neg}")
    print(f"      Rejeitadas: {peab_rej}")
    
    print("\n   PULP:")
    pulp_pos = pulp_stats.get('positiva', {}).get('instancias', 0)
    pulp_neg = pulp_stats.get('negativa', {}).get('instancias', 0)
    pulp_rej = pulp_stats.get('rejeitada', {}).get('instancias', 0)
    print(f"      Positivas: {pulp_pos}")
    print(f"      Negativas: {pulp_neg}")
    print(f"      Rejeitadas: {pulp_rej}")
    
    if pulp_pos == 0 and peab_pos > 0:
        print(f"   ⚠️  PULP NÃO TEM POSITIVAS MAS PEAB TEM {peab_pos}!")
        problemas.append(f"{dataset}: PULP sem positivas (PEAB tem {peab_pos})")
    
    # 3. COMPARAR TAMANHOS MÉDIOS
    print("\n3️⃣ TAMANHOS MÉDIOS:")
    
    # Classificadas PEAB (média ponderada)
    if (peab_pos + peab_neg) > 0:
        peab_classif_size = (
            peab_stats['positive']['mean_length'] * peab_pos +
            peab_stats['negative']['mean_length'] * peab_neg
        ) / (peab_pos + peab_neg)
    else:
        peab_classif_size = 0
    
    # Rejeitadas PEAB
    peab_rej_size = peab_stats['rejected']['mean_length']
    
    # Classificadas PULP (média ponderada se houver)
    if (pulp_pos + pulp_neg) > 0:
        pos_size = pulp_stats.get('positiva', {}).get('tamanho_medio', 0)
        neg_size = pulp_stats.get('negativa', {}).get('tamanho_medio', 0)
        pulp_classif_size = (pos_size * pulp_pos + neg_size * pulp_neg) / (pulp_pos + pulp_neg)
    else:
        pulp_classif_size = 0
    
    # Rejeitadas PULP
    pulp_rej_size = pulp_stats.get('rejeitada', {}).get('tamanho_medio', 0)
    
    print(f"\n   Classificadas:")
    print(f"      PEAB: {peab_classif_size:.2f} features")
    print(f"      PULP: {pulp_classif_size:.2f} features")
    
    if pulp_classif_size > peab_classif_size and pulp_classif_size > 0:
        diff = pulp_classif_size - peab_classif_size
        print(f"      ❌ PULP MAIOR que PEAB (+{diff:.2f}) - ISSO NÃO DEVERIA ACONTECER!")
        problemas.append(f"{dataset}: PULP ({pulp_classif_size:.2f}) > PEAB ({peab_classif_size:.2f}) em classificadas")
    elif pulp_classif_size > 0:
        diff = peab_classif_size - pulp_classif_size
        print(f"      ✅ PULP menor que PEAB (-{diff:.2f})")
    
    print(f"\n   Rejeitadas:")
    print(f"      PEAB: {peab_rej_size:.2f} features")
    print(f"      PULP: {pulp_rej_size:.2f} features")
    
    if pulp_rej_size > peab_rej_size and pulp_rej_size > 0:
        diff = pulp_rej_size - peab_rej_size
        print(f"      ❌ PULP MAIOR que PEAB (+{diff:.2f}) - ISSO NÃO DEVERIA ACONTECER!")
        problemas.append(f"{dataset}: PULP ({pulp_rej_size:.2f}) > PEAB ({peab_rej_size:.2f}) em rejeitadas")
    elif pulp_rej_size > 0:
        diff = peab_rej_size - pulp_rej_size
        print(f"      ✅ PULP menor que PEAB (-{diff:.2f})")

print("\n" + "="*80)
print("📋 RESUMO DOS PROBLEMAS ENCONTRADOS")
print("="*80)

if problemas:
    for i, p in enumerate(problemas, 1):
        print(f"{i}. {p}")
    
    print(f"\n⚠️  Total: {len(problemas)} problema(s) detectado(s)")
else:
    print("✅ Nenhum problema detectado!")

print("\n" + "="*80)
print("POSSÍVEIS CAUSAS:")
print("="*80)
print("""
1. BREAST CANCER sem positivas no PULP:
   → O solver pode estar sendo MUITO conservador
   → As restrições podem estar muito apertadas
   → Pode haver diferença nos thresholds usados

2. PULP maior que PEAB (quando deveria ser o contrário):
   → Heurística gulosa do PEAB pode estar encontrando soluções melhores
   → PULP pode não estar convergindo para o ótimo global
   → Pode haver diferença no espaço de features ou normalização

3. SOLUÇÃO SEM MODIFICAR TUDO:
   → Analisar instâncias específicas que divergem
   → Verificar logs do solver PULP
   → Comparar feature por feature entre PEAB e PULP
""")
print("="*80)
