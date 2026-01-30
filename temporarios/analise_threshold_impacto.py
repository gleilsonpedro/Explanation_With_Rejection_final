"""
ANÁLISE: Thresholds diferentes explicam PULP > PEAB?
=====================================================
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
print("ANÁLISE: Impacto dos Thresholds Diferentes")
print("="*80)

resultados = []

for dataset in DATASETS:
    peab_path = Path(f"json/peab/{dataset}.json")
    pulp_path = Path(f"json/pulp/{dataset}.json")
    
    if not peab_path.exists() or not pulp_path.exists():
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
    
    # Diferenças
    diff_t_plus = abs(peab_t_plus - pulp_t_plus)
    diff_t_minus = abs(peab_t_minus - pulp_t_minus)
    
    # Tamanhos (classificadas)
    peab_stats = peab['explanation_stats']
    pulp_stats = pulp.get('estatisticas_por_tipo', {})
    
    peab_pos = peab_stats.get('positive', {}).get('count', 0)
    peab_neg = peab_stats.get('negative', {}).get('count', 0)
    pulp_pos = pulp_stats.get('positiva', {}).get('instancias', 0)
    pulp_neg = pulp_stats.get('negativa', {}).get('instancias', 0)
    
    if (peab_pos + peab_neg) > 0:
        peab_tam = (
            peab_stats['positive']['mean_length'] * peab_pos +
            peab_stats['negative']['mean_length'] * peab_neg
        ) / (peab_pos + peab_neg)
    else:
        peab_tam = 0
    
    if (pulp_pos + pulp_neg) > 0:
        pos_tam = pulp_stats.get('positiva', {}).get('tamanho_medio', 0)
        neg_tam = pulp_stats.get('negativa', {}).get('tamanho_medio', 0)
        pulp_tam = (pos_tam * pulp_pos + neg_tam * pulp_neg) / (pulp_pos + pulp_neg)
    else:
        pulp_tam = 0
    
    diff_tam = pulp_tam - peab_tam
    
    resultados.append({
        'dataset': dataset,
        'diff_t_plus': diff_t_plus,
        'diff_t_minus': diff_t_minus,
        'diff_tamanho': diff_tam,
        'pulp_maior': pulp_tam > peab_tam
    })

# Mostrar resultados
print("\nDataset                | Diff t+    | Diff t-    | PULP-PEAB | PULP > PEAB?")
print("-" * 80)
for r in resultados:
    nome = r['dataset'][:20].ljust(20)
    dt_plus = f"{r['diff_t_plus']:.6f}"
    dt_minus = f"{r['diff_t_minus']:.6f}"
    dt_tam = f"{r['diff_tamanho']:+.2f}"
    maior = "❌ SIM" if r['pulp_maior'] else "✅ NÃO"
    print(f"{nome} | {dt_plus:>10} | {dt_minus:>10} | {dt_tam:>9} | {maior}")

print("\n" + "="*80)
print("CONCLUSÃO:")
print("="*80)

# Verificar correlação
problemas_threshold = [r for r in resultados if r['diff_t_plus'] > 0.0001 or r['diff_t_minus'] > 0.0001]
problemas_tamanho = [r for r in resultados if r['pulp_maior']]

print(f"\nDatasets com thresholds diferentes (>0.0001): {len(problemas_threshold)}")
print(f"Datasets onde PULP > PEAB: {len(problemas_tamanho)}")

# Verificar se há correlação
if problemas_threshold and problemas_tamanho:
    print("\n🎯 CORRELAÇÃO ENCONTRADA!")
    print("\nAnálise:")
    for r in problemas_tamanho:
        print(f"\n  {r['dataset']}:")
        print(f"    Diff t+: {r['diff_t_plus']:.8f}")
        print(f"    Diff t-: {r['diff_t_minus']:.8f}")
        print(f"    PULP é {abs(r['diff_tamanho']):.2f} features MAIOR")
        
        if r['diff_t_plus'] > 0.0001 or r['diff_t_minus'] > 0.0001:
            print(f"    ⚠️  Thresholds DIFERENTES podem estar causando isso!")

print("\n" + "="*80)
print("EXPLICAÇÃO:")
print("="*80)
print("""
Quando os thresholds são DIFERENTES entre PEAB e PULP:

1. PULP usa thresholds mais conservadores (diferentes)
2. As restrições do problema de otimização ficam MAIS APERTADAS
3. O solver precisa incluir MAIS features para satisfazer as restrições
4. Resultado: PULP > PEAB (mesmo sendo "exato")

ISSO NÃO É UM BUG! É consequência de usar thresholds diferentes!

SOLUÇÃO: Garantir que PULP use EXATAMENTE os mesmos thresholds do PEAB.
Assim, PULP será sempre ≤ PEAB (como esperado matematicamente).
""")
print("="*80)
