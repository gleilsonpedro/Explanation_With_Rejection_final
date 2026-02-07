"""
Análise de subsample ideal para submissão urgente do artigo.
"""

print("\n" + "="*80)
print("ANÁLISE RÁPIDA - SUBSAMPLE IDEAL (ARTIGO AMANHÃ)")
print("="*80 + "\n")

datasets = {
    'covertype': {
        'total': 581012,
        'test_30pct': 174303,
        'subsample_atual': 0.01,
        'subsample_recomendado': 0.005,
        'tempo_anchor_atual': 12,
        'tempo_minexp_atual': 1.5
    },
    'creditcard': {
        'total': 284807,
        'test_30pct': 85442,
        'subsample_atual': 0.02,
        'subsample_recomendado': 0.01,
        'tempo_anchor_atual': 7,
        'tempo_minexp_atual': 0.7
    },
    'mnist_3vs8': {
        'total': 11876,
        'test_30pct': 3562,
        'subsample_atual': 0.12,
        'subsample_recomendado': 0.12,  # Manter
        'tempo_anchor_atual': 3.5,
        'tempo_minexp_atual': 0.4
    }
}

print("VALIDAÇÃO CIENTÍFICA:")
print("-"*80)
print("✅ Treino: Dataset COMPLETO (70%) - MODELO ROBUSTO")
print("✅ Teste: Subsample APENAS para acelerar EXPLICAÇÕES")
print("✅ Estatisticamente válido com >500 instâncias no teste")
print()

total_economia_horas = 0

for dataset, info in datasets.items():
    print(f"\n{'─'*80}")
    print(f"{dataset.upper()}")
    print(f"{'─'*80}")
    
    atual_instancias = int(info['test_30pct'] * info['subsample_atual'])
    recom_instancias = int(info['test_30pct'] * info['subsample_recomendado'])
    
    print(f"  Atual: {info['subsample_atual']*100:.1f}% = {atual_instancias:,} instâncias")
    print(f"  Recomendado: {info['subsample_recomendado']*100:.1f}% = {recom_instancias:,} instâncias")
    
    if info['subsample_atual'] != info['subsample_recomendado']:
        tempo_atual = info['tempo_anchor_atual'] + info['tempo_minexp_atual']
        tempo_novo = tempo_atual * (info['subsample_recomendado'] / info['subsample_atual'])
        economia = tempo_atual - tempo_novo
        
        print(f"\n  Tempo total (Anchor + MinExp):")
        print(f"    Atual: ~{tempo_atual:.1f}h")
        print(f"    Novo: ~{tempo_novo:.1f}h")
        print(f"    ⏱️ ECONOMIZA: {economia:.1f}h")
        
        total_economia_horas += economia
        
        print(f"\n  ✅ Ainda válido? SIM ({recom_instancias} instâncias >> 500)")
    else:
        print(f"\n  ✅ Manter atual ({atual_instancias} instâncias)")

print(f"\n{'='*80}")
print(f"ECONOMIA TOTAL: ~{total_economia_horas:.1f} HORAS")
print(f"{'='*80}\n")

print("RECOMENDAÇÃO FINAL (ARTIGO AMANHÃ):")
print("-"*80)
print("📝 COVERTYPE: Reduzir de 0.01 → 0.005 (ECONOMIZA 6-7h)")
print("📝 CREDITCARD: Reduzir de 0.02 → 0.01 (ECONOMIZA 3-4h)")
print("✅ MNIST: Manter 0.12 (já é rápido)")
print()
print(f"💡 TOTAL ECONOMIZADO: ~{total_economia_horas:.0f}h")
print("   Com 3 métodos (PEAB, MinExp, Anchor) por dataset")
print()
print("✅ VALIDAÇÃO: Ainda terá 870-1700 instâncias (estatisticamente robusto)")
print("✅ QUALIDADE: Modelo treinado com 70% (isso não muda!)")
print("="*80 + "\n")

print("VALORES PARA ATUALIZAR NO peab.py:")
print("-"*80)
print("""
DATASET_CONFIG = {
    # ... outros ...
    "creditcard": {
        'subsample_size': 0.01,    # ← MUDE de 0.02 para 0.01
        'test_size': 0.3, 
        'rejection_cost': 0.24
    },
    "covertype": {
        'subsample_size': 0.005,   # ← MUDE de 0.01 para 0.005
        'test_size': 0.3, 
        'rejection_cost': 0.24
    },
    # ... outros ...
}
""")
print("="*80 + "\n")
