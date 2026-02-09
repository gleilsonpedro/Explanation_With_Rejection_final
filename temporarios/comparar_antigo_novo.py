"""
Compara valores antigos (comentados no arquivo) vs novos (atuais)
para identificar todas as discrepâncias.
"""

# Valores ANTIGOS (da tabela comentada)
antigos = {
    "Banknote": {"MINABRO_C": 5.6, "MINABRO_R": 40.8, "Anchors_C": 123.8, "Anchors_R": 58.9, "AbLinRO_C": 148.2, "AbLinRO_R": 230.3},
    "Vertebral Column": {"MINABRO_C": 6.2, "MINABRO_R": 42.4, "Anchors_C": 308.3, "Anchors_R": 123.8, "AbLinRO_C": 232.3, "AbLinRO_R": 370.5},
    "Pima Indians": {"MINABRO_C": 6.5, "MINABRO_R": 42.7, "Anchors_C": 330.7, "Anchors_R": 1120.9, "AbLinRO_C": 289.2, "AbLinRO_R": 305.6},
    "Heart Disease": {"MINABRO_C": 4.8, "MINABRO_R": 44.1, "Anchors_C": 86.3, "Anchors_R": 103.0, "AbLinRO_C": 476.2, "AbLinRO_R": 842.0},
    "Credit Card": {"MINABRO_C": 1.7, "MINABRO_R": 40.5, "Anchors_C": 513.4, "Anchors_R": 1555.7, "AbLinRO_C": 1146.3, "AbLinRO_R": 1585.5},
    "Breast Cancer": {"MINABRO_C": 5.1, "MINABRO_R": 44.6, "Anchors_C": 4765.0, "Anchors_R": 3547.5, "AbLinRO_C": 595.2, "AbLinRO_R": 1030.1},
    "Covertype": {"MINABRO_C": 2.2, "MINABRO_R": 46.6, "Anchors_C": 3038.7, "Anchors_R": 4492.0, "AbLinRO_C": 2073.0, "AbLinRO_R": 3607.6},
    "Spambase": {"MINABRO_C": 6.8, "MINABRO_R": 44.7, "Anchors_C": 202.6, "Anchors_R": 577.1, "AbLinRO_C": 2335.6, "AbLinRO_R": 3709.6},
    "Sonar": {"MINABRO_C": 10.7, "MINABRO_R": 48.7, "Anchors_C": 24247.1, "Anchors_R": 5788.6, "AbLinRO_C": 2390.1, "AbLinRO_R": 4785.2},
    "MNIST (3 vs 8)": {"MINABRO_C": 3.9, "MINABRO_R": 51.2, "Anchors_C": 64536.7, "Anchors_R": 81369.7, "AbLinRO_C": 9371.1, "AbLinRO_R": 13493.9},
}

# Valores NOVOS (da tabela atual - apenas a média, ignorando ± desvio)
novos = {
    "Banknote": {"MINABRO_C": 1.38, "MINABRO_R": 1.47, "Anchors_C": 140.89, "Anchors_R": 64.97, "AbLinRO_C": 167.94, "AbLinRO_R": 237.13},
    "Vertebral Column": {"MINABRO_C": 1.43, "MINABRO_R": 1.61, "Anchors_C": 422.66, "Anchors_R": 159.62, "AbLinRO_C": 284.73, "AbLinRO_R": 383.43},
    "Pima Indians": {"MINABRO_C": 1.64, "MINABRO_R": 1.85, "Anchors_C": 501.58, "Anchors_R": 1332.36, "AbLinRO_C": 345.53, "AbLinRO_R": 332.56},
    "Heart Disease": {"MINABRO_C": 1.64, "MINABRO_R": 1.96, "Anchors_C": 1099.30, "Anchors_R": 1174.24, "AbLinRO_C": 528.83, "AbLinRO_R": 973.86},
    "Credit Card": {"MINABRO_C": 1.92, "MINABRO_R": 2.80, "Anchors_C": 189.01, "Anchors_R": 32682.01, "AbLinRO_C": 1192.66, "AbLinRO_R": 1678.53},
    "Breast Cancer": {"MINABRO_C": 1.60, "MINABRO_R": 1.80, "Anchors_C": 6526.21, "Anchors_R": 5003.79, "AbLinRO_C": 781.94, "AbLinRO_R": 1427.27},
    "Covertype": {"MINABRO_C": 2.25, "MINABRO_R": 4.78, "Anchors_C": 34569.58, "Anchors_R": 67035.24, "AbLinRO_C": 2554.02, "AbLinRO_R": 3359.55},
    "Spambase": {"MINABRO_C": 2.92, "MINABRO_R": 5.51, "Anchors_C": 5291.85, "Anchors_R": 14507.73, "AbLinRO_C": 2413.84, "AbLinRO_R": 3785.25},
    "Sonar": {"MINABRO_C": 3.44, "MINABRO_R": 4.96, "Anchors_C": 32436.80, "Anchors_R": 8098.75, "AbLinRO_C": 2699.58, "AbLinRO_R": 4572.22},
    "MNIST (3 vs 8)": {"MINABRO_C": 23.24, "MINABRO_R": 167.26, "Anchors_C": 257871.79, "Anchors_R": 300590.33, "AbLinRO_C": 67574.06, "AbLinRO_R": 64727.33},
}

print("=" * 120)
print("ANÁLISE DE DISCREPÂNCIAS: VALORES ANTIGOS vs NOVOS")
print("=" * 120)
print()
print("Formato: Dataset | Métrica | Antigo → Novo | Diferença (%) | Status")
print("=" * 120)

limiar_mudanca = 10  # % para considerar mudança significativa

mudancas_por_metodo = {
    "MINABRO": [],
    "Anchors": [],
    "AbLinRO": []
}

for dataset in antigos.keys():
    print(f"\n{dataset}:")
    print("-" * 120)
    
    teve_mudanca = False
    
    for metrica in ["MINABRO_C", "MINABRO_R", "Anchors_C", "Anchors_R", "AbLinRO_C", "AbLinRO_R"]:
        antigo = antigos[dataset][metrica]
        novo = novos[dataset][metrica]
        
        # Calcular diferença percentual
        if antigo > 0:
            diff_pct = ((novo - antigo) / antigo) * 100
        else:
            diff_pct = 0 if novo == 0 else 999
        
        metodo = metrica.split("_")[0]
        tipo = "Classif" if metrica.endswith("_C") else "Rejeita"
        
        # Status
        if abs(diff_pct) < limiar_mudanca:
            status = "✓ OK (pequena mudança)"
        elif diff_pct > 0:
            status = f"⚠ AUMENTOU {abs(diff_pct):.1f}%"
        else:
            status = f"⬇ DIMINUIU {abs(diff_pct):.1f}%"
        
        mudou = abs(diff_pct) >= limiar_mudanca
        
        if mudou:
            teve_mudanca = True
            mudancas_por_metodo[metodo].append((dataset, tipo, antigo, novo, diff_pct))
            marker = "►►"
        else:
            marker = "  "
        
        print(f"{marker} {metodo:8} {tipo:8} | {antigo:10.2f} → {novo:10.2f} | {diff_pct:+7.1f}% | {status}")
    
    if not teve_mudanca:
        print("  → Sem mudanças significativas")

print("\n" + "=" * 120)
print("RESUMO POR MÉTODO")
print("=" * 120)

for metodo in ["MINABRO", "Anchors", "AbLinRO"]:
    print(f"\n{metodo}:")
    if mudancas_por_metodo[metodo]:
        print(f"  Total de mudanças significativas: {len(mudancas_por_metodo[metodo])}")
        for dataset, tipo, antigo, novo, diff_pct in mudancas_por_metodo[metodo]:
            direcao = "↑" if diff_pct > 0 else "↓"
            print(f"    {direcao} {dataset:20} {tipo:8}: {antigo:10.2f} → {novo:10.2f} ({diff_pct:+.1f}%)")
    else:
        print("  → Nenhuma mudança significativa")

print("\n" + "=" * 120)
print("CONCLUSÃO")
print("=" * 120)

total_mudancas = sum(len(v) for v in mudancas_por_metodo.values())
print(f"\nTotal de mudanças significativas (>{limiar_mudanca}%): {total_mudancas}")

# Identificar padrões
print("\nPadrões identificados:")
if all(len(v) > 0 for v in mudancas_por_metodo.values()):
    print("  • TODOS os métodos tiveram mudanças")
elif any(len(v) > 0 for v in mudancas_por_metodo.values()):
    metodos_mudaram = [k for k, v in mudancas_por_metodo.items() if len(v) > 0]
    print(f"  • Apenas {', '.join(metodos_mudaram)} tiveram mudanças")
else:
    print("  • Nenhum método teve mudanças significativas")

# Verificar se afeta todos os datasets
datasets_afetados = set()
for mudancas in mudancas_por_metodo.values():
    for dataset, _, _, _, _ in mudancas:
        datasets_afetados.add(dataset)

if len(datasets_afetados) == len(antigos):
    print("  • TODOS os datasets foram afetados")
elif len(datasets_afetados) > 0:
    print(f"  • Apenas {len(datasets_afetados)} datasets afetados: {', '.join(sorted(datasets_afetados))}")
