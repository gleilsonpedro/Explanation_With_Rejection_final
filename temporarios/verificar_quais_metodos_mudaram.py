"""
CRÍTICO: Comparar se TODOS os métodos mudaram ou só o PEAB
Para responder ao professor que vai perguntar: "Por que só PEAB mudou?"
"""

# Tabela ANTIGA (runtime_unified_with_std.tex - linha 20)
# Banknote & 1.4 ± 0.3 & 1.4 ± 0.4 & 123.8 & 58.9 & 144.3 ± 4.1 & 211.1 ± 4.5

antigos = {
    "Banknote": {
        "PEAB_C": 1.4, "PEAB_R": 1.4,
        "Anchors_C": 123.8, "Anchors_R": 58.9,
        "MinExp_C": 144.3, "MinExp_R": 211.1
    },
    "Vertebral Column": {
        "PEAB_C": 1.6, "PEAB_R": 1.7,
        "Anchors_C": 308.3, "Anchors_R": 123.8,
        "MinExp_C": 232.3, "MinExp_R": 370.5
    },
    "Pima Indians": {
        "PEAB_C": 1.5, "PEAB_R": 2.2,
        "Anchors_C": 330.7, "Anchors_R": 1120.9,
        "MinExp_C": 289.2, "MinExp_R": 305.6
    },
}

# Tabela NOVA (atual - mnist_runtime_unified.tex)
novos = {
    "Banknote": {
        "PEAB_C": 1.38, "PEAB_R": 1.47,
        "Anchors_C": 140.89, "Anchors_R": 64.97,
        "MinExp_C": 167.94, "MinExp_R": 237.13
    },
    "Vertebral Column": {
        "PEAB_C": 1.43, "PEAB_R": 1.61,
        "Anchors_C": 422.66, "Anchors_R": 159.62,
        "MinExp_C": 284.73, "MinExp_R": 383.43
    },
    "Pima Indians": {
        "PEAB_C": 1.64, "PEAB_R": 1.85,
        "Anchors_C": 501.58, "Anchors_R": 1332.36,
        "MinExp_C": 345.53, "MinExp_R": 332.56
    },
}

print("=" * 130)
print("ANÁLISE CRÍTICA: QUAIS MÉTODOS MUDARAM?")
print("=" * 130)
print("\nSe o professor perguntar: 'Por que só PEAB mudou?'")
print("Esta análise mostra se os OUTROS métodos também mudaram\n")

mudancas_por_metodo = {"PEAB": [], "Anchors": [], "MinExp": []}

for dataset in antigos.keys():
    print(f"\n{'=' * 130}")
    print(f"{dataset}")
    print(f"{'=' * 130}")
    
    for metodo in ["PEAB", "Anchors", "MinExp"]:
        for tipo in ["C", "R"]:
            key = f"{metodo}_{tipo}"
            antigo = antigos[dataset][key]
            novo = novos[dataset][key]
            
            diff = novo - antigo
            pct = (diff / antigo) * 100 if antigo > 0 else 0
            
            tipo_nome = "Classif" if tipo == "C" else "Rejeita"
            
            if abs(pct) > 5:  # Mudança > 5%
                status = "🔴 MUDOU"
                mudancas_por_metodo[metodo].append((dataset, tipo_nome, pct))
            else:
                status = "✓ Igual"
            
            print(f"  {metodo:8} {tipo_nome:8}: {antigo:8.2f} → {novo:8.2f} | "
                  f"Diff: {diff:+8.2f} ({pct:+6.1f}%) | {status}")

print("\n" + "=" * 130)
print("RESUMO: QUANTAS MUDANÇAS POR MÉTODO?")
print("=" * 130)

for metodo, mudancas in mudancas_por_metodo.items():
    print(f"\n{metodo}:")
    print(f"  Total de mudanças > 5%: {len(mudancas)}")
    if mudancas:
        for dataset, tipo, pct in mudancas:
            direcao = "↑" if pct > 0 else "↓"
            print(f"    {direcao} {dataset:20} {tipo:8}: {pct:+6.1f}%")

print("\n" + "=" * 130)
print("RESPOSTA PARA O PROFESSOR")
print("=" * 130)

todos_mudaram = all(len(m) > 0 for m in mudancas_por_metodo.values())
so_peab = len(mudancas_por_metodo["PEAB"]) > 0 and len(mudancas_por_metodo["Anchors"]) == 0 and len(mudancas_por_metodo["MinExp"]) == 0

if todos_mudaram:
    print("""
✓ TODOS os 3 métodos (PEAB, Anchors, MinExp) MUDARAM!

Resposta ao professor:
"Professor, TODOS os métodos mudaram, não apenas o PEAB:
 - PEAB mudou X vezes
 - Anchors mudou Y vezes  
 - MinExp mudou Z vezes

Todos usavam o mesmo método de cálculo (agregado) e todos foram 
recalculados com o novo método (per_instance)."
""")
elif so_peab:
    print("""
⚠️ SÓ o PEAB mudou! Anchors e MinExp ficaram iguais!

Isso significa que:
1. Os valores agregados de Anchors/MinExp JÁ estavam corretos (batiam com per_instance)
2. Apenas o PEAB tinha valores agregados ERRADOS no JSON antigo
3. Quando regenerou com per_instance, só o PEAB mudou

Resposta ao professor:
"Professor, apenas o PEAB mudou porque apenas os valores agregados 
do PEAB estavam incorretos nos JSONs antigos. Os outros métodos 
(Anchors e MinExp) já tinham valores agregados corretos que batiam 
com os dados per_instance."
""")
else:
    # Análise detalhada
    print(f"""
📊 ANÁLISE MISTA:
- PEAB: {len(mudancas_por_metodo["PEAB"])} mudanças
- Anchors: {len(mudancas_por_metodo["Anchors"])} mudanças
- MinExp: {len(mudancas_por_metodo["MinExp"])} mudanças

Todos os métodos mudaram, mas em proporções diferentes.
""")
