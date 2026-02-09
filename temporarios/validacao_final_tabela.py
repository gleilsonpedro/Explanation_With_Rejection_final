"""
VALIDAÇÃO FINAL: Verifica se os valores da tabela atual batem com o cálculo direto do JSON per_instance
Isso prova se os valores atuais estão corretos ou não.
"""
import json
import numpy as np

# Valores da TABELA ATUAL (mnist_runtime_unified.tex)
tabela_atual = {
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

# Mapeamento de nomes de datasets para arquivos JSON
dataset_files = {
    "Banknote": "banknote.json",
    "Vertebral Column": "vertebral_column.json",
    "Pima Indians": "pima_indians_diabetes.json",
    "Heart Disease": "heart_disease.json",
    "Credit Card": "creditcard.json",
    "Breast Cancer": "breast_cancer.json",
    "Covertype": "covertype.json",
    "Spambase": "spambase.json",
    "Sonar": "sonar.json",
    "MNIST (3 vs 8)": "mnist_3_vs_8.json",
}

metodos = {
    "MINABRO": "peab",
    "Anchors": "anchor",
    "AbLinRO": "minexp"
}

print("=" * 130)
print("VALIDAÇÃO FINAL: Tabela Atual vs Cálculo Direto do JSON per_instance")
print("=" * 130)
print()

erros_encontrados = []
valores_corretos = 0
total_verificacoes = 0

for dataset_name, json_file in dataset_files.items():
    print(f"\n{'=' * 130}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 130}")
    
    for metodo_nome, metodo_pasta in metodos.items():
        print(f"\n{'-' * 130}")
        print(f"MÉTODO: {metodo_nome}")
        print(f"{'-' * 130}")
        
        try:
            # Carregar JSON
            json_path = f"json/{metodo_pasta}/{json_file}"
            with open(json_path) as f:
                data = json.load(f)
            
            # Calcular do per_instance
            per_instance = data.get("per_instance", [])
            
            if not per_instance:
                print(f"  ⚠ Sem dados per_instance")
                continue
            
            classif_times = [p.get("computation_time", 0) for p in per_instance if not p.get("rejected", False)]
            rej_times = [p.get("computation_time", 0) for p in per_instance if p.get("rejected", False)]
            
            # Calcular médias
            if classif_times:
                classif_calc = np.mean(classif_times) * 1000  # converter para ms
                classif_tabela = tabela_atual[dataset_name][f"{metodo_nome.split()[0]}_C"]
                diff_c = abs(classif_calc - classif_tabela)
                total_verificacoes += 1
                
                if diff_c < 0.01:  # tolerância de 0.01 ms
                    print(f"  ✓ Classificadas: Tabela={classif_tabela:.2f} ms | Calculado={classif_calc:.2f} ms | Diff={diff_c:.4f} ms")
                    valores_corretos += 1
                else:
                    print(f"  ✗ Classificadas: Tabela={classif_tabela:.2f} ms | Calculado={classif_calc:.2f} ms | Diff={diff_c:.4f} ms ⚠ ERRO!")
                    erros_encontrados.append((dataset_name, metodo_nome, "Classif", classif_tabela, classif_calc))
            
            if rej_times:
                rej_calc = np.mean(rej_times) * 1000  # converter para ms
                rej_tabela = tabela_atual[dataset_name][f"{metodo_nome.split()[0]}_R"]
                diff_r = abs(rej_calc - rej_tabela)
                total_verificacoes += 1
                
                if diff_r < 0.01:  # tolerância de 0.01 ms
                    print(f"  ✓ Rejeitadas:    Tabela={rej_tabela:.2f} ms | Calculado={rej_calc:.2f} ms | Diff={diff_r:.4f} ms")
                    valores_corretos += 1
                else:
                    print(f"  ✗ Rejeitadas:    Tabela={rej_tabela:.2f} ms | Calculado={rej_calc:.2f} ms | Diff={diff_r:.4f} ms ⚠ ERRO!")
                    erros_encontrados.append((dataset_name, metodo_nome, "Rejeita", rej_tabela, rej_calc))
            
        except FileNotFoundError:
            print(f"  ⚠ Arquivo não encontrado: {json_path}")
        except Exception as e:
            print(f"  ✗ Erro ao processar: {e}")

print("\n" + "=" * 130)
print("RESUMO DA VALIDAÇÃO")
print("=" * 130)
print(f"\nTotal de verificações: {total_verificacoes}")
print(f"Valores CORRETOS: {valores_corretos} ({valores_corretos/total_verificacoes*100:.1f}%)")
print(f"Valores INCORRETOS: {len(erros_encontrados)} ({len(erros_encontrados)/total_verificacoes*100:.1f}%)")

if erros_encontrados:
    print("\n⚠ ERROS ENCONTRADOS:")
    print("-" * 130)
    for dataset, metodo, tipo, tabela, calc in erros_encontrados:
        print(f"  {dataset:20} | {metodo:10} | {tipo:8} | Tabela: {tabela:10.2f} | Calculado: {calc:10.2f}")
else:
    print("\n✓ TODOS OS VALORES DA TABELA ESTÃO CORRETOS!")
    print("  → Os valores batem PERFEITAMENTE com o cálculo direto do per_instance")
    print("  → Você pode responder ao professor com segurança que os valores estão corretos")

print("\n" + "=" * 130)
print("CONCLUSÃO PARA O PROFESSOR")
print("=" * 130)
print("""
Se TODOS os valores estão corretos (100%):
├─ Os valores atuais da tabela foram calculados CORRETAMENTE do JSON per_instance
├─ O método de cálculo (gerar_tabelas_mnist.py) está funcionando perfeitamente
└─ Qualquer diferença em relação a tabelas anteriores significa que:
   • Os valores ANTERIORES estavam incorretos (calculados de forma agregada errada)
   • OU os dados JSON foram atualizados/melhorados
   • EM AMBOS OS CASOS: os valores ATUAIS são os CORRETOS

Se houver ERROS:
├─ Alguns valores da tabela NÃO batem com os dados do JSON
├─ Pode ter havido problema na geração da tabela
└─ Precisa investigar e regenerar a tabela
""")
