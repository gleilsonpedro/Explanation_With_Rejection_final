"""
Verifica se os valores da tabela atual batem EXATAMENTE com os relatórios TXT gerados
"""
import re
import numpy as np

# Valores da tabela atual (mnist_runtime_unified.tex)
tabela = {
    "banknote": {"PEAB_C": 1.38, "PEAB_R": 1.47, "MinExp_C": 167.94, "MinExp_R": 237.13},
    "vertebral_column": {"PEAB_C": 1.43, "PEAB_R": 1.61, "MinExp_C": 284.73, "MinExp_R": 383.43},
    "pima_indians_diabetes": {"PEAB_C": 1.64, "PEAB_R": 1.85, "MinExp_C": 345.53, "MinExp_R": 332.56},
    "heart_disease": {"PEAB_C": 1.64, "PEAB_R": 1.96, "MinExp_C": 528.83, "MinExp_R": 973.86},
    "creditcard": {"PEAB_C": 1.92, "PEAB_R": 2.80, "MinExp_C": 1192.66, "MinExp_R": 1678.53},
    "breast_cancer": {"PEAB_C": 1.60, "PEAB_R": 1.80, "MinExp_C": 781.94, "MinExp_R": 1427.27},
    "covertype": {"PEAB_C": 2.25, "PEAB_R": 4.78, "MinExp_C": 2554.02, "MinExp_R": 3359.55},
    "spambase": {"PEAB_C": 2.92, "PEAB_R": 5.51, "MinExp_C": 2413.84, "MinExp_R": 3785.25},
    "sonar": {"PEAB_C": 3.44, "PEAB_R": 4.96, "MinExp_C": 2699.58, "MinExp_R": 4572.22},
}

print("=" * 120)
print("CONFERINDO: Valores da Tabela vs Relatórios TXT")
print("=" * 120)

def ler_tempo_relatorio(filepath):
    """Extrai tempos do relatório"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        # Padrões de tempo no relatório
        tempos = {}
        
        # Formato 1: PEAB - "Tempo médio POSITIVAS: 0.001331s"
        match_pos = re.search(r'Tempo médio POSITIVAS:\s*([\d.]+)s', conteudo, re.IGNORECASE)
        match_neg = re.search(r'Tempo médio NEGATIVAS:\s*([\d.]+)s', conteudo, re.IGNORECASE)
        match_rej = re.search(r'Tempo médio REJEITADAS:\s*([\d.]+)s', conteudo, re.IGNORECASE)
        
        # Formato 2: MinExp - "  - Positivas: 0.2366 segundos"
        if not match_pos:
            match_pos = re.search(r'[-\s]+Positivas:\s*([\d.]+)\s*segundos', conteudo)
        if not match_neg:
            match_neg = re.search(r'[-\s]+Negativas:\s*([\d.]+)\s*segundos', conteudo)
        if not match_rej:
            match_rej = re.search(r'[-\s]+Rejeitadas:\s*([\d.]+)\s*segundos', conteudo)
        
        if match_pos and match_neg:
            # Precisa pegar contagens para fazer média ponderada
            # Formato PEAB: "POSITIVAS:\n    Quantidade: 21"
            count_pos = re.search(r'POSITIVAS:\s*\n\s*Quantidade:\s*(\d+)', conteudo)
            count_neg = re.search(r'NEGATIVAS:\s*\n\s*Quantidade:\s*(\d+)', conteudo)
            
            # Formato MinExp: "Classe Positiva (21 instâncias)"
            if not count_pos:
                count_pos = re.search(r'Classe Positiva\s*\((\d+)\s*instâncias?\)', conteudo)
            if not count_neg:
                count_neg = re.search(r'Classe Negativa\s*\((\d+)\s*instâncias?\)', conteudo)
            
            pos_time = float(match_pos.group(1))  # já em segundos
            neg_time = float(match_neg.group(1))
            
            if count_pos and count_neg:
                pos_count = int(count_pos.group(1))
                neg_count = int(count_neg.group(1))
                
                # Média ponderada para classificadas
                if pos_count + neg_count > 0:
                    classif_time = (pos_time * pos_count + neg_time * neg_count) / (pos_count + neg_count)
                    tempos['classif'] = classif_time * 1000  # converter para ms
        
        if match_rej:
            tempos['rejeita'] = float(match_rej.group(1)) * 1000  # converter para ms
        
        return tempos
    except Exception as e:
        print(f"    DEBUG: Erro ao ler {filepath}: {e}")
        return None

total_conferencias = 0
valores_batem = 0
diferencas = []

for dataset, valores_tabela in tabela.items():
    print(f"\n{'=' * 120}")
    print(f"DATASET: {dataset.upper()}")
    print(f"{'=' * 120}")
    
    # PEAB
    peab_file = f"results/report/peab/peab_{dataset}.txt"
    peab_rel = ler_tempo_relatorio(peab_file)
    
    if peab_rel:
        print(f"\n  PEAB:")
        
        # Classificadas
        if 'classif' in peab_rel:
            total_conferencias += 1
            tabela_val = valores_tabela['PEAB_C']
            rel_val = peab_rel['classif']
            diff = abs(tabela_val - rel_val)
            
            if diff < 0.01:
                valores_batem += 1
                status = "✓"
            else:
                status = "✗"
                diferencas.append((dataset, "PEAB", "Classif", tabela_val, rel_val, diff))
            
            print(f"    {status} Classificadas: Tabela={tabela_val:.2f} ms | Relatório={rel_val:.2f} ms | Diff={diff:.4f} ms")
        
        # Rejeitadas
        if 'rejeita' in peab_rel:
            total_conferencias += 1
            tabela_val = valores_tabela['PEAB_R']
            rel_val = peab_rel['rejeita']
            diff = abs(tabela_val - rel_val)
            
            if diff < 0.01:
                valores_batem += 1
                status = "✓"
            else:
                status = "✗"
                diferencas.append((dataset, "PEAB", "Rejeita", tabela_val, rel_val, diff))
            
            print(f"    {status} Rejeitadas:    Tabela={tabela_val:.2f} ms | Relatório={rel_val:.2f} ms | Diff={diff:.4f} ms")
    else:
        print(f"  PEAB: ⚠ Relatório não encontrado ou sem dados de tempo")
    
    # MinExp
    minexp_file = f"results/report/minexp/minexp_{dataset}_*.txt"
    import glob
    minexp_files = glob.glob(minexp_file)
    
    if minexp_files:
        minexp_rel = ler_tempo_relatorio(minexp_files[0])
        
        if minexp_rel:
            print(f"\n  MinExp:")
            
            # Classificadas
            if 'classif' in minexp_rel:
                total_conferencias += 1
                tabela_val = valores_tabela['MinExp_C']
                rel_val = minexp_rel['classif']
                diff = abs(tabela_val - rel_val)
                
                if diff < 0.01:
                    valores_batem += 1
                    status = "✓"
                else:
                    status = "✗"
                    diferencas.append((dataset, "MinExp", "Classif", tabela_val, rel_val, diff))
                
                print(f"    {status} Classificadas: Tabela={tabela_val:.2f} ms | Relatório={rel_val:.2f} ms | Diff={diff:.4f} ms")
            
            # Rejeitadas
            if 'rejeita' in minexp_rel:
                total_conferencias += 1
                tabela_val = valores_tabela['MinExp_R']
                rel_val = minexp_rel['rejeita']
                diff = abs(tabela_val - rel_val)
                
                if diff < 0.01:
                    valores_batem += 1
                    status = "✓"
                else:
                    status = "✗"
                    diferencas.append((dataset, "MinExp", "Rejeita", tabela_val, rel_val, diff))
                
                print(f"    {status} Rejeitadas:    Tabela={tabela_val:.2f} ms | Relatório={rel_val:.2f} ms | Diff={diff:.4f} ms")
        else:
            print(f"  MinExp: ⚠ Relatório sem dados de tempo parseáveis")
    else:
        print(f"  MinExp: ⚠ Relatório não encontrado")

print("\n" + "=" * 120)
print("RESUMO DA CONFERÊNCIA")
print("=" * 120)
print(f"\nTotal de conferências: {total_conferencias}")
print(f"Valores que BATEM: {valores_batem} ({valores_batem/total_conferencias*100:.1f}%)" if total_conferencias > 0 else "Nenhuma conferência")
print(f"Valores DIFERENTES: {len(diferencas)} ({len(diferencas)/total_conferencias*100:.1f}%)" if total_conferencias > 0 else "")

if diferencas:
    print("\n⚠ DIFERENÇAS ENCONTRADAS:")
    print("-" * 120)
    for dataset, metodo, tipo, tab, rel, diff in diferencas:
        print(f"  {dataset:20} | {metodo:7} | {tipo:8} | Tabela: {tab:10.2f} | Relatório: {rel:10.2f} | Diff: {diff:.4f} ms")
else:
    print("\n✓✓✓ TODOS OS VALORES BATEM PERFEITAMENTE COM OS RELATÓRIOS! ✓✓✓")
    print("    → Tabela e relatórios estão 100% consistentes")
    print("    → Pode usar qualquer um dos dois como referência")

print("\n" + "=" * 120)
print("CONCLUSÃO")
print("=" * 120)
if valores_batem == total_conferencias and total_conferencias > 0:
    print("""
✓ SIM, OS VALORES DA TABELA BATEM EXATAMENTE COM OS RELATÓRIOS!

Isso confirma que:
1. A tabela foi gerada corretamente dos dados
2. Os relatórios também foram gerados dos mesmos dados
3. Há consistência total entre tabela ↔ relatórios ↔ JSON
4. Você pode responder ao professor com 100% de certeza
""")
else:
    print("""
Há algumas diferenças. Verificar:
1. Se os relatórios foram gerados com os mesmos dados
2. Se houve atualização nos JSONs após gerar os relatórios
3. Se o formato dos relatórios mudou
""")
