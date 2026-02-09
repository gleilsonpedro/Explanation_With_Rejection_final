"""
Verifica se TODOS os métodos (PEAB, Anchor, MinExp) estavam usando valores agregados
E qual teve maior discrepância
"""
import json
import numpy as np

datasets = [
    ("Banknote", "banknote.json"),
    ("Vertebral", "vertebral_column.json"),
    ("Pima", "pima_indians_diabetes.json"),
]

metodos = {
    "PEAB": "peab",
    "Anchors": "anchor", 
    "MinExp": "minexp"
}

print("=" * 120)
print("COMPARAÇÃO: Valores AGREGADOS (antigo) vs PER_INSTANCE (novo) - TODOS OS MÉTODOS")
print("=" * 120)

todas_diferencas = {"PEAB": [], "Anchors": [], "MinExp": []}

for dataset_nome, arquivo in datasets:
    print(f"\n{'=' * 120}")
    print(f"DATASET: {dataset_nome}")
    print(f"{'=' * 120}")
    
    for metodo_nome, metodo_pasta in metodos.items():
        print(f"\n  {metodo_nome}:")
        
        try:
            json_path = f"json/{metodo_pasta}/{arquivo}"
            with open(json_path) as f:
                data = json.load(f)
            
            # VALORES AGREGADOS (o que o script ANTIGO usava)
            comp_time = data.get("computation_time", {})
            pos_agg = comp_time.get("positive")
            neg_agg = comp_time.get("negative")
            rej_agg = comp_time.get("rejected")
            
            # Contagens
            stats = data.get("explanation_stats", {})
            pos_count = stats.get("positive", {}).get("count", 0)
            neg_count = stats.get("negative", {}).get("count", 0)
            
            # Calcular média ponderada COMO O SCRIPT ANTIGO FAZIA
            if pos_agg and neg_agg and (pos_count + neg_count) > 0:
                classif_agregado = (pos_agg * pos_count + neg_agg * neg_count) / (pos_count + neg_count)
                classif_agregado_ms = classif_agregado * 1000
            else:
                classif_agregado_ms = None
            
            rej_agregado_ms = rej_agg * 1000 if rej_agg else None
            
            # VALORES PER_INSTANCE (o que o script NOVO usa)
            per_instance = data.get("per_instance", [])
            classif_times = [p["computation_time"] for p in per_instance if not p.get("rejected")]
            rej_times = [p["computation_time"] for p in per_instance if p.get("rejected")]
            
            classif_per_instance_ms = np.mean(classif_times) * 1000 if classif_times else None
            rej_per_instance_ms = np.mean(rej_times) * 1000 if rej_times else None
            
            # COMPARAR
            if classif_agregado_ms and classif_per_instance_ms:
                diff_c = classif_agregado_ms - classif_per_instance_ms
                pct_c = (diff_c / classif_agregado_ms) * 100
                
                print(f"    Classificadas:")
                print(f"      Agregado (antigo): {classif_agregado_ms:10.2f} ms")
                print(f"      Per_instance (novo): {classif_per_instance_ms:10.2f} ms")
                print(f"      Diferença: {diff_c:+10.2f} ms ({pct_c:+.1f}%)")
                
                todas_diferencas[metodo_nome].append(abs(pct_c))
                
                if abs(diff_c) > 0.1:
                    if diff_c > 0:
                        print(f"      → Agregado estava MAIOR (valores antigos INFLACIONADOS)")
                    else:
                        print(f"      → Agregado estava MENOR")
            
            if rej_agregado_ms and rej_per_instance_ms:
                diff_r = rej_agregado_ms - rej_per_instance_ms
                pct_r = (diff_r / rej_agregado_ms) * 100
                
                print(f"    Rejeitadas:")
                print(f"      Agregado (antigo): {rej_agregado_ms:10.2f} ms")
                print(f"      Per_instance (novo): {rej_per_instance_ms:10.2f} ms")
                print(f"      Diferença: {diff_r:+10.2f} ms ({pct_r:+.1f}%)")
                
                todas_diferencas[metodo_nome].append(abs(pct_r))
                
                if abs(diff_r) > 0.1:
                    if diff_r > 0:
                        print(f"      → Agregado estava MAIOR (valores antigos INFLACIONADOS)")
                    else:
                        print(f"      → Agregado estava MENOR")
        
        except FileNotFoundError:
            print(f"    ⚠ Arquivo não encontrado")
        except Exception as e:
            print(f"    ✗ Erro: {e}")

print("\n" + "=" * 120)
print("RESUMO: Qual método teve MAIOR discrepância?")
print("=" * 120)

for metodo, diferencas in todas_diferencas.items():
    if diferencas:
        media = np.mean(diferencas)
        maxima = max(diferencas)
        print(f"\n{metodo}:")
        print(f"  Diferença média: {media:.2f}%")
        print(f"  Diferença máxima: {maxima:.2f}%")
        print(f"  Total de comparações: {len(diferencas)}")

print("\n" + "=" * 120)
print("CONCLUSÃO")
print("=" * 120)

# Identificar qual teve maior discrepância
medias = {m: np.mean(d) if d else 0 for m, d in todas_diferencas.items()}
maior_discrepancia = max(medias, key=medias.get)

print(f"""
TODOS os 3 métodos (PEAB, Anchors, MinExp) usavam o mesmo cálculo ACUMULATIVO
no script antigo (gerar_tabelas_analise.py):

    classif = (pos_time * pos_count + neg_time * neg_count) / total

Porém, a MAGNITUDE da discrepância foi diferente:

    {maior_discrepancia}: {medias[maior_discrepancia]:.2f}% de diferença média ← MAIOR IMPACTO
    
Isso acontece porque os valores 'positive', 'negative', 'rejected' salvos no JSON
estavam com erros/desatualizações DIFERENTES para cada método.

Para {maior_discrepancia} especificamente, os valores agregados estavam mais distantes
dos valores reais (per_instance).
""")
