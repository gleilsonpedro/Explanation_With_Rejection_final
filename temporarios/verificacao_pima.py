"""
Verificação para PIMA - onde houve mudança de ~46ms para ~3.5ms no PEAB  
"""
import json
import numpy as np

print("=" * 100)
print("DATASET: PIMA INDIANS - Onde houve GRANDE MUDANÇA no PEAB")
print("=" * 100)

# ===== PEAB (MINABRO) =====
print("\n" + "-" * 100)
print("MÉTODO: PEAB (MINABRO)")
print("-" * 100)

with open("json/peab/pima_indians_diabetes.json") as f:
    peab_data = json.load(f)

# 1. JSON AGREGADO (o que gerar_tabelas_analise.py usava)
comp_time = peab_data.get("computation_time", {})
pos_time_agg = comp_time.get("positive", None)
neg_time_agg = comp_time.get("negative", None)
rej_time_agg = comp_time.get("rejected", None)

print("\n1. JSON computation_time AGREGADO (usado por gerar_tabelas_analise.py - ANTIGO):")
print(f"   - positive: {pos_time_agg:.6f} s = {pos_time_agg*1000:.2f} ms")
print(f"   - negative: {neg_time_agg:.6f} s = {neg_time_agg*1000:.2f} ms")
print(f"   - rejected: {rej_time_agg:.6f} s = {rej_time_agg*1000:.2f} ms")

# Calcular média ponderada (como o script antigo fazia)
stats = peab_data.get("explanation_stats", {})
pos_count = stats.get("positive", {}).get("count", 0)
neg_count = stats.get("negative", {}).get("count", 0)

print(f"\n   Contagens: {pos_count} positivas, {neg_count} negativas")

if pos_count + neg_count > 0:
    classif_time_agg = (pos_time_agg * pos_count + neg_time_agg * neg_count) / (pos_count + neg_count)
    print(f"\n   CÁLCULO ANTIGO (média ponderada):")
    print(f"   classif = ({pos_time_agg:.6f} * {pos_count} + {neg_time_agg:.6f} * {neg_count}) / {pos_count + neg_count}")
    print(f"   classif = {classif_time_agg:.6f} s = {classif_time_agg*1000:.2f} ms")
    print(f"   rejected = {rej_time_agg*1000:.2f} ms")

# 2. JSON PER_INSTANCE (o que gerar_tabelas_mnist.py usa - NOVO)
per_instance = peab_data.get("per_instance", [])

classif_times = [p.get("computation_time", 0) for p in per_instance if not p.get("rejected", False)]
rej_times = [p.get("computation_time", 0) for p in per_instance if p.get("rejected", False)]

print(f"\n2. JSON per_instance (usado por gerar_tabelas_mnist.py - NOVO):")
print(f"   Total de instâncias: {len(per_instance)}")
print(f"   Classificadas: {len(classif_times)} instâncias")
print(f"   Rejeitadas: {len(rej_times)} instâncias")

if classif_times:
    classif_mean = np.mean(classif_times) * 1000
    classif_std = np.std(classif_times, ddof=1) * 1000 if len(classif_times) > 1 else 0.0
    print(f"\n   CÁLCULO NOVO (média direta):")
    print(f"   - Classificadas: {classif_mean:.2f} ± {classif_std:.2f} ms")
    print(f"     Range: {min(classif_times)*1000:.2f} - {max(classif_times)*1000:.2f} ms")
    print(f"     Primeiras 10: {[round(t*1000, 2) for t in classif_times[:10]]}")

if rej_times:
    rej_mean = np.mean(rej_times) * 1000
    rej_std = np.std(rej_times, ddof=1) * 1000 if len(rej_times) > 1 else 0.0
    print(f"   - Rejeitadas: {rej_mean:.2f} ± {rej_std:.2f} ms")
    print(f"     Range: {min(rej_times)*1000:.2f} - {max(rej_times)*1000:.2f} ms")
    print(f"     Primeiras 10: {[round(t*1000, 2) for t in rej_times[:10]]}")

# 3. COMPARAÇÃO DETALHADA
print("\n" + "!" * 100)
print("ANÁLISE DA DISCREPÂNCIA")
print("!" * 100)
print(f"\nCLASSIFICADAS:")
print(f"  Valor AGREGADO (antigo): {classif_time_agg*1000:.2f} ms")
print(f"  Valor PER_INSTANCE (novo): {classif_mean:.2f} ms")
print(f"  Diferença: {abs(classif_time_agg*1000 - classif_mean):.2f} ms")
print(f"  Mudança: {((classif_mean - classif_time_agg*1000) / (classif_time_agg*1000)) * 100:.1f}%")

print(f"\nREJEITADAS:")
print(f"  Valor AGREGADO (antigo): {rej_time_agg*1000:.2f} ms")
print(f"  Valor PER_INSTANCE (novo): {rej_mean:.2f} ms")
print(f"  Diferença: {abs(rej_time_agg*1000 - rej_mean):.2f} ms")
print(f"  Mudança: {((rej_mean - rej_time_agg*1000) / (rej_time_agg*1000)) * 100:.1f}%")

# 4. VERIFICAR SE OS VALORES AGREGADOS ESTAVAM CORRETOS
print("\n" + "!" * 100)
print("VERIFICAÇÃO: Os valores agregados batem com a média real dos per_instance?")
print("!" * 100)

# Calcular média REAL de todas as instâncias positivas e negativas
pos_times_real = []
neg_times_real = []

for p in per_instance:
    if not p.get("rejected", False):
        pred = p.get("prediction", -1)
        true_label = p.get("true_label", -1)
        
        # Positivo = predição correta
        if pred == true_label:
            pos_times_real.append(p.get("computation_time", 0))
        else:
            neg_times_real.append(p.get("computation_time", 0))

if pos_times_real:
    pos_mean_real = np.mean(pos_times_real)
    print(f"\nPOSITIVAS (predição correta):")
    print(f"  JSON agregado diz: {pos_time_agg:.6f} s = {pos_time_agg*1000:.2f} ms")
    print(f"  Média REAL do per_instance: {pos_mean_real:.6f} s = {pos_mean_real*1000:.2f} ms")
    print(f"  Diferença: {abs(pos_time_agg - pos_mean_real)*1000:.2f} ms")
    print(f"  BATEM? {abs(pos_time_agg - pos_mean_real) < 0.000001}")

if neg_times_real:
    neg_mean_real = np.mean(neg_times_real)
    print(f"\nNEGATIVAS (predição incorreta):")
    print(f"  JSON agregado diz: {neg_time_agg:.6f} s = {neg_time_agg*1000:.2f} ms")
    print(f"  Média REAL do per_instance: {neg_mean_real:.6f} s = {neg_mean_real*1000:.2f} ms")
    print(f"  Diferença: {abs(neg_time_agg - neg_mean_real)*1000:.2f} ms")
    print(f"  BATEM? {abs(neg_time_agg - neg_mean_real) < 0.000001}")

if rej_times:
    rej_mean_real = np.mean(rej_times)
    print(f"\nREJEITADAS:")
    print(f"  JSON agregado diz: {rej_time_agg:.6f} s = {rej_time_agg*1000:.2f} ms")
    print(f"  Média REAL do per_instance: {rej_mean_real:.6f} s = {rej_mean_real*1000:.2f} ms")
    print(f"  Diferença: {abs(rej_time_agg - rej_mean_real)*1000:.2f} ms")
    print(f"  BATEM? {abs(rej_time_agg - rej_mean_real) < 0.000001}")

print("\n" + "=" * 100)
print("CONCLUSÃO")
print("=" * 100)
print("""
Se os valores agregados NÃO batem com a média real do per_instance:
→ Os valores agregados foram calculados ERRADOS na hora de salvar o JSON
→ A tabela ANTIGA (que usava os valores agregados) estava ERRADA
→ A tabela NOVA (que recalcula do per_instance) está CERTA

Se os valores agregados BATEM:
→ Então o problema é na FORMA DE CALCULAR a média ponderada
→ Precisa investigar mais profundamente
""")
