"""
Cálculo de tempo de execução: Anchor + MNIST com subsample_size = 0.05
"""

print("="*80)
print("CÁLCULO DE TEMPO: Anchor + MNIST (subsample_size = 0.05)")
print("="*80)

# Dados do MNIST
print("\n📊 DADOS DO MNIST:")
print("-"*80)

mnist_total_instances = 2000  # Após filtrar para 2 dígitos (3 vs 8)
mnist_features = 784  # Pixels (28x28)
subsample_size = 0.05  # 5% do dataset

# Cálculo do número de instâncias
instances_after_subsample = int(mnist_total_instances * subsample_size)
print(f"  Total de instâncias no MNIST:     {mnist_total_instances}")
print(f"  Subsample (5%):                   {instances_after_subsample} instâncias")
print(f"  Features (pixels):                {mnist_features}")

# Com test_size = 0.3, apenas 30% vai para teste
test_size = 0.3
instances_test = int(instances_after_subsample * test_size)
instances_train = instances_after_subsample - instances_test

print(f"\n  Após split train/test (30% teste):")
print(f"    Treino:                         {instances_train} instâncias")
print(f"    Teste:                          {instances_test} instâncias")

# Tempo estimado por instância
print("\n⏱️  TEMPO ESTIMADO POR INSTÂNCIA:")
print("-"*80)

tempo_por_instancia = 24.0  # segundos (baseado em testes anteriores)
print(f"  Anchor no MNIST:                  ~{tempo_por_instancia}s por instância")
print(f"  (Com otimizações: threshold=0.90, batch_size=200, beam_size=2)")

# Cálculo do tempo total
print("\n🚀 TEMPO TOTAL ESTIMADO:")
print("-"*80)

tempo_total_segundos = instances_test * tempo_por_instancia
tempo_total_minutos = tempo_total_segundos / 60
tempo_total_horas = tempo_total_minutos / 60

print(f"\n  Instâncias de teste: {instances_test}")
print(f"  Tempo por instância: {tempo_por_instancia}s")
print(f"  {'='*40}")
print(f"  Tempo total:         {tempo_total_segundos:.0f} segundos")
print(f"                       {tempo_total_minutos:.1f} minutos")
if tempo_total_horas >= 1:
    print(f"                       {tempo_total_horas:.2f} horas")

# Comparação com outros cenários
print("\n📊 COMPARAÇÃO COM OUTROS CENÁRIOS:")
print("-"*80)

scenarios = [
    ("Subsample 0.05 (atual)", 0.05),
    ("Subsample 0.10 (dobro)", 0.10),
    ("Subsample 0.20 (4x)", 0.20),
    ("Limite manual 200", None),  # Limite fixo
    ("Dataset completo", 1.0),
]

print(f"{'Cenário':<30} {'Instâncias':>12} {'Tempo Total':>15}")
print("-"*60)

for nome, subsample in scenarios:
    if subsample is None:
        # Limite fixo de 200
        inst = min(200, mnist_total_instances)
        inst_test = int(inst * test_size)
    else:
        inst = int(mnist_total_instances * subsample)
        inst_test = int(inst * test_size)
    
    tempo_s = inst_test * tempo_por_instancia
    tempo_m = tempo_s / 60
    
    if tempo_m < 60:
        tempo_str = f"{tempo_m:.1f} min"
    else:
        tempo_str = f"{tempo_m/60:.1f}h"
    
    print(f"{nome:<30} {inst_test:>12} {tempo_str:>15}")

# Análise de viabilidade
print("\n✅ ANÁLISE DE VIABILIDADE:")
print("-"*80)

if tempo_total_minutos <= 10:
    status = "✅ RÁPIDO"
    recomendacao = "Execução imediata viável"
elif tempo_total_minutos <= 30:
    status = "✅ VIÁVEL"
    recomendacao = "Pode executar normalmente"
elif tempo_total_minutos <= 60:
    status = "⚠️  MODERADO"
    recomendacao = "Reserve tempo para execução"
elif tempo_total_minutos <= 120:
    status = "⚠️  DEMORADO"
    recomendacao = "Execute em período livre"
else:
    status = "❌ MUITO LONGO"
    recomendacao = "Considere reduzir subsample ou executar overnight"

print(f"\n  Status: {status}")
print(f"  Tempo: ~{tempo_total_minutos:.1f} minutos ({tempo_total_segundos:.0f}s)")
print(f"  Recomendação: {recomendacao}")

# Comandos para executar
print("\n🔧 COMO EXECUTAR:")
print("-"*80)
print("""
1. O subsample_size já está configurado em peab.py:
   MNIST_CONFIG = {
       'subsample_size': 0.05  # ← Já está configurado!
   }

2. Execute o Anchor normalmente:
   python anchor.py
   (escolher MNIST no menu)

3. Ou use o script do menu:
   from data.datasets import set_mnist_options
   set_mnist_options('raw', (3, 8))
   # Depois execute anchor.py

4. O shared_training.py vai automaticamente:
   - Fazer subsample de 5% (100 instâncias)
   - Split 70/30 train/test (30 instâncias de teste)
   - Passar para o Anchor explicar apenas as 30 instâncias
""")

# Detalhes adicionais
print("\n📝 OBSERVAÇÕES IMPORTANTES:")
print("-"*80)
print(f"""
• O subsample acontece ANTES do split train/test
• Anchor explica apenas instâncias de TESTE (não treino)
• Com subsample 0.05: {instances_after_subsample} total → {instances_test} teste
• Tempo pode variar ±20% dependendo da complexidade das instâncias
• Barra de progresso mostrará tempo restante durante execução
""")

print("\n" + "="*80)
print("RESUMO FINAL")
print("="*80)
print(f"\n  ✓ Subsample 0.05 = {instances_test} instâncias de teste")
print(f"  ✓ Tempo estimado: ~{tempo_total_minutos:.1f} minutos")
print(f"  ✓ Status: {status}")
print(f"  ✓ {recomendacao}")
print("\n" + "="*80)
