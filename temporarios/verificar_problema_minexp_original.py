import json
import numpy as np

print('=' * 80)
print('VERIFICAÇÃO DO PROBLEMA ORIGINAL DO MINEXP')
print('=' * 80)

print('\nPROBLEMA ORIGINAL:')
print('-' * 80)
print('''
O MinExp estava processando em CHUNKS e dividindo o tempo igualmente:

  for start in range(0, len(idx_array), chunk_size):
      chunk_indices = idx_array[start:start + chunk_size]
      tempo_chunk = time.perf_counter() - start_chunk
      tempo_por_instancia = tempo_chunk / len(chunk_indices)  ← PROBLEMA!
      
  → TODAS as instâncias do chunk recebiam o MESMO tempo
  → Desvio padrão das REJEITADAS = 0.00 (todas instâncias no mesmo chunk)
  
SOLUÇÃO APLICADA:
  Mudou para INSTANCE-BY-INSTANCE (uma por vez):
  
  for idx in idx_array:
      start_instance = time.perf_counter()
      # ... processar instância idx ...
      tempos_individuais[idx] = time.perf_counter() - start_instance
      
  → Cada instância tem seu PRÓPRIO tempo medido
  → Desvio padrão REAL (não zero)
''')

print('\n' + '=' * 80)
print('VERIFICAÇÃO ATUAL - REJEITADAS COM STD > 0?')
print('=' * 80)

datasets = [
    ('banknote', 'Banknote'),
    ('vertebral_column', 'Vertebral Column'),
    ('pima_indians_diabetes', 'Pima Indians'),
    ('heart_disease', 'Heart Disease'),
    ('creditcard', 'Credit Card'),
    ('breast_cancer', 'Breast Cancer'),
    ('covertype', 'Covertype'),
    ('spambase', 'Spambase'),
    ('sonar', 'Sonar'),
    ('mnist', 'MNIST 3 vs 8')
]

print(f'\n{"Dataset":<20} {"Rejeitadas":<12} {"Mean (ms)":<15} {"Std (ms)":<15} {"Status"}')
print('-' * 80)

problemas = []
ok_count = 0

for dataset_file, dataset_nome in datasets:
    try:
        data = json.load(open(f'json/minexp/{dataset_file}.json'))
        per_inst = data.get('per_instance', [])
        
        # Filtrar rejeitadas
        rejeitadas = [inst for inst in per_inst if inst.get('rejected', False)]
        
        if not rejeitadas:
            print(f'{dataset_nome:<20} {"0":<12} {"-":<15} {"-":<15} (sem rejeitadas)')
            continue
        
        # Calcular tempos
        tempos = [inst.get('computation_time', 0) * 1000 for inst in rejeitadas]  # ms
        mean_tempo = np.mean(tempos)
        std_tempo = np.std(tempos, ddof=1) if len(tempos) > 1 else 0
        
        # Verificar se todas têm o mesmo tempo (problema original)
        tempos_unicos = len(set(tempos))
        
        if std_tempo == 0.0:
            status = "⚠️ STD = 0 (PROBLEMA!)"
            problemas.append(dataset_nome)
        elif std_tempo < 0.01:
            status = "⚠️ STD muito baixo"
            problemas.append(dataset_nome)
        elif tempos_unicos == 1:
            status = "⚠️ Todas iguais!"
            problemas.append(dataset_nome)
        else:
            status = f"✓ OK ({tempos_unicos} valores únicos)"
            ok_count += 1
        
        print(f'{dataset_nome:<20} {len(rejeitadas):<12} {mean_tempo:<15.2f} {std_tempo:<15.2f} {status}')
        
    except Exception as e:
        print(f'{dataset_nome:<20} {"ERROR":<12} {"-":<15} {"-":<15} {str(e)[:30]}')

print('\n' + '=' * 80)
print('RESUMO:')
print('=' * 80)
print(f'✓ Datasets OK (std > 0): {ok_count}/10')
print(f'⚠️ Datasets com problema: {len(problemas)}/10')

if problemas:
    print('\nDatasets ainda com STD = 0:')
    for d in problemas:
        print(f'  - {d}')
    print('\n⚠️ AÇÃO: Regenerar esses datasets')
else:
    print('\n✅ PROBLEMA DO MINEXP FOI TOTALMENTE CORRIGIDO!')
    print('   Todos os datasets têm desvio padrão realista.')

print('=' * 80)

# Exemplo detalhado de 1 dataset
print('\nEXEMPLO DETALHADO (Banknote - primeiras 10 rejeitadas):')
print('-' * 80)
try:
    data = json.load(open('json/minexp/banknote.json'))
    per_inst = data.get('per_instance', [])
    rejeitadas = [inst for inst in per_inst if inst.get('rejected', False)][:10]
    
    for i, inst in enumerate(rejeitadas):
        tempo_ms = inst.get('computation_time', 0) * 1000
        print(f'  Rej [{i}]: {tempo_ms:.2f} ms')
    
    tempos_todas = [inst.get('computation_time', 0) * 1000 
                    for inst in per_inst if inst.get('rejected', False)]
    
    if len(tempos_todas) > 1:
        # Ver se há variação
        tempo_min = min(tempos_todas)
        tempo_max = max(tempos_todas)
        tempo_range = tempo_max - tempo_min
        
        print(f'\n  Range: {tempo_min:.2f} - {tempo_max:.2f} ms (Δ = {tempo_range:.2f} ms)')
        
        if tempo_range > 1.0:
            print('  ✓ Boa variação! Tempos medidos individualmente.')
        else:
            print('  ⚠️ Pouca variação. Pode ser problema de chunk.')
except Exception as e:
    print(f'  Erro: {e}')

print('=' * 80)
