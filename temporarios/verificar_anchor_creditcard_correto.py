import json
import numpy as np

print('=' * 80)
print('VERIFICAÇÃO CORRETA DO ANCHOR (usando explanation_size ao invés de explanation_stats)')
print('=' * 80)

# Testar com Credit Card
data = json.load(open('json/anchor/creditcard.json'))

# Stats agregados
stats = data['explanation_stats']
neg_mean_agg = stats['negative'].get('mean_length', 0)
neg_std_agg = stats['negative'].get('std_length', 0)

print(f'Stats agregados (negative):')
print(f'  mean: {neg_mean_agg:.2f}')
print(f'  std: {neg_std_agg:.2f}')

# Per instance - usando a chave CORRETA!
per_inst = data.get('per_instance', [])
classif_inst = [inst for inst in per_inst if not inst.get('rejected', False)]

print(f'\nPer instance:')
print(f'  Total classificadas: {len(classif_inst)}')

if classif_inst:
    # Usar explanation_size (não explanation_stats.size!)
    tamanhos = [inst.get('explanation_size', 0) for inst in classif_inst]
    
    mean_real = np.mean(tamanhos)
    std_real = np.std(tamanhos, ddof=1) if len(tamanhos) > 1 else 0
    
    print(f'  Mean: {mean_real:.2f}')
    print(f'  Std: {std_real:.2f}')
    print(f'  Min: {min(tamanhos)}')
    print(f'  Max: {max(tamanhos)}')
    print(f'  Zeros: {tamanhos.count(0)} ({tamanhos.count(0)/len(tamanhos)*100:.1f}%)')
    
    # Primeiros 20 exemplos
    print(f'\nPrimeiras 20 instâncias:')
    for i, inst in enumerate(classif_inst[:20]):
        tamanho = inst.get('explanation_size', 0)
        rej = inst.get('rejected', False)
        print(f'  [{i}] size={tamanho}, rejected={rej}')
    
    # Verificar consistência
    print(f'\n' + '=' * 80)
    if abs(mean_real - neg_mean_agg) < 0.01 and abs(std_real - neg_std_agg) < 0.01:
        print('✓ VALORES CONSISTENTES!')
    else:
        print('❌ INCONSISTÊNCIA!')
        print(f'   Diferença mean: {abs(mean_real - neg_mean_agg):.4f}')
        print(f'   Diferença std: {abs(std_real - neg_std_agg):.4f}')
        
        if mean_real == 0:
            print('\n⚠️  PROBLEMA: Todas as explicações no per_instance têm tamanho ZERO!')
            print('   Isso significa que o JSON foi gerado com bug.')
            print('   SOLUÇÃO: Regenerar o Anchor para este dataset.')
else:
    print('⚠️  Sem per_instance')

print('=' * 80)
