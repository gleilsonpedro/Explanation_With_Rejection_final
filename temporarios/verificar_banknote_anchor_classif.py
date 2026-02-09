import json

print('=' * 80)
print('INVESTIGAÇÃO: Banknote Anchor Classificadas (std=0.00)')
print('=' * 80)

data = json.load(open('json/anchor/banknote.json'))
stats = data.get('explanation_stats', {})
per_inst = data.get('per_instance', [])

# Stats agregados
pos = stats.get('positive', {})
neg = stats.get('negative', {})

print(f'\nESTATÍSTICAS AGREGADAS:')
print(f'  Positivas: {pos.get("count", 0)} instâncias')
print(f'    Mean: {pos.get("mean_size", pos.get("mean_length", 0)):.4f}')
print(f'    Std:  {pos.get("std_size", pos.get("std_length", 0)):.4f}')

print(f'\n  Negativas: {neg.get("count", 0)} instâncias')
print(f'    Mean: {neg.get("mean_size", neg.get("mean_length", 0)):.4f}')
print(f'    Std:  {neg.get("std_size", neg.get("std_length", 0)):.4f}')

# Pooled std
pos_count = pos.get('count', 0)
neg_count = neg.get('count', 0)
pos_std = pos.get('std_size', pos.get('std_length', 0))
neg_std = neg.get('std_size', neg.get('std_length', 0))

if (pos_count + neg_count) > 0:
    pooled_std = ((pos_std**2 * pos_count + neg_std**2 * neg_count) / (pos_count + neg_count)) ** 0.5
    print(f'\n  Pooled Std (Classif.): {pooled_std:.4f}')

# Valores individuais
classificadas = [inst for inst in per_inst if not inst.get('rejected', False)]
tamanhos = [inst.get('explanation_size', inst.get('explanation_stats', {}).get('size', 0)) 
           for inst in classificadas]

print(f'\n\nVALORES INDIVIDUAIS:')
print(f'  Total: {len(tamanhos)} instâncias classificadas')
print(f'  Valores únicos: {len(set(tamanhos))}')
print(f'  Distribuição: {set(tamanhos)}')

if tamanhos:
    from collections import Counter
    contagem = Counter(tamanhos)
    print(f'\n  Contagem por tamanho:')
    for tam, count in sorted(contagem.items()):
        print(f'    {tam} features: {count} instâncias ({100*count/len(tamanhos):.1f}%)')
    
    # Calcular std manual
    import numpy as np
    std_calculado = np.std(tamanhos, ddof=1)
    mean_calculado = np.mean(tamanhos)
    print(f'\n  Mean calculado: {mean_calculado:.4f}')
    print(f'  Std calculado: {std_calculado:.4f}')
    
    if std_calculado < 0.01:
        print(f'\n  ✓ LEGÍTIMO: Todas têm exatamente {int(tamanhos[0])} features!')
    else:
        print(f'\n  ⚠️ ATENÇÃO: Há variação mas na tabela std=0.00!')

print('\n' + '=' * 80)
