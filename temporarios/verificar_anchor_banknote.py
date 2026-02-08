import json

# Verificar Anchor Banknote
data = json.load(open('json/anchor/banknote.json'))
stats = data['explanation_stats']

print('=' * 70)
print('Anchor Banknote - Dados do JSON:')
print('=' * 70)

# Positivas
pos_mean = stats['positive'].get('mean_size', stats['positive'].get('mean_length', 0))
pos_std = stats['positive'].get('std_size', stats['positive'].get('std_length', 0))
pos_count = stats['positive']['count']

# Negativas
neg_mean = stats['negative'].get('mean_size', stats['negative'].get('mean_length', 0))
neg_std = stats['negative'].get('std_size', stats['negative'].get('std_length', 0))
neg_count = stats['negative']['count']

# Rejeitadas
rej_mean = stats['rejected'].get('mean_size', stats['rejected'].get('mean_length', 0))
rej_std = stats['rejected'].get('std_size', stats['rejected'].get('std_length', 0))
rej_count = stats['rejected']['count']

print(f'Positivas: mean={pos_mean:.2f}, std={pos_std:.2f}, count={pos_count}')
print(f'Negativas: mean={neg_mean:.2f}, std={neg_std:.2f}, count={neg_count}')
print(f'Rejeitadas: mean={rej_mean:.2f}, std={rej_std:.2f}, count={rej_count}')

# Calcular classificadas (pooled)
if (pos_count + neg_count) > 0:
    classif_mean = (pos_mean * pos_count + neg_mean * neg_count) / (pos_count + neg_count)
    classif_std = ((pos_std**2 * pos_count + neg_std**2 * neg_count) / (pos_count + neg_count)) ** 0.5
else:
    classif_mean, classif_std = 0, 0

print(f'\nCalculado Classif: {classif_mean:.2f} ± {classif_std:.2f}')
print(f'Valor na tabela: 1.37 ± 0.00')

# Verificar se positivas realmente tem std = 0
if pos_std == 0.0:
    print(f'\n✓ SIM, positivas tem std = 0.00')
    print(f'  Isso significa que TODAS as {pos_count} instâncias positivas')
    print(f'  têm explicações com exatamente {pos_mean:.2f} features')
    print(f'  (sem variação alguma)')
else:
    print(f'\n✗ NÃO, positivas tem std = {pos_std:.2f}')

# Ver algumas instâncias para confirmar
print('\n' + '=' * 70)
print('Verificando instâncias individuais (positivas):')
print('=' * 70)
per_instance = data.get('per_instance', [])
pos_instances = [inst for inst in per_instance if inst.get('prediction_type') == 'positive']
print(f'Total de instâncias positivas: {len(pos_instances)}')

if len(pos_instances) > 0:
    tamanhos = []
    for inst in pos_instances[:10]:  # Mostrar primeiras 10
        exp_stats = inst.get('explanation_stats', {})
        tamanho = exp_stats.get('size', exp_stats.get('length', 0))
        tamanhos.append(tamanho)
        print(f'  Instância: tamanho = {tamanho}')
    
    if len(set(tamanhos)) == 1:
        print(f'\n✓ CONFIRMADO: Todas têm tamanho {tamanhos[0]} (sem variação)')
    else:
        print(f'\n✗ ATENÇÃO: Tamanhos variam: {set(tamanhos)}')
