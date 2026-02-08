import json

# Verificar Anchor Credit Card
data = json.load(open('json/anchor/creditcard.json'))
stats = data['explanation_stats']

print('=' * 70)
print('Anchor Credit Card - Dados do JSON:')
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

print(f'\n' + '=' * 70)
print(f'Calculado Classif: {classif_mean:.2f} ± {classif_std:.2f}')
print(f'Valor na tabela: 0.19 ± 0.39')
print('=' * 70)

if abs(classif_mean - 0.19) < 0.01 and abs(classif_std - 0.39) < 0.01:
    print('✓ VALORES ESTÃO CORRETOS MATEMATICAMENTE')
else:
    print('✗ VALORES NÃO BATEM!')

# Ver distribuição real das instâncias
print('\n' + '=' * 70)
print('Analisando instâncias individuais (primeiras 20):')
print('=' * 70)
per_instance = data.get('per_instance', [])
pos_instances = [inst for inst in per_instance if inst.get('prediction_type') == 'positive']
neg_instances = [inst for inst in per_instance if inst.get('prediction_type') == 'negative']

print(f'\nPositivas ({len(pos_instances)} total):')
tamanhos_pos = []
for i, inst in enumerate(pos_instances[:20]):
    exp_stats = inst.get('explanation_stats', {})
    tamanho = exp_stats.get('size', exp_stats.get('length', 0))
    tamanhos_pos.append(tamanho)
    print(f'  [{i+1}] tamanho = {tamanho}')

print(f'\nNegativas ({len(neg_instances)} total):')
tamanhos_neg = []
for i, inst in enumerate(neg_instances[:20]):
    exp_stats = inst.get('explanation_stats', {})
    tamanho = exp_stats.get('size', exp_stats.get('length', 0))
    tamanhos_neg.append(tamanho)
    print(f'  [{i+1}] tamanho = {tamanho}')

# Estatísticas descritivas
if tamanhos_pos:
    print(f'\nPositivas: min={min(tamanhos_pos)}, max={max(tamanhos_pos)}, quantos zeros={tamanhos_pos.count(0)}')
if tamanhos_neg:
    print(f'Negativas: min={min(tamanhos_neg)}, max={max(tamanhos_neg)}, quantos zeros={tamanhos_neg.count(0)}')

print('\n' + '=' * 70)
print('VERIFICAÇÃO DIRETA DOS TAMANHOS:')
print('=' * 70)

# Calcular a partir dos dados reais
import numpy as np
per_inst = data.get('per_instance', [])
classif_inst = [inst for inst in per_inst if not inst.get('rejected', False)]
tamanhos_reais = [inst.get('explanation_stats', {}).get('size', inst.get('explanation_stats', {}).get('length', 0)) 
                  for inst in classif_inst]

if tamanhos_reais:
    mean_real = np.mean(tamanhos_reais)
    std_real = np.std(tamanhos_reais, ddof=1) if len(tamanhos_reais) > 1 else 0
    print(f'Calculado dos per_instance:')
    print(f'  Total: {len(tamanhos_reais)} instâncias')
    print(f'  Mean: {mean_real:.4f}')
    print(f'  Std: {std_real:.4f}')
    print(f'  Zeros: {tamanhos_reais.count(0)} ({tamanhos_reais.count(0)/len(tamanhos_reais)*100:.1f}%)')
    
    print(f'\nJSON aggregated stats:')
    print(f'  Mean: {classif_mean:.4f}')
    print(f'  Std: {classif_std:.4f}')
    
    if abs(mean_real - classif_mean) > 0.01:
        print(f'\n⚠️  INCONSISTÊNCIA DETECTADA!')
        print(f'  JSON diz mean={classif_mean:.2f}, mas cálculo real={mean_real:.2f}')
    else:
        print(f'\n✓ Valores consistentes')
else:
    print('Sem instâncias classificadas nos per_instance!')

print('\n' + '=' * 70)
print('INTERPRETAÇÃO:')
print('=' * 70)
print('Se a média é 0.19, significa que a MAIORIA das explicações')
print('tem tamanho ZERO (explicação vazia), e algumas poucas têm 1+ features.')
print('Isso é ESTRANHO e pode indicar problema no Anchor para esse dataset!')
