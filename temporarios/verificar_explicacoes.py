import json

# Verificar Banknote PEAB
data = json.load(open('json/peab/banknote.json'))
stats = data['explanation_stats']

print('=' * 70)
print('PEAB Banknote - Dados do JSON:')
print('=' * 70)

# Verificar qual campo existe (mean_size ou mean_length)
pos_mean = stats["positive"].get("mean_size", stats["positive"].get("mean_length", 0))
pos_std = stats["positive"].get("std_size", stats["positive"].get("std_length", 0))
neg_mean = stats["negative"].get("mean_size", stats["negative"].get("mean_length", 0))
neg_std = stats["negative"].get("std_size", stats["negative"].get("std_length", 0))
rej_mean = stats["rejected"].get("mean_size", stats["rejected"].get("mean_length", 0))
rej_std = stats["rejected"].get("std_size", stats["rejected"].get("std_length", 0))

print(f'Positivas: mean={pos_mean:.2f}, std={pos_std:.2f}, count={stats["positive"]["count"]}')
print(f'Negativas: mean={neg_mean:.2f}, std={neg_std:.2f}, count={stats["negative"]["count"]}')
print(f'Rejeitadas: mean={rej_mean:.2f}, std={rej_std:.2f}, count={stats["rejected"]["count"]}')

# Calcular classificadas (pos + neg combinados)
pos_c = stats['positive']['count']
neg_c = stats['negative']['count']
pos_m = pos_mean
neg_m = neg_mean
pos_s = pos_std
neg_s = neg_std

classif_mean = (pos_m * pos_c + neg_m * neg_c) / (pos_c + neg_c)
classif_std = ((pos_s**2 * pos_c + neg_s**2 * neg_c) / (pos_c + neg_c)) ** 0.5

print(f'\nClassificadas (Calculado): mean={classif_mean:.2f}, std={classif_std:.2f}')

# Verificar valores na tabela
print('\n' + '=' * 70)
print('Valores na tabela mnist_explicacoes.tex:')
print('=' * 70)
print('Banknote PEAB: Clas. = 2.86 ± 0.62, Rej. = 2.60 ± 0.70')

print(f'\n✓ Classificadas: {classif_mean:.2f} ± {classif_std:.2f} → {"CORRETO" if abs(classif_mean - 2.86) < 0.01 and abs(classif_std - 0.62) < 0.01 else "INCORRETO"}')
print(f'✓ Rejeitadas: {rej_mean:.2f} ± {rej_std:.2f} → {"CORRETO" if abs(rej_mean - 2.60) < 0.01 and abs(rej_std - 0.70) < 0.01 else "INCORRETO"}')
