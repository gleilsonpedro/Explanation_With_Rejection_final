import json

print('=' * 80)
print('VERIFICAÇÃO RÁPIDA: QUAIS DATASETS AINDA TÊM STD = 0.00?')
print('=' * 80)

datasets = [
    'banknote', 'vertebral_column', 'pima_indians_diabetes', 'heart_disease',
    'creditcard', 'breast_cancer', 'covertype', 'spambase', 'sonar', 'mnist'
]

print('\nANCHOR:')
anchor_problemas = []
for d in datasets:
    try:
        data = json.load(open(f'json/anchor/{d}.json'))
        per_inst = data.get('per_instance', [])
        tempos = [inst.get('computation_time', 0) for inst in per_inst]
        zeros = sum(1 for t in tempos if t == 0.0)
        if zeros == len(tempos):
            anchor_problemas.append(d)
            print(f'  ⚠️ {d}: ALL ZERO ({len(tempos)} instâncias)')
        else:
            print(f'  ✓ {d}: OK')
    except:
        print(f'  ? {d}: Erro ao ler')

print('\nMINEXP:')
minexp_problemas = []
for d in datasets:
    try:
        data = json.load(open(f'json/minexp/{d}.json'))
        per_inst = data.get('per_instance', [])
        
        # Classificadas
        classif = [inst for inst in per_inst if not inst.get('rejected', False)]
        classif_tempos = [inst.get('computation_time', 0) for inst in classif]
        classif_zeros = sum(1 for t in classif_tempos if t == 0.0)
        
        # Rejeitadas
        rejeit = [inst for inst in per_inst if inst.get('rejected', False)]
        rejeit_tempos = [inst.get('computation_time', 0) for inst in rejeit]
        rejeit_zeros = sum(1 for t in rejeit_tempos if t == 0.0)
        
        problema_classif = classif_zeros == len(classif_tempos) if classif_tempos else False
        problema_rejeit = rejeit_zeros == len(rejeit_tempos) if rejeit_tempos else False
        
        if problema_classif or problema_rejeit:
            minexp_problemas.append(d)
            status = []
            if problema_classif:
                status.append(f'Classif ALL ZERO')
            if problema_rejeit:
                status.append(f'Rejeit ALL ZERO')
            print(f'  ⚠️ {d}: {", ".join(status)}')
        else:
            print(f'  ✓ {d}: OK')
    except Exception as e:
        print(f'  ? {d}: Erro - {str(e)[:40]}')

print('\n' + '=' * 80)
print('RESUMO:')
print('=' * 80)
print(f'Anchor com problema: {len(anchor_problemas)}')
for d in anchor_problemas:
    print(f'  - {d}')

print(f'\nMinExp com problema: {len(minexp_problemas)}')
for d in minexp_problemas:
    print(f'  - {d}')

total_problemas = len(anchor_problemas) + len(minexp_problemas)
print(f'\n{"="*80}')
if total_problemas == 0:
    print('✅ TUDO PRONTO PARA SUBMISSÃO!')
    print('   Todas as tabelas têm desvio padrão correto.')
elif total_problemas == 1:
    print('⚠️ 1 dataset precisa ser regenerado')
    print(f'   Mas você pode submeter com fallback (std=0 só neste dataset)')
else:
    print(f'⚠️ {total_problemas} datasets ainda precisam ser regenerados')
    print('   Opções:')
    print('   1) Submeter assim (alguns terão std=0)')
    print('   2) Aguardar regeneração terminar')
print('=' * 80)
