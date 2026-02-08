import json
import os

print('=' * 80)
print('VERIFICAÇÃO DE TEMPOS POR INSTÂNCIA - ANCHOR')
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

problemas = []
ok_datasets = []

print('\nAnalisando per_instance.computation_time em cada dataset:\n')
print(f'{"Dataset":<20} {"Status":<15} {"Zeros":<10} {"Non-zeros":<12} {"Problema"}')
print('-' * 80)

for dataset_file, dataset_nome in datasets:
    try:
        arquivo = f'json/anchor/{dataset_file}.json'
        if not os.path.exists(arquivo):
            print(f'{dataset_nome:<20} {"MISSING":<15} {"-":<10} {"-":<12} Arquivo não existe')
            problemas.append((dataset_nome, 'Arquivo não existe'))
            continue
        
        data = json.load(open(arquivo))
        per_inst = data.get('per_instance', [])
        
        if not per_inst:
            print(f'{dataset_nome:<20} {"NO DATA":<15} {"-":<10} {"-":<12} per_instance vazio')
            problemas.append((dataset_nome, 'per_instance vazio'))
            continue
        
        # Contar tempos zerados
        tempos = [inst.get('computation_time', 0) for inst in per_inst]
        zeros = sum(1 for t in tempos if t == 0.0)
        non_zeros = len(tempos) - zeros
        
        # Verificar se todos ou maioria são zero
        if zeros == len(tempos):
            status = "ALL_ZERO"
            problema = "⚠️ TODOS ZERADOS"
            problemas.append((dataset_nome, 'Todos os tempos = 0'))
        elif zeros > len(tempos) * 0.9:
            status = "MOSTLY_ZERO"
            problema = "⚠️ >90% zerados"
            problemas.append((dataset_nome, f'{zeros}/{len(tempos)} zerados'))
        elif zeros > 0:
            status = "PARTIAL_ZERO"
            problema = f"⚠️ {zeros} zerados"
            problemas.append((dataset_nome, f'{zeros}/{len(tempos)} zerados'))
        else:
            status = "OK"
            problema = "✓"
            ok_datasets.append(dataset_nome)
        
        print(f'{dataset_nome:<20} {status:<15} {zeros:<10} {non_zeros:<12} {problema}')
        
    except Exception as e:
        print(f'{dataset_nome:<20} {"ERROR":<15} {"-":<10} {"-":<12} Erro: {str(e)[:30]}')
        problemas.append((dataset_nome, f'Erro: {str(e)}'))

print('\n' + '=' * 80)
print('RESUMO:')
print('=' * 80)
print(f'✓ Datasets OK: {len(ok_datasets)}')
for d in ok_datasets:
    print(f'  - {d}')

print(f'\n⚠️ Datasets com problemas: {len(problemas)}')
for d, motivo in problemas:
    print(f'  - {d}: {motivo}')

print('\n' + '=' * 80)
print('AÇÃO NECESSÁRIA:')
print('=' * 80)

if len(problemas) == 0:
    print('✓ Todos os datasets estão OK! Pode gerar a tabela com desvio padrão.')
elif len(problemas) == len(datasets):
    print('⚠️ TODOS OS DATASETS TÊM PROBLEMA!')
    print('   → Necessário regenerar TODOS os Anchor')
elif len(problemas) == 1:
    print(f'⚠️ Apenas {problemas[0][0]} tem problema')
    print(f'   → Regenerar apenas {problemas[0][0]}')
else:
    print(f'⚠️ {len(problemas)} datasets precisam ser regenerados:')
    for d, _ in problemas:
        print(f'   - {d}')
    print('\n   → Rodar Anchor em modo múltiplo com esses datasets')

print('=' * 80)
