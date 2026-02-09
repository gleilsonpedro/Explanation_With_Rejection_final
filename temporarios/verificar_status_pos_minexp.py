import json
import os
from datetime import datetime, timedelta

print('=' * 80)
print('STATUS APÓS EXECUTAR MINEXP EM TODOS OS DATASETS')
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

print('\n1️⃣ VERIFICANDO MINEXP JSONs:')
print('-' * 80)

minexp_ok = []
minexp_problemas = []

for dataset_file, dataset_nome in datasets:
    json_path = f'json/minexp/{dataset_file}.json'
    
    if not os.path.exists(json_path):
        minexp_problemas.append(f'{dataset_nome}: ❌ JSON não encontrado')
        continue
    
    # Timestamp
    mtime = os.path.getmtime(json_path)
    mod_time = datetime.fromtimestamp(mtime)
    age = datetime.now() - mod_time
    
    # Verificar rejeitadas
    try:
        data = json.load(open(json_path))
        per_inst = data.get('per_instance', [])
        rejeitadas = [inst for inst in per_inst if inst.get('rejected', False)]
        
        if rejeitadas:
            tempos = [inst.get('computation_time', 0) for inst in rejeitadas]
            tempos_unicos = len(set(tempos))
            
            if tempos_unicos == 1 and len(rejeitadas) > 1:
                minexp_problemas.append(
                    f'{dataset_nome}: ⚠️ {len(rejeitadas)} rejeitadas com tempos idênticos (std=0) - '
                    f'Atualizado há {age.total_seconds()/3600:.1f}h'
                )
            elif age.total_seconds() < 3600:  # Menos de 1h
                minexp_ok.append(
                    f'{dataset_nome}: ✅ {len(rejeitadas)} rejeitadas OK - '
                    f'Atualizado há {age.total_seconds()/60:.0f} minutos'
                )
            else:
                minexp_ok.append(
                    f'{dataset_nome}: ✅ OK - Atualizado há {age.total_seconds()/3600:.1f}h'
                )
        else:
            minexp_ok.append(f'{dataset_nome}: ✅ Sem rejeitadas')
            
    except Exception as e:
        minexp_problemas.append(f'{dataset_nome}: ❌ Erro ao ler JSON: {str(e)[:40]}')

print('\n✅ MinExp OK:')
for msg in minexp_ok:
    print(f'  {msg}')

if minexp_problemas:
    print('\n⚠️ MinExp com problemas:')
    for msg in minexp_problemas:
        print(f'  {msg}')

print('\n\n2️⃣ VERIFICANDO ANCHOR - COVERTYPE:')
print('-' * 80)

covertype_json = 'json/anchor/covertype.json'

if os.path.exists(covertype_json):
    mtime = os.path.getmtime(covertype_json)
    mod_time = datetime.fromtimestamp(mtime)
    age = datetime.now() - mod_time
    
    data = json.load(open(covertype_json))
    per_inst = data.get('per_instance', [])
    
    tempos = [inst.get('computation_time', 0) for inst in per_inst]
    zeros = sum(1 for t in tempos if t == 0.0)
    
    print(f'\nArquivo: json/anchor/covertype.json')
    print(f'Última atualização: {mod_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Idade: {age.total_seconds()/3600:.1f} horas atrás')
    print(f'\nStatus dos tempos:')
    print(f'  Total de instâncias: {len(tempos)}')
    print(f'  Tempos = 0.0: {zeros} ({100*zeros/len(tempos):.1f}%)')
    
    if zeros == len(tempos):
        print(f'\n  ❌ PROBLEMA: Todos os tempos são 0.0!')
        print(f'  ⏳ AÇÃO NECESSÁRIA: Executar Anchor no Covertype')
    elif zeros > 0:
        print(f'\n  ⚠️ ATENÇÃO: {zeros} instâncias com tempo=0')
    else:
        if age.total_seconds() < 3600:
            print(f'\n  ✅ OK: Nenhum tempo zerado (atualizado há {age.total_seconds()/60:.0f} min)')
        else:
            print(f'\n  ⚠️ OK mas antigo (atualizado há {age.total_seconds()/3600:.1f}h)')
else:
    print(f'\n  ❌ JSON não encontrado: {covertype_json}')

print('\n\n3️⃣ PRÓXIMO PASSO:')
print('=' * 80)

if minexp_problemas:
    print('''
⚠️ MinExp ainda tem problemas!

Os JSONs do MinExp ainda não estão todos atualizados corretamente.
Possíveis causas:
  • Execução ainda em andamento
  • Alguns datasets falharam
  • JSONs não foram salvos corretamente

AÇÃO: Verificar se o MinExp terminou e se gerou os JSONs corretamente.
''')
elif zeros == len(tempos):
    print('''
✅ MinExp: COMPLETO e OK!

⏳ Próximo: Executar ANCHOR no COVERTYPE

COMANDO:
    env\\Scripts\\python.exe anchor.py

OU se tiver script específico:
    env\\Scripts\\python.exe scripts/run_anchor_covertype.py

Após o Anchor terminar:
    env\\Scripts\\python.exe gerar_tabelas_mnist.py

Isso atualizará as últimas 3 células da tabela de tempo!
''')
else:
    print('''
✅ MinExp: COMPLETO e OK!
✅ Anchor Covertype: JÁ ESTÁ OK!

🎉 PRÓXIMO: REGENERAR AS TABELAS!

COMANDO:
    env\\Scripts\\python.exe gerar_tabelas_mnist.py

Isso atualizará TODAS as tabelas com os dados corretos:
  • Tabela de tempo: 100% completa (60/60 células)
  • Tabela de explicações: 100% completa (já corrigida)

Após isso, TODAS as tabelas estarão prontas para o artigo!
''')

print('=' * 80)
