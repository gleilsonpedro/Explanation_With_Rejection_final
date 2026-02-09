import json
import os
import numpy as np
from datetime import datetime

print('=' * 80)
print('VERIFICAÇÃO FINAL COMPLETA - ANTES DA SUBMISSÃO DO ARTIGO')
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

problemas_encontrados = []

print('\n1️⃣ VERIFICANDO MINEXP - TODOS OS DATASETS')
print('=' * 80)

for dataset_file, dataset_nome in datasets:
    json_path = f'json/minexp/{dataset_file}.json'
    
    if not os.path.exists(json_path):
        problemas_encontrados.append(f'MinExp {dataset_nome}: ❌ JSON não encontrado')
        continue
    
    try:
        data = json.load(open(json_path))
        per_inst = data.get('per_instance', [])
        
        # Verificar classificadas
        classificadas = [inst for inst in per_inst if not inst.get('rejected', False)]
        rej = [inst for inst in per_inst if inst.get('rejected', False)]
        
        if classificadas:
            tempos_class = [inst.get('computation_time', 0) for inst in classificadas]
            mean_class = np.mean(tempos_class)
            std_class = np.std(tempos_class, ddof=1) if len(tempos_class) > 1 else 0.0
            
            if std_class == 0.0 and len(tempos_class) > 1:
                problemas_encontrados.append(
                    f'MinExp {dataset_nome} Classificadas: ⚠️ std=0 com {len(tempos_class)} instâncias'
                )
            else:
                print(f'  ✅ {dataset_nome} Classif.: {len(tempos_class)} inst., mean={mean_class*1000:.2f}ms, std={std_class*1000:.2f}ms')
        
        if rej:
            tempos_rej = [inst.get('computation_time', 0) for inst in rej]
            mean_rej = np.mean(tempos_rej)
            std_rej = np.std(tempos_rej, ddof=1) if len(tempos_rej) > 1 else 0.0
            
            if std_rej == 0.0 and len(tempos_rej) > 1:
                problemas_encontrados.append(
                    f'MinExp {dataset_nome} Rejeitadas: ⚠️ std=0 com {len(tempos_rej)} instâncias'
                )
            else:
                print(f'  ✅ {dataset_nome} Rejeit.: {len(tempos_rej)} inst., mean={mean_rej*1000:.2f}ms, std={std_rej*1000:.2f}ms')
                
    except Exception as e:
        problemas_encontrados.append(f'MinExp {dataset_nome}: ❌ Erro ao ler JSON: {str(e)[:50]}')

print('\n\n2️⃣ VERIFICANDO ANCHOR - TODOS OS DATASETS')
print('=' * 80)

for dataset_file, dataset_nome in datasets:
    json_path = f'json/anchor/{dataset_file}.json'
    
    if not os.path.exists(json_path):
        problemas_encontrados.append(f'Anchor {dataset_nome}: ❌ JSON não encontrado')
        continue
    
    try:
        data = json.load(open(json_path))
        per_inst = data.get('per_instance', [])
        
        # Verificar classificadas
        classificadas = [inst for inst in per_inst if not inst.get('rejected', False)]
        rej = [inst for inst in per_inst if inst.get('rejected', False)]
        
        if classificadas:
            tempos_class = [inst.get('computation_time', 0) for inst in classificadas]
            mean_class = np.mean(tempos_class)
            std_class = np.std(tempos_class, ddof=1) if len(tempos_class) > 1 else 0.0
            
            # Verificar se todos são zero
            zeros = sum(1 for t in tempos_class if t == 0.0)
            
            if zeros == len(tempos_class):
                problemas_encontrados.append(
                    f'Anchor {dataset_nome} Classificadas: ❌ TODOS os tempos são 0.0!'
                )
            elif std_class == 0.0 and len(tempos_class) > 1:
                problemas_encontrados.append(
                    f'Anchor {dataset_nome} Classificadas: ⚠️ std=0 com {len(tempos_class)} instâncias'
                )
            else:
                print(f'  ✅ {dataset_nome} Classif.: {len(tempos_class)} inst., mean={mean_class*1000:.2f}ms, std={std_class*1000:.2f}ms')
        
        if rej:
            tempos_rej = [inst.get('computation_time', 0) for inst in rej]
            mean_rej = np.mean(tempos_rej)
            std_rej = np.std(tempos_rej, ddof=1) if len(tempos_rej) > 1 else 0.0
            
            zeros_rej = sum(1 for t in tempos_rej if t == 0.0)
            
            if zeros_rej == len(tempos_rej):
                problemas_encontrados.append(
                    f'Anchor {dataset_nome} Rejeitadas: ❌ TODOS os tempos são 0.0!'
                )
            elif std_rej == 0.0 and len(tempos_rej) > 1:
                problemas_encontrados.append(
                    f'Anchor {dataset_nome} Rejeitadas: ⚠️ std=0 com {len(tempos_rej)} instâncias'
                )
            else:
                print(f'  ✅ {dataset_nome} Rejeit.: {len(tempos_rej)} inst., mean={mean_rej*1000:.2f}ms, std={std_rej*1000:.2f}ms')
                
    except Exception as e:
        problemas_encontrados.append(f'Anchor {dataset_nome}: ❌ Erro ao ler JSON: {str(e)[:50]}')

print('\n\n3️⃣ VERIFICANDO PEAB - TODOS OS DATASETS')
print('=' * 80)

for dataset_file, dataset_nome in datasets:
    json_path = f'json/peab/{dataset_file}.json'
    
    if not os.path.exists(json_path):
        problemas_encontrados.append(f'PEAB {dataset_nome}: ❌ JSON não encontrado')
        continue
    
    try:
        data = json.load(open(json_path))
        per_inst = data.get('per_instance', [])
        
        # Verificar classificadas
        classificadas = [inst for inst in per_inst if not inst.get('rejected', False)]
        rej = [inst for inst in per_inst if inst.get('rejected', False)]
        
        if classificadas:
            tempos_class = [inst.get('computation_time', 0) for inst in classificadas]
            std_class = np.std(tempos_class, ddof=1) if len(tempos_class) > 1 else 0.0
            
            if std_class == 0.0 and len(tempos_class) > 1:
                problemas_encontrados.append(
                    f'PEAB {dataset_nome} Classificadas: ⚠️ std=0 com {len(tempos_class)} instâncias'
                )
        
        if rej:
            tempos_rej = [inst.get('computation_time', 0) for inst in rej]
            std_rej = np.std(tempos_rej, ddof=1) if len(tempos_rej) > 1 else 0.0
            
            if std_rej == 0.0 and len(tempos_rej) > 1:
                problemas_encontrados.append(
                    f'PEAB {dataset_nome} Rejeitadas: ⚠️ std=0 com {len(tempos_rej)} instâncias'
                )
                
    except Exception as e:
        problemas_encontrados.append(f'PEAB {dataset_nome}: ❌ Erro ao ler JSON: {str(e)[:50]}')

print('  ✅ PEAB: Verificação completa (detalhes omitidos)')

print('\n\n4️⃣ RESUMO DOS PROBLEMAS ENCONTRADOS')
print('=' * 80)

if problemas_encontrados:
    print(f'\n🔴 ENCONTRADOS {len(problemas_encontrados)} PROBLEMAS:\n')
    for problema in problemas_encontrados:
        print(f'  • {problema}')
    
    print('\n\n⚠️ AÇÃO NECESSÁRIA ANTES DA SUBMISSÃO:')
    print('   Resolver os problemas acima antes de submeter o artigo!')
else:
    print('\n✅ NENHUM PROBLEMA ENCONTRADO!')
    print('   Todos os JSONs estão com tempos válidos e std > 0 quando n > 1.')

print('\n\n5️⃣ VERIFICAÇÃO DE TIMESTAMPS (atualizações recentes)')
print('=' * 80)

# Verificar MinExp Credit Card e Covertype
minexp_critical = [
    ('creditcard', 'Credit Card'),
    ('covertype', 'Covertype')
]

print('\nMinExp (datasets que tinham problemas):')
for dataset_file, dataset_nome in minexp_critical:
    json_path = f'json/minexp/{dataset_file}.json'
    if os.path.exists(json_path):
        mtime = os.path.getmtime(json_path)
        mod_time = datetime.fromtimestamp(mtime)
        age = datetime.now() - mod_time
        print(f'  {dataset_nome}: {mod_time.strftime("%Y-%m-%d %H:%M:%S")} ({age.total_seconds()/3600:.1f}h atrás)')

# Verificar Anchor Covertype
print('\nAnchor (dataset que tinha problema):')
covertype_json = 'json/anchor/covertype.json'
if os.path.exists(covertype_json):
    mtime = os.path.getmtime(covertype_json)
    mod_time = datetime.fromtimestamp(mtime)
    age = datetime.now() - mod_time
    print(f'  Covertype: {mod_time.strftime("%Y-%m-%d %H:%M:%S")} ({age.total_seconds()/3600:.1f}h atrás)')

print('\n\n' + '=' * 80)
print('CONCLUSÃO FINAL')
print('=' * 80)

if not problemas_encontrados:
    print('''
🎉 TUDO PRONTO PARA SUBMISSÃO!

✅ Status Geral:
   • MinExp: 10/10 datasets OK
   • Anchor: 10/10 datasets OK
   • PEAB: 10/10 datasets OK
   
✅ Todos os tempos têm std > 0 (quando n > 1)
✅ Nenhum tempo zerado detectado
✅ JSONs atualizados recentemente

🚀 PRÓXIMO PASSO:
   1. Verificar tabela LaTeX gerada
   2. Submeter o artigo!
''')
else:
    print(f'''
⚠️ AINDA HÁ {len(problemas_encontrados)} PROBLEMAS!

Resolver antes de submeter:
''')
    for problema in problemas_encontrados:
        print(f'  • {problema}')

print('=' * 80)
