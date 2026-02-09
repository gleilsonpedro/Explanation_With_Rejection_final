import os
import json
from datetime import datetime

print('=' * 80)
print('VERIFICAÇÃO RÁPIDA: JSONs foram atualizados?')
print('=' * 80)

arquivos = [
    'json/minexp/creditcard.json',
    'json/anchor/covertype.json'
]

print(f'\n{"Arquivo":<40} {"Última modificação":<25} {"Status"}')
print('-' * 80)

todos_atualizados = True
agora = datetime.now()

for arquivo in arquivos:
    if os.path.exists(arquivo):
        timestamp = datetime.fromtimestamp(os.path.getmtime(arquivo))
        diff_minutos = (agora - timestamp).total_seconds() / 60
        
        if diff_minutos < 5:
            status = "✅ RECENTE (<5min)"
            atualizado = True
        elif diff_minutos < 60:
            status = f"⚠️ {int(diff_minutos)} minutos atrás"
            atualizado = False
        else:
            diff_horas = diff_minutos / 60
            status = f"⚠️ {diff_horas:.1f} horas atrás"
            atualizado = False
        
        todos_atualizados = todos_atualizados and atualizado
        print(f'{arquivo:<40} {timestamp.strftime("%Y-%m-%d %H:%M:%S"):<25} {status}')
    else:
        print(f'{arquivo:<40} {"N/A":<25} ❌ NÃO EXISTE')
        todos_atualizados = False

print('\n' + '=' * 80)

if todos_atualizados:
    print('✅ TODOS OS ARQUIVOS ESTÃO ATUALIZADOS!')
    print('\n📊 PRÓXIMO PASSO:')
    print('   env/Scripts/python.exe gerar_tabelas_mnist.py')
    print('\n   Isso vai regenerar todas as tabelas com os novos dados!')
else:
    print('⏳ Aguardando execuções terminarem...')
    print('   Execute este script novamente após MinExp/Anchor terminarem')

print('=' * 80)
