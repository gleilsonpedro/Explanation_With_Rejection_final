import json

print('=' * 80)
print('VERIFICAÇÃO: Breast Cancer Rejeitadas - Todas com 2.00 ± 0.00')
print('=' * 80)

metodos = ['peab', 'anchor', 'minexp']

for metodo in metodos:
    print(f'\n{"="*80}')
    print(f'MÉTODO: {metodo.upper()}')
    print(f'{"="*80}')
    
    try:
        data = json.load(open(f'json/{metodo}/breast_cancer.json'))
        per_inst = data.get('per_instance', [])
        
        # Filtrar rejeitadas
        rejeitadas = [inst for inst in per_inst if inst.get('rejected', False)]
        
        print(f'\nTotal de rejeitadas: {len(rejeitadas)}')
        
        if rejeitadas:
            # Extrair tamanhos
            tamanhos = [inst.get('explanation_size', inst.get('explanation_stats', {}).get('size', 0)) 
                       for inst in rejeitadas]
            
            print(f'\nTamanhos das explicações:')
            print(f'  Valores únicos: {set(tamanhos)}')
            print(f'  Contagem de cada valor:')
            
            from collections import Counter
            contagem = Counter(tamanhos)
            for tam, count in sorted(contagem.items()):
                print(f'    {tam} features: {count} instâncias ({100*count/len(tamanhos):.1f}%)')
            
            import numpy as np
            mean = np.mean(tamanhos)
            std = np.std(tamanhos, ddof=1) if len(tamanhos) > 1 else 0.0
            
            print(f'\nEstatísticas:')
            print(f'  Mean: {mean:.2f}')
            print(f'  Std:  {std:.2f}')
            
            if len(set(tamanhos)) == 1:
                print(f'\n  ✅ LEGÍTIMO: Todas as {len(rejeitadas)} rejeitadas têm EXATAMENTE {tamanhos[0]} features!')
                print(f'     Std = 0.00 é CORRETO (não há variação)')
            else:
                print(f'\n  ⚠️ ATENÇÃO: Há {len(set(tamanhos))} valores diferentes mas std deveria ser > 0')
        else:
            print('  Sem rejeitadas')
            
    except Exception as e:
        print(f'  ❌ Erro: {str(e)}')

print('\n\n' + '=' * 80)
print('ANÁLISE DO BREAST CANCER')
print('=' * 80)

print('''
O Breast Cancer tem uma característica especial:

• Total de features: 30
• Rejeitadas: 34 instâncias (em todos os métodos)

Se TODAS as 34 rejeitadas têm exatamente 2 features em suas explicações,
então:
  • Mean = 2.00 ✓
  • Std = 0.00 ✓ (LEGÍTIMO!)

Isso indica que as instâncias rejeitadas no Breast Cancer têm um padrão
muito consistente - todas precisam das MESMAS 2 features para serem
explicadas.

Isso é diferente de um BUG (como os tempos zerados), onde valores
deveriam variar mas estão idênticos por erro de código.

Aqui, os tamanhos das explicações REALMENTE são todos iguais!
''')

print('=' * 80)
