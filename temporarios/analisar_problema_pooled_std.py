import json
import numpy as np

print('=' * 80)
print('ANÁLISE: Problema do Pooled Std com Médias Diferentes')
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

metodos = ['peab', 'anchor', 'minexp']

print('\nO pooled std atual usa: sqrt((s1²×n1 + s2²×n2)/(n1+n2))')
print('Isso só funciona quando mean1 = mean2!')
print('\nQuando mean1 ≠ mean2, deve usar a variância total dos dados combinados.')
print('-' * 80)

problemas = []

for dataset_file, dataset_nome in datasets:
    for metodo in metodos:
        try:
            data = json.load(open(f'json/{metodo}/{dataset_file}.json'))
            stats = data.get('explanation_stats', {})
            per_inst = data.get('per_instance', [])
            
            pos = stats.get('positive', {})
            neg = stats.get('negative', {})
            
            pos_mean = pos.get('mean_size', pos.get('mean_length', 0))
            pos_std = pos.get('std_size', pos.get('std_length', 0))
            pos_count = pos.get('count', 0)
            
            neg_mean = neg.get('mean_size', neg.get('mean_length', 0))
            neg_std = neg.get('std_size', neg.get('std_length', 0))
            neg_count = neg.get('count', 0)
            
            # Pooled std ATUAL (errado quando médias diferentes)
            if (pos_count + neg_count) > 0:
                pooled_std_atual = ((pos_std**2 * pos_count + neg_std**2 * neg_count) / (pos_count + neg_count)) ** 0.5
            else:
                continue
            
            # Calcular std CORRETO dos valores individuais
            classificadas = [inst for inst in per_inst if not inst.get('rejected', False)]
            tamanhos = [inst.get('explanation_size', inst.get('explanation_stats', {}).get('size', 0)) 
                       for inst in classificadas]
            
            if len(tamanhos) > 1:
                std_correto = np.std(tamanhos, ddof=1)
                diferenca = abs(std_correto - pooled_std_atual)
                
                # Se diferença > 0.01, é um problema
                if diferenca > 0.01:
                    problemas.append({
                        'dataset': dataset_nome,
                        'metodo': metodo.upper(),
                        'pos_mean': pos_mean,
                        'neg_mean': neg_mean,
                        'pos_count': pos_count,
                        'neg_count': neg_count,
                        'std_atual': pooled_std_atual,
                        'std_correto': std_correto,
                        'diferenca': diferenca
                    })
                    
        except Exception as e:
            pass

if problemas:
    print(f'\n🔴 ENCONTRADOS {len(problemas)} CASOS COM STD INCORRETO!')
    print('=' * 80)
    
    for p in problemas:
        print(f'\n{p["dataset"]} - {p["metodo"]}:')
        print(f'  Positivas: n={p["pos_count"]}, mean={p["pos_mean"]:.2f}')
        print(f'  Negativas: n={p["neg_count"]}, mean={p["neg_mean"]:.2f}')
        print(f'  Std ATUAL (tabela):  {p["std_atual"]:.4f}')
        print(f'  Std CORRETO:         {p["std_correto"]:.4f}')
        print(f'  Diferença:           {p["diferenca"]:.4f} {"⚠️" if p["diferenca"] > 0.1 else ""}')
    
    print('\n' + '=' * 80)
    print('CONCLUSÃO:')
    print('=' * 80)
    print(f'''
🔴 BUG ENCONTRADO na tabela de EXPLICAÇÕES!

Problema: O pooled std está usando fórmula simplificada que assume
          que positivas e negativas têm a MESMA média.

Casos afetados: {len(problemas)} células

Solução: Calcular std direto dos valores individuais de per_instance.

Impacto: Tabela de explicações precisa ser REGENERADA com cálculo correto!
''')
    
    print('\n' + '=' * 80)
    print('CASOS MAIS AFETADOS (diferença > 0.1):')
    print('=' * 80)
    
    criticos = [p for p in problemas if p['diferenca'] > 0.1]
    if criticos:
        for p in sorted(criticos, key=lambda x: x['diferenca'], reverse=True):
            print(f'  • {p["dataset"]} - {p["metodo"]}: Δ = {p["diferenca"]:.2f} (atual: {p["std_atual"]:.2f}, correto: {p["std_correto"]:.2f})')
    else:
        print('  (Nenhum caso com diferença > 0.1)')
        
else:
    print('\n✅ TODOS OS CASOS ESTÃO CORRETOS!')
    print('   (Pooled std coincide com std real)')

print('\n' + '=' * 80)
