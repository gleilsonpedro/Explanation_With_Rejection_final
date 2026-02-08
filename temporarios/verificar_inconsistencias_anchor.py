import json
import numpy as np

print('=' * 80)
print('VERIFICAÇÃO DE INCONSISTÊNCIAS NOS DADOS AGREGADOS DO ANCHOR')
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

for dataset_file, dataset_nome in datasets:
    try:
        data = json.load(open(f'json/anchor/{dataset_file}.json'))
        stats = data['explanation_stats']
        
        # Stats agregados
        neg_mean_agg = stats['negative'].get('mean_size', stats['negative'].get('mean_length', 0))
        neg_std_agg = stats['negative'].get('std_size', stats['negative'].get('std_length', 0))
        neg_count = stats['negative']['count']
        
        # Calcular dos per_instance
        per_inst = data.get('per_instance', [])
        
        # Todas as instâncias classificadas (não rejeitadas)
        classif_instances = [inst for inst in per_inst if not inst.get('rejected', False)]
        
        if classif_instances:
            tamanhos = [inst.get('explanation_stats', {}).get('size', 
                                          inst.get('explanation_stats', {}).get('length', 0)) 
                       for inst in classif_instances]
            mean_real = np.mean(tamanhos)
            std_real = np.std(tamanhos, ddof=1) if len(tamanhos) > 1 else 0
            
            # Como neg_count domina, vamos comparar com esse
            # Verificar inconsistência
            if abs(mean_real - neg_mean_agg) > 0.01 or abs(std_real - neg_std_agg) > 0.01:
                problemas.append({
                    'dataset': dataset_nome,
                    'agg_mean': neg_mean_agg,
                    'agg_std': neg_std_agg,
                    'real_mean': mean_real,
                    'real_std': std_real,
                    'count': len(tamanhos)
                })
                print(f'\n❌ {dataset_nome}:')
                print(f'   Agregado: {neg_mean_agg:.2f} ± {neg_std_agg:.2f}')
                print(f'   Real: {mean_real:.2f} ± {std_real:.2f}')
                print(f'   Diferença: {abs(mean_real - neg_mean_agg):.4f} (mean), {abs(std_real - neg_std_agg):.4f} (std)')
            else:
                print(f'✓ {dataset_nome}: OK (agregado={neg_mean_agg:.2f}±{neg_std_agg:.2f}, real={mean_real:.2f}±{std_real:.2f})')
        else:
            print(f'⚠️  {dataset_nome}: Sem per_instance')
    except Exception as e:
        print(f'⚠️  {dataset_nome}: Erro ao processar - {e}')

print('\n' + '=' * 80)
print(f'RESUMO: {len(problemas)} datasets com inconsistências')
print('=' * 80)

if problemas:
    print('\nDATASETS PROBLEMÁTICOS:')
    for p in problemas:
        print(f"  - {p['dataset']}: agregado={p['agg_mean']:.2f}±{p['agg_std']:.2f}, real={p['real_mean']:.2f}±{p['real_std']:.2f}")
    
    print('\n⚠️  AÇÃO NECESSÁRIA: Regenerar Anchor para esses datasets!')
else:
    print('\n✓ Todos os datasets estão consistentes!')
