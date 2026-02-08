import json

def verificar_metodo(metodo, dataset, nome_exibicao, valor_tabela_classif, valor_tabela_rej):
    """Verifica se os valores na tabela correspondem aos cálculos do JSON"""
    
    if metodo == "pulp":
        arquivo = f'json/pulp/{dataset}.json'
        data = json.load(open(arquivo))
        stats = data.get('estatisticas_por_tipo', {})
        
        # Positivas
        pos_m = stats.get('positiva', {}).get('tamanho_medio', 0)
        pos_s = stats.get('positiva', {}).get('desvio_padrao', 0)
        pos_c = stats.get('positiva', {}).get('instancias', 0)
        
        # Negativas
        neg_m = stats.get('negativa', {}).get('tamanho_medio', 0)
        neg_s = stats.get('negativa', {}).get('desvio_padrao', 0)
        neg_c = stats.get('negativa', {}).get('instancias', 0)
        
        # Rejeitadas
        rej_m = stats.get('rejeitada', {}).get('tamanho_medio', 0)
        rej_s = stats.get('rejeitada', {}).get('desvio_padrao', 0)
        
    else:  # peab, anchor, minexp
        arquivo = f'json/{metodo}/{dataset}.json'
        data = json.load(open(arquivo))
        stats = data.get('explanation_stats', {})
        
        # Positivas
        pos_m = stats.get('positive', {}).get('mean_size', stats.get('positive', {}).get('mean_length', 0))
        pos_s = stats.get('positive', {}).get('std_size', stats.get('positive', {}).get('std_length', 0))
        pos_c = stats.get('positive', {}).get('count', 0)
        
        # Negativas
        neg_m = stats.get('negative', {}).get('mean_size', stats.get('negative', {}).get('mean_length', 0))
        neg_s = stats.get('negative', {}).get('std_size', stats.get('negative', {}).get('std_length', 0))
        neg_c = stats.get('negative', {}).get('count', 0)
        
        # Rejeitadas
        rej_m = stats.get('rejected', {}).get('mean_size', stats.get('rejected', {}).get('mean_length', 0))
        rej_s = stats.get('rejected', {}).get('std_size', stats.get('rejected', {}).get('std_length', 0))
    
    # Calcular classificadas (pooled)
    if (pos_c + neg_c) > 0:
        classif_m = (pos_m * pos_c + neg_m * neg_c) / (pos_c + neg_c)
        classif_s = ((pos_s**2 * pos_c + neg_s**2 * neg_c) / (pos_c + neg_c)) ** 0.5
    else:
        classif_m, classif_s = 0, 0
    
    # Comparar com valores da tabela
    classif_ok = abs(classif_m - valor_tabela_classif[0]) < 0.01 and abs(classif_s - valor_tabela_classif[1]) < 0.01
    rej_ok = abs(rej_m - valor_tabela_rej[0]) < 0.01 and abs(rej_s - valor_tabela_rej[1]) < 0.01
    
    print(f'\n{nome_exibicao}:')
    print(f'  Positivas: mean={pos_m:.2f}, std={pos_s:.2f}, count={pos_c}')
    print(f'  Negativas: mean={neg_m:.2f}, std={neg_s:.2f}, count={neg_c}')
    print(f'  Calculado Classif: {classif_m:.2f} ± {classif_s:.2f} → {"✓ CORRETO" if classif_ok else "✗ INCORRETO"}')
    print(f'  Rejeitadas: {rej_m:.2f} ± {rej_s:.2f} → {"✓ CORRETO" if rej_ok else "✗ INCORRETO"}')
    
    return classif_ok and rej_ok

print('=' * 80)
print('VERIFICAÇÃO DA TABELA mnist_explicacoes.tex')
print('=' * 80)

# Verificar vários casos de diferentes métodos e datasets
resultados = []

# Banknote
resultados.append(verificar_metodo('peab', 'banknote', 'Banknote PEAB', (2.86, 0.62), (2.60, 0.70)))
resultados.append(verificar_metodo('pulp', 'banknote', 'Banknote PULP', (2.87, 0.00), (2.61, 0.00)))
resultados.append(verificar_metodo('anchor', 'banknote', 'Banknote Anchor', (1.37, 0.00), (1.04, 0.34)))
resultados.append(verificar_metodo('minexp', 'banknote', 'Banknote MinExp', (2.86, 0.62), (2.73, 0.67)))

# Vertebral Column
resultados.append(verificar_metodo('peab', 'vertebral_column', 'Vertebral Column PEAB', (4.34, 0.80), (5.06, 0.23)))

# Heart Disease
resultados.append(verificar_metodo('minexp', 'heart_disease', 'Heart Disease MinExp', (7.44, 1.41), (6.61, 1.51)))

print('\n' + '=' * 80)
print(f'RESUMO: {sum(resultados)}/{len(resultados)} verificações passaram')
print('=' * 80)

if all(resultados):
    print('✓ TODOS OS VALORES ESTÃO CORRETOS!')
else:
    print('✗ Alguns valores estão incorretos. Verifique os cálculos.')
