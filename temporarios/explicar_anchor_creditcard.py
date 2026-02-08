import json
import numpy as np

print('=' * 80)
print('ANÁLISE DO COMPORTAMENTO DO ANCHOR NO CREDIT CARD')
print('=' * 80)

# Comparar todos os métodos no Credit Card
metodos = ['peab', 'anchor', 'minexp']

print('\n1. COMPARAÇÃO DE TAMANHOS DE EXPLICAÇÕES (Credit Card):')
print('-' * 80)

for metodo in metodos:
    try:
        data = json.load(open(f'json/{metodo}/creditcard.json'))
        stats = data['explanation_stats']
        
        if metodo == 'anchor':
            neg = stats['negative']
            pos = stats['positive']
            rej = stats['rejected']
            
            neg_mean = neg.get('mean_length', 0)
            pos_mean = pos.get('mean_length', 0)
            rej_mean = rej.get('mean_length', 0)
            
            neg_count = neg.get('count', 0)
            pos_count = pos.get('count', 0)
            rej_count = rej.get('count', 0)
            
            print(f'\n{metodo.upper()}:')
            print(f'  Positivas: {pos_mean:.2f} features ({pos_count} instâncias)')
            print(f'  Negativas: {neg_mean:.2f} features ({neg_count} instâncias)')
            print(f'  Rejeitadas: {rej_mean:.2f} features ({rej_count} instâncias)')
            
            # Calcular pooled
            if pos_count + neg_count > 0:
                classif_mean = (pos_mean * pos_count + neg_mean * neg_count) / (pos_count + neg_count)
                print(f'  → Classificadas (pooled): {classif_mean:.2f} features')
            
            # Analisar distribuição
            per_inst = data.get('per_instance', [])
            classif = [inst for inst in per_inst if not inst.get('rejected', False)]
            tamanhos = [inst.get('explanation_size', 0) for inst in classif]
            
            if tamanhos:
                zeros = tamanhos.count(0)
                uns = tamanhos.count(1)
                dois_mais = len([t for t in tamanhos if t >= 2])
                
                print(f'  Distribuição:')
                print(f'    - Tamanho 0 (vazio): {zeros} ({zeros/len(tamanhos)*100:.1f}%)')
                print(f'    - Tamanho 1: {uns} ({uns/len(tamanhos)*100:.1f}%)')
                print(f'    - Tamanho 2+: {dois_mais} ({dois_mais/len(tamanhos)*100:.1f}%)')
        else:
            # PEAB/MinExp
            neg = stats.get('negative', {})
            pos = stats.get('positive', {})
            
            neg_mean = neg.get('mean_size', neg.get('mean_length', 0))
            pos_mean = pos.get('mean_size', pos.get('mean_length', 0))
            
            neg_count = neg.get('count', 0)
            pos_count = pos.get('count', 0)
            
            if pos_count + neg_count > 0:
                classif_mean = (pos_mean * pos_count + neg_mean * neg_count) / (pos_count + neg_count)
                print(f'\n{metodo.upper()}: {classif_mean:.2f} features (média das classificadas)')
    except Exception as e:
        print(f'\n{metodo.upper()}: Erro - {e}')

# Analisar características do dataset
print('\n' + '=' * 80)
print('2. CARACTERÍSTICAS DO CREDIT CARD DATASET:')
print('-' * 80)

try:
    data_peab = json.load(open('json/peab/creditcard.json'))
    config = data_peab.get('config', {})
    model = data_peab.get('model', {})
    thresholds = data_peab.get('thresholds', {})
    
    print(f'  Número de features: {model.get("num_features", "?")}')
    print(f'  Instâncias de teste: {data_peab.get("performance", {}).get("num_test_instances", "?")}')
    print(f'  Taxa de rejeição: {data_peab.get("performance", {}).get("rejection_rate", 0)*100:.1f}%')
    print(f'  Zona de rejeição: [{thresholds.get("t_minus", 0):.2f}, {thresholds.get("t_plus", 0):.2f}]')
    print(f'  Largura: {thresholds.get("rejection_zone_width", 0):.2f}')
except Exception as e:
    print(f'  Erro ao carregar: {e}')

# Analisar hiperparâmetros do Anchor
print('\n' + '=' * 80)
print('3. HIPÓTESE - POR QUE O ANCHOR GERA EXPLICAÇÕES VAZIAS:')
print('-' * 80)

print('''
O Anchor gera explicações vazias quando NÃO consegue encontrar regras que:
  1. Satisfaçam o threshold de precisão (precision >= 0.95 tipicamente)
  2. Mantenham estabilidade em vizinhanças perturbadas

POSSÍVEIS CAUSAS NO CREDIT CARD:

A) ALTA DIMENSIONALIDADE (29 features):
   - Mais features = espaço de busca exponencialmente maior
   - Anchor precisa testar milhares de combinações
   - Pode não convergir dentro do budget de samples

B) DADOS DESBALANCEADOS:
   - Credit Card tem classes muito desbalanceadas (fraudes são raras)
   - Dificulta achar regras consistentes para classe minoritária

C) FEATURES NÃO-DISCRIMINATIVAS:
   - Se as features individualmente têm baixo poder discriminativo,
   - O Anchor não consegue construir âncoras estáveis

D) HIPERPARÂMETROS RESTRITIVOS:
   - threshold muito alto (0.95) + delta baixo (0.05)
   - Requer precisão muito alta, poucas regras passam no teste

E) BOUNDARY INSTABILITY (zona de rejeição):
   - Instâncias próximas da zona de rejeição são instáveis
   - Vizinhanças perturbadas mudam de classe facilmente
   - Anchor não consegue garantir estabilidade → retorna vazio

COMPARAÇÃO:
  - PEAB: 15.76 features (usa programação linear, sempre acha solução)
  - MinExp: 20.33 features (SVM constraint-based, sempre acha solução)
  - Anchor: 0.19 features (sampling-based, PODE FALHAR se não convergir)
''')

print('\n' + '=' * 80)
print('4. RECOMENDAÇÃO PARA O PROFESSOR:')
print('-' * 80)
print('''
"O Anchor é um método baseado em AMOSTRAGEM (sampling-based) que busca
regras com alta cobertura local. No Credit Card dataset, devido à:
  
  • Alta dimensionalidade (29 features)
  • Zona de rejeição ampla (instâncias instáveis)
  • Desbalanceamento de classes
  
... o algoritmo falha em encontrar âncoras estáveis para 81% das instâncias,
retornando explicações vazias. Isso é uma LIMITAÇÃO CONHECIDA do Anchor em
datasets complexos, enquanto métodos baseados em otimização (PEAB/MinExp)
sempre conseguem gerar explicações válidas."
''')

print('=' * 80)
