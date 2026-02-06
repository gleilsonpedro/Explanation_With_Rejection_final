"""
Script para gerar resumo executivo conciso dos experimentos.
Formato compacto ideal para colar direto no artigo.
"""
import os
import json
import glob

DATASET_SOURCES_SHORT = {
    'mnist': 'MNIST',
    'breast_cancer': 'Breast Cancer (UCI)',
    'pima_indians_diabetes': 'Pima Diabetes (UCI)',
    'vertebral_column': 'Vertebral Column (UCI)',
    'sonar': 'Sonar (UCI)',
    'spambase': 'Spambase (UCI)',
    'banknote': 'Banknote (UCI)',
    'heart_disease': 'Heart Disease (UCI)',
    'creditcard': 'Credit Card Fraud',
    'covertype': 'Covertype (UCI)',
    'gas_sensor': 'Gas Sensor (UCI)',
    'newsgroups': '20 Newsgroups',
    'rcv1': 'RCV1'
}

def extract_info(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = data.get('config', {})
        performance = data.get('performance', {})
        model = data.get('model', {})
        return {
            'dataset_name': config.get('dataset_name'),
            'test_size': config.get('test_size'),
            'subsample_size': config.get('subsample_size'),
            'num_features': model.get('num_features'),
            'acc_wo': performance.get('accuracy_without_rejection'),
            'acc_w': performance.get('accuracy_with_rejection'),
            'rej_rate': performance.get('rejection_rate'),
        }
    except:
        return None

print("\n" + "="*90)
print("RESUMO EXECUTIVO - CONFIGURACAO E RESULTADOS DOS EXPERIMENTOS")
print("="*90 + "\n")

json_files = glob.glob('json/peab/*.json')
infos = []

for jf in sorted(json_files):
    info = extract_info(jf)
    if info:
        infos.append(info)

# Tabela compacta
print("Dataset                    Features  Split    Subsample  Acc.Base  Acc.Rej  RejRate  Ganho")
print("-" * 90)

for info in infos:
    name = info['dataset_name']
    name_short = DATASET_SOURCES_SHORT.get(name, name)[:25]
    
    features = info['num_features'] if info['num_features'] else '-'
    
    split = f"70/30" if info['test_size'] == 0.3 else "N/A"
    
    subsample = f"{int(info['subsample_size']*100)}%" if info['subsample_size'] else "100%"
    
    acc_wo = f"{info['acc_wo']:.1f}" if info['acc_wo'] else "--"
    acc_w = f"{info['acc_w']:.1f}" if info['acc_w'] else "--"
    rej = f"{info['rej_rate']:.1f}" if info['rej_rate'] else "--"
    
    gain = ""
    if info['acc_wo'] and info['acc_w']:
        g = info['acc_w'] - info['acc_wo']
        gain = f"{g:+.1f}"
    
    print(f"{name_short:<26} {str(features):<9} {split:<8} {subsample:<10} {acc_wo:>7}%  {acc_w:>7}%  {rej:>6}%  {gain:>5}pp")

print("-" * 90)

# Estatísticas
gains = [info['acc_w'] - info['acc_wo'] for info in infos if info['acc_w'] and info['acc_wo']]
rej_rates = [info['rej_rate'] for info in infos if info['rej_rate']]

print(f"\nESTATISTICAS GERAIS:")
print(f"  Total datasets: {len(infos)}")
print(f"  Datasets com subsample: {sum(1 for i in infos if i['subsample_size'])}")
print(f"  Ganho medio de acuracia: {sum(gains)/len(gains):.2f} pontos percentuais")
print(f"  Taxa media de rejeicao: {sum(rej_rates)/len(rej_rates):.2f}%")

print("\nCONFIGURACOES ESPECIAIS:")
for info in infos:
    if info['subsample_size']:
        pct = int(info['subsample_size'] * 100)
        print(f"  - {info['dataset_name']}: subsample {pct}%")

# Para MNIST
mnist_info = next((i for i in infos if 'mnist' in i['dataset_name'].lower()), None)
if mnist_info and mnist_info['num_features'] == 784:
    print(f"  - MNIST: raw features (28x28 pixels), digitos 3 vs 8")
elif mnist_info and mnist_info['num_features'] == 196:
    print(f"  - MNIST: pooling 2x2 (14x14), digitos 3 vs 8")

print("\n" + "="*90)
print("Fim do resumo executivo")
print("="*90 + "\n")
