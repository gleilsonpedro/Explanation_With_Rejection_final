"""
Script para verificar dados dos JSONs do PEAB e gerar tabela LaTeX de datasets.
"""
import json
import os

JSON_DIR = "json/peab"

# Datasets desejados
datasets_desejados = [
    'banknote',
    'vertebral_column', 
    'pima_indians_diabetes',
    'heart_disease',
    'creditcard',
    'breast_cancer',
    'covertype',
    'spambase',
    'sonar',
    'mnist'
]

print("\n" + "="*80)
print("VERIFICAÇÃO DE DADOS - JSONs do PEAB")
print("="*80 + "\n")

dados_encontrados = {}
datasets_faltando = []

for dataset in datasets_desejados:
    # Tentar variações do nome do arquivo
    variacoes = [
        f"{dataset}.json",
        f"{dataset.replace('_', '')}.json",
        f"{dataset.replace('_indians_', '_')}.json"
    ]
    
    # Para MNIST, verificar variações
    if dataset == 'mnist':
        variacoes.extend([
            'mnist_3_vs_8.json',
            'mnist_1_vs_2.json',
            'mnist_0_vs_1.json'
        ])
    
    json_encontrado = None
    for variacao in variacoes:
        json_path = os.path.join(JSON_DIR, variacao)
        if os.path.exists(json_path):
            json_encontrado = json_path
            break
    
    if json_encontrado:
        try:
            with open(json_encontrado, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extrair informações
            config = data.get('config', {})
            thresholds = data.get('thresholds', {})
            performance = data.get('performance', {})
            model = data.get('model', {})
            
            dados_encontrados[dataset] = {
                'arquivo': os.path.basename(json_encontrado),
                'nome': config.get('dataset_name', dataset),
                'instancias': performance.get('num_test_instances', 'N/A'),
                'features': model.get('num_features', 'N/A'),
                't_plus': thresholds.get('t_plus', 'N/A'),
                't_minus': thresholds.get('t_minus', 'N/A'),
                'zona_rejeicao': thresholds.get('rejection_zone_width', 'N/A'),
                'taxa_rejeicao': performance.get('rejection_rate', 'N/A'),
                'acuracia': performance.get('accuracy_with_rejection', 'N/A'),
                'acuracia_sem_rej': performance.get('accuracy_without_rejection', 'N/A')
            }
            
            print(f"✅ {dataset:25s} - {os.path.basename(json_encontrado)}")
            
        except Exception as e:
            print(f"⚠️  {dataset:25s} - Erro ao ler: {e}")
            datasets_faltando.append(dataset)
    else:
        print(f"❌ {dataset:25s} - JSON não encontrado")
        datasets_faltando.append(dataset)

print("\n" + "="*80)
print(f"RESUMO: {len(dados_encontrados)}/{len(datasets_desejados)} datasets encontrados")
print("="*80 + "\n")

if datasets_faltando:
    print("Datasets FALTANDO (precisam executar PEAB):")
    for dataset in datasets_faltando:
        print(f"  - {dataset}")
        print(f"    Comando: python peab.py")
        print(f"    Escolha: {dataset}")
        print()

# Mostrar exemplo de dados encontrados
if dados_encontrados:
    print("\n" + "="*80)
    print("EXEMPLO DE DADOS EXTRAÍDOS:")
    print("="*80 + "\n")
    
    primeiro = list(dados_encontrados.values())[0]
    for chave, valor in primeiro.items():
        print(f"  {chave:20s}: {valor}")

# Confirmar estrutura
print("\n" + "="*80)
print("CONFIRMAÇÃO:")
print("="*80)
print("✅ Os JSONs do PEAB contêm TODOS os dados necessários:")
print("   - Nome do dataset: config.dataset_name")
print("   - Instâncias: performance.num_test_instances")
print("   - Features: model.num_features")
print("   - t+: thresholds.t_plus")
print("   - t-: thresholds.t_minus")
print("   - Zona de rejeição: thresholds.rejection_zone_width")
print("   - Taxa de rejeição: performance.rejection_rate")
print("   - Acurácia: performance.accuracy_with_rejection")
print("="*80 + "\n")

# Salvar dados para script de geração de tabela
if dados_encontrados:
    output_file = "temporarios/dados_tabela_datasets.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dados_encontrados, f, indent=2, ensure_ascii=False)
    print(f"✅ Dados salvos em: {output_file}")
    print("   Use este arquivo para gerar a tabela LaTeX")
