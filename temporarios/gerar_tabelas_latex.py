"""
Script para gerar pipeline em formato LaTeX para o artigo.
Gera tabelas e descrições prontas para copiar no artigo.
"""
import os
import json
import glob

# Mapeamento de fontes dos datasets
DATASET_SOURCES = {
    'mnist': 'LeCun et al.~\\cite{lecun1998mnist}',
    'breast_cancer': 'UCI ML Repository~\\cite{uci_breast_cancer}',
    'pima_indians_diabetes': 'UCI ML Repository~\\cite{uci_pima}',
    'vertebral_column': 'UCI ML Repository~\\cite{uci_vertebral}',
    'sonar': 'UCI ML Repository~\\cite{uci_sonar}',
    'spambase': 'UCI ML Repository~\\cite{uci_spambase}',
    'banknote': 'UCI ML Repository~\\cite{uci_banknote}',
    'heart_disease': 'UCI ML Repository~\\cite{uci_heart}',
    'creditcard': 'Kaggle~\\cite{kaggle_creditcard}',
    'covertype': 'UCI ML Repository~\\cite{uci_covertype}',
    'gas_sensor': 'UCI ML Repository~\\cite{uci_gas_sensor}',
    'newsgroups': '20 Newsgroups~\\cite{newsgroups}',
    'rcv1': 'RCV1~\\cite{rcv1}'
}

DATASET_NAMES_LATEX = {
    'mnist': 'MNIST (3 vs 8)',
    'breast_cancer': 'Breast Cancer',
    'pima_indians_diabetes': 'Pima Indians Diabetes',
    'vertebral_column': 'Vertebral Column',
    'sonar': 'Sonar',
    'spambase': 'Spambase',
    'banknote': 'Banknote',
    'heart_disease': 'Heart Disease',
    'creditcard': 'Credit Card Fraud',
    'covertype': 'Covertype',
    'gas_sensor': 'Gas Sensor',
    'newsgroups': '20 Newsgroups',
    'rcv1': 'RCV1'
}

def extract_info_from_json(json_path):
    """Extrai informações relevantes do arquivo JSON."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = data.get('config', {})
        performance = data.get('performance', {})
        model = data.get('model', {})
        
        return {
            'dataset_name': config.get('dataset_name', 'Unknown'),
            'test_size': config.get('test_size'),
            'subsample_size': config.get('subsample_size'),
            'rejection_cost': config.get('rejection_cost'),
            'num_features': model.get('num_features'),
            'accuracy_without_rejection': performance.get('accuracy_without_rejection'),
            'accuracy_with_rejection': performance.get('accuracy_with_rejection'),
            'rejection_rate': performance.get('rejection_rate'),
            'num_test_instances': performance.get('num_test_instances'),
        }
    except Exception as e:
        return None

def generate_latex_table():
    """Gera tabela LaTeX com configurações dos datasets."""
    json_dir = 'json/peab'
    
    if not os.path.exists(json_dir):
        print("❌ Diretório json/peab não encontrado!")
        return
    
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    datasets_info = {}
    
    for json_file in json_files:
        info = extract_info_from_json(json_file)
        if not info:
            continue
        
        dataset_name = info['dataset_name']
        datasets_info[dataset_name] = info
    
    print("\n% ============================================")
    print("% TABELA LATEX - CONFIGURAÇÃO DOS DATASETS")
    print("% ============================================\n")
    
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Configuração experimental dos datasets utilizados.}")
    print("\\label{tab:dataset_config}")
    print("\\begin{tabular}{lrrrrr}")
    print("\\toprule")
    print("\\textbf{Dataset} & \\textbf{Features} & \\textbf{Train/Test} & \\textbf{Subsample} & \\textbf{Test Inst.} & \\textbf{Rej. Cost} \\\\")
    print("\\midrule")
    
    for dataset_name in sorted(datasets_info.keys()):
        info = datasets_info[dataset_name]
        
        name_latex = DATASET_NAMES_LATEX.get(dataset_name, dataset_name)
        features = info['num_features'] if info['num_features'] else '--'
        
        if info['test_size']:
            train_pct = int((1 - info['test_size']) * 100)
            test_pct = int(info['test_size'] * 100)
            split = f"{train_pct}/{test_pct}"
        else:
            split = "--"
        
        if info['subsample_size']:
            subsample = f"{int(info['subsample_size'] * 100)}\\%"
        else:
            subsample = "100\\%"
        
        test_inst = info['num_test_instances'] if info['num_test_instances'] else '--'
        rej_cost = f"{info['rejection_cost']:.2f}" if info['rejection_cost'] else '--'
        
        print(f"{name_latex} & {features} & {split} & {subsample} & {test_inst} & {rej_cost} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

def generate_results_latex_table():
    """Gera tabela LaTeX com resultados de desempenho."""
    json_dir = 'json/peab'
    
    if not os.path.exists(json_dir):
        return
    
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    datasets_info = {}
    
    for json_file in json_files:
        info = extract_info_from_json(json_file)
        if not info:
            continue
        
        dataset_name = info['dataset_name']
        datasets_info[dataset_name] = info
    
    print("\n% ============================================")
    print("% TABELA LATEX - RESULTADOS DE DESEMPENHO")
    print("% ============================================\n")
    
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Resultados de desempenho do método PEAB nos datasets avaliados.}")
    print("\\label{tab:peab_results}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("\\textbf{Dataset} & \\textbf{Acc. s/ Rej.} & \\textbf{Acc. c/ Rej.} & \\textbf{Taxa Rej.} & \\textbf{Ganho} \\\\")
    print(" & (\\%) & (\\%) & (\\%) & (p.p.) \\\\")
    print("\\midrule")
    
    for dataset_name in sorted(datasets_info.keys()):
        info = datasets_info[dataset_name]
        
        name_latex = DATASET_NAMES_LATEX.get(dataset_name, dataset_name)
        
        acc_wo = f"{info['accuracy_without_rejection']:.2f}" if info['accuracy_without_rejection'] else '--'
        acc_w = f"{info['accuracy_with_rejection']:.2f}" if info['accuracy_with_rejection'] else '--'
        rej_rate = f"{info['rejection_rate']:.2f}" if info['rejection_rate'] else '--'
        
        if info['accuracy_without_rejection'] and info['accuracy_with_rejection']:
            gain = info['accuracy_with_rejection'] - info['accuracy_without_rejection']
            gain_str = f"{gain:+.2f}"
        else:
            gain_str = '--'
        
        print(f"{name_latex} & {acc_wo} & {acc_w} & {rej_rate} & {gain_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

def generate_text_description():
    """Gera descrição textual para a seção de metodologia do artigo."""
    json_dir = 'json/peab'
    
    if not os.path.exists(json_dir):
        return
    
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    datasets_info = {}
    
    for json_file in json_files:
        info = extract_info_from_json(json_file)
        if not info:
            continue
        
        dataset_name = info['dataset_name']
        datasets_info[dataset_name] = info
    
    print("\n% ============================================")
    print("% DESCRIÇÃO TEXTUAL PARA METODOLOGIA")
    print("% ============================================\n")
    
    total_datasets = len(datasets_info)
    datasets_with_subsample = sum(1 for info in datasets_info.values() if info['subsample_size'])
    total_test = sum(info['num_test_instances'] for info in datasets_info.values() if info['num_test_instances'])
    
    print(f"% Para o artigo, seção de metodologia/experimentos:\n")
    print(f"Foram avaliados {total_datasets} datasets públicos amplamente utilizados na literatura")
    print(f"de aprendizado de máquina. O protocolo experimental consistiu em dividir cada dataset")
    print(f"em conjuntos de treinamento (70\\%) e teste (30\\%), utilizando estratificação para")
    print(f"preservar a distribuição de classes. Para datasets de grande porte, foi aplicado")
    print(f"subsampling estratificado ({datasets_with_subsample} datasets), mantendo a proporção")
    print(f"entre as classes. No total, foram testadas {total_test:,} instâncias.\n")
    
    print("% Configurações especiais:")
    for dataset_name, info in datasets_info.items():
        if info['subsample_size']:
            pct = int(info['subsample_size'] * 100)
            print(f"% - {dataset_name}: subsample de {pct}% do dataset original")
    
    # Estatísticas de desempenho
    gains = [info['accuracy_with_rejection'] - info['accuracy_without_rejection'] 
             for info in datasets_info.values() 
             if info['accuracy_with_rejection'] and info['accuracy_without_rejection']]
    
    if gains:
        mean_gain = sum(gains) / len(gains)
        max_gain = max(gains)
        min_gain = min(gains)
        
        print(f"\n% Resultados gerais:")
        print(f"% - Ganho médio de acurácia: {mean_gain:.2f} pontos percentuais")
        print(f"% - Ganho máximo: {max_gain:.2f} p.p.")
        print(f"% - Ganho mínimo: {min_gain:.2f} p.p.")
        
        print(f"\nO método PEAB apresentou ganho médio de {mean_gain:.2f} pontos percentuais")
        print(f"na acurácia, variando de {min_gain:.2f} a {max_gain:.2f} pontos percentuais")
        print(f"dependendo do dataset.\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GERADOR DE TABELAS LATEX PARA ARTIGO")
    print("="*80)
    
    generate_latex_table()
    generate_results_latex_table()
    generate_text_description()
    
    print("\n" + "="*80)
    print("✅ Tabelas LaTeX geradas com sucesso!")
    print("💡 Copie as tabelas acima diretamente para o seu arquivo .tex")
    print("💡 Não esqueça de adicionar \\usepackage{booktabs} no preâmbulo")
    print("="*80 + "\n")
