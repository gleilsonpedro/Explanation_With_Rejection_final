"""
Script para gerar pipeline completo dos experimentos para inclusão no artigo (versão texto puro).
Exibe informações sobre fonte do dataset, split treino/teste, subsample, pooling,
configurações especiais do MNIST e métricas principais.
"""
import os
import json
import glob

# Mapeamento de fontes dos datasets
DATASET_SOURCES = {
    'mnist': 'LeCun et al. (1998) - MNIST Database',
    'breast_cancer': 'UCI ML Repository - Breast Cancer Wisconsin (Diagnostic)',
    'pima_indians_diabetes': 'UCI ML Repository - Pima Indians Diabetes',
    'vertebral_column': 'UCI ML Repository - Vertebral Column',
    'sonar': 'UCI ML Repository - Connectionist Bench (Sonar, Mines vs. Rocks)',
    'spambase': 'UCI ML Repository - Spambase',
    'banknote': 'UCI ML Repository - Banknote Authentication',
    'heart_disease': 'UCI ML Repository - Heart Disease (Cleveland)',
    'creditcard': 'Kaggle/OpenML - Credit Card Fraud Detection',
    'covertype': 'UCI ML Repository - Covertype',
    'gas_sensor': 'UCI ML Repository - Gas Sensor Array Drift',
    'newsgroups': '20 Newsgroups Dataset',
    'rcv1': 'Reuters Corpus Volume I (RCV1)'
}

def format_percentage(value):
    """Formata valores decimais como porcentagem."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"

def extract_info_from_json(json_path):
    """Extrai informações relevantes do arquivo JSON de resultados."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = data.get('config', {})
        performance = data.get('performance', {})
        model = data.get('model', {})
        
        info = {
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
        
        return info
    except Exception as e:
        print(f"Erro ao ler {json_path}: {e}")
        return None

def detect_mnist_config(json_path):
    """Detecta configurações especiais do MNIST (feature mode, digit pair)."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_features = data.get('model', {}).get('num_features')
        
        # Detecta feature mode baseado no número de features
        feature_mode = None
        if num_features == 784:
            feature_mode = 'raw (28x28 pixels)'
        elif num_features == 196:
            feature_mode = 'pool2x2 (14x14, pooling 2x2)'
        
        # Tenta extrair digit pair do nome do arquivo
        filename = os.path.basename(json_path)
        digit_pair = None
        if 'mnist' in filename.lower():
            import re
            match = re.search(r'mnist_(\d+)_vs_(\d+)', filename)
            if match:
                digit_pair = f"Digitos {match.group(1)} vs {match.group(2)}"
        
        return feature_mode, digit_pair
    except Exception:
        return None, None

def generate_pipeline_section(method_name='peab'):
    """Gera seção do pipeline para um método específico."""
    json_dir = f'json/{method_name}'
    
    if not os.path.exists(json_dir):
        print(f"[ERRO] Diretorio {json_dir} nao encontrado!")
        return
    
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    
    if not json_files:
        print(f"[ERRO] Nenhum arquivo JSON encontrado em {json_dir}")
        return
    
    print("\n" + "="*100)
    print(f"PIPELINE EXPERIMENTAL - METODO: {method_name.upper()}")
    print("="*100)
    
    for json_file in sorted(json_files):
        info = extract_info_from_json(json_file)
        
        if not info:
            continue
        
        dataset_name = info['dataset_name']
        
        print(f"\n{'-'*100}")
        print(f"[DATASET] {dataset_name.upper()}")
        print(f"{'-'*100}")
        
        # Fonte do dataset
        source = DATASET_SOURCES.get(dataset_name, 'Unknown Source')
        print(f"Fonte: {source}")
        
        # Split treino/teste
        test_size = info['test_size']
        if test_size:
            train_size = 1 - test_size
            print(f"Split Treino/Teste: {format_percentage(train_size)} / {format_percentage(test_size)}")
        
        # Subsample
        subsample = info['subsample_size']
        if subsample:
            print(f"Subsample: {format_percentage(subsample)} do dataset original")
        
        # Número de features
        num_features = info['num_features']
        if num_features:
            print(f"Numero de Features: {num_features}")
        
        # Configurações especiais MNIST
        if 'mnist' in dataset_name.lower():
            feature_mode, digit_pair = detect_mnist_config(json_file)
            if feature_mode:
                print(f"Feature Mode (MNIST): {feature_mode}")
            if digit_pair:
                print(f"Classificacao Binaria: {digit_pair}")
        
        # Rejection cost
        rejection_cost = info['rejection_cost']
        if rejection_cost:
            print(f"Custo de Rejeicao: {rejection_cost}")
        
        # Métricas de performance
        print(f"\n[RESULTADOS]")
        print(f"  - Instancias de Teste: {info['num_test_instances']}")
        
        acc_wo = info['accuracy_without_rejection']
        if acc_wo:
            print(f"  - Acuracia sem rejeicao: {acc_wo:.2f}%")
        
        acc_w = info['accuracy_with_rejection']
        if acc_w:
            print(f"  - Acuracia com rejeicao: {acc_w:.2f}%")
        
        rej_rate = info['rejection_rate']
        if rej_rate:
            print(f"  - Taxa de Rejeicao: {rej_rate:.2f}%")
        
        # Ganho de acurácia
        if acc_wo and acc_w:
            gain = acc_w - acc_wo
            if gain > 0:
                print(f"  - Ganho de Acuracia: +{gain:.2f} pontos percentuais")
            else:
                print(f"  - Ganho de Acuracia: {gain:.2f} pontos percentuais")
        
        # Observações
        if rej_rate and rej_rate > 50:
            print(f"  [OBSERVACAO] Alta taxa de rejeicao (>50%)")
        
        if acc_w and acc_w > 95:
            print(f"  [OBSERVACAO] Alta acuracia final (>95%)")
    
    print("\n" + "="*100)
    print("FIM DO PIPELINE")
    print("="*100 + "\n")

def generate_comparison_table():
    """Gera tabela comparativa consolidada de todos os datasets."""
    json_dir = 'json/peab'
    
    if not os.path.exists(json_dir):
        return
    
    print("\n" + "="*100)
    print("TABELA RESUMO - CONFIGURACAO EXPERIMENTAL DOS DATASETS")
    print("="*100 + "\n")
    
    # Coleta informações
    datasets_info = {}
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    
    for json_file in json_files:
        info = extract_info_from_json(json_file)
        if not info:
            continue
        
        dataset_name = info['dataset_name']
        
        if dataset_name not in datasets_info:
            datasets_info[dataset_name] = {
                'source': DATASET_SOURCES.get(dataset_name, 'Unknown'),
                'test_size': info['test_size'],
                'subsample_size': info['subsample_size'],
                'rejection_cost': info['rejection_cost'],
                'num_features': info['num_features']
            }
    
    # Imprime tabela formatada
    print(f"{'Dataset':<25} {'Split (Train/Test)':<20} {'Subsample':<12} {'Features':<10} {'Rej.Cost':<10}")
    print("-" * 100)
    
    for dataset_name, info in sorted(datasets_info.items()):
        test_split = format_percentage(info['test_size']) if info['test_size'] else "N/A"
        train_split = format_percentage(1 - info['test_size']) if info['test_size'] else "N/A"
        split_str = f"{train_split}/{test_split}"
        subsample = format_percentage(info['subsample_size']) if info['subsample_size'] else "100%"
        features = str(info['num_features']) if info['num_features'] else "N/A"
        rej_cost = f"{info['rejection_cost']:.2f}" if info['rejection_cost'] else "N/A"
        
        print(f"{dataset_name:<25} {split_str:<20} {subsample:<12} {features:<10} {rej_cost:<10}")
    
    print("\n" + "-" * 100)
    print("Legenda: Split = Treino/Teste | Subsample = % do dataset original usado | Rej.Cost = Custo de Rejeicao")
    print("="*100 + "\n")

if __name__ == "__main__":
    print("\n" + "="*100)
    print("GERADOR DE PIPELINE PARA ARTIGO CIENTIFICO")
    print("="*100 + "\n")
    
    # Verifica métodos disponíveis
    available_methods = []
    for method in ['peab', 'pulp', 'anchor', 'minexp']:
        json_dir = f'json/{method}'
        if os.path.exists(json_dir):
            json_files = glob.glob(os.path.join(json_dir, '*.json'))
            if json_files:
                available_methods.append(method)
    
    if not available_methods:
        print("[ERRO] Nenhum resultado encontrado nos diretorios json/")
        exit(1)
    
    print(f"Metodos encontrados: {', '.join(available_methods)}\n")
    
    # Gera pipeline para o primeiro método disponível (ou PEAB se existir)
    primary_method = 'peab' if 'peab' in available_methods else available_methods[0]
    generate_pipeline_section(primary_method)
    
    # Gera tabela comparativa consolidada
    generate_comparison_table()
    
    # Estatísticas gerais
    json_dir = f'json/{primary_method}'
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    
    print("\n" + "="*100)
    print("ESTATISTICAS GERAIS DO EXPERIMENTO")
    print("="*100 + "\n")
    
    total_datasets = len(json_files)
    datasets_with_subsample = 0
    total_test_instances = 0
    avg_accuracy_gain = []
    avg_rejection_rate = []
    
    for json_file in json_files:
        info = extract_info_from_json(json_file)
        if not info:
            continue
        
        if info['subsample_size']:
            datasets_with_subsample += 1
        
        if info['num_test_instances']:
            total_test_instances += info['num_test_instances']
        
        if info['accuracy_without_rejection'] and info['accuracy_with_rejection']:
            gain = info['accuracy_with_rejection'] - info['accuracy_without_rejection']
            avg_accuracy_gain.append(gain)
        
        if info['rejection_rate']:
            avg_rejection_rate.append(info['rejection_rate'])
    
    print(f"Total de Datasets Avaliados: {total_datasets}")
    print(f"Datasets com Subsample: {datasets_with_subsample}")
    print(f"Total de Instancias de Teste: {total_test_instances:,}")
    
    if avg_accuracy_gain:
        mean_gain = sum(avg_accuracy_gain) / len(avg_accuracy_gain)
        max_gain = max(avg_accuracy_gain)
        min_gain = min(avg_accuracy_gain)
        print(f"\nGanho Medio de Acuracia: {mean_gain:.2f} pontos percentuais")
        print(f"Ganho Maximo de Acuracia: {max_gain:.2f} pontos percentuais")
        print(f"Ganho Minimo de Acuracia: {min_gain:.2f} pontos percentuais")
    
    if avg_rejection_rate:
        mean_rej = sum(avg_rejection_rate) / len(avg_rejection_rate)
        max_rej = max(avg_rejection_rate)
        min_rej = min(avg_rejection_rate)
        print(f"\nTaxa Media de Rejeicao: {mean_rej:.2f}%")
        print(f"Taxa Maxima de Rejeicao: {max_rej:.2f}%")
        print(f"Taxa Minima de Rejeicao: {min_rej:.2f}%")
    
    print("\n" + "="*100)
    
    print("\n[SUCESSO] Pipeline gerado com sucesso!")
    print("[INFO] Voce pode copiar a saida acima diretamente para seu artigo.")
    print("[INFO] Para salvar em arquivo: python temporarios\\gerar_pipeline_artigo_txt.py > pipeline.txt\n")
