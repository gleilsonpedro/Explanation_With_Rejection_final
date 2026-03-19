# datasets.py (MODIFICADO)
import os
import io
import zipfile
import time
import re
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_openml, load_breast_cancer, fetch_covtype
from typing import List, Tuple, Dict, Any, Optional

# ------------------------ Utilitários de download/cache ------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _parse_dataset_selection(input_str: str, max_index: int) -> List[int]:
    """
    Parse string de seleção de datasets.
    
    Exemplos:
        "0,2,3" -> [0, 2, 3]
        "0-5" -> [0, 1, 2, 3, 4, 5]
        "0,2-4,7" -> [0, 2, 3, 4, 7]
    
    Args:
        input_str: String com seleção (ex: "0,2,3" ou "0-5")
        max_index: Índice máximo permitido
    
    Returns:
        Lista de índices válidos
    """
    indices = set()
    parts = input_str.split(',')
    
    for part in parts:
        part = part.strip()
        
        # Verifica se é um range (ex: "0-5")
        if '-' in part:
            try:
                start, end = part.split('-')
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                
                for i in range(start_idx, end_idx + 1):
                    if 0 <= i <= max_index:
                        indices.add(i)
            except ValueError:
                continue
        
        # Índice simples
        else:
            try:
                idx = int(part)
                if 0 <= idx <= max_index:
                    indices.add(idx)
            except ValueError:
                continue
    
    return sorted(list(indices))


def _download_with_retries(url: str, dest_path: str, max_retries: int = 3, timeout: int = 20) -> bool:
    """Tenta baixar um arquivo com cabeçalho User-Agent e backoff exponencial.
    Retorna True se salvou em dest_path, False caso contrário.
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    backoff = 1.5
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            # Respeita 429 Too Many Requests com pequeno backoff
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_s = min(int(retry_after), 10)
                else:
                    wait_s = min(int(delay), 10)
                time.sleep(wait_s)
                delay *= backoff
                continue

            if 200 <= resp.status_code < 300:
                _ensure_dir(dest_path)
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                return True
            # Em outros códigos, tenta novamente com backoff leve
            time.sleep(min(int(delay), 5))
            delay *= backoff
        except requests.RequestException:
            time.sleep(min(int(delay), 5))
            delay *= backoff
    return False


def _read_csv_with_cache(urls: List[str], local_path: str, **read_csv_kwargs) -> pd.DataFrame:
    """Lê CSV de um caminho local (se existir) ou tenta baixar de uma lista de URLs.
    Em caso de falha no download, levanta uma exceção com instruções para cache manual.
    """
    # 1) Usa cache local se existir
    if os.path.exists(local_path):
        return pd.read_csv(local_path, **read_csv_kwargs)

    # 2) Tenta baixar de alguma URL
    for url in urls:
        if _download_with_retries(url, local_path):
            try:
                return pd.read_csv(local_path, **read_csv_kwargs)
            except Exception:
                # Arquivo baixado pode não estar conforme esperado; tenta próxima URL
                try:
                    os.remove(local_path)
                except OSError:
                    pass
                continue

    # 3) Falhou totalmente
    raise RuntimeError(
        (
            f"Não foi possível obter o dataset a partir das URLs fornecidas. "
            f"Por favor, baixe manualmente o arquivo e salve em '{local_path}'."
        )
    )


def _fetch_openml_with_retries(dataset_name: str, version: int = 1, as_frame: bool = True, max_retries: int = 3):
    """Wrapper para fetch_openml com tentativas e backoff simples."""
    delay = 1.0
    backoff = 1.7
    last_err = None
    for _ in range(max_retries):
        try:
            return fetch_openml(dataset_name, version=version, as_frame=as_frame, parser='auto')
        except Exception as e:
            last_err = e
            time.sleep(min(int(delay), 5))
            delay *= backoff
    raise last_err if last_err else RuntimeError(f"Falha ao baixar dataset OpenML: {dataset_name}")

RANDOM_STATE: int = 42

# ------------------------ Opções globais específicas do MNIST ------------------------
# Modo de features para MNIST: 'raw' (784) ou 'pool2x2' (196)
MNIST_FEATURE_MODE: str = 'raw'
# Par de dígitos selecionados (classe A vs classe B). None => usar todas as classes originais
MNIST_SELECTED_PAIR: Optional[Tuple[int, int]] = None


def set_mnist_options(feature_mode: str, pair: Optional[Tuple[int, int]]):
    """Define opções globais de MNIST que serão respeitadas por carregar_dataset.
    feature_mode: 'raw' ou 'pool2x2'; pair: (dA, dB) ou None.
    """
    global MNIST_FEATURE_MODE, MNIST_SELECTED_PAIR
    if feature_mode not in ('raw', 'pool2x2'):
        feature_mode = 'raw'
    MNIST_FEATURE_MODE = feature_mode
    MNIST_SELECTED_PAIR = pair


def _pool2x2(arr28: np.ndarray) -> np.ndarray:
    """Aplica pooling 2x2 por média em uma imagem 28x28, resultando em 14x14 (196)."""
    out = np.zeros((14, 14), dtype=float)
    for r in range(14):
        for c in range(14):
            block = arr28[2*r:2*r+2, 2*c:2*c+2]
            out[r, c] = float(block.mean())
    return out


def _apply_pooling_df(X: pd.DataFrame) -> pd.DataFrame:
    """Converte DataFrame com 784 colunas (28x28) para 196 (14x14) via pooling 2x2."""
    # cria mapeamento de ordem de colunas para 28x28
    # assume colunas ordenadas por índice (feature_0..feature_783 ou similares)
    arr = X.values.astype(float)
    n = arr.shape[0]
    pooled = np.zeros((n, 14*14), dtype=float)
    for i in range(n):
        img = arr[i].reshape(28, 28)
        pooled[i] = _pool2x2(img).reshape(-1)
    cols = [f"bin_{r}_{c}" for r in range(14) for c in range(14)]
    return pd.DataFrame(pooled, index=X.index, columns=cols)

# --- Funções de Carregamento e Preparação de Dataset ---
def carregar_dataset(nome_dataset: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]: #
    try:
        X, y_series, class_names_list = None, None, None
        # --- Iris Dataset (Classes 0 e 1 apenas) ---
        if nome_dataset == 'iris':
            data = load_iris()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y_series = pd.Series(data.target, name='target')
            class_names_list = list(data.target_names)

            # Binarização OvR (classe 0 vs resto)
            y_series = np.where(y_series == 0, 0, 1)  # Setosa (0) vs Versicolor +      Virginica (1)
            class_names_list = [class_names_list[0], "Não-Setosa"]  # ["setosa",        "Não-Setosa"]
        elif nome_dataset == 'sonar':
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
            ]
            local_path = os.path.join('data', 'sonar.all-data')
            # As features não têm nomes, então vamos nomeá-las genericamente
            col_names = [f"feature_{i}" for i in range(60)] + ["target"]
            data_df = _read_csv_with_cache(urls, local_path, header=None, names=col_names)
            # Converte a classe alvo (M para mina, R para rocha) em 1 e 0
            data_df["target"] = data_df["target"].apply(lambda x: 1 if x == 'M' else 0)
            X = data_df.drop("target", axis=1)
            y_series = data_df["target"]
            class_names_list = ["Rocha", "Mina (Metal)"]
        elif nome_dataset == 'pima_indians_diabetes':
            # URLs alternativas para reduzir risco de 429 e quedas de serviço
            urls = [
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
            ]
            local_path = os.path.join('data', 'pima-indians-diabetes.csv')
            col_names = ['num_gravidezes', 'glicose', 'pressao_sangue', 'espessura_pele',
                         'insulina', 'imc', 'diabetes_pedigree', 'idade', 'target']
            # Usa cache local se existir; caso contrário, tenta baixar de uma das URLs
            data_df = _read_csv_with_cache(urls, local_path, header=None, names=col_names)
            
            # --- SEÇÃO DE TRATAMENTO REMOVIDA PARA COMPARAÇÃO DIRETA COM O DO MATEUS ---
            # As linhas abaixo, que limpavam os zeros inválidos, foram removidas.
            # colunas_com_zero_invalido = ['glicose', 'pressao_sangue', 'espessura_pele', 'insulina', 'imc']
            # data_df[colunas_com_zero_invalido] = data_df[colunas_com_zero_invalido].replace(0, np.nan)
            # data_df.dropna(inplace=True)
            # data_df.reset_index(drop=True, inplace=True)
            # --- FIM DA SEÇÃO REMOVIDA ---

            X = data_df.drop('target', axis=1)
            y_series = data_df['target'].astype(int)
            class_names_list = ['Não Diabético', 'Diabético']
            
        elif nome_dataset == 'breast_cancer':
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y_series = pd.Series(data.target, name='target')
            class_names_list = list(data.target_names)
        elif nome_dataset == 'banknote':
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
            ]
            local_path = os.path.join('data', 'data_banknote_authentication.txt')
            col_names = ["variance", "skewness", "curtosis", "entropy", "target"]
            data_df = _read_csv_with_cache(urls, local_path, names=col_names)
            X = data_df.drop("target", axis=1)
            y_series = data_df["target"].astype(int)
            class_names_list = ["Autêntica", "Falsificada"]
        elif nome_dataset == 'heart_disease':
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            ]
            col_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                         "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
            local_path = os.path.join('data', 'processed.cleveland.data')
            data_df = _read_csv_with_cache(urls, local_path, names=col_names, na_values="?")
            data_df.dropna(inplace=True)
            data_df["target"] = pd.to_numeric(data_df["target"], errors='coerce')
            data_df.dropna(subset=["target"], inplace=True)
            data_df["target"] = data_df["target"].astype(int).apply(lambda x: 1 if x > 0 else 0)
            X = data_df.drop("target", axis=1).astype(float)
            y_series = data_df["target"]
            class_names_list = ["Sem Doença", "Com Doença"]
        elif nome_dataset == 'creditcard':
            # Apenas carrega o dataset completo e o retorna.
            # A amostragem será feita no script principal.
            data_openml = _fetch_openml_with_retries('creditcard', version=1, as_frame=True)
            X = data_openml.data
            y_series = data_openml.target.astype(int)
            class_names_list = ['Normal', 'Fraude']
        elif nome_dataset == 'covertype':
            print("Carregando Covertype (581k instances, 54 features)...")
            from sklearn.datasets import fetch_covtype
            data = fetch_covtype()
            
            # Nomes reais das 54 features geográficas
            col_names = [
                "Elevation", "Aspect", "Slope", 
                "Horiz_Dist_To_Hydrology", "Vert_Dist_To_Hydrology", 
                "Horiz_Dist_To_Roadways", "Hillshade_9am", 
                "Hillshade_Noon", "Hillshade_3pm", "Horiz_Dist_To_Fire_Points"
            ]
            col_names += [f"Wilderness_Area_{i}" for i in range(1, 5)]
            col_names += [f"Soil_Type_{i}" for i in range(1, 41)]
            
            X = pd.DataFrame(data.data, columns=col_names)
            y_all = pd.Series(data.target, name='target')
            
            # Convert to binary: class 1 vs class 2 (most common classes)
            mask = (y_all == 1) | (y_all == 2)
            X = X[mask].copy()
            y_series = pd.Series(np.where(y_all[mask] == 1, 0, 1), index=X.index, name='target')
            class_names_list = ['Spruce/Fir', 'Lodgepole Pine']
            print(f"Covertype binário: {X.shape[0]} amostras ({sum(y_series==0)} classe 0, {sum(y_series==1)} classe 1)")
        elif nome_dataset == 'gas_sensor':
            # Gas Sensor Array Drift: 13.9k instances, 128 features, 6 classes
            # Binarization: 2 most frequent classes
            print("Carregando Gas Sensor (13.9k instances, 128 features)...")
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip"
            ]
            local_path = os.path.join('data', 'gas_sensor.zip')
            extracted_dir = os.path.join('data', 'gas_sensor_data')
            
            # Download and extract if not exists
            if not os.path.exists(extracted_dir):
                ok = _download_with_retries(urls[0], local_path)
                if not ok:
                    raise RuntimeError("Falha ao baixar gas_sensor. Faça o download manual e coloque em 'data/gas_sensor.zip'")
                with zipfile.ZipFile(local_path, 'r') as zf:
                    zf.extractall(extracted_dir)
            
            # Os arquivos batch estão dentro de Dataset/ após extração
            dataset_subdir = os.path.join(extracted_dir, 'Dataset')
            if not os.path.exists(dataset_subdir):
                dataset_subdir = extracted_dir  # Fallback se não houver subpasta
            
            # Load all batch files and concatenate
            batch_files = []
            for i in range(1, 11):  # 10 batches
                batch_path = os.path.join(dataset_subdir, f"batch{i}.dat")
                if os.path.exists(batch_path):
                    batch_files.append(batch_path)
            
            if not batch_files:
                raise RuntimeError(f"Nenhum arquivo batch encontrado em {dataset_subdir}")
            
            # Read and concatenate all batches (formato LIBSVM: class idx:val idx:val ...)
            all_rows = []
            all_labels = []
            num_features = 128  # Gas sensor tem 128 features
            
            for batch_file in batch_files:
                with open(batch_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        # Primeiro elemento é a classe
                        label = int(parts[0])
                        # Resto são pares idx:valor
                        row = np.zeros(num_features)
                        for pair in parts[1:]:
                            if ':' in pair:
                                idx_str, val_str = pair.split(':', 1)
                                idx = int(idx_str) - 1  # LIBSVM é 1-indexed
                                if 0 <= idx < num_features:
                                    row[idx] = float(val_str)
                        all_rows.append(row)
                        all_labels.append(label)
            
            X_all = pd.DataFrame(all_rows)
            y_all = pd.Series(all_labels)
            
            # Binarize: class 1 vs class 3 (2 most common)
            mask = (y_all == 1) | (y_all == 3)
            X = X_all[mask].copy()
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]
            y_series = pd.Series(np.where(y_all[mask] == 1, 0, 1), index=X.index, name='target')
            class_names_list = ['Ethanol', 'Ammonia']
            print(f"Gas Sensor binário: {X.shape[0]} amostras ({sum(y_series==0)} classe 0, {sum(y_series==1)} classe 1)")
        elif nome_dataset == 'vertebral_column':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
            col_names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", 
                         "sacral_slope", "pelvic_radius", "spondylolisthesis_grade", "target"]
            file_path = os.path.join('data', 'column_2C.dat')
            # Faz download e extrai se não existir localmente
            if not os.path.exists(file_path):
                zip_path = os.path.join('data', 'vertebral_column_data.zip')
                ok = _download_with_retries(url, zip_path)
                if not ok:
                    raise RuntimeError("Falha ao baixar vertebral_column_data.zip. Faça o download manual e coloque em 'data/'.")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Extrai apenas o arquivo 2 classes (column_2C.dat)
                    for name in zf.namelist():
                        if name.endswith('column_2C.dat'):
                            zf.extract(name, 'data')
                            # Mover/renomear se estiver em subpasta
                            src = os.path.join('data', name)
                            if src != file_path:
                                os.replace(src, file_path)
                            break
            data_df = pd.read_csv(file_path, header=None, names=col_names, sep=r'\s+')
            data_df["target"] = data_df["target"].apply(lambda x: 0 if x == 'NO' else 1)
            X = data_df.drop("target", axis=1)
            y_series = data_df["target"]
            class_names_list = ["Normal", "Anormal"]
        elif nome_dataset == 'spambase':
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
            ]
            local_path = os.path.join('data', 'spambase.data')
            
            # Nomes das 57 features, seguidos pela coluna 'target'
            # (Conforme a documentação oficial do dataset)
            col_names = [
                "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", 
                "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", 
                "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", 
                "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", 
                "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", 
                "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", 
                "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650", 
                "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", 
                "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology", 
                "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct", 
                "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", 
                "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", 
                "char_freq_semicolon", "char_freq_parentheses", "char_freq_bracket", 
                "char_freq_exclamation", "char_freq_dollar", "char_freq_hash", 
                "capital_run_length_average", "capital_run_length_longest", 
                "capital_run_length_total", "target"
            ]
            
            # Carrega os dados usando pandas
            data_df = _read_csv_with_cache(urls, local_path, names=col_names, header=None)
            
            # Separa os alvos (y) e as features (X)
            y_series = data_df["target"].astype(int) # Alvo já é 0 (não spam) ou 1 (spam)
            X = data_df.drop("target", axis=1)
            
            # Define os nomes das classes para o menu
            class_names_list = ["Não Spam", "Spam"]
        elif nome_dataset == 'mnist':
            # MNIST via OpenML (10 classes), com cache do scikit-learn e retries
            data_openml = _fetch_openml_with_retries('mnist_784', version=1, as_frame=True)
            X = data_openml.data
            y_all = pd.Series(data_openml.target.astype(int), name='target')
            class_names_list = [str(d) for d in range(10)]

            # Aplicar seleção de par (classe vs classe) se definida globalmente
            global MNIST_SELECTED_PAIR, MNIST_FEATURE_MODE
            if MNIST_SELECTED_PAIR is not None:
                a, b = MNIST_SELECTED_PAIR
                mask = (y_all == a) | (y_all == b)
                X = X[mask].copy()
                y_bin_np = np.where(y_all[mask] == a, 0, 1)
                y_series = pd.Series(y_bin_np, index=X.index, name='target')
                class_names_list = [str(a), str(b)]
            else:
                # Sem par definido: retorna multiclasse original
                y_series = y_all

            # Aplicar pooling 2x2 (196 features) se requisitado
            if MNIST_FEATURE_MODE == 'pool2x2':
                X = _apply_pooling_df(X)
        elif nome_dataset == 'newsgroups':
            # 20 Newsgroups - Dataset de texto com TF-IDF (~50k features esparsas)
            # Ideal para testar escalabilidade: PuLP não resolve, PEAB sim
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            print("Carregando 20 Newsgroups TF-IDF (~50k features)...")
            
            # Binarização: sci.* (ciência) vs rec.* (recreação)
            categories_pos = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
            categories_neg = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
            
            newsgroups = fetch_20newsgroups(
                subset='all',
                categories=categories_pos + categories_neg,
                remove=('headers', 'footers', 'quotes'),
                random_state=RANDOM_STATE
            )
            
            # Vetorização TF-IDF (limitar a 10k features para não explodir demais)
            # Mas ainda assim é muito para MILP
            vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
            X_sparse = vectorizer.fit_transform(newsgroups.data)
            X = pd.DataFrame(X_sparse.toarray(), columns=vectorizer.get_feature_names_out())
            
            # Labels binários: sci.* = 1, rec.* = 0
            y_binary = np.array([1 if newsgroups.target_names[t].startswith('sci') else 0 
                                for t in newsgroups.target])
            y_series = pd.Series(y_binary, name='target')
            class_names_list = ['Recreation', 'Science']
            
            print(f"20 Newsgroups binário: {X.shape[0]} amostras, {X.shape[1]} features "
                  f"({sum(y_series==0)} rec, {sum(y_series==1)} sci)")
        
        elif nome_dataset == 'rcv1':
            # RCV1 Reuters Corpus - Dataset de texto (~47k features, 800k+ instâncias)
            # Usamos subamostra para viabilidade
            from sklearn.datasets import fetch_rcv1
            
            print("Carregando RCV1 Reuters (~47k features, subamostra de 5k)...")
            
            rcv1 = fetch_rcv1()
            
            # RCV1 é multilabel, vamos binarizar pegando 2 categorias frequentes
            # CCAT (Corporate/Industrial) vs GCAT (Government/Social)
            # Índices das categorias no target
            target_names = list(rcv1.target_names)
            
            if 'CCAT' in target_names and 'GCAT' in target_names:
                ccat_idx = target_names.index('CCAT')
                gcat_idx = target_names.index('GCAT')
                
                # Filtrar instâncias que são CCAT ou GCAT (mas não ambos)
                is_ccat = np.array(rcv1.target[:, ccat_idx].toarray().flatten(), dtype=bool)
                is_gcat = np.array(rcv1.target[:, gcat_idx].toarray().flatten(), dtype=bool)
                mask = (is_ccat | is_gcat) & ~(is_ccat & is_gcat)
                
                X_sparse = rcv1.data[mask]
                y_binary = np.where(is_ccat[mask], 1, 0)
            else:
                # Fallback: usar primeiras 2 categorias
                cat1 = np.array(rcv1.target[:, 0].toarray().flatten(), dtype=bool)
                cat2 = np.array(rcv1.target[:, 1].toarray().flatten(), dtype=bool)
                mask = (cat1 | cat2) & ~(cat1 & cat2)
                X_sparse = rcv1.data[mask]
                y_binary = np.where(cat1[mask], 1, 0)
            
            # Subamostra de 5000 instâncias para viabilidade
            n_samples = min(5000, X_sparse.shape[0])
            np.random.seed(RANDOM_STATE)
            indices = np.random.choice(X_sparse.shape[0], size=n_samples, replace=False)
            X_sparse = X_sparse[indices]
            y_binary = y_binary[indices]
            
            # Converter para denso e criar DataFrame
            X = pd.DataFrame(X_sparse.toarray())
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]
            y_series = pd.Series(y_binary, name='target')
            class_names_list = ['GCAT', 'CCAT']
            
            print(f"RCV1 binário: {X.shape[0]} amostras, {X.shape[1]} features "
                  f"({sum(y_series==0)} GCAT, {sum(y_series==1)} CCAT)")
        
        else:
            raise ValueError(f"Dataset '{nome_dataset}' não suportado.")

        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        if not isinstance(y_series, pd.Series): y_series = pd.Series(y_series, name='target')
        
        if not hasattr(X.columns, 'tolist') or not X.columns.tolist() or all(isinstance(c, int) for c in X.columns):
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y_series, class_names_list
    except Exception as e:
        print(f"\nErro ao carregar o dataset '{nome_dataset}': {str(e)}")
        return None, None, None

def selecionar_datasets_multiplos() -> Optional[List[str]]:
    """
    Permite selecionar múltiplos datasets para execução em sequência.
    
    Returns:
        Lista de nomes de datasets ou None se cancelado
    """
    nomes_datasets = [
        'breast_cancer', 'mnist', 'pima_indians_diabetes', 'sonar', 
        'vertebral_column', 'banknote', 'heart_disease',
        'spambase', 'creditcard', 'covertype', 'gas_sensor',
        'newsgroups', 'rcv1'
    ]
    
    menu = '''
    | ***************** MENU DE DATASETS DO EXPERIMENTO ****************** |
    | Datasets Clássicos para Comparação:                                  |
    | [0] Breast Cancer (569x30x2)       | [1] MNIST (70k x 784 x 10)      |
    | [2] Pima Diabetes (392x8x2)        | [3] Sonar (208x60x2)            |
    | [4] Vertebral Column (310x6x2)     | [5] Banknote Auth (1372x4x2)    |
    |----------------------------------------------------------------------|
    | Outros Datasets Disponíveis:                                         |
    | [6] Heart Disease (297x13x2)       | [7] Spambase (4601 x 57 x 2)    |
    | [8] Creditcard Fraud (Amostra)     | [9] Covertype (581k x 54 x 2)   |
    | [10] Gas Sensor (13.9k x 128 x 2)                                    |
    |----------------------------------------------------------------------|
    | Datasets Alta Dimensionalidade (para testar escalabilidade):         |
    | [11] 20 Newsgroups TF-IDF (~7k x 10k features)  [PuLP ESTOURA]       |
    | [12] RCV1 Reuters (~5k x 47k features)          [PuLP ESTOURA]       |
    |----------------------------------------------------------------------|
    '''
    print(menu)
    print("\n SELEÇÃO MÚLTIPLA DE DATASETS")
    print(" Exemplos:")
    print("   • Específicos: 0,2,3       (breast_cancer, pima, sonar)")
    print("   • Range:       0-5         (datasets 0 até 5)")
    print("   • Misto:       0,2-4,7     (0, 2, 3, 4, 7)")
    print("   • Todos:       A ou ALL")
    print("   • Cancelar:    Q\n")
    
    while True:
        seleção = input("Digite sua seleção: ").upper().strip()
        
        if seleção == 'Q':
            return None
        
        if seleção in ['A', 'ALL']:
            print(f"\n✓ Selecionados TODOS os {len(nomes_datasets)} datasets:")
            for i, nome in enumerate(nomes_datasets):
                print(f"   [{i}] {nome}")
            confirmar = input("\nConfirmar execução em sequência? (S/N): ").upper().strip()
            if confirmar == 'S':
                return nomes_datasets
            continue
        
        # Parse da seleção
        indices = _parse_dataset_selection(seleção, len(nomes_datasets) - 1)
        
        if not indices:
            print("❌ Seleção inválida. Tente novamente.")
            continue
        
        datasets_selecionados = [nomes_datasets[i] for i in indices]
        
        print(f"\n✓ Selecionados {len(datasets_selecionados)} datasets:")
        for i, nome in zip(indices, datasets_selecionados):
            print(f"   [{i}] {nome}")
        
        confirmar = input("\nConfirmar execução em sequência? (S/N): ").upper().strip()
        if confirmar == 'S':
            return datasets_selecionados
        else:
            print("\nTente novamente com nova seleção.\n")


def selecionar_dataset_e_classe() -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]:
    # [MODIFICAÇÃO IMPORTANTE] Menu atualizado para incluir Iris e reorganizar as opções.
    menu = '''
    | ******************* MENU DE DATASETS DO EXPERIMENTO ****************** |
    | Datasets Clássicos para Comparação:                                  |
    | [0] Breast Cancer (569x30x2)       | [1] MNIST (70k x 784 x 10)       |
    | [2] Pima Diabetes (392x8x2)        | [3] Sonar (208x60x2)            |
    | [4] Vertebral Column (310x6x2)     | [5] Banknote Auth (1372x4x2)    |
    |----------------------------------------------------------------------|
    | Outros Datasets Disponíveis:                                         |
    | [6] Heart Disease (297x13x2)       | [7] Spambase (4601 x 57 x 2)    |
    | [8] Creditcard Fraud (Amostra)     | [9] Covertype (581k x 54 x 2)   |
    | [10] Gas Sensor (13.9k x 128 x 2)                                    |
    |----------------------------------------------------------------------|
    | Datasets Alta Dimensionalidade (para testar escalabilidade):         |
    | [11] 20 Newsgroups TF-IDF (~7k x 10k features)  [PuLP ESTOURA]       |
    | [12] RCV1 Reuters (~5k x 47k features)          [PuLP ESTOURA]       |
    |----------------------------------------------------------------------|
    | [M] MÚLTIPLOS DATASETS (Ex: 0,2,3 ou 0-5)                           |
    | [A] TODOS OS DATASETS                                                |
    | [Q] SAIR                                                             |
    |----------------------------------------------------------------------|
    '''
    print(menu)
    # [MODIFICAÇÃO IMPORTANTE] Lista de datasets atualizada para corresponder ao menu.
    nomes_datasets = [
        'breast_cancer', 'mnist', 'pima_indians_diabetes', 'sonar', 
        'vertebral_column', 'banknote', 'heart_disease',
        'spambase', 'creditcard', 'covertype', 'gas_sensor',
        'newsgroups', 'rcv1'
    ]
    while True:
        opcao = input("\nDigite o número do dataset, 'M' para múltiplos, 'A' para todos, ou 'Q' para sair: ").upper().strip()
        if opcao == 'Q': return None, None, None, None, None
        
        # Seleção múltipla
        if opcao == 'M':
            datasets_selecionados = selecionar_datasets_multiplos()
            if datasets_selecionados:
                # Retorna lista de datasets como string especial para indicar múltiplos
                return ('__MULTIPLE__', None, None, None, datasets_selecionados)
            continue
        
        # Todos os datasets
        if opcao == 'A':
            print(f"\n✓ Selecionados TODOS os {len(nomes_datasets)} datasets")
            confirmar = input("Confirmar execução em sequência? (S/N): ").upper().strip()
            if confirmar == 'S':
                return ('__MULTIPLE__', None, None, None, nomes_datasets)
            continue

        if opcao.isdigit() and 0 <= int(opcao) < len(nomes_datasets):
            nome_dataset_selecionado = nomes_datasets[int(opcao)]
            print(f"\nCarregando {nome_dataset_selecionado}...")
            # Para MNIST, as configurações são automáticas (definidas em peab_2.MNIST_CONFIG)
            # Não há menu interativo - tudo é configurado via código
            if nome_dataset_selecionado == 'mnist':
                # Importar e aplicar configurações de MNIST_CONFIG
                try:
                    import sys
                    import os
                    # Adicionar diretório raiz ao path se necessário
                    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if root_dir not in sys.path:
                        sys.path.insert(0, root_dir)
                    
                    try:
                        from peab import MNIST_CONFIG
                        print("\n[INFO] MNIST detectado. Usando configurações automáticas de peab.MNIST_CONFIG")
                    except ImportError:
                        try:
                            from peab import MNIST_CONFIG
                            print("\n[INFO] MNIST detectado. Usando configurações automáticas de peab_2.MNIST_CONFIG")
                        except ImportError:
                            raise ImportError("Não foi possível importar MNIST_CONFIG de peab.py nem de peab_2.py")
                    
                    # Aplicar configurações ANTES de carregar
                    feature_mode = MNIST_CONFIG.get('feature_mode', 'raw')
                    digit_pair = MNIST_CONFIG.get('digit_pair', None)
                    
                    if digit_pair is None:
                        print("[ERRO] MNIST_CONFIG['digit_pair'] não está definido!")
                        return None, None, None, None, None
                    
                    print(f"[INFO] Configurações: feature_mode='{feature_mode}', digit_pair={digit_pair}")
                    
                    # Configurar opções globais ANTES de carregar
                    set_mnist_options(feature_mode, digit_pair)
                    
                except ImportError as e:
                    print(f"[ERRO] Falha na importação de configuração MNIST: {e}")
                    print("[INFO] Usando configuração padrão: raw, (3, 8)")
                    set_mnist_options('raw', (3, 8))
                
                # Agora carregar o dataset com as opções já configuradas
                X_original_completo, y_original_completo, classes_originais_nomes = carregar_dataset('mnist')
            else:
                X_original_completo, y_original_completo, classes_originais_nomes = carregar_dataset(nome_dataset_selecionado)

            if X_original_completo is None or y_original_completo is None or classes_originais_nomes is None:
                print("Falha ao carregar dataset. Tente novamente.")
                continue

            print(f"Dataset carregado! (Total Amostras: {X_original_completo.shape[0]}, Features: {X_original_completo.shape[1]})")
            
            # Para MNIST, se as opções globais já estão configuradas (par selecionado), NÃO pedir menu
            if nome_dataset_selecionado == 'mnist' and MNIST_SELECTED_PAIR is not None:
                print(f"\n[INFO] Par de classes já configurado automaticamente: {MNIST_SELECTED_PAIR[0]} vs {MNIST_SELECTED_PAIR[1]}")
                print("[INFO] Dataset MNIST já filtrado para classificação binária.")
                # O dataset já foi carregado filtrado em carregar_dataset()
                # Retornar diretamente
                return nome_dataset_selecionado, str(MNIST_SELECTED_PAIR[0]), X_original_completo, y_original_completo, classes_originais_nomes
            
            print("\nClasses disponíveis no dataset original:")
            for i, nome_classe in enumerate(classes_originais_nomes):
                print(f"   [{i}] - {nome_classe} (Total: {sum(y_original_completo == i)})")

            # Se o dataset só tiver 2 classes, automatiza a seleção
            if len(classes_originais_nomes) == 2:
                print("\nDataset binário detectado. Configurando automaticamente Classe 0 vs Classe 1.")
                indice_classe_0 = 0
                indice_classe_1 = 1
                nome_classe_0_final = classes_originais_nomes[indice_classe_0]

            # Se tiver mais de 2 classes, pede ao usuário para escolher
            else:
                indice_classe_0 = -1
                while True:
                    entrada_classe_0 = input(f"\nDigite o NÚMERO da classe a ser CLASSE 0 (0-{len(classes_originais_nomes)-1}): ").strip()
                    if entrada_classe_0.isdigit() and 0 <= int(entrada_classe_0) < len(classes_originais_nomes):
                        indice_classe_0 = int(entrada_classe_0)
                        break
                    else:
                        print("Número inválido!")
                
                nome_classe_0_final = classes_originais_nomes[indice_classe_0]
                indice_classe_1 = -1

                print(f"\nClasses restantes para CLASSE 1 (não pode ser '{nome_classe_0_final}'):")
                indices_disponiveis_classe_1 = [i for i, _ in enumerate(classes_originais_nomes) if i != indice_classe_0]
                for i_disp in indices_disponiveis_classe_1:
                    print(f"   [{i_disp}] - {classes_originais_nomes[i_disp]} (Total: {sum(y_original_completo == i_disp)})")
                
                while True:
                    entrada_classe_1 = input(f"\nDigite o NÚMERO da CLASSE 1: ").strip()
                    if entrada_classe_1.isdigit() and int(entrada_classe_1) in indices_disponiveis_classe_1:
                        indice_classe_1 = int(entrada_classe_1)
                        break
                    else:
                        print("Número inválido ou classe já escolhida!")
            
            # Filtragem e Mapeamento para Binário
            mascara_classe_0 = (y_original_completo == indice_classe_0)
            mascara_classe_1 = (y_original_completo == indice_classe_1)
            mascara_combinada = mascara_classe_0 | mascara_classe_1

            X_filtrado = X_original_completo[mascara_combinada].copy()
            y_filtrado = y_original_completo[mascara_combinada].copy()

            # Para MNIST, pooling já foi aplicado em carregar_dataset conforme opção global
            
            # Mapeia para 0 e 1 (LogisticRegression prefere isso)
            y_binario_np = np.where(y_filtrado == indice_classe_0, 0, 1)
            y_binario = pd.Series(y_binario_np, index=X_filtrado.index, name='target_binario')
            nomes_classes_binarias_finais = [classes_originais_nomes[indice_classe_0], classes_originais_nomes[indice_classe_1]]

            print(f"\n🔹 Dataset '{nome_dataset_selecionado}' configurado para classificação binária:")
            print(f"   Classe 0 mapeada para: '{nomes_classes_binarias_finais[0]}' (Originalmente índice {indice_classe_0})")
            print(f"   Classe 1 mapeada para: '{nomes_classes_binarias_finais[1]}' (Originalmente índice {indice_classe_1})")
            print(f"   Total de amostras para o problema binário: {X_filtrado.shape[0]}\n")

            return nome_dataset_selecionado, nome_classe_0_final, X_filtrado, y_binario, nomes_classes_binarias_finais
        else:
            print("Opção inválida.")