# datasets.py (MODIFICADO)
import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_openml, load_breast_cancer, load_wine
from typing import List, Tuple, Dict, Any, Optional

RANDOM_STATE: int = 42

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
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
            # As features não têm nomes, então vamos nomeá-las genericamente
            col_names = [f"feature_{i}" for i in range(60)] + ["target"]
            data_df = pd.read_csv(url, header=None, names=col_names)
            # Converte a classe alvo (M para mina, R para rocha) em 1 e 0
            data_df["target"] = data_df["target"].apply(lambda x: 1 if x == 'M' else 0)
            X = data_df.drop("target", axis=1)
            y_series = data_df["target"]
            class_names_list = ["Rocha", "Mina (Metal)"]
        elif nome_dataset == 'pima_indians_diabetes':
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
            col_names = ['num_gravidezes', 'glicose', 'pressao_sangue', 'espessura_pele',
                         'insulina', 'imc', 'diabetes_pedigree', 'idade', 'target']
            data_df = pd.read_csv(url, header=None, names=col_names)
            
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
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
            col_names = ["variance", "skewness", "curtosis", "entropy", "target"]
            data_df = pd.read_csv(url, names=col_names)
            X = data_df.drop("target", axis=1)
            y_series = data_df["target"].astype(int)
            class_names_list = ["Autêntica", "Falsificada"]
        elif nome_dataset == 'heart_disease':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            col_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                         "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
            data_df = pd.read_csv(url, names=col_names, na_values="?")
            data_df.dropna(inplace=True)
            data_df["target"] = pd.to_numeric(data_df["target"], errors='coerce')
            data_df.dropna(subset=["target"], inplace=True)
            data_df["target"] = data_df["target"].astype(int).apply(lambda x: 1 if x > 0 else 0)
            X = data_df.drop("target", axis=1).astype(float)
            y_series = data_df["target"]
            class_names_list = ["Sem Doença", "Com Doença"]
        elif nome_dataset == 'wine_quality':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            data_df = pd.read_csv(url, sep=";")
            data_df["target"] = data_df["quality"].apply(lambda x: 1 if x >= 7 else 0)
            X = data_df.drop(["quality", "target"], axis=1)
            y_series = data_df["target"]
            class_names_list = ["Baixa Qualidade", "Alta Qualidade"]
        elif nome_dataset == 'creditcard':
            # Apenas carrega o dataset completo e o retorna.
            # A amostragem será feita no script principal.
            data_openml = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')
            X = data_openml.data
            y_series = data_openml.target.astype(int)
            class_names_list = ['Normal', 'Fraude']
        elif nome_dataset == 'wine':
            data = load_wine()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y_series = pd.Series(data.target, name='target')
            class_names_list = list(data.target_names)
            
            # Binarização OvR (classe 0 vs resto)
            y_series = np.where(y_series == 0, 0, 1)  # class_0 (0) vs class_1 + class_2        (1)
            class_names_list = [class_names_list[0], "Não-class_0"]  # ["class_0",      "Não-class_0"]
        elif nome_dataset == 'vertebral_column':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
            col_names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", 
                         "sacral_slope", "pelvic_radius", "spondylolisthesis_grade", "target"]
            file_path = os.path.join('datasets', 'column_2C.dat')
            # (Código original de download e carga)
            data_df = pd.read_csv(file_path, header=None, names=col_names, sep=r'\s+')
            data_df["target"] = data_df["target"].apply(lambda x: 0 if x == 'NO' else 1)
            X = data_df.drop("target", axis=1)
            y_series = data_df["target"]
            class_names_list = ["Normal", "Anormal"]
        elif nome_dataset == 'spambase':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
            
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
            data_df = pd.read_csv(url, names=col_names, header=None)
            
            # Separa os alvos (y) e as features (X)
            y_series = data_df["target"].astype(int) # Alvo já é 0 (não spam) ou 1 (spam)
            X = data_df.drop("target", axis=1)
            
            # Define os nomes das classes para o menu
            class_names_list = ["Não Spam", "Spam"]
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

def selecionar_dataset_e_classe() -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]:
    # [MODIFICAÇÃO IMPORTANTE] Menu atualizado para incluir Iris e reorganizar as opções.
    menu = '''
    | ******************* MENU DE DATASETS DO EXPERIMENTO ****************** |
    | Datasets Clássicos para Comparação:                                  |
    | [0] Breast Cancer (569x30x2)       | [1] Iris (150x4x3)              |
    | [2] Pima Diabetes (392x8x2)        | [3] Sonar (208x60x2)            |
    | [4] Vertebral Column (310x6x2)     | [5] Wine (178x13x3)             |
    |----------------------------------------------------------------------|
    | Outros Datasets Disponíveis:                                         |
    | [6] Banknote Auth (1372x4x2)       | [7] Heart Disease (297x13x2)    |
    | [8] Wine Quality (Red) (1599x11x2) | [9] Spambase (210x7x3)             |
    | [10] Creditcard Fraud (Amostra)                                      |
    |----------------------------------------------------------------------|
    | [Q] SAIR                                                             |
    |----------------------------------------------------------------------|
    '''
    print(menu)
    # [MODIFICAÇÃO IMPORTANTE] Lista de datasets atualizada para corresponder ao menu.
    nomes_datasets = [
        'breast_cancer', 'iris', 'pima_indians_diabetes', 'sonar', 
        'vertebral_column', 'wine', 'banknote', 'heart_disease',
        'wine_quality', 'spambase', 'creditcard'
    ]
    while True:
        opcao = input("\nDigite o número do dataset ou 'Q' para sair: ").upper().strip()
        if opcao == 'Q': return None, None, None, None, None

        if opcao.isdigit() and 0 <= int(opcao) < len(nomes_datasets):
            nome_dataset_selecionado = nomes_datasets[int(opcao)]
            print(f"\nCarregando {nome_dataset_selecionado}...")
            X_original_completo, y_original_completo, classes_originais_nomes = carregar_dataset(nome_dataset_selecionado)

            if X_original_completo is None or y_original_completo is None or classes_originais_nomes is None:
                print("Falha ao carregar dataset. Tente novamente.")
                continue

            print(f"Dataset carregado! (Total Amostras: {X_original_completo.shape[0]}, Features: {X_original_completo.shape[1]})")
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