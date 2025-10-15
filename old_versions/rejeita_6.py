import os
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_openml, load_breast_cancer, load_wine
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Docstring do Módulo
"""
Este script implementa a geração e avaliação de PI-Explicações para modelos
de Regressão Logística, com foco em robustez, minimalidade e a opção de rejeição.
As explicações são geradas através de um processo de três fases:
1. Geração da Explicação Inicial (one_explanation).
2. Refinamento Aditivo para Robustez.
3. Minimização Subtrativa para Minimalidade.

O script permite carregar diversos datasets, treinar um modelo, calcular
thresholds de rejeição, gerar explicações para instâncias de teste e
produzir relatórios (resumido e detalhado) e visualizações.
"""
""" NOVAS FUNÇÕES DE PLOTAGEM (ou mova as existentes para cá e adicione novas) """
DIRETORIO_VISUALIZACOES = "visualizacoes_combinadas" # Pode mudar o nome do diretório

# --- Constantes Globais ---
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
WR_REJECTION_COST: float = 0.24
EPSILON: float = 1e-9
DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'liblinear',
    'max_iter': 200
}
# Defina como 0 ou None para mostrar todas, ou um número para limitar
MAX_FEATURES_DETAILED_REPORT: Optional[int] = 10

# --- Funções de Persistência de Artefatos ---
def salvar_artefatos(artefatos: dict, nome_dataset: str, diretorio: str = "artefatos_comparacao") -> None:
    os.makedirs(diretorio, exist_ok=True)
    caminho_arquivo = os.path.join(diretorio, f"artefatos_{nome_dataset}.joblib")
    joblib.dump(artefatos, caminho_arquivo)
    print(f"Artefatos para '{nome_dataset}' salvos em: {caminho_arquivo}")

def carregar_artefatos_salvos(nome_dataset: str, diretorio: str = "artefatos_comparacao") -> Optional[dict]:
    caminho_arquivo = os.path.join(diretorio, f"artefatos_{nome_dataset}.joblib")
    if os.path.exists(caminho_arquivo):
        print(f"Carregando artefatos de: {caminho_arquivo}")
        return joblib.load(caminho_arquivo)
    else:
        print(f"ARQUIVO DE ARTEFATOS NÃO ENCONTRADO: {caminho_arquivo}")
        return None

# --- Funções de Carregamento e Preparação de Dataset ---
def carregar_dataset(nome_dataset: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]: #
    try:
        X, y_series, class_names_list = None, None, None
        if nome_dataset == 'iris':
            data = load_iris()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y_series = pd.Series(data.target, name='target')
            class_names_list = list(data.target_names)
        elif nome_dataset == 'pima_indians_diabetes':
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
            col_names = ['num_gravidezes', 'glicose', 'pressao_sangue', 'espessura_pele',
                         'insulina', 'imc', 'diabetes_pedigree', 'idade', 'target']
            data_df = pd.read_csv(url, header=None, names=col_names)
            data_df = data_df.apply(pd.to_numeric, errors='coerce').dropna()
            X = data_df.iloc[:, :-1]
            y_series = data_df.iloc[:, -1].astype(int)
            class_names_list = ['Não Diabético', 'Diabético']
            X = pd.DataFrame(X.values, columns=col_names[:-1])
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
            data_openml = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')
            full_X, full_y = data_openml.data, data_openml.target.astype(int)
            # Amostragem reduzida para 'creditcard' para tornar execuções mais rápidas
            X, _, y_series, _ = train_test_split(
                full_X, full_y, train_size=0.005, stratify=full_y, random_state=RANDOM_STATE # 0.5%
            )
            class_names_list = ['Normal', 'Fraude']
            if X is None or y_series is None :
                 raise ValueError("Amostragem do Creditcard resultou em X ou y nulos.")
            X = pd.DataFrame(X)
            y_series = pd.Series(y_series) # Certificar que y_series é pd.Series
        elif nome_dataset == 'wine':
            data = load_wine()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y_series = pd.Series(data.target, name='target')
            class_names_list = list(data.target_names)
        elif nome_dataset == 'haberman':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
            col_names = ["age", "year_of_op", "positive_nodes", "target"]
            data_df = pd.read_csv(url, header=None, names=col_names)
            data_df["target"] = data_df["target"].astype(int) - 1 # 1 (sobreviveu) e 2 (morreu) para 0 e 1
            X = data_df.drop("target", axis=1)
            y_series = data_df["target"]
            class_names_list = ["Sobreviveu 5+ anos", "Morreu antes de 5 anos"]
        elif nome_dataset == 'seeds':
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
            col_names = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "target"]
            data_df = pd.read_csv(url, sep=r"\s+", names=col_names, header=None)
            data_df.dropna(inplace=True)
            y_series = data_df["target"].astype(int) - 1 # Classes 1,2,3 para 0,1,2
            X = data_df.drop("target", axis=1)
            class_names_list = ["Kama", "Rosa", "Canadian"]
        else:
            raise ValueError(f"Dataset '{nome_dataset}' não suportado.")

        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        if not isinstance(y_series, pd.Series): y_series = pd.Series(y_series, name='target')
        
        # Garante nomes de colunas se não existirem
        if not hasattr(X.columns, 'tolist') or not X.columns.tolist() or all(isinstance(c, int) for c in X.columns):
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y_series, class_names_list
    except Exception as e:
        print(f"\nErro ao carregar o dataset '{nome_dataset}': {str(e)}")
        return None, None, None

def selecionar_dataset_e_classe() -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]: #
    menu = '''
    | ******************* MENU DE DATASETS CONFIÁVEIS ******************** |
    | [0] Iris (150x4x3)                 | [1] Pima Diabetes (768x8x2)     |
    | [2] Breast Cancer (569x30x2)       | [3] Creditcard Fraud (Amostra)  |
    | [4] Banknote Auth (1372x4x2)       | [5] Heart Disease (303x13x2)    |
    | [6] Wine Quality (Red) (1599x11x2) | [7] Wine (178x13x3)             |
    | [8] Haberman Survival (306x3x2)    | [9] Seeds (210x7x3)             |
    | [Q] SAIR                                                             |
    |----------------------------------------------------------------------|
    '''
    print(menu)
    nomes_datasets = [
        'iris', 'pima_indians_diabetes', 'breast_cancer',
        'creditcard', 'banknote', 'heart_disease',
        'wine_quality', 'wine', 'haberman', 'seeds'
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
            nome_classe_1_final = ""

            if len(classes_originais_nomes) == 2:
                print(f"\nDataset com duas classes. '{nome_classe_0_final}' foi selecionada como CLASSE 0.")
                disponiveis_para_classe1 = [i for i, _ in enumerate(classes_originais_nomes) if i != indice_classe_0]
                indice_classe_1 = disponiveis_para_classe1[0]
                nome_classe_1_final = classes_originais_nomes[indice_classe_1]
                print(f"CLASSE 1 definida automaticamente como: '{nome_classe_1_final}' (Índice original: {indice_classe_1})")
            
            elif len(classes_originais_nomes) > 2:
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
                nome_classe_1_final = classes_originais_nomes[indice_classe_1]
            else: 
                print("Dataset com uma ou nenhuma classe. Impossível criar problema binário.")
                continue

            mascara_classe_0 = (y_original_completo == indice_classe_0)
            mascara_classe_1 = (y_original_completo == indice_classe_1)
            mascara_combinada = mascara_classe_0 | mascara_classe_1

            X_filtrado = X_original_completo[mascara_combinada].copy()
            y_filtrado = y_original_completo[mascara_combinada].copy()

            y_binario_np = np.where(y_filtrado == indice_classe_0, 0, 1)
            y_binario = pd.Series(y_binario_np, index=X_filtrado.index, name='target_binario')
            nomes_classes_binarias_finais = [nome_classe_0_final, nome_classe_1_final]

            print(f"\n🔹 Dataset '{nome_dataset_selecionado}' configurado para classificação binária:")
            print(f"   Classe 0 mapeada para: '{nomes_classes_binarias_finais[0]}' (Originalmente índice {indice_classe_0})")
            print(f"   Classe 1 mapeada para: '{nomes_classes_binarias_finais[1]}' (Originalmente índice {indice_classe_1})")
            print(f"   Total de amostras para o problema binário: {X_filtrado.shape[0]}\n")

            return nome_dataset_selecionado, nome_classe_0_final, X_filtrado, y_binario, nomes_classes_binarias_finais
        else:
            print("Opção inválida.")

def calcular_thresholds(modelo: LogisticRegression, X_train: pd.DataFrame, y_train: pd.Series,
                        custo_rejeicao_wr: float = WR_REJECTION_COST) -> Tuple[float, float]: #
    decision_scores = modelo.decision_function(X_train) 
    if len(decision_scores) < 2 : return 0.1, -0.1 
    score_std = np.std(decision_scores)                 
    limite_superior_scores: float
    limite_inferior_scores: float

    if score_std < EPSILON or len(np.unique(decision_scores)) < 2:
        limite_superior_scores, limite_inferior_scores = 0.5, -0.5
    else:
        limite_superior_scores, limite_inferior_scores = np.max(decision_scores), np.min(decision_scores)

    if limite_superior_scores <= limite_inferior_scores + EPSILON:
        mediana_scores = np.median(decision_scores)
        if limite_superior_scores <= mediana_scores + EPSILON:
            limite_superior_scores = mediana_scores + 0.1 if mediana_scores + 0.1 > limite_inferior_scores else limite_inferior_scores + 0.2
        if limite_inferior_scores >= mediana_scores - EPSILON:
            limite_inferior_scores = mediana_scores - 0.1 if mediana_scores - 0.1 < limite_superior_scores else limite_superior_scores - 0.2
        if limite_superior_scores <= limite_inferior_scores + EPSILON:
            limite_inferior_scores, limite_superior_scores = -0.1, 0.1

    num_pontos_candidatos = 100
    t_plus_candidatos: List[float] = []
    t_minus_candidatos: List[float] = []

    if limite_superior_scores > EPSILON:
        t_plus_candidatos.extend(list(limite_superior_scores * np.linspace(0.01, 1.0, num_pontos_candidatos)))
    else:
        t_plus_candidatos.extend(list(np.linspace(EPSILON, 0.1, num_pontos_candidatos)))

    if limite_inferior_scores < -EPSILON:
        t_minus_candidatos.extend(list(limite_inferior_scores * np.linspace(0.01, 1.0, num_pontos_candidatos)))
    else:
        t_minus_candidatos.extend(list(np.linspace(-0.1, -EPSILON, num_pontos_candidatos)))

    t_plus_unicos = np.unique(np.array(t_plus_candidatos))
    t_minus_unicos = np.unique(np.array(t_minus_candidatos))

    min_custo_total = float('inf')
    melhor_t_plus = (np.max(t_plus_unicos) if t_plus_unicos.size > 0 else 0.1)
    melhor_t_minus = (np.min(t_minus_unicos) if t_minus_unicos.size > 0 else -0.1)

    if not t_plus_unicos.size or not t_minus_unicos.size:
        return 0.1, -0.1

    for t_p_cand in t_plus_unicos:
        for t_m_cand in t_minus_unicos:
            if t_m_cand >= t_p_cand - EPSILON:
                continue
            mascara_rejeicao = (decision_scores <= t_p_cand) & (decision_scores >= t_m_cand)
            mascara_aceitas = ~mascara_rejeicao
            taxa_rejeicao_parcial = np.mean(mascara_rejeicao)
            if np.sum(mascara_aceitas) == 0:
                taxa_erro_aceitas_parcial = 1.0
            else:
                predicoes_aceitas = modelo.predict(X_train[mascara_aceitas])
                y_verdadeiro_aceitas = y_train[mascara_aceitas]
                taxa_erro_aceitas_parcial = np.mean(predicoes_aceitas != y_verdadeiro_aceitas.values)
            custo_total_candidato = taxa_erro_aceitas_parcial + custo_rejeicao_wr * taxa_rejeicao_parcial
            if custo_total_candidato < min_custo_total:
                min_custo_total = custo_total_candidato
                melhor_t_plus = t_p_cand
                melhor_t_minus = t_m_cand

    if melhor_t_plus <= melhor_t_minus + EPSILON:
        mediana_scores, std_dev_scores = np.median(decision_scores), np.std(decision_scores)
        if std_dev_scores > EPSILON:
            melhor_t_plus = min(limite_superior_scores, mediana_scores + 0.25 * std_dev_scores)
            melhor_t_minus = max(limite_inferior_scores, mediana_scores - 0.25 * std_dev_scores)
        else:
            melhor_t_plus, melhor_t_minus = 0.05, -0.05
        if melhor_t_plus <= melhor_t_minus + EPSILON:
             melhor_t_plus, melhor_t_minus = 0.01, -0.01
    return melhor_t_plus, melhor_t_minus

def calculate_deltas(modelo: LogisticRegression, instance_df: pd.DataFrame, X_train: pd.DataFrame) -> np.ndarray: #
    coeficientes = modelo.coef_[0]
    pred_class = modelo.predict(instance_df)[0]
    instance_valores = instance_df.iloc[0]
    deltas = np.zeros_like(coeficientes)
    for i, feature_name in enumerate(X_train.columns):
        valor_feature = instance_valores[feature_name]
        coef_feature = coeficientes[i]
        if pred_class == 1:
            pior_valor = X_train[feature_name].min() if coef_feature > 0 else X_train[feature_name].max()
        else: # pred_class == 0
            pior_valor = X_train[feature_name].max() if coef_feature > 0 else X_train[feature_name].min()
        deltas[i] = (valor_feature - pior_valor) * coef_feature
    return deltas

# --- Funções de Explicação Refatoradas ---

def one_explanation(modelo: LogisticRegression,
                    instance_df: pd.DataFrame,
                    X_train: pd.DataFrame,
                    t_plus: Optional[float],
                    t_minus: Optional[float]) -> List[str]: # Lógica baseada em com remoção da margem
    """
    Gera uma explicação inicial (Fase 0) para uma única instância,
    baseado na lógica de `rejeita_5.py` mas sem a margem de segurança explícita.
    """
    score_original = modelo.decision_function(instance_df)[0]
    pred_class_original = modelo.predict(instance_df)[0]
    
    deltas_calculados = calculate_deltas(modelo, instance_df, X_train) #
    indices_ordenados = np.argsort(-np.abs(deltas_calculados)) #
    features_ordenadas = X_train.columns[indices_ordenados] #
    deltas_ordenados = deltas_calculados[indices_ordenados] #

    explicacao: List[str] = []
    foi_rejeitada_originalmente = False
    if t_plus is not None and t_minus is not None:
        foi_rejeitada_originalmente = (t_minus <= score_original <= t_plus) #

    if foi_rejeitada_originalmente: #
        phi_rejection_target = abs(score_original) #
        soma_cumulativa_abs_deltas = 0.0 #
        for i in range(len(features_ordenadas)): #
            feature_atual_nome = features_ordenadas[i] #
            valor_feature_atual = instance_df.iloc[0][feature_atual_nome] #
            delta_abs_atual = abs(deltas_ordenados[i]) #
            if soma_cumulativa_abs_deltas <= phi_rejection_target + EPSILON: #
                explicacao.append(f"{feature_atual_nome} = {valor_feature_atual:.4f}") #
                soma_cumulativa_abs_deltas += delta_abs_atual #
            else:
                break #
        if not explicacao and len(deltas_ordenados) > 0: #
             explicacao.append(f"{features_ordenadas[0]} = {instance_df.iloc[0][features_ordenadas[0]]:.4f}") #
    else: # Lógica para instâncias CLASSIFICADAS
        soma_total_deltas = np.sum(deltas_calculados) #
        score_base = score_original - soma_total_deltas #

        if pred_class_original == 1: #
            effective_floor_class1 = 0.0 #
            if t_plus is not None and score_original > t_plus: #
                effective_floor_class1 = t_plus #
            # limiar_alvo_score SEM a margem de segurança
            limiar_alvo_score = effective_floor_class1 # MODIFICADO: Remoção da margem
            
            phi_target_sum_contrib = limiar_alvo_score - score_base + EPSILON # (fórmula mantida, mas limiar_alvo_score mudou)
            soma_cumulativa_contrib_deltas = 0.0 #
            
            for i in range(len(deltas_ordenados)): #
                feature_atual_nome = features_ordenadas[i] #
                valor_feature_atual = instance_df.iloc[0][feature_atual_nome] #
                delta_atual_com_sinal = deltas_ordenados[i] #
                
                if soma_cumulativa_contrib_deltas <= phi_target_sum_contrib: #
                    explicacao.append(f"{feature_atual_nome} = {valor_feature_atual:.4f}") #
                    soma_cumulativa_contrib_deltas += delta_atual_com_sinal #
                elif not explicacao: #
                    explicacao.append(f"{feature_atual_nome} = {valor_feature_atual:.4f}") #
                    soma_cumulativa_contrib_deltas += delta_atual_com_sinal #
                    break #
                else:
                    break #
            if soma_cumulativa_contrib_deltas <= phi_target_sum_contrib and explicacao and len(explicacao) < len(features_ordenadas): #
                idx_prox_feature_ordenada = len(explicacao) #
                explicacao.append(f"{features_ordenadas[idx_prox_feature_ordenada]} = {instance_df.iloc[0][features_ordenadas[idx_prox_feature_ordenada]]:.4f}") #

        else: # pred_class_original == 0
            effective_ceiling_class0 = 0.0 #
            if t_minus is not None and score_original < t_minus: #
                effective_ceiling_class0 = t_minus #
            # limiar_alvo_score SEM a margem de segurança
            limiar_alvo_score = effective_ceiling_class0 # MODIFICADO: Remoção da margem
            
            # Fórmula de phi_target_sum_abs_deltas como em rejeita_5.py, mas com limiar_alvo_score modificado
            phi_target_sum_abs_deltas = score_base - limiar_alvo_score + EPSILON # (fórmula mantida, mas limiar_alvo_score mudou)
            soma_cumulativa_abs_deltas = 0.0 #

            for i in range(len(deltas_ordenados)): #
                feature_atual_nome = features_ordenadas[i] #
                valor_feature_atual = instance_df.iloc[0][feature_atual_nome] #
                delta_abs_atual = abs(deltas_ordenados[i]) #
                if soma_cumulativa_abs_deltas <= phi_target_sum_abs_deltas: #
                    explicacao.append(f"{feature_atual_nome} = {valor_feature_atual:.4f}") #
                    soma_cumulativa_abs_deltas += delta_abs_atual #
                elif not explicacao: #
                    explicacao.append(f"{feature_atual_nome} = {valor_feature_atual:.4f}") #
                    soma_cumulativa_abs_deltas += delta_abs_atual #
                    break #
                else:
                    break #
            if soma_cumulativa_abs_deltas <= phi_target_sum_abs_deltas and explicacao and len(explicacao) < len(features_ordenadas): #
                idx_prox_feature_ordenada = len(explicacao) #
                explicacao.append(f"{features_ordenadas[idx_prox_feature_ordenada]} = {instance_df.iloc[0][features_ordenadas[idx_prox_feature_ordenada]]:.4f}") #
    return explicacao


def perturbar_features(modelo: LogisticRegression, instance_original_df: pd.DataFrame,
                       explicacao_features_valores: List[str], X_train: pd.DataFrame,
                       t_plus: Optional[float], t_minus: Optional[float],
                       class_names_binario: List[str]) -> Tuple[pd.DataFrame, str, str]: # Modificado o tipo de retorno
    """
    Perturba as features não presentes na explicação, avalia a predição e retorna
    a instância perturbada, o status da validação e a string da predição/score perturbado.
    """
    inst_pert = instance_original_df.copy()
    nomes_features_explicacao = [f.split(' = ')[0] for f in explicacao_features_valores]
    features_para_perturbar = [feat_n for feat_n in X_train.columns if feat_n not in nomes_features_explicacao]
    
    existem_features_para_perturbar = bool(features_para_perturbar)

    if existem_features_para_perturbar:
        pred_class_original_para_pert = modelo.predict(instance_original_df)[0]
        for feat_nome_perturbar in features_para_perturbar:
            try:
                coef_idx = X_train.columns.get_loc(feat_nome_perturbar)
                coef_feature = modelo.coef_[0][coef_idx]
                if pred_class_original_para_pert == 1:
                    valor_perturbado = X_train[feat_nome_perturbar].min() if coef_feature > 0 else X_train[feat_nome_perturbar].max()
                else: # Originalmente classe 0
                    valor_perturbado = X_train[feat_nome_perturbar].max() if coef_feature > 0 else X_train[feat_nome_perturbar].min()
                inst_pert.loc[inst_pert.index[0], feat_nome_perturbar] = valor_perturbado
            except KeyError:
                # Feature pode não estar no X_train se X_train for um subconjunto, embora raro neste contexto
                continue

    score_perturbado = modelo.decision_function(inst_pert)[0]
    pred_classe_perturbada_idx = modelo.predict(inst_pert)[0]
    score_original = modelo.decision_function(instance_original_df)[0]
    pred_classe_original_idx = modelo.predict(instance_original_df)[0]

    original_foi_rejeitada = (t_plus is not None and t_minus is not None and (t_minus <= score_original <= t_plus))
    
    status_validacao = "INVÁLIDA (STATUS NÃO DETERMINADO)"
    if original_foi_rejeitada:
        # Verifica se a instância perturbada também foi rejeitada
        perturbada_tambem_rejeitada_check = (t_plus is not None and t_minus is not None and (t_minus <= score_perturbado <= t_plus))
        if not existem_features_para_perturbar: # Se não há features para perturbar, ela deveria se manter rejeitada
            status_validacao = "VÁLIDA (REJEIÇÃO MANTEVE-SE REJEITADA)" # ou o status que já tinha
        else:
            status_validacao = "VÁLIDA (REJEIÇÃO MANTEVE-SE REJEITADA)" if perturbada_tambem_rejeitada_check else "INVÁLIDA (REJEIÇÃO SAIU DA ZONA DE REJEIÇÃO)"
    else: # Instância original foi CLASSIFICADA
        perturbada_caiu_na_rejeicao_check = (t_plus is not None and t_minus is not None and (t_minus <= score_perturbado <= t_plus))
        if perturbada_caiu_na_rejeicao_check:
            status_validacao = "INVÁLIDA (CLASSIFICADA CAIU NA REJEIÇÃO)"
        elif pred_classe_perturbada_idx != pred_classe_original_idx:
            status_validacao = (f"INVÁLIDA (CLASSIFICADA MUDOU DE CLASSE DE "
                                f"{class_names_binario[pred_classe_original_idx]} PARA "
                                f"{class_names_binario[pred_classe_perturbada_idx]})")
        else: # Manteve a classe e não caiu na rejeição
            status_validacao = "VÁLIDA (CLASSIFICADA MANTEVE CLASSE E NÃO FOI REJEITADA)"

    # String de log detalhada da predição perturbada
    perturbada_foi_rejeitada_log = (t_plus is not None and t_minus is not None and (t_minus <= score_perturbado <= t_plus))
    pred_str_log_detalhado = (f"REJEITADA (Score: {score_perturbado:.4f})" if perturbada_foi_rejeitada_log
                              else f"{class_names_binario[pred_classe_perturbada_idx]} (Score: {score_perturbado:.4f})")
    
    return inst_pert, status_validacao, pred_str_log_detalhado

# --- Novas Funções de Fase ---

def executar_fase_0_explicacao_inicial(
    modelo: LogisticRegression,
    instancia_df: pd.DataFrame,
    X_train: pd.DataFrame,
    t_plus: Optional[float],
    t_minus: Optional[float]
) -> Tuple[List[str], List[str]]:
    """Gera a explicação inicial (Fase 0)."""
    log_detalhes_fase_0 = [f"FASE 0: Geração da Explicação Inicial"]
    
    expl_inicial = one_explanation(modelo, instancia_df, X_train, t_plus, t_minus)
    
    log_detalhes_fase_0.append(f"  Explicação Inicial (one_explanation) ({len(expl_inicial)} feature):")
    if expl_inicial:
        for i, fv_str in enumerate(expl_inicial):
            log_detalhes_fase_0.append(f"    {i+1}. {fv_str}")
    else:
        log_detalhes_fase_0.append("    Explicação inicial vazia.")
    return expl_inicial, log_detalhes_fase_0

def executar_fase_1_refinamento_robustez(
    modelo: LogisticRegression,
    instancia_df: pd.DataFrame,
    explicacao_candidata_fase0: List[str], # Explicação vinda da Fase 0
    X_train: pd.DataFrame,
    t_plus: Optional[float],
    t_minus: Optional[float],
    class_names_binario: List[str]
) -> Tuple[List[str], str, pd.DataFrame, List[str], int]:
    """
    Fase 1: Testa a robustez da explicação inicial e a reforça aditivamente se necessário.
    Retorna a explicação robusta, seu status, a instância perturbada correspondente,
    o log da fase e o número de features adicionadas.
    """
    log_detalhes_fase_1 = ["FASE 1: Teste de Robustez e Reforço da Explicação"] # Nome da fase atualizado
    
    expl_robusta = list(explicacao_candidata_fase0) # Começa com a explicação da Fase 0
    
    # 1. Teste inicial de robustez na explicação da Fase 0
    inst_pert_robusta, status_robusto, pred_str_log_inicial = perturbar_features(
        modelo, instancia_df, expl_robusta, X_train, t_plus, t_minus, class_names_binario
    )
    log_detalhes_fase_1.append(f"  Teste de robustez na explicação da Fase 0 ({len(expl_robusta)} feats):")
    log_detalhes_fase_1.append(f"    Status do teste: '{status_robusto}'")
    log_detalhes_fase_1.append(f"    Predição da Instância Perturbada (com expl. Fase 0): {pred_str_log_inicial}")
    log_detalhes_fase_1.append("")

    eh_valida_inicialmente = status_robusto.startswith("VÁLIDA")
    explicacao_esta_completa = (len(expl_robusta) == X_train.shape[1])
    features_adicionadas_fase1 = 0
    len_antes_fase1 = len(expl_robusta)

    # 2. Reforço aditivo, se necessário
    precisa_refinar_aditivamente = (not eh_valida_inicialmente) and (not explicacao_esta_completa)
    pred_str_log_apos_adicao = pred_str_log_inicial # Para manter o último log de predição

    if precisa_refinar_aditivamente:
        log_detalhes_fase_1.append(f"  Explicação ({len(expl_robusta)} feats) NÃO é robusta. Iniciando reforço aditivo...")
        todos_deltas_instancia = calculate_deltas(modelo, instancia_df, X_train)
        indices_features_ordenadas = np.argsort(-np.abs(todos_deltas_instancia))
        nomes_features_todas_ordenadas = X_train.columns[indices_features_ordenadas]
        features_ja_na_explicacao = set([fs.split(' = ')[0] for fs in expl_robusta])

        for nome_feature_a_adicionar in nomes_features_todas_ordenadas:
            if nome_feature_a_adicionar not in features_ja_na_explicacao:
                log_detalhes_fase_1.append(f"    Tentando adicionar: {nome_feature_a_adicionar}")
                valor_feature_a_adicionar = instancia_df.iloc[0][nome_feature_a_adicionar]
                expl_tentativa_refinada = expl_robusta + [f"{nome_feature_a_adicionar} = {valor_feature_a_adicionar:.4f}"]
                
                inst_pert_refinada_loop, status_refinado_loop, pred_str_log_loop = perturbar_features(
                    modelo, instancia_df, expl_tentativa_refinada, X_train, t_plus, t_minus, class_names_binario
                )
                
                expl_robusta = expl_tentativa_refinada
                status_robusto = status_refinado_loop # Atualiza o status principal
                inst_pert_robusta = inst_pert_refinada_loop # Atualiza a instância perturbada
                pred_str_log_apos_adicao = pred_str_log_loop # Atualiza o log da predição

                features_ja_na_explicacao.add(nome_feature_a_adicionar)
                log_detalhes_fase_1.append(f"      Explicação atualizada para ({len(expl_robusta)} feats).")
                log_detalhes_fase_1.append(f"        Status após adição: '{status_robusto}'")
                log_detalhes_fase_1.append(f"        Predição da Instância Perturbada: {pred_str_log_apos_adicao}")

                if status_robusto.startswith("VÁLIDA"):
                    log_detalhes_fase_1.append("      Explicação tornou-se robusta com esta adição.")
                    break 
                if len(expl_robusta) == X_train.shape[1]:
                    log_detalhes_fase_1.append("      Todas as features foram adicionadas. Parando reforço.")
                    break
        features_adicionadas_fase1 = len(expl_robusta) - len_antes_fase1
        log_detalhes_fase_1.append(f"  Fim do reforço aditivo. Features adicionadas nesta fase: {features_adicionadas_fase1}")
    else:
        if eh_valida_inicialmente:
            log_detalhes_fase_1.append("  Explicação da Fase 0 JÁ é robusta. Reforço aditivo não necessário.")
        elif explicacao_esta_completa and not eh_valida_inicialmente: 
             log_detalhes_fase_1.append("  Explicação da Fase 0 já contém todas as features, mas não é robusta. Impossível reforçar aditivamente.")
        elif explicacao_esta_completa:
            log_detalhes_fase_1.append("  Explicação da Fase 0 já contém todas as features e é robusta.")

    log_detalhes_fase_1.append(f"  Explicação final da Fase 1 ({len(expl_robusta)} feats):")
    log_detalhes_fase_1.append(f"    Status: {status_robusto}")
    log_detalhes_fase_1.append(f"    Predição da Inst. Perturbada (com expl. da Fase 1): {pred_str_log_apos_adicao}")
    
    return expl_robusta, status_robusto, inst_pert_robusta, log_detalhes_fase_1, features_adicionadas_fase1

def executar_fase_2_minimizacao_minimalidade(
    modelo: LogisticRegression,
    instancia_df: pd.DataFrame,
    explicacao_candidata_robusta: List[str], # Explicação vinda da Fase 1
    X_train: pd.DataFrame,
    t_plus: Optional[float],
    t_minus: Optional[float],
    class_names_binario: List[str],
    status_entrada_fase2: str,        # Status da explicação que entra nesta fase
    inst_pert_entrada_fase2: pd.DataFrame # Instância perturbada correspondente à explicação de entrada
) -> Tuple[List[str], str, pd.DataFrame, List[str], int]:
    """
    Fase 2: Tenta simplificar (minimizar) a explicação robusta da Fase 1, mantendo a robustez.
    """
    log_detalhes_fase_2 = ["FASE 2: Simplificação da Explicação Robusta (Remoção Subtrativa)"] # Nome da fase atualizado
    
    expl_final_min = list(explicacao_candidata_robusta)
    status_final_min = status_entrada_fase2
    inst_pert_final_min = inst_pert_entrada_fase2.copy()
    # Obter a string de predição da explicação que entra na Fase 2
    _, _, pred_str_log_entrada_f2 = perturbar_features(
        modelo, instancia_df, expl_final_min, X_train, t_plus, t_minus, class_names_binario
    )

    log_detalhes_fase_2.append(f"  Explicação robusta de entrada ({len(expl_final_min)} feats):")
    log_detalhes_fase_2.append(f"    Status de entrada: '{status_final_min}'")
    log_detalhes_fase_2.append(f"    Predição da Inst. Perturbada (com expl. de entrada): {pred_str_log_entrada_f2}")

    features_removidas_fase2 = 0
    len_antes_fase2 = len(expl_final_min)

    deve_minimizar = status_entrada_fase2.startswith("VÁLIDA") and bool(expl_final_min)

    if deve_minimizar:
        log_detalhes_fase_2.append(f"  Iniciando minimização subtrativa...")
        
        melhor_explicacao_minima_valida = list(expl_final_min)
        melhor_status_minimo_valido = status_final_min
        melhor_inst_pert_minima_valida = inst_pert_final_min.copy()
        # Predição da melhor explicação válida encontrada até agora (inicialmente é a de entrada)
        pred_str_log_melhor_minima = pred_str_log_entrada_f2


        # Criar mapa nome_feature -> string completa "feature = valor" para reconstrução
        map_nome_feature_para_strcompleta = {
            feat_str.split(' = ')[0]: feat_str for feat_str in expl_final_min
        }
        
        # Ordenar APENAS as features presentes na explicação atual para tentativa de remoção
        features_na_explicacao_para_ordenar = list(map_nome_feature_para_strcompleta.keys())
        
        deltas_instancia_completa = calculate_deltas(modelo, instancia_df, X_train)
        # Filtrar deltas apenas para as features na explicação e criar dict para sorting
        deltas_features_na_explicacao = {
            nome_feat: deltas_instancia_completa[X_train.columns.get_loc(nome_feat)]
            for nome_feat in features_na_explicacao_para_ordenar
        }
        
        features_ordenadas_para_remocao = sorted(
            features_na_explicacao_para_ordenar,
            key=lambda nome_feat: abs(deltas_features_na_explicacao[nome_feat]) # Menor |delta| primeiro
        )
        log_detalhes_fase_2.append(f"    Ordem de tentativa de remoção (features da expl. atual, menor |delta| primeiro): {', '.join(features_ordenadas_para_remocao)}")

        for nome_feature_tentar_remover in features_ordenadas_para_remocao:
            # Verifica se a feature ainda está na explicação que está sendo iterativamente minimizada
            nomes_atuais_expl_min = set([fs.split(' = ')[0] for fs in expl_final_min])
            if nome_feature_tentar_remover not in nomes_atuais_expl_min:
                continue 
            
            expl_temporaria_formatada = [
                map_nome_feature_para_strcompleta[nome_f]
                for nome_f in nomes_atuais_expl_min
                if nome_f != nome_feature_tentar_remover
            ]
            
            log_detalhes_fase_2.append(f"    Tentando remover: '{nome_feature_tentar_remover}'. Explicação atual ({len(expl_final_min)} feats).")
            log_detalhes_fase_2.append(f"      Explicação temporária seria ({len(expl_temporaria_formatada)} feats).")

            if not expl_temporaria_formatada and X_train.shape[1] > 0 : # Evita explicações vazias se ainda houver features para perturbar
                 log_detalhes_fase_2.append("      Explicação temporária vazia, mantendo feature para evitar robustez trivial indesejada (a menos que seja a única feature).")
                 # (você pode adicionar uma lógica mais sofisticada aqui se uma explicação vazia for aceitável em alguns casos)
                 status_tentativa_min = "INVÁLIDA (EXPLICAÇÃO VAZIA)" # Status artificial para não remover
                 pred_str_log_tentativa = "N/A (Explicação vazia)"
            else:
                inst_pert_tentativa_min, status_tentativa_min, pred_str_log_tentativa = perturbar_features(
                    modelo, instancia_df, expl_temporaria_formatada, X_train, t_plus, t_minus, class_names_binario
                )
            
            log_detalhes_fase_2.append(f"      Status após perturbação da temp.: '{status_tentativa_min}'")
            log_detalhes_fase_2.append(f"      Predição da Instância Perturbada (temp.): {pred_str_log_tentativa}")

            if status_tentativa_min.startswith("VÁLIDA"):
                log_detalhes_fase_2.append(f"      REMOÇÃO BEM-SUCEDIDA de '{nome_feature_tentar_remover}'. Explicação permanece válida.")
                expl_final_min = expl_temporaria_formatada
                status_final_min = status_tentativa_min
                inst_pert_final_min = inst_pert_tentativa_min
                
                # Atualiza a "melhor encontrada até agora"
                melhor_explicacao_minima_valida = list(expl_final_min)
                melhor_status_minimo_valido = status_final_min
                melhor_inst_pert_minima_valida = inst_pert_final_min.copy()
                pred_str_log_melhor_minima = pred_str_log_tentativa # Atualiza a predição da melhor
            else:
                log_detalhes_fase_2.append(f"      FALHA NA REMOÇÃO de '{nome_feature_tentar_remover}'. Explicação tornou-se inválida. Mantendo feature.")
        
        # Garante que retornamos a última explicação válida e mínima encontrada
        expl_final_min = melhor_explicacao_minima_valida
        status_final_min = melhor_status_minimo_valido
        inst_pert_final_min = melhor_inst_pert_minima_valida
        pred_str_log_final_da_fase2 = pred_str_log_melhor_minima # Predição correspondente

        features_removidas_fase2 = len_antes_fase2 - len(expl_final_min)
        log_detalhes_fase_2.append(f"  Fim da minimização subtrativa. Features removidas nesta fase: {features_removidas_fase2}")
    elif not expl_final_min:
        log_detalhes_fase_2.append("  Explicação robusta de entrada está vazia. Minimização não aplicável.")
        pred_str_log_final_da_fase2 = "N/A (Explicação vazia na entrada)"
    else: # Não era válida inicialmente
        log_detalhes_fase_2.append(f"  Explicação robusta de entrada NÃO é válida (Status: {status_entrada_fase2}). Minimização não aplicável.")
        pred_str_log_final_da_fase2 = pred_str_log_entrada_f2 # Mantém a predição da entrada

    log_detalhes_fase_2.append(f"  Explicação final da Fase 2 ({len(expl_final_min)} feats).")
    log_detalhes_fase_2.append(f"    Status: {status_final_min}")
    log_detalhes_fase_2.append(f"    Predição da Inst. Perturbada (com expl. final da Fase 2): {pred_str_log_final_da_fase2}")
    
    return expl_final_min, status_final_min, inst_pert_final_min, log_detalhes_fase_2, features_removidas_fase2


# --- Novas Funções de Relatório ---

def gerar_relatorio_resumido(
    modelo: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame,
    class_names_binario: List[str], nome_dataset_original: str,
    t_plus: Optional[float], t_minus: Optional[float]
) -> str:
    """Gera um relatório resumido com estatísticas agregadas."""
    os.makedirs("relatorios_refatorados", exist_ok=True)
    caminho_arquivo_relatorio = os.path.join("relatorios_refatorados", f"relatorio_resumido_{nome_dataset_original}.txt")
    
    scores_decisao_teste = modelo.decision_function(X_test) #
    contagem_validacao_final = Counter() #
    lista_tamanhos_explicacoes_validas = [] #
    lista_todas_features_explicacoes_validas = [] #
    total_features_adicionadas_fase1 = 0 #
    instancias_com_adicao_fase1 = 0 #
    total_features_removidas_fase2 = 0 #
    instancias_com_remocao_fase2 = 0 #

    print(f"Iniciando geração de relatório resumido para {len(X_test)} instâncias de teste...")
    for i in range(len(X_test)):
        if (i + 1) % 50 == 0: # Log de progresso
            print(f"  Processando instância {i+1}/{len(X_test)}...")
        instancia_atual_df = X_test.iloc[[i]] #
        
        expl_fase0, _ = executar_fase_0_explicacao_inicial(modelo, instancia_atual_df, X_train, t_plus, t_minus)
        
        expl_fase1, status_fase1, inst_pert_fase1, _, features_add_f1 = executar_fase_1_refinamento_robustez(
            modelo, instancia_atual_df, expl_fase0, X_train, t_plus, t_minus, class_names_binario
        )
        if features_add_f1 > 0: # (lógica similar)
            total_features_adicionadas_fase1 += features_add_f1
            instancias_com_adicao_fase1 += 1
            
        expl_final, status_final, _, _, features_rem_f2 = executar_fase_2_minimizacao_minimalidade(
            modelo, instancia_atual_df, expl_fase1, X_train, t_plus, t_minus, class_names_binario, status_fase1, inst_pert_fase1
        )
        if features_rem_f2 > 0: # (lógica similar)
            total_features_removidas_fase2 += features_rem_f2
            instancias_com_remocao_fase2 += 1

        contagem_validacao_final[status_final] += 1 #
        if status_final.startswith("VÁLIDA"): #
            lista_tamanhos_explicacoes_validas.append(len(expl_final)) #
            if expl_final: #
                for feat_val_str in expl_final: #
                    lista_todas_features_explicacoes_validas.append(feat_val_str.split(' = ')[0]) #

    with open(caminho_arquivo_relatorio, "w", encoding="utf-8") as f: #
        f.write("="*80 + f"\nRELATÓRIO RESUMIDO DE PI-EXPLICAÇÕES - {nome_dataset_original.upper()}\n" + "="*80 + "\n\n") #
        f.write("[CONFIGURAÇÕES GERAIS]\n") #
        f.write(f"  Total de instâncias de teste: {len(X_test)}\n") #
        f.write(f"  Número total de features no modelo: {X_train.shape[1]}\n") #
        f.write(f"  Acurácia do modelo (teste, sem rejeição): {modelo.score(X_test, y_test):.2%}\n") #
        if t_plus is not None and t_minus is not None: #
            rejeitadas_mask_teste = (scores_decisao_teste <= t_plus) & (scores_decisao_teste >= t_minus) #
            f.write(f"  Taxa de rejeição (teste): {np.mean(rejeitadas_mask_teste):.2%}\n") #
            f.write(f"  Thresholds de Rejeição: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n") #
            aceitas_mask_teste = ~rejeitadas_mask_teste #
            if np.sum(aceitas_mask_teste) > 0: #
                f.write(f"  Acurácia do modelo (instâncias ACEITAS do teste): {modelo.score(X_test[aceitas_mask_teste], y_test[aceitas_mask_teste]):.2%}\n") #
            else:
                f.write("  Acurácia do modelo (instâncias ACEITAS do teste): N/A (todas rejeitadas no teste)\n") #
        else:
            f.write("  Opção de Rejeição: Não Ativada\n") #
        f.write("\nNota: Explicações justificam a PREDIÇÃO DO MODELO.\n\n") #

        f.write("\n\n" + "="*80 + "\nRESUMO DAS VALIDAÇÕES (CONJUNTO DE TESTE - REFINAMENTO COMPLETO)\n" + "="*80 + "\n") #
        total_instancias_processadas = len(X_test) #
        f.write(f"Total de instâncias processadas para explicação: {total_instancias_processadas}\n\n") #
        
        f.write("Contagem de Status das Explicações Finais:\n") #
        for status_key, count in contagem_validacao_final.items(): #
            percentual = (count / total_instancias_processadas * 100) if total_instancias_processadas > 0 else 0 #
            f.write(f"  - {status_key}: {count} ({percentual:.2f}%)\n") #

        media_feats_validas = np.mean(lista_tamanhos_explicacoes_validas) if lista_tamanhos_explicacoes_validas else 0 #
        min_feats_validas = np.min(lista_tamanhos_explicacoes_validas) if lista_tamanhos_explicacoes_validas else 0 #
        max_feats_validas = np.max(lista_tamanhos_explicacoes_validas) if lista_tamanhos_explicacoes_validas else 0 #
        distribuicao_tamanhos_str = "N/A" #
        if lista_tamanhos_explicacoes_validas: #
            dist_counter = Counter(lista_tamanhos_explicacoes_validas) #
            distribuicao_tamanhos_str = ", ".join([f"{size}f: {c} ({c/len(lista_tamanhos_explicacoes_validas)*100:.1f}%)" for size, c in sorted(dist_counter.items())]) #

        f.write("\nEstatísticas do Tamanho das Explicações Válidas Finais:\n") #
        f.write(f"  - Total de Explicações Válidas Consideradas: {len(lista_tamanhos_explicacoes_validas)}\n") #
        f.write(f"  - Média de features: {media_feats_validas:.2f}\n") #
        f.write(f"  - Mínimo de features: {min_feats_validas}\n") #
        f.write(f"  - Máximo de features: {max_feats_validas}\n") #
        f.write(f"  - Distribuição (Tamanho: Qtd (%)): {distribuicao_tamanhos_str}\n") #

        perc_fase1_acionada = (instancias_com_adicao_fase1 / total_instancias_processadas * 100) if total_instancias_processadas > 0 else 0 #
        media_features_adicionadas_fase1 = (total_features_adicionadas_fase1 / instancias_com_adicao_fase1) if instancias_com_adicao_fase1 > 0 else 0 #
        perc_fase2_com_remocao = (instancias_com_remocao_fase2 / total_instancias_processadas * 100) if total_instancias_processadas > 0 else 0 #
        media_features_removidas_fase2 = (total_features_removidas_fase2 / instancias_com_remocao_fase2) if instancias_com_remocao_fase2 > 0 else 0 #
        
        f.write("\nEstatísticas das Fases de Refinamento/Minimização:\n") #
        f.write(f"  - Instâncias que passaram pela Fase 1 (Ref. Aditivo): {instancias_com_adicao_fase1} ({perc_fase1_acionada:.2f}%)\n") #
        f.write(f"  - Média de features adicionadas na Fase 1 (quando acionada): {media_features_adicionadas_fase1:.2f}\n") #
        f.write(f"  - Instâncias com remoção efetiva de features na Fase 2: {instancias_com_remocao_fase2} ({perc_fase2_com_remocao:.2f}%)\n") #
        f.write(f"  - Média de features removidas na Fase 2 (quando houve remoção): {media_features_removidas_fase2:.2f}\n") #

        num_top_features = 10 #
        if lista_todas_features_explicacoes_validas: #
            contagem_features_agg = Counter(lista_todas_features_explicacoes_validas) #
            f.write(f"\nTop {num_top_features} Features Mais Frequentes em Explicações Válidas (Não Vazias):\n") #
            for feature, count in contagem_features_agg.most_common(num_top_features): #
                f.write(f"  - {feature}: {count} ocorrências\n") #
        else:
            f.write("\nTop Features Mais Frequentes: Nenhuma explicação válida não vazia para análise.\n") #

        f.write("\nLembretes sobre os Status de Validação no Relatório Final:\n" #
                "  - VÁLIDA (REJEIÇÃO MANTEVE-SE REJEITADA): Explicação manteve a instância na zona de rejeição após perturbação.\n" #
                "  - VÁLIDA (CLASSIFICADA MANTEVE CLASSE E NÃO FOI REJEITADA): Explicação manteve a classe original e não caiu na zona de rejeição.\n" #
                "  - INVÁLIDA (CLASSIFICADA CAIU NA REJEIÇÃO): Instância originalmente classificada foi para a zona de rejeição após perturbação.\n" #
                # ... (outros lembretes como no original)
                "  - EXPLICAÇÃO VAZIA E INVÁLIDA: A explicação final é vazia E não foi robusta.\n") #
        f.write("\nNota sobre 'Explicação vazia (instância inerentemente robusta à perturbação total)':\n" #
                "  Esta mensagem indica que, mesmo sem fixar features, a perturbação de todas as outras ainda resulta em predição válida.\n") #
        f.write("\n" + "="*80 + "\nFIM DO RELATÓRIO RESUMIDO\n" + "="*80 + "\n") #
    print(f"Relatório resumido para '{nome_dataset_original}' gerado.")
    return caminho_arquivo_relatorio

def gerar_relatorio_detalhado_por_instancia(
    modelo: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    class_names_binario: List[str],
    nome_dataset_original: str,
    t_plus: Optional[float],
    t_minus: Optional[float],
    wr_cost_usado: Optional[float] = None # NOVO PARÂMETRO para exibir o WR_COST
) -> str:
    """
    Gera um relatório detalhado mostrando cada passo da execução por instância,
    com melhor espaçamento e clareza nos logs das fases.
    """
    # Define o diretório para salvar relatórios (pode ser uma constante global)
    diretorio_relatorios = "relatorios_refatorados"
    os.makedirs(diretorio_relatorios, exist_ok=True)
    caminho_arquivo_relatorio = os.path.join(diretorio_relatorios, f"6_{nome_dataset_original}.txt")
    
    # Calcula os scores de decisão para o conjunto de teste uma vez
    scores_decisao_teste = modelo.decision_function(X_test)

    print(f"Iniciando geração de relatório detalhado para {len(X_test)} instâncias de teste...")
    with open(caminho_arquivo_relatorio, "w", encoding="utf-8") as f:
        # Cabeçalho do Relatório
        f.write("="*80 + f"\nRELATÓRIO DETALHADO DE PI-EXPLICAÇÕES - {nome_dataset_original.upper()}".center(80)+"\n" + "="*80 + "\n\n")
        
        # Seção de Configurações Gerais
        f.write("DADOS DATASET\n")
        f.write(f"  Instâncias de teste: {len(X_test)}\n")
        f.write(f"  Features do modelo: {X_train.shape[1]}\n")
        if wr_cost_usado is not None: # Exibe o WR_COST efetivo que foi usado
            f.write(f"  WR_REJECTION_COST efetivo utilizado: {wr_cost_usado:.4f}\n")
        if t_plus is not None and t_minus is not None:
            f.write(f"  Zona de Rejeição(Thresholds): t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n")
        else:
            f.write("  Opção de Rejeição: Não Ativada ou Thresholds não calculados.\n")
        # Você pode adicionar mais informações gerais aqui, como acurácia do modelo, etc.
        f.write("\n") # Linha em branco após configurações gerais

        f.write("DETALHAMENTO INSTÂNCIAS DE TESTE".center(80) + "\n" + "-" * 80 + "\n")

        # Loop através de cada instância de teste
        for i in range(len(X_test)):
            if (i + 1) % 25 == 0 or i == len(X_test) -1 :
                print(f"  Detalhando instância {i+1}/{len(X_test)}...")
            
            instancia_atual_df = X_test.iloc[[i]]
            score_orig_inst = scores_decisao_teste[i]
            pred_orig_inst_idx = modelo.predict(instancia_atual_df)[0]
            
            original_foi_rejeitada = (t_plus is not None and t_minus is not None and (t_minus <= score_orig_inst <= t_plus))
            pred_orig_inst_str = (f"REJEITADA (Score: {score_orig_inst:.4f})" if original_foi_rejeitada
                                  else f"{class_names_binario[pred_orig_inst_idx]} (Score: {score_orig_inst:.4f})")
            
            f.write(f"\n--- INSTÂNCIA # {i} ---\n") # Linha em branco antes de cada nova instância
            
            # Exibição dos valores da instância (com limite opcional)
            num_features_total = len(instancia_atual_df.columns)
            # MAX_FEATURES_DETAILED_REPORT deve ser uma constante global ou passada como parâmetro
            limitar_exibicao = isinstance(MAX_FEATURES_DETAILED_REPORT, int) and MAX_FEATURES_DETAILED_REPORT > 0
            
            if limitar_exibicao and num_features_total > MAX_FEATURES_DETAILED_REPORT:
                f.write(f"  Dados da Instância (primeiras {MAX_FEATURES_DETAILED_REPORT} de {num_features_total} features):\n")
            else:
                f.write(f"  Dados da Instância ({num_features_total} features):\n")

            for col_idx, col_name in enumerate(instancia_atual_df.columns):
                if limitar_exibicao and col_idx >= MAX_FEATURES_DETAILED_REPORT:
                    f.write(f"    ... (e mais {num_features_total - MAX_FEATURES_DETAILED_REPORT} features não exibidas aqui)\n")
                    break
                f.write(f"    {col_name}: {instancia_atual_df.iloc[0, col_idx]:.4f}\n")
            
            # Informações da predição original e classe real
            if t_plus is not None and t_minus is not None:
                 f.write(f"  (Thresholds de referência: t+={t_plus:.4f}, t-={t_minus:.4f})\n")
            f.write(f"  Classe Real: {class_names_binario[y_test.iloc[i]]}\n")
            f.write(f"  Predição Original do Modelo: {pred_orig_inst_str}\n\n") # Linha em branco após info da instância

            # --- FASE 0: Geração da Explicação Inicial ---
            expl_fase0, log_f0 = executar_fase_0_explicacao_inicial(modelo, instancia_atual_df, X_train, t_plus, t_minus)
            for log_line in log_f0: 
                f.write(f"{log_line}\n")
            
            # Adiciona linha em branco APÓS os logs da Fase 0, se houver logs/explicação
            if expl_fase0 and log_f0: 
                f.write("\n") 
            
            # --- FASE 1: Teste de Robustez e Reforço da Explicação ---
            expl_fase1, status_fase1, inst_pert_fase1, log_f1, _ = executar_fase_1_refinamento_robustez(
                modelo, instancia_atual_df, expl_fase0, X_train, t_plus, t_minus, class_names_binario
            )
            for log_line in log_f1: 
                f.write(f"{log_line}\n")
            # Adiciona linha em branco APÓS os logs da Fase 1, se houver logs
            if log_f1:
                f.write("\n")

            # --- FASE 2: Simplificação da Explicação Robusta ---
            expl_final, status_final, inst_pert_final, log_f2, _ = executar_fase_2_minimizacao_minimalidade(
                modelo, instancia_atual_df, expl_fase1, X_train, t_plus, t_minus, class_names_binario, 
                status_fase1, inst_pert_fase1 
            )
            for log_line in log_f2: 
                f.write(f"{log_line}\n")
            # Adiciona linha em branco APÓS os logs da Fase 2, se houver logs
            if log_f2:
                 f.write("\n")

            # --- RESULTADO FINAL PARA A INSTÂNCIA ---
            f.write("  RESULTADO FINAL PARA A INSTÂNCIA:\n")
            f.write("    PI-EXPLICAÇÃO FINAL:\n")
            if not expl_final: # Se a explicação final for vazia
                if status_final.startswith("VÁLIDA"):
                    f.write("      Explicação vazia (instância inerentemente robusta à perturbação total).\n")
                else:
                    f.write("      Nenhuma explicação robusta encontrada (explicação final é vazia e inválida).\n")
            else: # Se houver explicação final
                for idx_f, fv_str in enumerate(expl_final):
                    f.write(f"      {idx_f+1}. {fv_str}\n")

            # Reavalia a predição da instância perturbada com a explicação final para o log do resultado
            # A função perturbar_features retorna a string de predição já formatada
            _, status_validacao_final_recheck, pred_str_final_perturbada = perturbar_features(
                 modelo, instancia_atual_df, expl_final, X_train, t_plus, t_minus, class_names_binario
            )

            f.write(f"    Predição da Instância Perturbada (usando PI-Explicação Final): {pred_str_final_perturbada}\n")
            f.write(f"    Status da Validação Final: {status_validacao_final_recheck}\n") # Usa o status rechecado para consistência
            f.write("-" * 70 + "\n") # Separador para a próxima instância
        
        # Rodapé do Relatório
        f.write("\n\n" + "="*80 + "\nFIM DO RELATÓRIO DETALHADO\n" + "="*80 + "\n")
    
    print(f"Relatório detalhado para '{nome_dataset_original}' gerado em: {caminho_arquivo_relatorio}")
    return caminho_arquivo_relatorio

def _salvar_plot(fig, nome_arquivo_sem_extensao: str, nome_dataset: str):
    """Helper para salvar plots no diretório correto."""
    os.makedirs(DIRETORIO_VISUALIZACOES, exist_ok=True)
    caminho = os.path.join(DIRETORIO_VISUALIZACOES, f"{nome_arquivo_sem_extensao}_{nome_dataset}.png")
    fig.savefig(caminho)
    plt.close(fig)
    print(f"Plot salvo em: {caminho}")


def plotar_distribuicao_scores_decisao(
    scores_decisao: np.ndarray,
    t_plus: Optional[float],
    t_minus: Optional[float],
    nome_dataset: str
):
    """
    Plota a distribuição dos scores de decisão com os limiares de rejeição.
    """
    if scores_decisao is None or len(scores_decisao) == 0:
        print(f"Sem scores de decisão para plotar para o dataset {nome_dataset}.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(scores_decisao, kde=True, ax=ax, label="Distribuição dos Scores de Decisão (Teste)", bins=50)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label="Fronteira de Decisão (Score = 0)")
    if t_plus is not None:
        ax.axvline(t_plus, color='red', linestyle=':', linewidth=2, label=f't+ ({t_plus:.2f})')
    if t_minus is not None:
        ax.axvline(t_minus, color='blue', linestyle=':', linewidth=2, label=f't- ({t_minus:.2f})')
    
    if t_plus is not None and t_minus is not None:
        ax.fill_betweenx(ax.get_ylim(), t_minus, t_plus, color='gray', alpha=0.2, label=f'Zona de Rejeição [{t_minus:.2f}, {t_plus:.2f}]')

    ax.set_title(f'Distribuição dos Scores de Decisão e Limiares\nDataset: {nome_dataset}')
    ax.set_xlabel('Score de Decisão')
    ax.set_ylabel('Frequência / Densidade')
    ax.legend()
    plt.tight_layout()
    _salvar_plot(fig, "dist_scores_decisao", nome_dataset)


def plotar_desempenho_fases_explicacao(
    features_adicionadas_fase1_lista: List[int], # Lista de contagem de features adicionadas na Fase 1 por instância
    features_removidas_fase2_lista: List[int],   # Lista de contagem de features removidas na Fase 2 por instância
    total_instancias: int,
    nome_dataset: str
):
    """
    Plota o desempenho das fases de refinamento (adição na Fase 1) e minimização (remoção na Fase 2).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Fase 1: Adição de Features
    instancias_com_adicao_f1 = [f for f in features_adicionadas_fase1_lista if f > 0]
    perc_adicao_f1 = (len(instancias_com_adicao_f1) / total_instancias * 100) if total_instancias > 0 else 0
    
    if instancias_com_adicao_f1:
        sns.histplot(instancias_com_adicao_f1, ax=axes[0], kde=False, bins=max(1, len(set(instancias_com_adicao_f1))))
        axes[0].set_title(f'Features Adicionadas na Fase 1 (Robustez)\n{perc_adicao_f1:.1f}% das instâncias tiveram adições')
        axes[0].set_xlabel('Número de Features Adicionadas')
    else:
        axes[0].text(0.5, 0.5, 'Nenhuma feature adicionada na Fase 1', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Fase 1: Adição de Features (Robustez)')
    axes[0].set_ylabel('Frequência')

    # Fase 2: Remoção de Features
    instancias_com_remocao_f2 = [f for f in features_removidas_fase2_lista if f > 0]
    perc_remocao_f2 = (len(instancias_com_remocao_f2) / total_instancias * 100) if total_instancias > 0 else 0

    if instancias_com_remocao_f2:
        sns.histplot(instancias_com_remocao_f2, ax=axes[1], kde=False, bins=max(1, len(set(instancias_com_remocao_f2))))
        axes[1].set_title(f'Features Removidas na Fase 2 (Minimalidade)\n{perc_remocao_f2:.1f}% das instâncias tiveram remoções')
        axes[1].set_xlabel('Número de Features Removidas')
    else:
        axes[1].text(0.5, 0.5, 'Nenhuma feature removida na Fase 2', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Fase 2: Remoção de Features (Minimalidade)')
    axes[1].set_ylabel('Frequência')

    fig.suptitle(f'Desempenho das Fases de Refinamento e Minimização\nDataset: {nome_dataset}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para o suptitle
    _salvar_plot(fig, "desempenho_fases", nome_dataset)


def plotar_correlacao_tamanho_confianca(
    tamanhos_explicacoes_validas: List[int], # Lista com tamanhos das explicações finais válidas
    distancias_ao_limiar: List[float],      # Lista com distâncias ao limiar para as mesmas instâncias
    nome_dataset: str
):
    """
    Plota a correlação entre o tamanho da explicação válida e a "confiança"
    (distância do score original ao limiar relevante).
    """
    if not tamanhos_explicacoes_validas or not distancias_ao_limiar or len(tamanhos_explicacoes_validas) != len(distancias_ao_limiar):
        print(f"Dados insuficientes ou inconsistentes para plot de correlação em {nome_dataset}.")
        return

    df_correlacao = pd.DataFrame({
        'Tamanho da Explicação': tamanhos_explicacoes_validas,
        'Distância ao Limiar (Confiança)': distancias_ao_limiar
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='Distância ao Limiar (Confiança)', y='Tamanho da Explicação', data=df_correlacao, ax=ax, alpha=0.6)
    
    # Adicionar linha de regressão (opcional, pode ser poluído se não houver correlação clara)
    # sns.regplot(x='Distância ao Limiar (Confiança)', y='Tamanho da Explicação', data=df_correlacao, ax=ax, scatter=False, color='red')
    
    try:
        correlation_coef = df_correlacao['Distância ao Limiar (Confiança)'].corr(df_correlacao['Tamanho da Explicação'])
        ax.set_title(f'Tamanho da Explicação vs. Confiança da Predição\nDataset: {nome_dataset} (Correlação: {correlation_coef:.2f})')
    except Exception: # Pode falhar se houver std dev zero
        ax.set_title(f'Tamanho da Explicação vs. Confiança da Predição\nDataset: {nome_dataset}')

    ax.set_xlabel('Distância do Score ao Limiar Relevante (Maior = Mais Confiante/Distante)')
    ax.set_ylabel('Número de Features na Explicação Válida')
    plt.tight_layout()
    _salvar_plot(fig, "correlacao_tamanho_confianca", nome_dataset)


def gerar_visualizacoes_completas(
    nome_dataset_original: str,
    lista_tamanhos_explicacoes_validas: List[int], # Geral, para plot 1
    lista_todas_features_explicacoes_validas: List[str], # Plot 2
    contagem_validacao_final: Counter, # Plot 3
    # Para Plot 4
    scores_decisao_teste: Optional[np.ndarray] = None,
    t_plus: Optional[float] = None,
    t_minus: Optional[float] = None,
    # Para Plot 5
    features_adicionadas_f1_todas_instancias: Optional[List[int]] = None,
    features_removidas_f2_todas_instancias: Optional[List[int]] = None,
    total_instancias_teste: Optional[int] = None,
    # Para Plot 6 (Correlação)
    tamanhos_explicacoes_para_correlacao: Optional[List[int]] = None, 
    distancias_ao_limiar_para_correlacao: Optional[List[float]] = None
):
    """
    Gera e salva todas as visualizações, incluindo as novas.
    Esta função deve ser chamada de rejeita_6.py
    """
    print(f"\n--- Gerando Visualizações Avançadas para: {nome_dataset_original} ---")

    # Plot 1: Distribuição do Tamanho das Explicações Válidas (EXISTENTE, mas usando helper)
    if lista_tamanhos_explicacoes_validas:
        fig_tamanho, ax_tamanho = plt.subplots(figsize=(10, 6))
        sns.histplot(lista_tamanhos_explicacoes_validas, kde=False, bins=max(1, len(set(lista_tamanhos_explicacoes_validas)) if lista_tamanhos_explicacoes_validas else 1), ax=ax_tamanho)
        ax_tamanho.set_title(f'Distribuição do Tamanho das Explicações Válidas\nDataset: {nome_dataset_original}')
        ax_tamanho.set_xlabel('Número de Features na Explicação')
        ax_tamanho.set_ylabel('Frequência')
        plt.tight_layout()
        _salvar_plot(fig_tamanho, "dist_tamanho_expl_validas", nome_dataset_original)

    # Plot 2: Features Mais Frequentes (EXISTENTE, mas usando helper)
    if lista_todas_features_explicacoes_validas:
        contagem_features = Counter(lista_todas_features_explicacoes_validas)
        features_comuns_df = pd.DataFrame(contagem_features.most_common(10), columns=['Feature', 'Frequência'])
        if not features_comuns_df.empty:
            fig_freq, ax_freq = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Frequência', y='Feature', data=features_comuns_df, palette="viridis", ax=ax_freq)
            ax_freq.set_title(f'Top 10 Features Mais Frequentes em Explicações Válidas\nDataset: {nome_dataset_original}')
            ax_freq.set_xlabel('Frequência')
            ax_freq.set_ylabel('Feature')
            plt.tight_layout()
            _salvar_plot(fig_freq, "freq_features_expl_validas", nome_dataset_original)

    # Plot 3: Contagem de Status de Validação Final (EXISTENTE, mas usando helper)
    if contagem_validacao_final:
        status_plot = {status: count for status, count in contagem_validacao_final.items() if count > 0}
        if status_plot:
            status_df = pd.DataFrame(status_plot.items(), columns=['Status', 'Contagem']).sort_values(by='Contagem', ascending=False)
            fig_status, ax_status = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Contagem', y='Status', data=status_df, palette="coolwarm", ax=ax_status)
            ax_status.set_title(f'Contagem de Status de Validação Final das Explicações\nDataset: {nome_dataset_original}')
            ax_status.set_xlabel('Contagem')
            ax_status.set_ylabel('Status de Validação')
            plt.tight_layout()
            _salvar_plot(fig_status, "status_validacao_final", nome_dataset_original)

    # Plot 4: Distribuição dos Scores de Decisão (NOVO)
    if scores_decisao_teste is not None:
        plotar_distribuicao_scores_decisao(scores_decisao_teste, t_plus, t_minus, nome_dataset_original)
    
    # Plot 5: Desempenho das Fases de Explicação (NOVO)
    if features_adicionadas_f1_todas_instancias is not None and \
       features_removidas_f2_todas_instancias is not None and \
       total_instancias_teste is not None: # Verifica se os dados foram passados
        plotar_desempenho_fases_explicacao(
            features_adicionadas_f1_todas_instancias,
            features_removidas_f2_todas_instancias,
            total_instancias_teste,
            nome_dataset_original
        )

    # Plot 6: Correlação Tamanho da Explicação vs. Confiança (NOVO)
    if tamanhos_explicacoes_para_correlacao and distancias_ao_limiar_para_correlacao:
         plotar_correlacao_tamanho_confianca(
             tamanhos_explicacoes_para_correlacao, # Passa a lista de tamanhos correta
             distancias_ao_limiar_para_correlacao, # Passa a lista de distâncias correta
             nome_dataset_original
         )
    
    print(f"--- Visualizações Avançadas para '{nome_dataset_original}' salvas na pasta '{DIRETORIO_VISUALIZACOES}'. ---")

def carregar_hiperparametros(nome_dataset: str, arquivo_json: str = 'hiperparam.json') -> Dict[str, Any]: #
    try:
        with open(arquivo_json, 'r') as f:
            todos_hiperparametros = json.load(f)
        return todos_hiperparametros.get(nome_dataset, {}).get('hiperparametros', {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {}

# --- Função Principal Modificada ---
def main(): #
    MODO_SALVAR_ARTEFATOS: bool = False #
    MODO_CARREGAR_ARTEFATOS: bool = False #
    # Para testar rapidamente, pode fixar um dataset ou deixar o menu interativo
    # nome_dataset_para_processar: str = "breast_cancer"

    modelo_logreg: Optional[LogisticRegression] = None #
    X_treino_df: Optional[pd.DataFrame] = None #
    y_treino_series: Optional[pd.Series] = None #
    X_teste_df: Optional[pd.DataFrame] = None #
    y_teste_series: Optional[pd.Series] = None #
    threshold_plus: Optional[float] = None #
    threshold_minus: Optional[float] = None #
    nomes_classes_binarias: Optional[List[str]] = None #
    nome_dataset_usado: Optional[str] = None #

    if MODO_CARREGAR_ARTEFATOS: #
        nome_dataset_para_processar_carregar = input("Digite o nome do dataset para carregar artefatos (ex: breast_cancer): ").strip()
        print(f"Modo Carregar Artefatos ATIVADO para: {nome_dataset_para_processar_carregar}") #
        artefatos_carregados = carregar_artefatos_salvos(nome_dataset_para_processar_carregar) #
        if artefatos_carregados: #
            modelo_logreg = artefatos_carregados.get("modelo") #
            X_treino_df = artefatos_carregados.get("X_train_df") #
            y_treino_series = artefatos_carregados.get("y_train") #
            X_teste_df = artefatos_carregados.get("X_test_df") #
            y_teste_series = artefatos_carregados.get("y_test") #
            threshold_plus = artefatos_carregados.get("t_plus") #
            threshold_minus = artefatos_carregados.get("t_minus") #
            nomes_classes_binarias = artefatos_carregados.get("class_names_binario") #
            nome_dataset_usado = artefatos_carregados.get("nome_dataset_original") #
        else:
            print(f"Falha ao carregar artefatos para '{nome_dataset_para_processar_carregar}'. Encerrando.") #
            return
    else:
        nome_ds_selecionado, _, X_data, y_data, nomes_cls_bin = selecionar_dataset_e_classe() #
        if nome_ds_selecionado is None or X_data is None or y_data is None or nomes_cls_bin is None: #
            print("Nenhum dataset selecionado ou falha na configuração. Encerrando.") #
            return
        nome_dataset_usado = nome_ds_selecionado #
        nomes_classes_binarias = nomes_cls_bin #
        if isinstance(y_data, np.ndarray): y_data = pd.Series(y_data, index=X_data.index) #

        X_treino_df, X_teste_arr, y_treino_series, y_teste_series_temp = train_test_split( #
            X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_data
        )
        # Assegurar que X_treino_df e X_teste_df sejam DataFrames com nomes de colunas corretos
        X_treino_df = pd.DataFrame(X_treino_df, columns=X_data.columns) #
        X_teste_df = pd.DataFrame(X_teste_arr, columns=X_data.columns) #
        y_teste_series = y_teste_series_temp #

        hiperparams_dataset = carregar_hiperparametros(nome_dataset_usado) #
        params_modelo = DEFAULT_LOGREG_PARAMS.copy() #
        params_modelo.update(hiperparams_dataset) #
        modelo_logreg = LogisticRegression(**params_modelo) #
        modelo_logreg.fit(X_treino_df, y_treino_series) #
        
        threshold_plus, threshold_minus = calcular_thresholds(modelo_logreg, X_treino_df, y_treino_series, custo_rejeicao_wr=WR_REJECTION_COST) #
        
        if MODO_SALVAR_ARTEFATOS: #
            artefatos_para_salvar = { #
                "modelo": modelo_logreg, "X_train_df": X_treino_df, "y_train": y_treino_series, #
                "X_test_df": X_teste_df, "y_test": y_teste_series, "t_plus": threshold_plus, #
                "t_minus": threshold_minus, "class_names_binario": nomes_classes_binarias, #
                "nome_dataset_original": nome_dataset_usado #
            }
            salvar_artefatos(artefatos_para_salvar, nome_dataset_usado) #
            print(">>> ARTEFATOS SALVOS. Encerrando para evitar reprocessamento imediato.") #
            return # Retorna após salvar para não continuar para relatórios/plots nesta execução

    # Verificação de segurança para garantir que as variáveis essenciais estão carregadas ou calculadas
    if not all([modelo_logreg, X_teste_df is not None, y_teste_series is not None, 
                X_treino_df is not None, nomes_classes_binarias, nome_dataset_usado]): #
        print("Erro: Uma ou mais variáveis essenciais não foram inicializadas. Encerrando.") #
        return

    print("\nGerando Relatório Resumido...")
    caminho_resumido = gerar_relatorio_resumido(
        modelo_logreg, X_teste_df, y_teste_series, X_treino_df,
        nomes_classes_binarias, nome_dataset_usado,
        threshold_plus, threshold_minus
    )
    print(f"Relatório Resumido salvo em: {caminho_resumido}")

    print("\nGerando Relatório Detalhado...")
    caminho_detalhado = gerar_relatorio_detalhado_por_instancia(
        modelo_logreg, X_teste_df, y_teste_series, X_treino_df,
        nomes_classes_binarias, nome_dataset_usado,
        threshold_plus, threshold_minus
    )
    print(f"Relatório Detalhado salvo em: {caminho_detalhado}")

    print("\nColetando dados para visualizações...")
    # Para Plot 3
    contagem_validacao_plot = Counter()
    # Para Plot 1 (todas explicações válidas)
    tamanhos_plot_geral = []
    # Para Plot 2 (features de todas explicações válidas)
    features_plot = []
    
    # Para Plot 4 (Scores de Decisão)
    # Certifique-se que X_teste_df não é None e tem linhas
    scores_teste_para_plot = np.array([])
    if X_teste_df is not None and not X_teste_df.empty:
        scores_teste_para_plot = modelo_logreg.decision_function(X_teste_df)
    
    # Para Plot 5 (Desempenho das Fases)
    lista_features_adicionadas_f1_plot = []
    lista_features_removidas_f2_plot = []
    
    # Para Plot 6 (Correlação Tamanho vs. Confiança)
    tamanhos_plot_para_correlacao = []    # Tamanhos das explicações de instâncias classificadas válidas
    distancias_ao_limiar_validas_plot = [] # Distâncias ao limiar para essas mesmas instâncias

    if X_teste_df is not None and not X_teste_df.empty: # Apenas itera se houver dados de teste
        for i in range(len(X_teste_df)):
            if (i + 1) % 25 == 0 or i == len(X_teste_df) -1 : # Log de progresso mais frequente
                print(f"  Coletando dados da instância {i+1}/{len(X_teste_df)} para plots...")
            instancia_atual_df = X_teste_df.iloc[[i]]
            
            expl_f0, _ = executar_fase_0_explicacao_inicial(modelo_logreg, instancia_atual_df, X_treino_df, threshold_plus, threshold_minus)
            
            expl_f1, status_f1, inst_pert_f1, _, features_add_f1 = executar_fase_1_refinamento_robustez(
                modelo_logreg, instancia_atual_df, expl_f0, X_treino_df, threshold_plus, threshold_minus, nomes_classes_binarias
            )
            lista_features_adicionadas_f1_plot.append(features_add_f1) # Coleta para Plot 5
            
            expl_final, status_final, _, _, features_rem_f2 = executar_fase_2_minimizacao_minimalidade(
                modelo_logreg, instancia_atual_df, expl_f1, X_treino_df, threshold_plus, threshold_minus, nomes_classes_binarias, status_f1, inst_pert_f1
            )
            lista_features_removidas_f2_plot.append(features_rem_f2) # Coleta para Plot 5
            
            contagem_validacao_plot[status_final] +=1 # Para Plot 3
            
            if status_final.startswith("VÁLIDA"): # Para Plot 1 e 2
                tamanhos_plot_geral.append(len(expl_final))
                if expl_final: # Apenas adiciona features se a explicação não for vazia
                    for fv_str in expl_final:
                        features_plot.append(fv_str.split(" = ")[0])

                # Coleta de dados para Plot 6 (Correlação)
                score_inst_original = modelo_logreg.decision_function(instancia_atual_df)[0]
                pred_classe_inst_original = modelo_logreg.predict(instancia_atual_df)[0]
                
                originalmente_classificada = True
                if threshold_plus is not None and threshold_minus is not None:
                    if threshold_minus <= score_inst_original <= threshold_plus:
                        originalmente_classificada = False
                
                if originalmente_classificada and status_final.startswith("VÁLIDA (CLASSIFICADA MANTEVE CLASSE"):
                    distancia_calculada = 0.0
                    if threshold_plus is not None and threshold_minus is not None: 
                        if pred_classe_inst_original == 1 and score_inst_original > threshold_plus:
                            distancia_calculada = score_inst_original - threshold_plus
                        elif pred_classe_inst_original == 0 and score_inst_original < threshold_minus:
                            distancia_calculada = threshold_minus - score_inst_original 
                        # Se classificada mas entre os thresholds (e.g. exatamente em t+ ou t- mas ainda considerada classificada)
                        # a distância seria 0, o que é ok.
                    else: 
                        distancia_calculada = abs(score_inst_original)
                    
                    # Adiciona mesmo se a distância for pequena ou zero, desde que as condições sejam atendidas.
                    # A função de plotagem pode depois decidir como lidar com muitos pontos em zero, se necessário.
                    distancias_ao_limiar_validas_plot.append(distancia_calculada)
                    tamanhos_plot_para_correlacao.append(len(expl_final))
    else:
        print("X_teste_df está vazio. Nenhum dado será coletado para visualizações.")

    print("\nGerando Visualizações...")
    if nome_dataset_usado:
        # Apenas chama a geração de visualizações se X_teste_df não estiver vazio
        if X_teste_df is not None and not X_teste_df.empty:
            gerar_visualizacoes_completas(
                nome_dataset_original=nome_dataset_usado,
                lista_tamanhos_explicacoes_validas=tamanhos_plot_geral,                 # Plot 1
                lista_todas_features_explicacoes_validas=features_plot,                 # Plot 2
                contagem_validacao_final=contagem_validacao_plot,                     # Plot 3
                scores_decisao_teste=scores_teste_para_plot if scores_teste_para_plot.size > 0 else None, # Plot 4
                t_plus=threshold_plus,                                                # Plot 4
                t_minus=threshold_minus,                                              # Plot 4
                features_adicionadas_f1_todas_instancias=lista_features_adicionadas_f1_plot, # Plot 5
                features_removidas_f2_todas_instancias=lista_features_removidas_f2_plot,     # Plot 5
                tamanhos_explicacoes_para_correlacao=tamanhos_plot_para_correlacao,   # Plot 6 (tamanhos)
                distancias_ao_limiar_para_correlacao=distancias_ao_limiar_validas_plot, # Plot 6 (distâncias)
                total_instancias_teste=len(X_teste_df) if X_teste_df is not None else 0 # Plot 5
            )
        else:
            print("Dados de teste (X_teste_df) estão vazios. Nenhuma visualização será gerada.")
    else:
        print("Nome do dataset não definido, pulando visualizações.")

if __name__ == "__main__":
    main()