import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split 

# ==============================================================================
# 1. AJUSTE DE CAMINHOS (PATH)
# ==============================================================================
# Como este script está em /utils/, subimos um nível para chegar à raiz do projeto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from data.datasets import carregar_dataset

# Silencia avisos de convergência durante o Grid Search
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==============================================================================
# 2. CONFIGURAÇÕES DA BUSCA
# ==============================================================================
# Note que os solvers precisam ser compatíveis com L1 e L2
PARAM_GRID = [
    {"penalty": ["l1"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "saga"], "max_iter": [500]},
    {"penalty": ["l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"], "max_iter": [500]}
]

# Lista de datasets para otimizar (adicione ou remova conforme necessário)
# Lista de todos os datasets do projeto para otimizar
DATASETS_PARA_OTIMIZAR = [
    "breast_cancer", 
    "pima_indians_diabetes", 
    "banknote",
    "heart_disease",
    "spambase",
    "vertebral_column",
    "sonar",
    "wine",
    "creditcard",
    "covertype",
    "gas_sensor",
    "mnist",
    "newsgroups",
    "rcv1"
]

# Caminho final onde o JSON será salvo
JSON_OUTPUT_DIR = os.path.join(ROOT_DIR, 'json')
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
JSON_OUTPUT_PATH = os.path.join(JSON_OUTPUT_DIR, 'best_hyperparameters.json')

# ==============================================================================
# 3. MOTOR DE OTIMIZAÇÃO
# ==============================================================================
def otimizar_dataset(dataset_name: str) -> dict:
    print(f"\n[INFO] Iniciando Grid Search para: {dataset_name.upper()}")
    
    try:
        retorno = carregar_dataset(dataset_name)
        # Verifica se o dataset não existe ou retornou vazio
        if retorno is None or len(retorno) < 2 or retorno[0] is None:
            print(f"[AVISO] Dataset '{dataset_name}' não encontrado ou inválido. Pulando...")
            return None
        X, y, _ = retorno
    except Exception as e:
        print(f"[ERRO] Falha ao carregar {dataset_name}: {e}")
        return None

    # Reduz o tamanho do dataset se for gigantesco, apenas para achar os melhores params
    if len(X) > 2000:
        print(f"[INFO] Reduzindo amostra de {dataset_name} para acelerar a busca...")
        _, X_sub, _, y_sub = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X, y = X_sub, y_sub

    melhor_score = -1.0
    melhores_params = {}
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Iterando manualmente sobre o grid para garantir controle total
    for grid_dict in PARAM_GRID:
        for penalty in grid_dict["penalty"]:
            for C in grid_dict["C"]:
                for solver in grid_dict["solver"]:
                    for max_iter in grid_dict["max_iter"]:
                        
                        params_atuais = {'penalty': penalty, 'C': C, 'solver': solver, 'max_iter': max_iter}
                        scores = []
                        
                        for train_idx, val_idx in cv.split(X, y):
                            X_tr, X_val = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
                            y_tr, y_val = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
                            
                            modelo = Pipeline([
                                ('scaler', MinMaxScaler()),
                                ('lr', LogisticRegression(random_state=42, **params_atuais))
                            ])
                            
                            try:
                                modelo.fit(X_tr, y_tr)
                                preds = modelo.predict(X_val)
                                scores.append(accuracy_score(y_val, preds))
                            except Exception:
                                scores.append(0.0) # Falha na convergência ou incompatibilidade
                                
                        media_score = np.mean(scores)
                        
                        if media_score > melhor_score:
                            melhor_score = media_score
                            melhores_params = params_atuais

    print(f"[OK] Melhor configuração para {dataset_name}: {melhores_params} (Acurácia: {melhor_score:.4f})")
    return melhores_params

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    resultados_globais = {}
    
    print("="*60)
    print("  OTIMIZAÇÃO DE HIPERPARÂMETROS (SURROGATE MINABRO)")
    print("="*60)
    
    for ds in DATASETS_PARA_OTIMIZAR:
        params = otimizar_dataset(ds)
        if params is not None:
            resultados_globais[ds] = params
            
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(resultados_globais, f, indent=4)
        
    print("="*60)
    print(f"[SUCESSO] Hiperparâmetros salvos em: {JSON_OUTPUT_PATH}")
    print("="*60)