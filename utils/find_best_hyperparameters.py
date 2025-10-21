import os
import json
import itertools
import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Importa a função de carregamento de datasets do seu script.
# Garante que o script está sendo executado no diretório correto.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.datasets_Mateus_comparation_complete import carregar_dataset

# Ignora avisos de convergência durante a busca para não poluir a saída.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- CONFIGURAÇÕES DA BUSCA ---

# Hiperparâmetros para a busca em grade (Grid Search)
PARAM_GRID = {
    "modelo__penalty": ["l1", "l2"],
    "modelo__C": [0.01, 0.1, 1, 10, 100],
    "modelo__max_iter": [200, 500, 1000],
    "modelo__solver": ["liblinear", "saga"]
}

# Lista de todos os datasets que serão processados automaticamente
DATASET_NAMES = [
    'sonar', 'pima_indians_diabetes', 'breast_cancer',
    'wine', 'vertebral_column', 'iris'#removido credicard
]

# Para datasets com mais de 2 classes, define um mapeamento padrão para o problema binário.
# Mapeia (classe a ser usada como 0, classe a ser usada como 1)
MULTICLASS_TO_BINARY_MAP = {
#    'iris': (0, 1),   # Ex: 'setosa' vs 'versicolor'
#    'wine': (0, 1),   # Ex: 'class_0' vs 'class_1'
#    'seeds': (0, 1)   # Ex: 'Kama' vs 'Rosa'
# em concnordancia com a comparação com o experimento do mateus
    'iris': (0, -1),   # Ex: 'setosa' vs rest
    'wine': (0, -1),   # Ex: 'class_0' vs rest
    'seeds': (0, -1)   # Ex: 'Kama' vs rest
}

# --- FUNÇÕES AUXILIARES ---

def gerar_combinacoes_de_parametros() -> List[Dict]:
    """Gera todas as combinações de hiperparâmetros a partir do PARAM_GRID."""
    keys, values = zip(*PARAM_GRID.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def preparar_dataset_para_binario(nome_dataset: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    Carrega um dataset e o transforma em um problema de classificação binária,
    seja por padrão ou usando o mapeamento definido.
    """
    X, y, _ = carregar_dataset(nome_dataset)
    if X is None or y is None:
        print(f"  -> Falha ao carregar {nome_dataset}. Pulando.")
        return None, None

    # Se o dataset tem mais de 2 classes, aplica o mapeamento.
    if nome_dataset in MULTICLASS_TO_BINARY_MAP:
        idx_classe_0, idx_classe_1 = MULTICLASS_TO_BINARY_MAP[nome_dataset]
        
        mascara_0 = (y == idx_classe_0)
        mascara_1 = (y == idx_classe_1)
        mascara_total = mascara_0 | mascara_1
        
        X_filtrado = X[mascara_total].copy()
        y_filtrado = y[mascara_total].copy()
        
        # Converte para 0 e 1
        y_final = np.where(y_filtrado == idx_classe_0, 0, 1)
        y_final = pd.Series(y_final, index=X_filtrado.index)
        
        print(f"  -> Dataset multi-classe. Usando classes {idx_classe_0} e {idx_classe_1} para binarização.")
        return X_filtrado, y_final

    # Se já for binário, retorna como está.
    return X, y

def avaliar_combinacao(args: Tuple) -> Optional[Dict]:
    params, X, y, splits = args
    accs, f1s, non_zero_coeffs = [], [], []

    # Cria a pipeline uma vez
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', LogisticRegression(random_state=42))
    ])

    # Define os parâmetros para a pipeline nesta iteração
    pipeline.set_params(**params)

    for train_idx, val_idx in splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        try:
            # Treina a pipeline inteira (scaler + modelo)
            pipeline.fit(X_train, y_train)
            
            # Prediz com a pipeline (ela aplica o scaler internamente)
            y_pred = pipeline.predict(X_val)

            # Acessa o modelo treinado dentro da pipeline para ver os coeficientes
            modelo_treinado = pipeline.named_steps['modelo']
            non_zero_coeffs.append(np.count_nonzero(np.abs(modelo_treinado.coef_) > 1e-5))
            accs.append(accuracy_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))

        except Exception:
            return None

    if not accs:
        return None

    return {
        # Retorna os parâmetros com o prefixo 'modelo__' removido para o JSON final
        "params": {key.replace('modelo__', ''): val for key, val in params.items()},
        "accuracy": np.mean(accs),
        "f1_score": np.mean(f1s),
        "avg_non_zero_coeffs": np.mean(non_zero_coeffs)
    }

# --- FUNÇÃO PRINCIPAL ---

def main():
    """
    Orquestra o processo de otimização para todos os datasets,
    gerando um JSON com os melhores hiperparâmetros e um relatório em TXT.
    """
    print("Iniciando busca otimizada de hiperparâmetros para Regressão Logística...")
    
    # Carrega o arquivo JSON existente para atualizá-lo
    caminho_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json", "hiperparametros.json")
    if os.path.exists(caminho_json):
        with open(caminho_json, 'r') as f:
            melhores_parametros_gerais = json.load(f)
    else:
        melhores_parametros_gerais = {}

    relatorio_linhas = [
        "===================================================================================",
        "        RELATÓRIO DE OTIMIZAÇÃO DE HIPERPARÂMETROS (Logistic Regression)",
        "===================================================================================\n"
        "Critérios de seleção (prioridade):\n"
        "  1. Maior Acurácia (accuracy)\n"
        "  2. Maior F1-Score (em caso de empate na acurácia)\n"
        "  3. Menor Número Médio de Coeficientes Não-Nulos (em caso de empate em ambos)\n"
    ]

    combinacoes = gerar_combinacoes_de_parametros()
    
    for nome_dataset in DATASET_NAMES:
        print(f"\nProcessando dataset: {nome_dataset.upper()}...")
        
        X, y = preparar_dataset_para_binario(nome_dataset)
        if X is None:
            continue

        n_folds = 5 if len(X) > 1000 else 10
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))

        args_list = [(params, X, y, splits) for params in combinacoes]

        # Usa multiprocessing para acelerar a busca
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            resultados = pool.map(avaliar_combinacao, args_list)

        # Filtra execuções que falharam e ordena os resultados
        resultados_validos = [r for r in resultados if r is not None]
        if not resultados_validos:
            print(f"  -> Nenhuma combinação de parâmetros funcionou para {nome_dataset}.")
            continue
            
        resultados_validos.sort(
            key=lambda x: (-x["accuracy"], -x["f1_score"], x["avg_non_zero_coeffs"]),
            reverse=False
        )

        melhor_resultado = resultados_validos[0]
        melhores_parametros_gerais[nome_dataset] = {"params": melhor_resultado["params"]}
        
        # Monta a seção do relatório para este dataset
        relatorio_linhas.append("\n" + "="*82)
        relatorio_linhas.append(f" DATASET: {nome_dataset.upper()}")
        relatorio_linhas.append("="*82)
        relatorio_linhas.append(
            f"{'Penalty':<8} | {'Solver':<10} | {'C':<7} | {'Max_Iter':<8} | "
            f"{'Acurácia':<12} | {'F1-Score':<12} | {'Avg Non-Zero Coeffs'}"
        )
        relatorio_linhas.append("-" * 82)

        for res in resultados_validos[:10]: # Mostra os top 10 resultados
            p = res['params']
            linha = (
                f"{p['penalty']:<8} | {p['solver']:<10} | {str(p['C']):<7} | {str(p['max_iter']):<8} | "
                f"{res['accuracy']:.4f}{'*' if res == melhor_resultado else ' ':<11} | "
                f"{res['f1_score']:.4f}{' ':<12} | "
                f"{res['avg_non_zero_coeffs']:.2f}"
            )
            relatorio_linhas.append(linha)
        relatorio_linhas.append(f"\n  -> MELHOR ESCOLHA PARA '{nome_dataset.upper()}': {melhor_resultado['params']} (*)")

    # Salva o arquivo JSON com os melhores parâmetros para todos os datasets
    with open(caminho_json, 'w') as f:
        json.dump(melhores_parametros_gerais, f, indent=4)
    print(f"\n\nArquivo de hiperparâmetros foi salvo com sucesso em: '{caminho_json}'")

    # Salva o relatório TXT
    caminho_relatorio = "relatorio_otimizacao.txt"
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("\n".join(relatorio_linhas))
    print(f"Relatório de otimização foi salvo em: '{caminho_relatorio}'")

if __name__ == '__main__':
    main()