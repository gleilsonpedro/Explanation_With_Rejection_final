import os
import json
import warnings
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datasets.datasets_Mateus_comparation_complete import selecionar_dataset_e_classe
from explanations.pi_explanation import one_explanation, verificar_explicacao_detalhada
from multiprocessing import Pool, cpu_count

"""
Realiza busca de hiperparâmetros para Regressão Logística com avaliação de PI-explicações.

Este script avalia diferentes combinações de hiperparâmetros utilizando validação cruzada estratificada
e mede a acurácia, precisão, recall, F1-score, AUC, tamanho médio das explicações e sua robustez.

Os melhores parâmetros encontrados são salvos em um arquivo JSON (`hiperparam.json`), permitindo
que o `main.py` utilize diretamente os hiperparâmetros otimizados por dataset.

Requisitos:
    - scikit-learn
    - numpy
    - pandas
    - módulo de PI-explicações personalizado

"""

warnings.filterwarnings("ignore")

# Hiperparâmetros para busca
param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 10, 100],
    "max_iter": [200, 500, 1000],
    "solver": ["liblinear", "lbfgs", "saga"]
}

def gerar_combinacoes_validas():
    """
    Gera combinações válidas de hiperparâmetros para Regressão Logística.

    Filtra combinações inválidas como:
      - penalty='l1' com solver incompatível (exceto 'liblinear' ou 'saga')
      - penalty='l2' com solver='liblinear' e C muito baixo (evita instabilidades)

    Returns:
        list of tuple: Lista de combinações válidas na forma (penalty, solver, C, max_iter).
    """

    combinacoes = []
    for p, c, m, s in itertools.product(param_grid["penalty"], param_grid["C"], param_grid["max_iter"], param_grid["solver"]):
        if (p == "l1" and s not in ["liblinear", "saga"]) or (p == "l2" and s == "liblinear" and c == 0.01):
            continue
        combinacoes.append((p, s, c, m))
    return combinacoes

def avaliar_modelo(args):
    """
    Avalia um modelo de Regressão Logística com uma combinação específica de hiperparâmetros.

    Utiliza validação cruzada estratificada e calcula métricas de desempenho,
    além de avaliar explicações PI geradas para amostras de teste.

    Args:
        args (tuple): Contém (X, y, penalty, solver, C, max_iter, nome_dataset, splits),
            onde:
            - X (DataFrame): Dados de entrada.
            - y (array): Rótulos binários.
            - penalty (str): Tipo de penalização ('l1' ou 'l2').
            - solver (str): Algoritmo de otimização.
            - C (float): Inverso da regularização.
            - max_iter (int): Número máximo de iterações.
            - nome_dataset (str): Nome do dataset atual.
            - splits (list): Lista de splits do StratifiedKFold.

    Returns:
        dict or None: Dicionário com métricas médias (se avaliação teve sucesso),
        ou None se todos os folds falharem.
    """

    X, y, p, s, c, m, nome_dataset, splits = args

    accs, feats, precs, recalls, f1s, rocs, robusts = [], [], [], [], [], [], []
    logs = [f"\nAvaliando {p}/{s}/C={c}/max_iter={m} com {len(splits)}-Fold..."]

    for fold, (train_index, test_index) in enumerate(splits, 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        try:
            modelo = LogisticRegression(penalty=p, C=c, solver=s, max_iter=m, random_state=42)
            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)
            y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else [0]*len(y_test)

            # Amostragem de até 20 instâncias para PI-explicações
            n_amostras = min(20, len(X_test))
            indices = np.random.choice(len(X_test), size=n_amostras, replace=False)

            tamanhos, validacoes = [], []
            for i in indices:
                exp = one_explanation(modelo, X_test.iloc[[i]], X_train)
                valid, _ = verificar_explicacao_detalhada(modelo, X_test.iloc[[i]], exp, X_train)
                tamanhos.append(len(exp))
                validacoes.append(valid)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
            tam = np.mean(tamanhos) if tamanhos else 0
            rob = 100 * sum(validacoes)/len(validacoes) if validacoes else 0

            accs.append(acc)
            precs.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            rocs.append(roc)
            feats.append(tam)
            robusts.append(rob)

            logs.append(f"  Fold {fold}: Acc={acc:.2%} | Features={tam:.1f} | Robustez={rob:.0f}%")

        except Exception as e:
            logs.append(f"  Fold {fold} falhou: {str(e)}")
            continue

    if not accs:
        logs.append(f"  Todos os folds falharam para {p}/{s}/C={c}/max_iter={m}")
        print("\n".join(logs))
        return None

    # Imprime tudo ao final
    print("\n".join(logs))

    return {
        "penalty": p,
        "solver": s,
        "C": c,
        "max_iter": m,
        "accuracy": np.mean(accs),
        "accuracy_std": np.std(accs),
        "precision": np.mean(precs),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
        "roc_auc": np.mean(rocs),
        "avg_feats": np.mean(feats),
        "avg_feats_std": np.std(feats),
        "robustez": np.mean(robusts)
    }

def main():
    """
    Executa o pipeline principal de busca de hiperparâmetros para Regressão Logística com avaliação de PI-explicações.

    Fluxo:
        1. Carrega o dataset e prepara os dados (via `selecionar_dataset_e_classe()`)
        2. Configura a validação cruzada estratificada (5 ou 10 folds conforme tamanho do dataset)
        3. Gera todas as combinações válidas de hiperparâmetros
        4. Avalia cada combinação em paralelo, calculando métricas de desempenho e explicações
        5. Ordena os resultados e salva os melhores hiperparâmetros em 'hiperparam.json'

    Métricas reportadas:

        TP - True Positive | TN - True Negativo | FP - Falso Positivo | FN - Falso Negativo
        - Accuracy (Acurácia):
            * O que é: Proporção de classificações corretas (verdadeiros positivos + verdadeiros negativos)
            * Importância: Quanto mais alto, melhor a confiabilidade geral do modelo
            * Fórmula: (TP + TN) / (TP + TN + FP + FN)

        - Precision (Precisão):
            * O que é: Proporção de verdadeiros positivos entre os casos classificados como positivos
            * Importância: Alto valor indica menos falsos positivos nas explicações
            * Fórmula: TP / (TP + FP)

        - Recall (Revocação):
            * O que é: Proporção de verdadeiros positivos identificados dentre todos os positivos reais
            * Importância: Alto valor significa que o modelo captura a maioria dos casos relevantes
            * Fórmula: TP / (TP + FN)

        - F1-Score:
            * O que é: Média harmônica entre Precisão e Recall (balanceia os dois aspectos)
            * Importância: Ideal para datasets desbalanceados, penaliza extremos
            * Fórmula: 2 * (Precision * Recall) / (Precision + Recall)

        - ROC AUC:
            * O que é: Área sob a curva ROC (mede capacidade de distinguir entre classes)
            * Importância: Quanto mais próximo de 1, melhor o modelo separa as classes
            * Escala: 0.5 (aleatório) a 1.0 (perfeito)

        - Avg_Feats (Tamanho médio das explicações):
            * O que é: Número médio de features usadas nas PI-explicações
            * Importância: Valores menores indicam explicações mais concisas e interpretáveis
            * Meta: Minimizar sem prejudicar a acurácia

        - Robustez:
            * O que é: Porcentagem de explicações que passam na verificação semântica
            * Importância: Garante que as explicações são consistentes e válidas
            * Ideal: 100%

    Ordem de prioridade para seleção do melhor modelo:
        1. Maior Accuracy (desempenho geral)
        2. Maior F1-Score (balanceamento precisão/recall)
        3. Menor Avg_Feats (explicações mais simples)

    Retorna:
        None (os resultados são salvos em 'hiperparam.json')
    """

    nome_dataset, _, X, y, class_names = selecionar_dataset_e_classe()
    if nome_dataset is None:
        return

    # Configuração automática de K-Fold
    n_samples = len(X)
    n_folds = 10 if n_samples < 500 else 5
    
    print(f"\nDataset: {nome_dataset}")
    print(f"  Samples: {n_samples} | Features: {X.shape[1]} | Classes: {len(class_names)}")
    print(f"  Estrategia: Stratified {n_folds}-Fold")
    print(f"  Buscando melhores hiperparametros...\n")

    # Gera todos os splits ANTES da paralelização
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))  # Convertendo para lista para reusar

    combinacoes = gerar_combinacoes_validas()
    
    # Prepara argumentos incluindo os splits pre-gerados
    args_list = [(X, y, p, s, c, m, nome_dataset, splits) 
                for p, s, c, m in combinacoes]

    # Paralelização da busca
    with Pool(processes=cpu_count()) as pool:
        resultados = pool.map(avaliar_modelo, args_list)

    # Filtra resultados válidos e ordena
    resultados = [r for r in resultados if r]
    resultados.sort(key=lambda x: (-x["accuracy"], -x["f1"], x["avg_feats"])) # sequencia de verificação acuracia -> f1 -> avg_feats

    print("\nLegenda das métricas:")
    print("  Accuracy    : Acurácia média de classificação")
    print("  Acc_Std     : Desvio padrão da acurácia entre os folds")
    print("  Precision   : Precisão (quantos positivos previstos são corretos)")
    print("  Recall      : Revocação (quantos positivos reais foram encontrados)")
    print("  F1          : Média harmônica entre precisão e recall")
    print("  ROC AUC     : Área sob a curva ROC (discriminação)")
    print("  Avg_Feats   : Tamanho médio das explicações PI")
    print("  Robustez    : Porcentagem das explicações consideradas robustas")


    # Exibe tabela de resultados
    print("\n" + "="*128)
    print("Penalty | Solver    | C    | Iter | Accuracy  | Acc_Std  | Precision | Recall    | F1        | ROC AUC   | Avg_Feats | Robustez")
    print("-"*128)
    for r in resultados[:10]:  # Mostra apenas os top 10
        print(
            f"{r['penalty']:<7} | {r['solver']:<9} | {r['C']:>4} | {r['max_iter']:>4} | "
            f"{r['accuracy']*100:>8.2f}% | {r['accuracy_std']*100:>7.2f}% | "
            f"{r['precision']*100:>8.2f}% | {r['recall']*100:>8.2f}% | "
            f"{r['f1']*100:>8.2f}% | {r['roc_auc']*100:>8.2f}% | "
            f"{r['avg_feats']:>9.2f} | {r['robustez']:>8.2f}%"
        )

    # Salva no JSON
    caminho_json = "hiperparam.json"
    parametros_salvos = {}
    
    if os.path.exists(caminho_json):
        with open(caminho_json, "r") as f:
            parametros_salvos = json.load(f)

    melhor = resultados[0]
    parametros_salvos[nome_dataset] = {
        "hiperparametros": {
            "penalty": melhor["penalty"],
            "solver": melhor["solver"],
            "C": melhor["C"],
            "max_iter": melhor["max_iter"]
        },
        "metricas": {
            "accuracy": melhor["accuracy"],
            "accuracy_std": melhor["accuracy_std"],
            "precision": melhor["precision"],
            "recall": melhor["recall"],
            "f1": melhor["f1"],
            "roc_auc": melhor["roc_auc"],
            "avg_feats": melhor["avg_feats"],
            "avg_feats_std": melhor["avg_feats_std"],
            "robustez": melhor["robustez"]
        }
    }

    with open(caminho_json, "w") as f:
        json.dump(parametros_salvos, f, indent=2)

    print(f"\nMelhores parametros salvos em {caminho_json}")

if __name__ == "__main__":
    main()