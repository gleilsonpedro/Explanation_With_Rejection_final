
'''Utiliza os datasets make_moons (2D) e make_classification 
(3D) para testar a "blindagem" do seu método. Ele verifica se 
a explicação local realmente cobre a área da MLP sem que a 
"curva" da rede invada o limite do plano local.'''

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Importa a sua arquitetura blindada
from MINABRO_MLP import (
    treinar_modelo_mlp, 
    MinabroMLPSurrogateExplainer, 
    MLP_PARAMS, 
    RANDOM_STATE
)

def executar_prova_de_conceito():
    print("="*75)
    print(" PROVA DE CONCEITO: FIDELIDADE EM ESPAÇOS 2D E 3D ")
    print("="*75)

    # Criamos os dois universos (2D com formato de luas e 3D com clusters)
    datasets_toy = [
        ("2D (Espaço Plano)", make_moons(n_samples=500, noise=0.15, random_state=RANDOM_STATE), ['Eixo_X', 'Eixo_Y']),
        ("3D (Espaço Volumétrico)", make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, random_state=RANDOM_STATE), ['Eixo_X', 'Eixo_Y', 'Eixo_Z'])
    ]

    for nome, (X_np, y_np), colunas in datasets_toy:
        print(f"\n>>> Analisando o cenário: {nome}")
        
        # O MinMaxScaler no Pipeline do MINABRO exige DataFrames para manter nomes
        df_X = pd.DataFrame(X_np, columns=colunas)
        y = pd.Series(y_np)

        X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=RANDOM_STATE)

        # 1. Treina o Oráculo (Caixa-Preta Curva)
        mlp = treinar_modelo_mlp(X_train, y_train, MLP_PARAMS)

        # 2. Prepara o Explicador (Plano Reto) - Usando L2 simples pois o espaço é limpo
        logreg_params = {'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear', 'max_iter': 500}
        
        # Custo de rejeição padrão (0.24), mas aqui o foco é a fidelidade das instâncias aceites
        explainer = MinabroMLPSurrogateExplainer(mlp, df_X, 0.24, logreg_params)

        # 3. Testar a fidelidade em 10 instâncias aleatórias
        amostra = X_test.sample(10, random_state=RANDOM_STATE)
        amostra_vals = amostra.values
        
        fidelidades = []
        tamanhos = []
        
        for i in range(len(amostra)):
            inst_vals = amostra_vals[i]
            original_pred = mlp.predict(pd.DataFrame([inst_vals], columns=colunas))[0]
            
            # Gera a explicação e afere a fidelidade de pior caso
            explicacao, pred_code, is_faithful = explainer.explain_instance(inst_vals)
            
            # Ignoramos casos rejeitados para focar na explicação da fronteira clássica
            if pred_code != 2:
                fidelidades.append(is_faithful)
                tamanhos.append(len(explicacao))
                
                # --- SAÍDA INTUITIVA E SÓBRIA PARA O TERMINAL ---
                status = "[SEGURO] 100% à prova de variações no pior cenário." if is_faithful else "[FALHA] A curva da MLP invadiu o limite do plano."
                
                print(f"\n[Instância {i+1}] {'-'*55}")
                print(f" Decisão Original : Classe {original_pred}")
                print(f" Explicação Local : {explicacao} (Tamanho: {len(explicacao)})")
                print(f" Fidelidade       : {status}")

        taxa_fidelidade = np.mean(fidelidades) * 100 if fidelidades else 0.0
        tamanho_medio = np.mean(tamanhos) if tamanhos else 0.0
        
        print("\n" + "="*50)
        print(f" RESUMO: {nome}")
        print(f" Fidelidade Abdutiva Média  : {taxa_fidelidade:.1f}%")
        print(f" Tamanho Médio da Explicação: {tamanho_medio:.1f} variáveis")
        print("="*50 + "\n")

if __name__ == '__main__':
    executar_prova_de_conceito()