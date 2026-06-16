import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from MINABRO_MLP import treinar_modelo_mlp, MinabroMLPSurrogateExplainer, MLP_PARAMS

def plotar_painel_multiplo_mnist():
    print(">>> Carregando o dataset MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=True, parser='auto')
    y = y.astype(int)
    
    X = X.iloc[:10000]
    y = y.iloc[:10000]
    
    print(">>> Dividindo dados e treinando a MLP...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_mlp = treinar_modelo_mlp(X_train, y_train, MLP_PARAMS)
    
    explainer_minabro = MinabroMLPSurrogateExplainer(modelo_mlp, X_train, 0.15, {})
    
    # =================================================================
    # CONTROLE DO PAINEL: Coloque aqui os índices das imagens que deseja plotar.
    # =================================================================
    INDICES_IMAGENS = [0, 32, 11]  
    
    # FIGURA COMPACTA: Tamanho ajustado para caber perfeitamente no PDF do LaTeX
    fig, axes = plt.subplots(len(INDICES_IMAGENS), 3, figsize=(9, 7))
    
    for i, idx in enumerate(INDICES_IMAGENS):
        inst_vals = X_test.iloc[idx].values
        predicao_original = modelo_mlp.predict([inst_vals])[0]
        
        print(f"\n>>> Analisando o Índice {idx} (A MLP diz que é: {predicao_original})...")
        
        exp_minabro, pred_code, is_faithful, bounds = explainer_minabro.explain_instance(inst_vals)
        
        ax_orig = axes[i, 0]
        ax_iso = axes[i, 1]
        ax_col = axes[i, 2]
        
        if pred_code == 2 or len(exp_minabro) == 0:
            print(f">>> [AVISO] Rejeição no índice {idx}. Troque o número.")
            continue

        print(f">>> Sucesso! Encontrados {len(exp_minabro)} pixels para a predição {predicao_original}.")

        img_original = inst_vals.reshape(28, 28).astype(float)
        if img_original.max() > 0:
            img_original = img_original / img_original.max()
        
        img_isolada = np.zeros((28, 28))
        img_colorida = np.stack((img_original, img_original, img_original), axis=-1)
        img_colorida *= 0.3  
        
        pixels_plotados = 0
        for feature in exp_minabro:
            idx_pixel = int(feature.replace('pixel', '')) - 1
            linha = idx_pixel // 28
            coluna = idx_pixel % 28
            
            img_isolada[linha, coluna] = img_original[linha, coluna]
            
            if img_original[linha, coluna] > 0.1:
                img_colorida[linha, coluna] = [1.0, 0.5, 0.0] # Laranja (Traço)
            else:
                img_colorida[linha, coluna] = [0.0, 0.5, 1.0] # Azul (Fundo Restritivo)
                
            pixels_plotados += 1

        # Plotagem com fontes reduzidas e minimalistas
        ax_orig.imshow(img_original, cmap='gray', vmin=0, vmax=1)
        ax_orig.set_title(f"Instância Alvo (Índ. {idx})\nPredição: {predicao_original}", fontsize=10)
        ax_orig.axis('off')
        
        ax_iso.imshow(img_isolada, cmap='gray', vmin=0, vmax=1)
        ax_iso.set_title(f"Subconjunto MINABRO\n({pixels_plotados} pixels)", fontsize=10)
        ax_iso.axis('off')
        
        if i == 0:
            ax_col.set_title(f"Localização da Explicação\n(Laranja: Traço | Azul: Fundo)", fontsize=10)
        else:
            ax_col.set_title(f"Localização da Explicação", fontsize=10)
            
        ax_col.imshow(img_colorida)
        ax_col.axis('off')

    # Ajuste de layout rigoroso para não sobrepor nada e grudar as imagens
    plt.tight_layout(pad=1.0, w_pad=0.1, h_pad=1.5)
    plt.show()

if __name__ == '__main__':
    plotar_painel_multiplo_mnist()