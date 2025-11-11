"""
Visualizador de Explicações Abdutivas do PEAB para MNIST

Este script analisa o arquivo comparative_results.json e gera visualizações
autoexplicativas mostrando como o método PEAB explica suas classificações.

Para cada experimento MNIST encontrado, gera 3 imagens que mostram:
- POSITIVA (AZUL): Pixels importantes para classificar como classe positiva
- NEGATIVA (VERMELHO): Pixels importantes para classificar como classe negativa
- REJEITADAS (MISTO): Pixels com evidências conflitantes

Autor: Sistema especialista em visualização de explicações abdutivas
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# CONSTANTES
# ==============================================================================
RESULTS_FILE = 'json/comparative_results.json'
OUTPUT_DIR = 'analysis_output/plots'
SAVE_PLOTS = True

# ==============================================================================
# FUNÇÕES
# ==============================================================================

def carregar_json(filepath: str) -> dict:
    """Carrega o arquivo JSON de resultados"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def encontrar_experimentos_mnist(data: dict) -> list:
    """Encontra todos os experimentos MNIST no JSON"""
    if 'peab' not in data:
        return []
    
    mnist_exp = [k for k in data['peab'].keys() if 'mnist' in k.lower()]
    return sorted(mnist_exp)


def calcular_deltas_da_explicacao(per_instance: list, model_coefs: list, 
                                   X_test, num_features: int) -> tuple:
    """
    Calcula os deltas (contribuições) a partir das explicações e do modelo.
    
    X_test pode vir em dois formatos:
    1. Lista de listas: [[feat1, feat2, ...], [feat1, feat2, ...], ...]
    2. Dict de listas: {'pixel1': [inst1, inst2, ...], 'pixel2': [...], ...}
    
    Returns:
        (heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej)
    """
    heatmap_pos = np.zeros(num_features, dtype=float)
    heatmap_neg = np.zeros(num_features, dtype=float)
    heatmap_rej = np.zeros(num_features, dtype=float)
    
    count_pos = 0
    count_neg = 0
    count_rej = 0
    
    # Converter coefs para numpy array
    coefs_arr = np.array(model_coefs)
    
    # Converter X_test para formato [instâncias x features]
    if isinstance(X_test, dict):
        # X_test é dict: {'pixel1': [inst1, inst2, ...], 'pixel2': [...], ...}
        # Precisamos transpor para [[inst1_feat1, inst1_feat2, ...], [inst2_feat1, ...], ...]
        
        # Ordenar chaves por número do pixel
        pixel_keys = sorted(X_test.keys(), key=lambda x: int(x.replace('pixel', '')))
        
        # Criar matriz transposta
        num_instances = len(X_test[pixel_keys[0]])
        X_test_arr = np.zeros((num_instances, num_features))
        
        for feat_idx, pixel_key in enumerate(pixel_keys):
            X_test_arr[:, feat_idx] = X_test[pixel_key]
    else:
        # X_test já é lista de listas
        X_test_arr = np.array(X_test)
    
    print(f"  📐 Dimensões: X_test={X_test_arr.shape}, coefs={coefs_arr.shape}")
    
    for idx, inst in enumerate(per_instance):
        # Pegar o ID da instância
        inst_id = int(inst['id'])
        
        # Usar o ID como índice (assumindo IDs sequenciais de 0)
        if inst_id < len(X_test_arr):
            x_vals = X_test_arr[inst_id]
        else:
            x_vals = np.zeros(num_features)
        
        # Calcular deltas: delta_i = x_i * w_i
        deltas = x_vals * coefs_arr
        
        # Agregar por classe
        if inst['rejected']:
            heatmap_rej += deltas
            count_rej += 1
        elif inst['y_pred'] == 0:
            heatmap_neg += deltas
            count_neg += 1
        elif inst['y_pred'] == 1:
            heatmap_pos += deltas
            count_pos += 1
    
    # Normalizar (calcular médias)
    if count_pos > 0:
        heatmap_pos /= count_pos
    if count_neg > 0:
        heatmap_neg /= count_neg
    if count_rej > 0:
        heatmap_rej /= count_rej
    
    return heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej


def agregar_deltas_por_classe(per_instance: list, num_features: int = 784) -> tuple:
    """
    Agrega os deltas (contribuições) por categoria de predição.
    
    Returns:
        (heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej)
    """
    heatmap_pos = np.zeros(num_features, dtype=float)
    heatmap_neg = np.zeros(num_features, dtype=float)
    heatmap_rej = np.zeros(num_features, dtype=float)
    
    count_pos = 0
    count_neg = 0
    count_rej = 0
    
    for inst in per_instance:
        deltas = np.array(inst['deltas'], dtype=float)
        
        if inst['rejected']:
            heatmap_rej += deltas
            count_rej += 1
        elif inst['y_pred'] == 0:
            heatmap_neg += deltas
            count_neg += 1
        elif inst['y_pred'] == 1:
            heatmap_pos += deltas
            count_pos += 1
    
    # Normalizar (calcular médias)
    if count_pos > 0:
        heatmap_pos /= count_pos
    if count_neg > 0:
        heatmap_neg /= count_neg
    if count_rej > 0:
        heatmap_rej /= count_rej
    
    return heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej


def criar_visualizacao(heatmap_pos, heatmap_neg, heatmap_rej,
                       class_names: list, counts: tuple,
                       experiment_name: str, img_shape=(28, 28)):
    """
    Cria visualização com 3 imagens lado a lado mostrando as explicações.
    
    INTERPRETAÇÃO:
    - AZUL (+): Evidência PARA a classe positiva (ex: "É um 7")
    - VERMELHO (-): Evidência CONTRA a classe positiva (ex: "NÃO é um 7")
    - BRANCO (0): Neutro, não contribui
    """
    count_pos, count_neg, count_rej = counts
    
    # Criar figura 1x3
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # Calcular escala global para cores uniformes
    all_values = []
    if count_neg > 0:
        all_values.extend(heatmap_neg.flatten())
    if count_pos > 0:
        all_values.extend(heatmap_pos.flatten())
    if count_rej > 0:
        all_values.extend(heatmap_rej.flatten())
    
    if len(all_values) > 0:
        vmax = np.max(np.abs(all_values))
    else:
        vmax = 1.0
    
    if vmax == 0:
        vmax = 1.0
    
    # Colormap: Vermelho (negativo) → Branco (zero) → Azul (positivo)
    cmap = 'RdBu_r'
    
    # ========================================
    # PLOT 1: CLASSE NEGATIVA (Classe 0)
    # ========================================
    if count_neg > 0:
        img_neg = heatmap_neg.reshape(img_shape)
    else:
        img_neg = np.zeros(img_shape)
    
    im0 = axes[0].imshow(img_neg, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[0].set_title(
        f'CLASSE NEGATIVA: {class_names[0]}\n'
        f'({count_neg} instâncias)\n\n'
        f'Vermelho = Evidência de que é {class_names[0]}',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[0].axis('off')
    
    # ========================================
    # PLOT 2: CLASSE POSITIVA (Classe 1)
    # ========================================
    if count_pos > 0:
        img_pos = heatmap_pos.reshape(img_shape)
    else:
        img_pos = np.zeros(img_shape)
    
    im1 = axes[1].imshow(img_pos, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[1].set_title(
        f'CLASSE POSITIVA: {class_names[1]}\n'
        f'({count_pos} instâncias)\n\n'
        f'Azul = Evidência de que é {class_names[1]}',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[1].axis('off')
    
    # ========================================
    # PLOT 3: REJEITADAS
    # ========================================
    if count_rej > 0:
        img_rej = heatmap_rej.reshape(img_shape)
    else:
        img_rej = np.zeros(img_shape)
    
    im2 = axes[2].imshow(img_rej, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[2].set_title(
        f'REJEITADAS\n'
        f'({count_rej} instâncias)\n\n'
        f'Cores mistas = Evidências conflitantes',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[2].axis('off')
    
    # ========================================
    # COLORBAR (Legenda)
    # ========================================
    fig.subplots_adjust(right=0.87, wspace=0.3)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label(
        'Contribuição (δ)\n\n'
        f'AZUL (+):\nEvidência para\n{class_names[1]}\n\n'
        f'VERMELHO (-):\nEvidência para\n{class_names[0]}\n\n'
        'BRANCO (0):\nNeutro',
        fontsize=10,
        rotation=0,
        labelpad=15,
        ha='left'
    )
    
    # Título principal
    fig.suptitle(
        f'Explicações Abdutivas do PEAB - {experiment_name}\n'
        f'Mapa de Calor de Evidências Médias por Classe',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.87, 0.95])
    
    # Salvar se configurado
    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f'explicacao_peab_{experiment_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Imagem salva: {filename}")
    
    plt.show()


def processar_experimento(data: dict, exp_key: str):
    """Processa um experimento e gera a visualização"""
    print(f"\n{'='*80}")
    print(f"Processando: {exp_key}")
    print(f"{'='*80}")
    
    try:
        exp_data = data['peab'][exp_key]
        
        # Verificar se tem estrutura necessária
        if 'per_instance' not in exp_data:
            print("\n❌ ERRO: Chave 'per_instance' não encontrada!")
            return
        
        if 'data' not in exp_data or 'model' not in exp_data:
            print("\n❌ ERRO: Estrutura incompleta no JSON!")
            return
        
        # Obter informações
        class_names = exp_data['data']['class_names']
        per_instance = exp_data['per_instance']
        X_test = exp_data['data']['X_test']
        model_coefs = exp_data['model']['coefs']
        
        if len(per_instance) == 0:
            print("⚠ Nenhuma instância encontrada. Pulando...")
            return
        
        # Determinar número de features
        num_features = len(model_coefs)
        
        print(f"\n✓ Classes: {class_names[0]} vs {class_names[1]}")
        print(f"✓ Total de instâncias: {len(per_instance)}")
        print(f"✓ Features: {num_features}")
        
        # Determinar shape da imagem
        if num_features == 784:
            img_shape = (28, 28)
            print("✓ Formato: 28x28 (MNIST raw)")
        elif num_features == 196:
            img_shape = (14, 14)
            print("✓ Formato: 14x14 (MNIST pooling)")
        else:
            lado = int(np.sqrt(num_features))
            if lado * lado == num_features:
                img_shape = (lado, lado)
            else:
                img_shape = (28, 28)  # fallback
            print(f"✓ Formato: {img_shape}")
        
        # Agregar evidências (agora calculando deltas)
        print("\n📊 Calculando e agregando evidências por classe...")
        heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej = \
            calcular_deltas_da_explicacao(per_instance, model_coefs, X_test, num_features)
        
        print(f"  • Negativas ({class_names[0]}): {count_neg} instâncias")
        print(f"  • Positivas ({class_names[1]}): {count_pos} instâncias")
        print(f"  • Rejeitadas: {count_rej} instâncias")
        
        # Gerar visualização
        print("\n🎨 Gerando visualização...")
        criar_visualizacao(
            heatmap_pos, heatmap_neg, heatmap_rej,
            class_names,
            (count_pos, count_neg, count_rej),
            exp_key,
            img_shape
        )
        
        print(f"\n✅ Visualização concluída para {exp_key}!")
        
    except KeyError as e:
        print(f"\n❌ ERRO: Chave não encontrada no JSON: {e}")
        print("\n💡 Dica: Verifique se o experimento foi executado corretamente")
        print("         e se o JSON foi gerado com todas as informações necessárias.")
    except Exception as e:
        print(f"\n❌ ERRO ao processar {exp_key}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*80)
    print("VISUALIZADOR DE EXPLICAÇÕES ABDUTIVAS DO PEAB")
    print("="*80)
    print(f"\n📂 Arquivo: {RESULTS_FILE}")
    
    try:
        # Carregar JSON
        data = carregar_json(RESULTS_FILE)
        print("✓ JSON carregado com sucesso")
        
        # Encontrar experimentos MNIST
        mnist_exp = encontrar_experimentos_mnist(data)
        
        if len(mnist_exp) == 0:
            print("\n❌ Nenhum experimento MNIST encontrado no JSON!")
            print(f"Experimentos disponíveis: {list(data.get('peab', {}).keys())}")
            return
        
        print(f"\n✓ Encontrado(s) {len(mnist_exp)} experimento(s) MNIST:")
        for exp in mnist_exp:
            print(f"  • {exp}")
        
        # Processar cada experimento automaticamente
        for exp_key in mnist_exp:
            processar_experimento(data, exp_key)
        
        print(f"\n{'='*80}")
        print(f"✅ TODOS OS EXPERIMENTOS PROCESSADOS COM SUCESSO!")
        print(f"{'='*80}")
        
        if SAVE_PLOTS:
            print(f"\n💾 Imagens salvas em: {OUTPUT_DIR}/")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
    except KeyError as e:
        print(f"\n❌ ERRO: Chave não encontrada no JSON: {e}")
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
