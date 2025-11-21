"""
Visualizador: Top-50 pixels da classe POSITIVA

Este script calcula as 50 features/pixels mais frequentes nas explicações
mínimas das instâncias preditas como POSITIVAS (y_pred == 1) para um
experimento PEAB (ex.: mnist_3_vs_8) e gera:

- Um heatmap agregado mostrando a frequência (normalizada) dos top-50
- Opcionalmente, um overlay sobre uma instância específica mostrando
  somente os pixels dentre o top-50 que aparecem naquela explicação

Uso:
    python visualizer_top50_positive.py --experiment mnist_3_vs_8 [--instance-idx 58] [--show]

Saídas:
    analysis_output/plots/top50_positive_{experiment}.png
    analysis_output/plots/top50_positive_{experiment}_inst_{idx}.png  (se --instance-idx)

"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

RESULTS_FILE = 'json/comparative_results.json'
OUTPUT_DIR = 'analysis_output/plots'
TOP_K = 50


def carregar_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return json.loads(p.read_text(encoding='utf-8'))


def extrair_topk_positive(per_instance, num_features, k=TOP_K):
    """Conta frequência de pixels nas explicações das instâncias positivas."""
    counts = np.zeros(num_features, dtype=int)
    total_pos = 0
    for inst in per_instance:
        if inst.get('y_pred') == 1:
            explanation = inst.get('explanation', []) or []
            for feat in explanation:
                try:
                    if isinstance(feat, str) and feat.startswith('pixel'):
                        idx = int(feat.replace('pixel', '')) - 1
                    elif isinstance(feat, (int, np.integer)):
                        idx = int(feat) - 1
                    else:
                        idx = None
                except Exception:
                    idx = None
                if idx is not None and 0 <= idx < num_features:
                    counts[idx] += 1
            total_pos += 1
    # Selecionar top-k por contagem
    if np.sum(counts) == 0:
        return counts, np.array([], dtype=int), total_pos
    topk_idx = np.argsort(counts)[-k:][::-1]
    return counts, topk_idx, total_pos


def plot_aggregated_topk(counts, topk_idx, num_features, experiment, img_shape=(28,28), show=False):
    # construir mapa de calor com valores reais de frequência
    freq = np.zeros(num_features, dtype=float)
    
    # Copiar as contagens reais para os top-k pixels
    freq[topk_idx] = counts[topk_idx].astype(float)
    
    # Normalizar APENAS os valores não-zero para ter gradiente visível
    max_count = np.max(freq)
    if max_count > 0:
        # Normalizar para [0.2, 1.0] para que até o pixel menos frequente seja visível
        min_nonzero = np.min(freq[freq > 0])
        # Escala: valores baixos → 0.2, valores altos → 1.0
        for idx in topk_idx:
            if freq[idx] > 0:
                # Normalização linear preservando gradiente
                freq[idx] = 0.2 + 0.8 * (freq[idx] - min_nonzero) / (max_count - min_nonzero + 1e-8)
    
    heat = freq.reshape(img_shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Painel 1: Heatmap com colormap 'hot' (preto → vermelho → amarelo → branco)
    im1 = axes[0].imshow(heat, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title(f'Heatmap Top-{len(topk_idx)} pixels\n(colormap: hot)', fontweight='bold')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Frequência relativa', rotation=270, labelpad=15)
    
    # Painel 2: Heatmap com colormap 'viridis' (alternativa mais contrastante)
    im2 = axes[1].imshow(heat, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title(f'Heatmap Top-{len(topk_idx)} pixels\n(colormap: viridis)', fontweight='bold')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Frequência relativa', rotation=270, labelpad=15)
    
    fig.suptitle(f'Top-{len(topk_idx)} Pixels Mais Frequentes - Classe Positiva\n{experiment}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR) / f'top50_positive_{experiment}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  💾 Salvo aggregate: {out}")
    print(f"     Estatísticas dos top-{len(topk_idx)}:")
    print(f"       - Freq. máxima: {int(max_count)}")
    print(f"       - Freq. mínima (top-{len(topk_idx)}): {int(np.min(counts[topk_idx]))}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_instance_overlay(inst, inst_idx, X_test, topk_idx, class_names, experiment, img_shape=(28,28), show=False):
    num_features = img_shape[0]*img_shape[1]
    x_vals = None
    # reconstruir vetor da instância
    if isinstance(X_test, dict):
        pixel_keys = sorted(X_test.keys(), key=lambda x: int(x.replace('pixel','')))
        num_instances = len(X_test[pixel_keys[0]])
        if inst_idx >= num_instances:
            print(f"Índice {inst_idx} fora do range (num_instances={num_instances})")
            return
        x_vals = np.zeros(num_features)
        for i, k in enumerate(pixel_keys):
            x_vals[i] = X_test[k][inst_idx]
    else:
        X_arr = np.array(X_test)
        if inst_idx < X_arr.shape[0]:
            x_vals = X_arr[inst_idx]
        else:
            print(f"Índice {inst_idx} fora do range do X_test")
            return

    # normalizar se necessário
    if x_vals.max() > 1.0:
        x_vals = x_vals / 255.0

    img_original = x_vals.reshape(img_shape)

    # máscara apenas com topk presentes na explicação dessa instância
    explanation = inst.get('explanation', []) or []
    mask = np.zeros(num_features, dtype=float)
    for feat in explanation:
        try:
            if isinstance(feat, str) and feat.startswith('pixel'):
                idx = int(feat.replace('pixel','')) - 1
            elif isinstance(feat, (int, np.integer)):
                idx = int(feat) - 1
            else:
                idx = None
        except Exception:
            idx = None
        if idx is not None and 0 <= idx < num_features and idx in topk_idx:
            mask[idx] = img_original[idx]

    mask_img = mask.reshape(img_shape)

    # overlay: vermelho para os top50
    img_rgb = np.stack([img_original, img_original, img_original], axis=-1)
    alpha = 0.8
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if mask_img[i,j] > 0:
                pixel_value = img_original[i,j]
                img_rgb[i,j,0] = alpha*1.0 + (1-alpha)*pixel_value
                img_rgb[i,j,1] = (1-alpha)*pixel_value*0.3
                img_rgb[i,j,2] = (1-alpha)*pixel_value*0.3
    
    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(img_original, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Original (inst {inst_idx})')
    ax1.axis('off')

    ax2 = plt.subplot(1,2,2)
    ax2.imshow(img_rgb, interpolation='nearest')
    ax2.set_title(f'Top-{len(topk_idx)} overlay (inst {inst_idx})')
    ax2.axis('off')

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR) / f'top50_positive_{experiment}_inst_{inst_idx}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  💾 Salvo instance overlay: {out}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=RESULTS_FILE)
    parser.add_argument('--experiment', type=str, default='mnist')
    parser.add_argument('--instance-idx', type=int, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    data = carregar_json(args.results)
    if 'peab' not in data or args.experiment not in data['peab']:
        print('Experimento não encontrado no JSON')
        return

    exp = data['peab'][args.experiment]
    per_instance = exp.get('per_instance', [])
    X_test = exp['data']['X_test']
    class_names = exp['data'].get('class_names', ['0','1'])
    # determinar num_features a partir do modelo ou X_test
    if 'model' in exp and 'coefs' in exp['model']:
        num_features = len(exp['model']['coefs'])
    else:
        # inferir de X_test
        if isinstance(X_test, dict):
            num_features = len(X_test.keys())
        else:
            num_features = np.array(X_test).shape[1]

    counts, topk_idx, total_pos = extrair_topk_positive(per_instance, num_features, k=TOP_K)
    print(f"Total de instâncias positivas analisadas: {total_pos}")
    print(f"Top-{len(topk_idx)} índices: {topk_idx}")

    # plot agregado
    plot_aggregated_topk(counts, topk_idx, num_features, args.experiment, img_shape=(28,28), show=args.show)

    # se instancia pedida, localizar a instância na lista e plotar overlay apenas com topk
    if args.instance_idx is not None:
        idx = args.instance_idx
        if idx < 0 or idx >= len(per_instance):
            print(f"Índice de instância {idx} fora do range (0..{len(per_instance)-1})")
        else:
            inst = per_instance[idx]
            plot_instance_overlay(inst, idx, X_test, topk_idx, class_names, args.experiment, img_shape=(28,28), show=args.show)

if __name__ == '__main__':
    main()
