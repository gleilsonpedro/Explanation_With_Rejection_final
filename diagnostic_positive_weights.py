"""
Diagnóstico Completo: Pesos e Pixels da Classe Positiva

Gera uma análise visual completa para detectar problemas de mapeamento:
1. Imagem original de um dígito "8" puro do MNIST (classe positiva)
2. Pixels ativos (não-zeros) dessa instância
3. Top-10 pixels por PESO ABSOLUTO do modelo (favorece classe positiva)
4. Top-10 pixels por FREQUÊNCIA nas explicações positivas
5. Tabela com índices, pesos e valores dos top-10 pixels

Uso:
    python diagnostic_positive_weights.py --experiment mnist_3_vs_8 --instance-idx 58

Saída:
    analysis_output/plots/diagnostic_weights_{experiment}_inst_{idx}.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

RESULTS_FILE = 'json/comparative_results.json'
OUTPUT_DIR = 'analysis_output/plots/individual_examples'
IDX_POSITIVA = 58  # MESMO DO VISUALIZER COPY 2


def carregar_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return json.loads(p.read_text(encoding='utf-8'))


def get_instance_vector(X_test, inst_idx, num_features):
    """Extrai vetor de uma instância."""
    if isinstance(X_test, dict):
        pixel_keys = sorted(X_test.keys(), key=lambda x: int(x.replace('pixel', '')))
        num_instances = len(X_test[pixel_keys[0]])
        if inst_idx >= num_instances:
            return None
        x_vals = np.zeros(num_features)
        for i, k in enumerate(pixel_keys):
            x_vals[i] = X_test[k][inst_idx]
        return x_vals
    else:
        X_arr = np.array(X_test)
        if inst_idx < X_arr.shape[0]:
            return X_arr[inst_idx]
        return None


def extract_top_freq_pixels(per_instance, num_features, k=10):
    """Conta frequência de pixels nas explicações positivas."""
    counts = np.zeros(num_features, dtype=int)
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
    topk_idx = np.argsort(counts)[-k:][::-1]
    return topk_idx, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=RESULTS_FILE)
    parser.add_argument('--experiment', type=str, default='mnist')
    parser.add_argument('--instance-idx', type=int, default=IDX_POSITIVA)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    data = carregar_json(args.results)
    if 'peab' not in data or args.experiment not in data['peab']:
        print(f"❌ Experimento '{args.experiment}' não encontrado no JSON")
        return

    exp = data['peab'][args.experiment]
    per_instance = exp.get('per_instance', [])
    X_test = exp['data']['X_test']
    class_names = exp['data'].get('class_names', ['0', '1'])
    
    # Obter coeficientes do modelo
    if 'model' not in exp or 'coefs' not in exp['model']:
        print("❌ Modelo não encontrado no JSON")
        return
    
    coefs = np.array(exp['model']['coefs'])
    num_features = len(coefs)
    img_shape = (28, 28)
    
    print(f"\n{'='*80}")
    print(f"DIAGNÓSTICO DE PESOS - {args.experiment}")
    print(f"{'='*80}")
    print(f"Instância: {args.instance_idx}")
    print(f"Classes: {class_names[0]} (negativa) vs {class_names[1]} (positiva)")
    print(f"Total de features: {num_features}")
    
    # Verificar se instância existe e é positiva
    if args.instance_idx >= len(per_instance):
        print(f"❌ Índice {args.instance_idx} fora do range (0..{len(per_instance)-1})")
        return
    
    inst = per_instance[args.instance_idx]
    y_pred = inst.get('y_pred', -1)
    y_true = inst.get('y_true', -1)
    
    print(f"\n📊 Informações da Instância:")
    print(f"  y_true: {y_true} ({class_names[y_true]})")
    print(f"  y_pred: {y_pred} ({class_names[y_pred] if y_pred in [0,1] else 'REJEITADA'})")
    print(f"  rejected: {inst.get('rejected', False)}")
    
    if y_pred != 1:
        print(f"\n⚠️ AVISO: Instância não é positiva (y_pred={y_pred})")
        print("   Continuando análise mesmo assim...")
    
    # Obter vetor da instância
    x_vals = get_instance_vector(X_test, args.instance_idx, num_features)
    if x_vals is None:
        print(f"❌ Erro ao extrair vetor da instância {args.instance_idx}")
        return
    
    # Normalizar se necessário
    if x_vals.max() > 1.0:
        x_vals = x_vals / 255.0
    
    img_original = x_vals.reshape(img_shape)
    
    # 1. Top-10 pixels por PESO ABSOLUTO do modelo
    abs_weights = np.abs(coefs)
    top10_weight_idx = np.argsort(abs_weights)[-10:][::-1]
    
    print(f"\n🔬 Top-10 Pixels por PESO ABSOLUTO (favorecem classe positiva):")
    print(f"{'Rank':<6}{'Índice':<10}{'Peso':<15}{'Valor Pixel':<15}{'(i,j)'}")
    print("-" * 60)
    for rank, idx in enumerate(top10_weight_idx, 1):
        i, j = idx // 28, idx % 28
        print(f"{rank:<6}{idx:<10}{coefs[idx]:+.6f}     {x_vals[idx]:.4f}          ({i},{j})")
    
    # 2. Top-10 pixels por FREQUÊNCIA nas explicações
    top10_freq_idx, counts = extract_top_freq_pixels(per_instance, num_features, k=10)
    
    print(f"\n📊 Top-10 Pixels por FREQUÊNCIA nas explicações positivas:")
    print(f"{'Rank':<6}{'Índice':<10}{'Freq':<10}{'Peso':<15}{'Valor Pixel':<15}{'(i,j)'}")
    print("-" * 70)
    for rank, idx in enumerate(top10_freq_idx, 1):
        i, j = idx // 28, idx % 28
        print(f"{rank:<6}{idx:<10}{counts[idx]:<10}{coefs[idx]:+.6f}     {x_vals[idx]:.4f}          ({i},{j})")
    
    # 3. Criar visualização comparativa
    fig = plt.figure(figsize=(20, 5))
    
    # Painel 1: Imagem original completa
    ax1 = plt.subplot(1, 5, 1)
    ax1.imshow(img_original, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'1. Original Completo\n(Inst {args.instance_idx})', fontweight='bold')
    ax1.axis('off')
    
    # Painel 2: Apenas pixels ativos (não-zeros)
    img_ativos = np.copy(img_original)
    img_ativos[img_ativos < 0.05] = 0  # Threshold para considerar "ativo"
    ax2 = plt.subplot(1, 5, 2)
    ax2.imshow(img_ativos, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('2. Pixels Ativos\n(valor > 0.05)', fontweight='bold', color='green')
    ax2.axis('off')
    
    # Painel 3: Top-10 por PESO (deve formar padrão do 8)
    mask_weight = np.zeros(num_features)
    mask_weight[top10_weight_idx] = x_vals[top10_weight_idx]
    img_weight = mask_weight.reshape(img_shape)
    ax3 = plt.subplot(1, 5, 3)
    ax3.imshow(img_weight, cmap='hot', vmin=0, vmax=1)
    ax3.set_title('3. Top-10 por PESO\n(deveria formar "8")', fontweight='bold', color='blue')
    ax3.axis('off')
    
    # Painel 4: Top-10 por FREQUÊNCIA nas explicações
    mask_freq = np.zeros(num_features)
    mask_freq[top10_freq_idx] = x_vals[top10_freq_idx]
    img_freq = mask_freq.reshape(img_shape)
    ax4 = plt.subplot(1, 5, 4)
    ax4.imshow(img_freq, cmap='hot', vmin=0, vmax=1)
    ax4.set_title('4. Top-10 por FREQUÊNCIA\n(das explicações)', fontweight='bold', color='orange')
    ax4.axis('off')
    
    # Painel 5: Mapa de PESOS do modelo
    weight_map = coefs.reshape(img_shape)
    ax5 = plt.subplot(1, 5, 5)
    im5 = ax5.imshow(weight_map, cmap='RdBu_r', vmin=-np.max(np.abs(coefs)), vmax=np.max(np.abs(coefs)))
    ax5.set_title('5. Pesos do Modelo\n(vermelho=positivo)', fontweight='bold', color='red')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    fig.suptitle(
        f'Diagnóstico Completo: Pesos e Pixels - {args.experiment}\n'
        f'Instância {args.instance_idx} | y_true={y_true} ({class_names[y_true]}) | '
        f'y_pred={y_pred} ({class_names[y_pred] if y_pred in [0,1] else "REJEITADA"})',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Salvar
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR) / f'diagnostic_weights_{args.experiment}_inst_{args.instance_idx}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n💾 Diagnóstico salvo: {out}")
    
    # Análise comparativa
    print(f"\n🔍 ANÁLISE COMPARATIVA:")
    print(f"  ✓ Painel 2 (ativos) deveria ser similar ao original")
    print(f"  ✓ Painel 3 (top peso) deveria formar o padrão do dígito '{class_names[1]}'")
    print(f"  ✓ Painel 4 (top freq) deveria ser similar ao Painel 3")
    print(f"  ✓ Painel 5 mostra onde o modelo 'olha' para decidir")
    print(f"\n  ⚠️ Se Painel 3 NÃO formar '{class_names[1]}', há problema de indexação!")
    print(f"  ⚠️ Se Painel 2 e 3 forem diferentes, explicação está pegando pixels errados!")
    
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()
