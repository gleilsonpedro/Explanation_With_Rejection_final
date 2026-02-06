"""
Script para identificar as melhores instâncias rejeitadas do MNIST
Foco: Rejeitadas com MENOR quantidade de features na explicação
"""
import json
import pandas as pd

# Carregar resultados do PEAB MNIST
with open('json/peab/mnist_3_vs_8.json', 'r') as f:
    data = json.load(f)

# Extrair instâncias rejeitadas COM ÍNDICE SEQUENCIAL
rejeitadas = []
for idx_sequencial, inst in enumerate(data['per_instance']):
    if inst['rejected']:
        rejeitadas.append({
            'idx_sequencial': idx_sequencial,  # 0-502 (índice na lista de teste)
            'id_original': inst['id'],  # ID original do dataset
            'y_true': inst['y_true'],
            'decision_score': inst['decision_score'],
            'explanation_size': inst['explanation_size'],
            'explanation': inst['explanation']
        })

# Ordenar por tamanho de explicação (menor primeiro)
rejeitadas_sorted = sorted(rejeitadas, key=lambda x: x['explanation_size'])

print("="*80)
print("MELHORES ÍNDICES PARA TESTE - MNIST REJEITADAS (MENOR EXPLICAÇÃO)")
print("="*80)
print(f"\nTotal de instâncias rejeitadas: {len(rejeitadas)}")
print(f"Média de features nas rejeitadas: {data['explanation_stats']['rejected']['mean_length']:.1f}")
print(f"Taxa de rejeição: {data['performance']['rejection_rate']:.2f}%")
print(f"\nThresholds: t- = {data['thresholds']['t_minus']:.4f}, t+ = {data['thresholds']['t_plus']:.4f}")

print("\n" + "-"*80)
print("TOP 20 REJEITADAS COM MENOR EXPLICAÇÃO")
print("-"*80)
print(f"{'Rank':<6} {'IDX':<8} {'ID_Orig':<10} {'Classe':<8} {'Score':<12} {'#Features':<12} {'% de 784':<10}")
print("-"*80)

# Salvar lista para uso fácil
melhores_idx = []
for i, inst in enumerate(rejeitadas_sorted[:20], 1):
    idx_seq = inst['idx_sequencial']  # ÍNDICE CORRETO (0-502)
    id_orig = inst['id_original']
    classe = inst['y_true']
    score = inst['decision_score']
    n_feat = inst['explanation_size']
    pct = (n_feat / 784) * 100
    
    print(f"{i:<6} {idx_seq:<8} {id_orig:<10} {classe:<8} {score:<12.4f} {n_feat:<12} {pct:<10.1f}%")
    melhores_idx.append(idx_seq)

print("\n" + "-"*80)
print("LISTA PYTHON PARA COPIAR:")
print("-"*80)
print(f"melhores_idx_rejeitadas = {melhores_idx[:10]}")
print(f"\n# Top 20 completo:")
print(f"top_20_rejeitadas = {melhores_idx}")

# Salvar em arquivo texto
with open('temporarios/lista_idx_rejeitadas_mnist.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("LISTA DE ÍNDICES - REJEITADAS MNIST (MENOR EXPLICAÇÃO)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total rejeitadas: {len(rejeitadas)}\n")
    f.write(f"Thresholds: t- = {data['thresholds']['t_minus']:.4f}, t+ = {data['thresholds']['t_plus']:.4f}\n\n")
    f.write("-"*80 + "\n")
    f.write("TOP 10 ÍNDICES RECOMENDADOS (copie e cole):\n")
    f.write("-"*80 + "\n")
    f.write(f"{melhores_idx[:10]}\n\n")
    f.write("-"*80 + "\n")
    f.write("TOP 20 COMPLETO:\n")
    f.write("-"*80 + "\n")
    for i, inst in enumerate(rejeitadas_sorted[:20], 1):
        f.write(f"{i}. IDX {inst['idx_sequencial']:<4} (ID_orig: {inst['id_original']}) | Classe {inst['y_true']} | Score {inst['decision_score']:8.4f} | {inst['explanation_size']:3} features ({inst['explanation_size']/784*100:.1f}%)\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("ESTATÍSTICAS DAS REJEITADAS:\n")
    f.write("-"*80 + "\n")
    tamanhos = [r['explanation_size'] for r in rejeitadas]
    f.write(f"Mínimo: {min(tamanhos)} features\n")
    f.write(f"Máximo: {max(tamanhos)} features\n")
    f.write(f"Média: {sum(tamanhos)/len(tamanhos):.1f} features\n")
    f.write(f"Mediana: {sorted(tamanhos)[len(tamanhos)//2]} features\n")

print("\n✓ Arquivo salvo em: temporarios/lista_idx_rejeitadas_mnist.txt")

# Mostrar distribuição
print("\n" + "-"*80)
print("DISTRIBUIÇÃO DE TAMANHOS DE EXPLICAÇÃO (REJEITADAS)")
print("-"*80)
tamanhos = [r['explanation_size'] for r in rejeitadas]
bins = [0, 200, 300, 400, 500, 600, 700, 784]
for i in range(len(bins)-1):
    count = sum(1 for t in tamanhos if bins[i] <= t < bins[i+1])
    print(f"{bins[i]:3d}-{bins[i+1]:3d} features: {count:3d} instâncias {'█'*count}")

print("\n" + "="*80)
