import matplotlib.pyplot as plt
import numpy as np

def plotar_barras_cenario2():
    # 1. Os dados exatos da sua Tabela 5
    datasets = ['Banknote', 'Breast\nCancer', 'Sonar', 'Spambase', 'Heart\nDisease']
    
    # Fidelidade Abdutiva
    fid_minabro = [100.0, 100.0, 84.0, 100.0, 78.0]
    fid_lime    = [100.0, 58.0, 70.0, 96.0, 62.0]
    fid_shap    = [58.0, 0.0, 0.0, 16.0, 0.0]
    
    # Estabilidade (Jaccard)
    jac_minabro = [100.0, 100.0, 100.0, 100.0, 100.0]
    jac_lime    = [100.0, 96.0, 96.1, 87.9, 88.5]
    jac_shap    = [90.8, 59.8, 89.4, 96.2, 66.4]

    # 2. Configurações de layout
    x = np.arange(len(datasets))  # Posição no eixo X
    width = 0.25  # Largura de cada barra
    
    # Cores acadêmicas elegantes
    cor_minabro = '#1f77b4' # Azul padrao matplotlib
    cor_lime = '#7f7f7f'    # Cinza neutro
    cor_shap = '#ff7f0e'    # Laranja
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ==========================================
    # PAINEL 1: Fidelidade Abdutiva
    # ==========================================
    ax1 = axes[0]
    rects1_m = ax1.bar(x - width, fid_minabro, width, label='MINABRO', color=cor_minabro, edgecolor='black')
    rects1_l = ax1.bar(x, fid_lime, width, label='LIME', color=cor_lime, edgecolor='black')
    rects1_s = ax1.bar(x + width, fid_shap, width, label='SHAP', color=cor_shap, edgecolor='black')
    
    ax1.set_ylabel('Fidelidade Abdutiva (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Desempenho no Teste Adversarial (Cenário 2)', fontsize=13, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=11)
    ax1.set_ylim(0, 115) # Dá espaço para a legenda não cobrir as barras
    
    # Adiciona os valores em cima das barras para facilitar a leitura
    ax1.bar_label(rects1_m, padding=3, fmt='%.0f', fontsize=9)
    ax1.bar_label(rects1_l, padding=3, fmt='%.0f', fontsize=9)
    ax1.bar_label(rects1_s, padding=3, fmt='%.0f', fontsize=9)
    
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # ==========================================
    # PAINEL 2: Estabilidade (Jaccard)
    # ==========================================
    ax2 = axes[1]
    rects2_m = ax2.bar(x - width, jac_minabro, width, label='MINABRO', color=cor_minabro, edgecolor='black')
    rects2_l = ax2.bar(x, jac_lime, width, label='LIME', color=cor_lime, edgecolor='black')
    rects2_s = ax2.bar(x + width, jac_shap, width, label='SHAP', color=cor_shap, edgecolor='black')
    
    ax2.set_ylabel('Índice de Jaccard (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Estabilidade de Execução (Cenário 2)', fontsize=13, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=11)
    ax2.set_ylim(0, 115)
    
    ax2.bar_label(rects2_m, padding=3, fmt='%.1f', fontsize=9)
    ax2.bar_label(rects2_l, padding=3, fmt='%.1f', fontsize=9)
    ax2.bar_label(rects2_s, padding=3, fmt='%.1f', fontsize=9)
    
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajuste final e plotagem
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plotar_barras_cenario2()