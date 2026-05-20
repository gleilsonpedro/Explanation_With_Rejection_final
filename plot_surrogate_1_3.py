import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def gerar_painel_populado():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor='white')
    fig.suptitle("Estratégias de Treinamento Local na Prática", fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Simulando a Fronteira de Decisão da MLP (Background)
    xx, yy = np.meshgrid(np.linspace(-1, 7, 100), np.linspace(-1, 7, 100))
    Z = xx + yy > 6  # Fronteira diagonal simples
    
    # Coordenadas do Paciente e do Inimigo
    alvo = np.array([1.5, 1.5])
    inimigo = np.array([4.5, 4.5])

    # 2. Gerando os Clones do Tubo (Seu Método Original com Ruído)
    np.random.seed(42)
    num_clones_tubo = 400
    alphas = np.random.uniform(0, 1, num_clones_tubo)[:, np.newaxis]
    linha_perfeita = alvo + alphas * (inimigo - alvo)
    
    # Ruído Gaussiano "Gordo" para espalhar os pontos
    ruido = np.random.normal(0, 0.35, size=linha_perfeita.shape)
    clones_tubo = linha_perfeita + ruido

    # =========================================================================
    # PAINEL A: Interpolação Populada (Tubo de Ruído)
    # =========================================================================
    ax_a = axes[0]
    ax_a.contourf(xx, yy, Z, alpha=0.15, cmap='coolwarm')
    ax_a.plot([alvo[0], inimigo[0]], [alvo[1], inimigo[1]], 'k--', lw=2, alpha=0.7, label='Eixo Direcional')
    
    # Plotando os clones com ruído
    ax_a.scatter(clones_tubo[:, 0], clones_tubo[:, 1], color='#ffc107', edgecolor='black', s=45, alpha=0.7)
    
    # Alvo e Inimigo
    ax_a.scatter(alvo[0], alvo[1], color='lime', s=350, marker='*', edgecolor='black', zorder=4)
    ax_a.scatter(inimigo[0], inimigo[1], color='magenta', s=200, marker='X', edgecolor='black', zorder=4)

    ax_a.set_title("A) Tubo de Interpolação (Com Ruído Gaussiano)", fontsize=14, fontweight='bold', pad=15)
    ax_a.set_xlim(-0.5, 6.5)
    ax_a.set_ylim(-0.5, 6.5)
    ax_a.set_xlabel("Feature 1", fontsize=12, fontweight='bold')
    ax_a.set_ylabel("Feature 2", fontsize=12, fontweight='bold')
    ax_a.grid(True, linestyle=':', alpha=0.6)

    # =========================================================================
    # PAINEL B: Interpolação + Piores Casos (O Método Completo)
    # =========================================================================
    ax_b = axes[1]
    ax_b.contourf(xx, yy, Z, alpha=0.15, cmap='coolwarm')
    ax_b.plot([alvo[0], inimigo[0]], [alvo[1], inimigo[1]], 'k--', lw=2, alpha=0.7)
    
    # Plotando os clones do tubo um pouco mais transparentes para destacar os extremos
    ax_b.scatter(clones_tubo[:, 0], clones_tubo[:, 1], color='#ffc107', edgecolor='black', s=45, alpha=0.4)

    # Calculando os limites da vizinhança (Bounding Box)
    b_min = clones_tubo.min(axis=0)
    b_max = clones_tubo.max(axis=0)

    # Desenhando o Retângulo da Vizinhança
    rect = patches.Rectangle((b_min[0], b_min[1]), b_max[0]-b_min[0], b_max[1]-b_min[1], 
                             fill=False, edgecolor='red', linestyle='--', lw=2, alpha=0.8)
    ax_b.add_patch(rect)

    # Gerando os Clones Adversariais (Piores Casos) nas bordas do retângulo
    clones_adv = []
    for _ in range(80): # 80 pontos vermelhos espalhados pelas quinas e bordas
        c = np.zeros(2)
        fix_idx = np.random.choice([0, 1])
        free_idx = 1 - fix_idx
        c[fix_idx] = alvo[fix_idx] # Fixa uma feature no valor do alvo
        # A feature livre vai pro máximo ou mínimo (borda do retângulo)
        c[free_idx] = b_max[free_idx] if np.random.rand() > 0.5 else b_min[free_idx]
        clones_adv.append(c)
    
    clones_adv = np.array(clones_adv)
    
    # Plotando os piores casos
    ax_b.scatter(clones_adv[:, 0], clones_adv[:, 1], color='red', marker='x', s=60, lw=2, zorder=5)

    # Alvo e Inimigo
    ax_b.scatter(alvo[0], alvo[1], color='lime', s=350, marker='*', edgecolor='black', zorder=6)
    ax_b.scatter(inimigo[0], inimigo[1], color='magenta', s=200, marker='X', edgecolor='black', zorder=6)

    ax_b.set_title("B) Amostragem Adversarial (Nas Bordas da Vizinhança)", fontsize=14, fontweight='bold', pad=15)
    ax_b.set_xlim(-0.5, 6.5)
    ax_b.set_ylim(-0.5, 6.5)
    ax_b.set_xlabel("Feature 1", fontsize=12, fontweight='bold')
    ax_b.set_ylabel("Feature 2", fontsize=12, fontweight='bold')
    ax_b.grid(True, linestyle=':', alpha=0.6)

    # =========================================================================
    # LEGENDA ÚNICA
    # =========================================================================
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Paciente Alvo', markerfacecolor='lime', markersize=18, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Inimigo Equidistante', markerfacecolor='magenta', markersize=14, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Clones c/ Ruído (Tubo)', markerfacecolor='#ffc107', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], marker='x', color='w', label='Piores Casos (Adversariais)', markerfacecolor='red', markeredgecolor='red', markersize=12, lw=2),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Limites da Vizinhança (Caixa)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig('amostragem_populada.png', dpi=300, bbox_inches='tight')
    print("[SUCESSO] Imagem 'amostragem_populada.png' gerada e pronta!")
    plt.show()

if __name__ == "__main__":
    gerar_painel_populado()