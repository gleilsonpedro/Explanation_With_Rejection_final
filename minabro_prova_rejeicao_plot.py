import numpy as np
import matplotlib.pyplot as plt

def plotar_prova_visual_rejeicao():
    # 1. Gerando o espaço amostral (Eixo X)
    x = np.linspace(-2, 2, 500)

    # 2. Simulando a fronteira da Caixa-Preta (MLP)
    # Uma equação altamente não-linear e ruidosa (senoidal com variação)
    y_mlp = 0.6 * np.sin(4 * x) - 0.2 * x**2 + 0.3 * np.cos(2 * x)

    # 3. Simulando a aproximação linear do MINABRO
    # Um hiperplano (reta) que tenta se ajustar localmente perto da origem
    y_minabro = 0.2 * x + 0.1 

    # 4. Definindo a "Zona de Perigo" (Os limiares t+ e t-)
    margem_rejeicao = 0.5
    y_t_plus = y_minabro + margem_rejeicao
    y_t_minus = y_minabro - margem_rejeicao

    # 5. A Instância Alvo (Caindo exatamente na indefinição)
    x_alvo = 0.1
    y_alvo = 0.12 # Bem próximo do hiperplano, no olho do furacão

    # --- PLOTAGEM ---
    fig, ax = plt.subplots(figsize=(9, 6))

    # Preenchendo a Zona de Rejeição primeiro para ficar no fundo
    ax.fill_between(x, y_t_minus, y_t_plus, color='#d3d3d3', alpha=0.5, 
                    label='Zona de Risco ($t^-$ $\leq f(\mathbf{x}) \leq$ $t^+$)')

    # Desenhando as margens t+ e t- (tracejadas)
    ax.plot(x, y_t_plus, color='gray', linestyle='--', linewidth=1.5)
    ax.plot(x, y_t_minus, color='gray', linestyle='--', linewidth=1.5)

    # Desenhando a Fronteira da MLP (Vermelha e Complexa)
    ax.plot(x, y_mlp, color='#d62728', linestyle='-', linewidth=2.5, 
            label='Fronteira Local da Caixa-Preta (MLP)')

    # Desenhando o Hiperplano do MINABRO (Azul e Reto)
    ax.plot(x, y_minabro, color='#1f77b4', linestyle='-', linewidth=2.5, 
            label='Hiperplano Substituto (MINABRO)')

    # Plotando a Instância Alvo (Verde)
    ax.scatter([x_alvo], [y_alvo], color='#2ca02c', s=150, zorder=5, 
               edgecolors='black', label='Instância Alvo $\mathbf{x}$')

    # Anotações textuais direto no gráfico para guiar a banca
    ax.annotate('$t^+$ (Limiar Superior)', xy=(1.2, 0.2 * 1.2 + 0.1 + margem_rejeicao + 0.05), 
                color='gray', fontsize=11, fontweight='bold')
    ax.annotate('$t^-$ (Limiar Inferior)', xy=(1.2, 0.2 * 1.2 + 0.1 - margem_rejeicao - 0.12), 
                color='gray', fontsize=11, fontweight='bold')
    
    # Seta apontando para a complexidade
    ax.annotate('Alta\nNão Linearidade', xy=(-0.8, -0.6), xytext=(-1.5, -1.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color='black')

    # Configurações visuais do gráfico
    ax.set_title('Acionamento da Opção de Rejeição por Instabilidade Geométrica', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Característica $x_1$', fontsize=12)
    ax.set_ylabel('Característica $x_2$', fontsize=12)
    
    # Configurando a legenda
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='black')
    
    # Grid e limites
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.5, 1.5)

    # Remove os números dos eixos para focar puramente no conceito geométrico
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plotar_prova_visual_rejeicao()