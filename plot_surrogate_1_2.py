import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def gerar_fluxograma():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')
    
    passos = [
        "[Instância X]\n(Paciente Alvo)",
        "[Busca pelo Vizinho Oposto]\n(Distância Euclidiana)",
        "[Projeção do Inimigo Equidistante]\n(Fronteira Exata + Simetria)",
        "[Geração de 1.000 Clones]\n(Interpolação + Ruído Gaussiano de 15%)",
        "[Rotulação pelo Oráculo]\n(MLP classifica os clones)",
        "[Treinamento da Regressão Logística]\n(Ajuste do Modelo Substituto)",
        "[Cálculo dos Limiares t+ e t-]\n(Otimização de Risco local)",
        "[Extração via Algoritmo Guloso]\n(Busca pelo Tamanho Mínimo)",
        "[Validação Abdutiva]\n(Teste no Pior Cenário Possível)",
        "[Explicação Final] ou [Rejeição]\n(Saída do Sistema)"
    ]

    num_passos = len(passos)
    y_pos = np.linspace(0.95, 0.05, num_passos)
    
    for i, passo in enumerate(passos):
        # Box de texto com design limpo e profissional
        bbox_props = dict(boxstyle="round,pad=0.6", fc="#f8f9fa", ec="#343a40", lw=2)
        ax.text(0.5, y_pos[i], passo, ha="center", va="center", size=12, fontweight='bold', bbox=bbox_props, color="#212529")
        
        # Setas de conexão
        if i < num_passos - 1:
            ax.annotate('', xy=(0.5, y_pos[i+1] + 0.04), xytext=(0.5, y_pos[i] - 0.04),
                        arrowprops=dict(arrowstyle="->", color="#343a40", lw=2.5))

    plt.title("Fluxograma do Ciclo de Explicação Local (MINABRO)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('fluxograma_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def gerar_paineis_amostragem():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Dados base simulados
    alvo = np.array([1.0, 1.0])
    inimigo = np.array([4.0, 4.0])
    alphas = np.linspace(0, 1, 150)[:, np.newaxis]
    linha_perfeita = alvo + alphas * (inimigo - alvo)

    # Background de fronteira simples para ilustrar
    x_bg = np.linspace(0, 5, 100)
    y_bg = np.linspace(0, 5, 100)
    X_bg, Y_bg = np.meshgrid(x_bg, y_bg)
    Z_bg = X_bg + Y_bg > 5

    # --- PAINEL 1: Linha Reta (Interpolação) ---
    ax1 = axes[0]
    ax1.contourf(X_bg, Y_bg, Z_bg, alpha=0.15, cmap='coolwarm')
    ax1.plot([alvo[0], inimigo[0]], [alvo[1], inimigo[1]], 'k--', lw=2, alpha=0.6, label='Eixo Direcional')
    ax1.scatter(linha_perfeita[:, 0], linha_perfeita[:, 1], color='#ffc107', edgecolor='black', s=40, label='Clones de Interpolação', zorder=2)
    ax1.scatter(alvo[0], alvo[1], color='#28a745', edgecolor='black', s=250, marker='*', zorder=3, label='Instância Alvo')
    ax1.scatter(inimigo[0], inimigo[1], color='#dc3545', edgecolor='black', s=150, marker='X', zorder=3, label='Inimigo Equidistante')
    ax1.set_title("Amostragem Base (Interpolação Direcional)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- PAINEL 2: Linha Reta + Ruído Gaussiano ---
    ax2 = axes[1]
    ax2.contourf(X_bg, Y_bg, Z_bg, alpha=0.15, cmap='coolwarm')
    
    np.random.seed(42)
    ruido = np.random.normal(0, 0.35, size=linha_perfeita.shape)
    clones_ruido = linha_perfeita + ruido
    
    ax2.plot([alvo[0], inimigo[0]], [alvo[1], inimigo[1]], 'k--', lw=2, alpha=0.6)
    ax2.scatter(clones_ruido[:, 0], clones_ruido[:, 1], color='#ffc107', edgecolor='black', s=40, alpha=0.8, label='Clones (Com Ruído)', zorder=2)
    ax2.scatter(alvo[0], alvo[1], color='#28a745', edgecolor='black', s=250, marker='*', zorder=3)
    ax2.scatter(inimigo[0], inimigo[1], color='#dc3545', edgecolor='black', s=150, marker='X', zorder=3)
    ax2.set_title("Vizinhança Expandida (Ruído Gaussiano)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Legenda única centralizada abaixo dos gráficos
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig('amostragem_espacial.png', dpi=300, bbox_inches='tight')
    plt.close()

def gerar_grafico_guloso():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    features = ['F12', 'F5', 'F22', 'F8', 'F1']
    ganhos = [0.60, 0.45, 0.20, 0.10, 0.05]
    acumulado = np.cumsum(ganhos)
    limiar_t_plus = 1.15

    cores = ['#28a745' if val >= limiar_t_plus else '#007bff' for val in acumulado]
    cores[0], cores[1] = '#007bff', '#007bff'
    cores[2] = '#28a745' # Bateu o limiar
    cores[3], cores[4] = '#6c757d', '#6c757d' # Descartadas

    ax.bar(features, acumulado, color=cores, edgecolor='black', alpha=0.85)
    ax.axhline(y=limiar_t_plus, color='#dc3545', linestyle='--', lw=3)
    
    ax.annotate('Limiar Atingido\n(Explicação Suficiente)', xy=(2, acumulado[2]), xytext=(2.2, 1.4),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                  fontsize=11, fontweight='bold', color='#28a745')
    
    ax.set_title("Otimização: Extração de Tamanho Mínimo", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Log-Odds Acumulado", fontweight='bold')
    ax.set_ylim(0, 1.6)
    
    legend_elements = [
        Line2D([0], [0], color='#dc3545', lw=3, linestyle='--', label='Limiar de Aceitação (t+)'),
        patches.Patch(facecolor='#007bff', edgecolor='black', label='Atributos Selecionados'),
        patches.Patch(facecolor='#6c757d', edgecolor='black', label='Atributos Descartados')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('extracao_gulosa.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Gerando imagens isoladas e profissionais...")
    gerar_fluxograma()
    gerar_paineis_amostragem()
    gerar_grafico_guloso()
    print("Sucesso! Verifique os 3 novos arquivos .png na sua pasta.")