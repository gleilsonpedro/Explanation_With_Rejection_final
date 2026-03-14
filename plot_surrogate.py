import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Função auxiliar do MINABRO para achar as bordas da Zona de Rejeição
def encontrar_thresholds_locais(modelo_lr, X_local, y_local, rejection_cost=0.24):
    probas = np.clip(modelo_lr.predict_proba(X_local), 1e-9, 1 - 1e-9)
    decision_scores = np.log(probas[:, 1] / probas[:, 0])
    
    scores_neg = decision_scores[decision_scores < 0]
    scores_pos = decision_scores[decision_scores > 0]
    
    t_minus_grid = np.linspace(scores_neg.min(), -0.001, 20) if len(scores_neg) > 0 else np.array([-0.1])
    t_plus_grid  = np.linspace(0.001, scores_pos.max(), 20)  if len(scores_pos) > 0 else np.array([0.1])
    
    best_risk, best_t_plus, best_t_minus = float('inf'), 0.1, -0.1
    for tm in t_minus_grid:
        for tp in t_plus_grid:
            if not (tm < 0 < tp): continue
            acc_mask = (decision_scores >= tp) | (decision_scores <= tm)
            preds = np.full(y_local.shape, -1)
            preds[decision_scores >= tp] = 1
            preds[decision_scores <= tm] = 0
            
            error = np.mean(preds[acc_mask] != y_local[acc_mask]) if np.any(acc_mask) else 0.0
            rejection_rate = 1.0 - np.mean(acc_mask)
            risk = error + rejection_cost * rejection_rate
            
            if risk < best_risk:
                best_risk, best_t_plus, best_t_minus = risk, tp, tm
                
    return best_t_plus, best_t_minus

def plot_surrogate(X, y, instancia_alvo, ax, title):
    # 1. Treina a MLP Global
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp.fit(X, y)
    classe_alvo = mlp.predict(instancia_alvo)[0]

    # 2. Acha a fronteira (Técnica do Espelho)
    opostos_idx = np.where(y != classe_alvo)[0]
    X_opostos = X[opostos_idx]
    distancias = np.linalg.norm(X_opostos - instancia_alvo, axis=1)
    inimigo_direcao = X_opostos[np.argmin(distancias)]

    passos = np.linspace(0, 1, 1000)[:, np.newaxis]
    caminho = instancia_alvo[0] + passos * (inimigo_direcao - instancia_alvo[0])
    preds_caminho = mlp.predict(caminho)

    try:
        idx_fronteira = np.where(preds_caminho != classe_alvo)[0][0]
        ponto_fronteira_exato = caminho[idx_fronteira]
        vetor_ate_fronteira = ponto_fronteira_exato - instancia_alvo[0]
        inimigo_equidistante = instancia_alvo[0] + 2.0 * vetor_ate_fronteira
    except IndexError:
        ponto_fronteira_exato = inimigo_direcao
        inimigo_equidistante = inimigo_direcao

    # 3. Nuvem Gorda MAIS ESPALHADA E DISTRIBUÍDA (Pedido do Professor)
    num_clones = 600
    
    np.random.seed(42) # Semente mágica: o gráfico agora vai sair SEMPRE idêntico!

    # Aumentamos drasticamente a área de atuação na linha (-0.5 a 1.5)
    alphas = np.random.uniform(-0.5, 1.5, num_clones)[:, np.newaxis]
    vetor_balanceado = inimigo_equidistante - instancia_alvo[0]
    linha_central = instancia_alvo[0] + alphas * vetor_balanceado

    # Aumentamos o espalhamento lateral (ruído) de 0.25 para 0.45
    # Agora eles vão se soltar de vez e formar uma nuvem bem larga
    std_train = X.std(axis=0)
    ruido_gordo = np.random.normal(0, std_train * 0.45, size=linha_central.shape)
    clones_finais = linha_central + ruido_gordo
    
    # Garantindo as âncoras na nuvem
    clones_finais[0] = instancia_alvo[0]
    clones_finais[1] = inimigo_equidistante

    y_oraculo = mlp.predict(clones_finais)

    # 4. Treina Regressão Logística e Acha a Rejeição
    logreg = LogisticRegression(C=1.0, solver='lbfgs')
    logreg.fit(clones_finais, y_oraculo)
    
    t_plus, t_minus = encontrar_thresholds_locais(logreg, clones_finais, y_oraculo)

    # 5. Desenha no subplot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Fundo da MLP
    Z_mlp = mlp.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z_mlp, alpha=0.2, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.15)

    # Clones mais espalhados
    ax.scatter(clones_finais[:, 0], clones_finais[:, 1], c='yellow', edgecolors='black', s=15, alpha=0.5)

    # Calcula os Scores para desenhar a Zona de Rejeição
    probas_grid = np.clip(logreg.predict_proba(grid), 1e-9, 1 - 1e-9)
    scores_grid = np.log(probas_grid[:, 1] / probas_grid[:, 0])
    
    # Máscara binária onde o score está entre t_minus e t_plus (Rejeição)
    Z_rej = ((scores_grid >= t_minus) & (scores_grid <= t_plus)).astype(int).reshape(xx.shape)
    
    # Pinta a Zona de Rejeição de Amarelo Transparente
    ax.contourf(xx, yy, Z_rej, levels=[0.5, 1.5], colors=['gold'], alpha=0.4)

    # Reta de Decisão (Fronteira Local)
    Z_lr = logreg.predict(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z_lr, colors='black', linewidths=3, linestyles='dashed')

    # Protagonistas
    ax.scatter(instancia_alvo[:, 0], instancia_alvo[:, 1], color='lime', edgecolors='black', s=250, marker='*', zorder=5)
    ax.scatter(ponto_fronteira_exato[0], ponto_fronteira_exato[1], color='white', edgecolors='black', s=80, marker='D', zorder=5)
    ax.scatter(inimigo_equidistante[0], inimigo_equidistante[1], color='magenta', edgecolors='black', s=150, marker='X', zorder=5)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# ==============================================================================
# CONFIGURAÇÃO DOS 4 EXPERIMENTOS E LEGENDA GLOBAL
# ==============================================================================
# Aumentei o espaço inferior (bottom=0.15) para caber a legenda
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plt.subplots_adjust(bottom=0.15)

# 1. Moons (Centro)
X1, y1 = make_moons(n_samples=500, noise=0.15, random_state=42)
plot_surrogate(X1, y1, np.array([[0.5, 0.0]]), axes[0, 0], "Moons: Alvo no Centro")

# 2. Moons (Borda)
plot_surrogate(X1, y1, np.array([[-0.5, 0.8]]), axes[0, 1], "Moons: Alvo na Borda Externa")

# 3. Círculos
X2, y2 = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
plot_surrogate(X2, y2, np.array([[0.0, 0.0]]), axes[1, 0], "Circles: Alvo no Núcleo")

# 4. Nuvens Lineares
X3, y3 = make_classification(n_samples=500, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
plot_surrogate(X3, y3, np.array([[0.0, -1.0]]), axes[1, 1], "Linear Blobs: Alvo Próximo à Divisa")

# ==============================================================================
# CRIANDO A LEGENDA GLOBAL (A sua ideia brilhante!)
# ==============================================================================
legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='Paciente (Alvo)', markerfacecolor='lime', markersize=16, markeredgecolor='k'),
    Line2D([0], [0], marker='X', color='w', label='Inimigo Equidistante', markerfacecolor='magenta', markersize=14, markeredgecolor='k'),
    Line2D([0], [0], marker='D', color='w', label='Ponto Exato da Fronteira', markerfacecolor='white', markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Clones Gerados (Vizinhança)', markerfacecolor='yellow', markersize=8, markeredgecolor='k'),
    Line2D([0], [0], color='black', lw=3, linestyle='dashed', label='Hiperplano Local (Seu Método)'),
    Patch(facecolor='gold', edgecolor='none', alpha=0.4, label='Zona de Rejeição (MINABRO)')
]

# Adiciona a legenda na figura principal, lá embaixo
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, frameon=True, shadow=True, bbox_to_anchor=(0.5, 0.02))

plt.show()