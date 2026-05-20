import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. FUNÇÕES DO MINABRO (Motor e Plotagem)
# ==============================================================================
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

def plot_surrogate(X, y, instancia_alvo, ax, title, usar_adversarial=False):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp.fit(X, y)
    classe_alvo = mlp.predict(instancia_alvo)[0]

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
        
    # OFFSET VISUAL
    direcao = inimigo_equidistante - instancia_alvo[0]
    norma = np.linalg.norm(direcao)
    offset = (direcao / norma) * 0.15 if norma > 0 else np.zeros_like(direcao)
    inimigo_plot = inimigo_equidistante + offset

    # --- PARTE 1: CLONES DO ESPELHO (Sua ideia Original) ---
    num_clones = 600
    np.random.seed(42)
    alphas = np.linspace(0.0, 1.0, num_clones)[:, np.newaxis]
    vetor_balanceado = inimigo_equidistante - instancia_alvo[0]
    linha_central = instancia_alvo[0] + alphas * vetor_balanceado

    std_train = X.std(axis=0)
    ruido_gordo = np.random.normal(0, std_train * 0.15, size=linha_central.shape)
    clones_espelho = linha_central + ruido_gordo
    
    clones_espelho[0] = instancia_alvo[0]
    clones_espelho[1] = inimigo_equidistante

    X_train_surrogate = clones_espelho

    # --- PARTE 2: CLONES ADVERSARIAIS (Ideia do Professor - Condicional) ---
    if usar_adversarial:
        bounds_min = clones_espelho.min(axis=0)
        bounds_max = clones_espelho.max(axis=0)
        
        clones_adv = np.zeros((100, 2))
        for i in range(100):
            fixed_idx = np.random.choice([0, 1])
            free_idx = 1 - fixed_idx
            
            clone = np.zeros(2)
            clone[fixed_idx] = instancia_alvo[0][fixed_idx] 
            clone[free_idx] = bounds_max[free_idx] if np.random.rand() > 0.5 else bounds_min[free_idx]
            clones_adv[i] = clone
            
        X_train_surrogate = np.vstack((clones_espelho, clones_adv))

    y_train_surrogate = mlp.predict(X_train_surrogate)

    logreg = Pipeline([('scaler', MinMaxScaler()), 
                   ('model', LogisticRegression(C=1.0, solver='lbfgs'))])
    logreg.fit(X_train_surrogate, y_train_surrogate)
    
    t_plus, t_minus = encontrar_thresholds_locais(logreg, X_train_surrogate, y_train_surrogate)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z_mlp = mlp.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z_mlp, alpha=0.2, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.15)

    ax.scatter(clones_espelho[:, 0], clones_espelho[:, 1], c='yellow', edgecolors='black', s=15, alpha=0.5)
    
    if usar_adversarial:
        ax.scatter(clones_adv[:, 0], clones_adv[:, 1], c='red', marker='x', s=40, alpha=0.9)

    probas_grid = np.clip(logreg.predict_proba(grid), 1e-9, 1 - 1e-9)
    scores_grid = np.log(probas_grid[:, 1] / probas_grid[:, 0])
    
    Z_rej = ((scores_grid >= t_minus) & (scores_grid <= t_plus)).astype(int).reshape(xx.shape)
    ax.contourf(xx, yy, Z_rej, levels=[0.5, 1.5], colors=['gold'], alpha=0.4)

    Z_lr = logreg.predict(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z_lr, colors='black', linewidths=3, linestyles='dashed')

    ax.scatter(instancia_alvo[:, 0], instancia_alvo[:, 1], color='lime', edgecolors='black', s=250, marker='*', zorder=5)
    ax.scatter(ponto_fronteira_exato[0], ponto_fronteira_exato[1], color='white', edgecolors='black', s=80, marker='D', zorder=5)
    ax.scatter(inimigo_plot[0], inimigo_plot[1], color='magenta', edgecolors='black', s=150, marker='X', zorder=5)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# ==============================================================================
# LEGENDA E PLOTAGEM
# ==============================================================================
legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='Paciente (Alvo)', markerfacecolor='lime', markersize=16, markeredgecolor='k'),
    Line2D([0], [0], marker='X', color='w', label='Inimigo Equidistante', markerfacecolor='magenta', markersize=14, markeredgecolor='k'),
    Line2D([0], [0], marker='D', color='w', label='Fronteira Exata', markerfacecolor='white', markersize=10, markeredgecolor='k'),
    Line2D([0], [0], color='black', lw=3, linestyle='dashed', label='Hiperplano Local'),
    Patch(facecolor='gold', edgecolor='none', alpha=0.4, label='Zona de Rejeição'),
    Line2D([0], [0], marker='o', color='w', label='Clones Padrão', markerfacecolor='yellow', markersize=8, markeredgecolor='k'),
    Line2D([0], [0], marker='x', color='w', label='Clones Pior Caso', markerfacecolor='red', markeredgecolor='red', markersize=10)
]

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))
plt.subplots_adjust(bottom=0.15)

X1, y1 = make_moons(n_samples=500, noise=0.15, random_state=42)
X2, y2 = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

# LINHA 1: Sem Adversarial (Só a sua ideia)
plot_surrogate(X1, y1, np.array([[0.1, 0.1]]), axes1[0, 0], "Moons: Apenas Interpolação Padrão", usar_adversarial=False)
plot_surrogate(X2, y2, np.array([[0.7, 0.3]]), axes1[0, 1], "Circles: Apenas Interpolação Padrão", usar_adversarial=False)

# LINHA 2: Com Adversarial (Professor)
plot_surrogate(X1, y1, np.array([[0.1, 0.1]]), axes1[1, 0], "Moons: Com Amostragem Adversarial Abdutiva", usar_adversarial=True)
plot_surrogate(X2, y2, np.array([[0.7, 0.3]]), axes1[1, 1], "Circles: Com Amostragem Adversarial Abdutiva", usar_adversarial=True)

fig1.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=True, shadow=True, bbox_to_anchor=(0.5, 0.01))

plt.show()