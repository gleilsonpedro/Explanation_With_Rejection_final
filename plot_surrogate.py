import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. GERAÇÃO DOS DADOS E TREINAMENTO GLOBAL (A MLP)
# ==============================================================================
print("[INFO] Gerando dataset artificial 'Moons'...")
X, y = make_moons(n_samples=500, noise=0.15, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X, y)

# ==============================================================================
# 2. VIZINHANÇA: A IDEIA DA "NUVEM GORDA / QUADRADO" (Espelho + Espalhamento)
# ==============================================================================
instancia_alvo = np.array([[0.5, 0.0]]) 
classe_alvo = mlp.predict(instancia_alvo)[0]

# 2.1 Acha a direção do inimigo e a fronteira exata (A Técnica do Espelho)
opostos_idx = np.where(y != classe_alvo)[0]
X_opostos = X[opostos_idx]
distancias = np.linalg.norm(X_opostos - instancia_alvo, axis=1)
inimigo_direcao = X_opostos[np.argmin(distancias)]

passos = np.linspace(0, 1, 1000)[:, np.newaxis]
caminho = instancia_alvo[0] + passos * (inimigo_direcao - instancia_alvo[0])
preds_caminho = mlp.predict(caminho)

idx_fronteira = np.where(preds_caminho != classe_alvo)[0][0]
ponto_fronteira_exato = caminho[idx_fronteira]

vetor_ate_fronteira = ponto_fronteira_exato - instancia_alvo[0]
inimigo_equidistante = instancia_alvo[0] + 2.0 * vetor_ate_fronteira

print("[INFO] Gerando clones espalhados (O Quadrado do Professor)...")
num_clones = 800 # Aumentei um pouco para a nuvem ficar mais densa
alphas = np.linspace(0.0, 1.0, num_clones)[:, np.newaxis]
vetor_balanceado = inimigo_equidistante - instancia_alvo[0]

# Clones no "esqueleto" da linha central
linha_central = instancia_alvo[0] + alphas * vetor_balanceado

# A MÁGICA ACONTECE AQUI:
# Em vez de um ruído de 0.05 (cilindro fino), usamos a distância entre os pontos 
# para criar um "espalhamento" (spread) proporcional, formando o quadrado/nuvem gorda.
distancia_entre_pontos = np.linalg.norm(inimigo_equidistante - instancia_alvo[0])
espalhamento = distancia_entre_pontos * 0.35 # 35% de espalhamento lateral

ruido_gordo = np.random.normal(0, espalhamento, size=linha_central.shape)
clones_finais = linha_central + ruido_gordo

# Garante que as estrelas principais continuem lá
clones_finais[0] = instancia_alvo[0]
clones_finais[1] = inimigo_equidistante

y_oraculo = mlp.predict(clones_finais)

logreg = LogisticRegression(C=1.0, solver='lbfgs')
logreg.fit(clones_finais, y_oraculo)

# ==============================================================================
# 3. VISUALIZAÇÃO 2D
# ==============================================================================
plt.figure(figsize=(10, 7))

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]

Z_mlp = mlp.predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, Z_mlp, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.15)

# A Linha Central que serviu de base (agora apenas como referência visual)
plt.plot([instancia_alvo[0,0], inimigo_equidistante[0]], 
         [instancia_alvo[0,1], inimigo_equidistante[1]], 
         color='gray', linestyle=':', linewidth=2, label='Eixo Central')

# A Nuvem Gorda de Clones
plt.scatter(clones_finais[:, 0], clones_finais[:, 1], c='yellow', edgecolors='black', 
            s=25, alpha=0.7, zorder=3, label='Clones (Espalhamento em Quadrado/Nuvem)')

Z_lr = logreg.predict(grid).reshape(xx.shape)
plt.contour(xx, yy, Z_lr, colors='black', linewidths=3, linestyles='dashed')

plt.scatter(instancia_alvo[:, 0], instancia_alvo[:, 1], color='lime', edgecolors='black', 
            s=300, marker='*', zorder=5, label='Paciente (Alvo)')
plt.scatter(ponto_fronteira_exato[0], ponto_fronteira_exato[1], color='white', edgecolors='black', 
            s=100, marker='D', zorder=5, label='Fronteira Exata')
plt.scatter(inimigo_equidistante[0], inimigo_equidistante[1], color='magenta', edgecolors='black', 
            s=200, marker='X', zorder=5, label='Inimigo Sintético')

plt.title("Aproximação Local 2D: Exploração de Vizinhança com Espalhamento", fontsize=14, fontweight='bold')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="best", framealpha=0.9)
plt.tight_layout()
plt.show()