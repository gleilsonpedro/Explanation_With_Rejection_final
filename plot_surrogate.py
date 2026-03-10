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

print("[INFO] Treinando a MLP (Caixa-Preta) Global...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X, y)

# ==============================================================================
# 2. VIZINHANÇA DIRECIONADA: A IDEIA DO "SNIPER" (Interpolação na Fronteira)
# ==============================================================================
# Escolhemos um paciente (Instância Alvo)
instancia_alvo = np.array([[0.5, 0.0]]) 
classe_alvo = mlp.predict(instancia_alvo)[0]

# Achamos os "inimigos" (dados da classe oposta)
opostos_idx = np.where(y != classe_alvo)[0]
X_opostos = X[opostos_idx]

# Calculamos a distância e pegamos o Inimigo Mais Próximo
distancias = np.linalg.norm(X_opostos - instancia_alvo, axis=1)
inimigo_mais_proximo = X_opostos[np.argmin(distancias)]

print("[INFO] Gerando clones na linha de fronteira...")
# Geramos os clones caminhando em linha reta entre o Alvo e o Inimigo
num_clones = 500
alphas = np.linspace(-0.2, 1.2, num_clones)[:, np.newaxis]
vetor_direcao = inimigo_mais_proximo - instancia_alvo[0]
clones_na_reta = instancia_alvo[0] + alphas * vetor_direcao

# Adicionamos um leve "cilindro de ruído" para a LogReg não bugar com a linha perfeita
std_train = X.std(axis=0)
ruido = np.random.normal(0, std_train * 0.05, size=clones_na_reta.shape)
clones_finais = clones_na_reta + ruido
clones_finais[0] = instancia_alvo[0]

# O Oráculo (MLP) classifica esses clones gerados
y_oraculo = mlp.predict(clones_finais)

print("[INFO] Treinando o Modelo Substituto Local (Regressão Logística)...")
logreg = LogisticRegression(C=1.0, solver='lbfgs')
logreg.fit(clones_finais, y_oraculo)

# ==============================================================================
# 3. FUNÇÃO DE VISUALIZAÇÃO 2D (Para a Dissertação)
# ==============================================================================
plt.figure(figsize=(10, 7))

# Cria a malha de fundo para pintar as regiões da MLP
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]

# Fundo colorido (A regra complexa da MLP)
Z_mlp = mlp.predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, Z_mlp, alpha=0.3, cmap='coolwarm')

# Plota os dados globais mais apagados
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.2, label='Dados Globais')

# Plota a "Rodovia" (Linha ligando os dois pontos)
plt.plot([instancia_alvo[0,0], inimigo_mais_proximo[0]], 
         [instancia_alvo[0,1], inimigo_mais_proximo[1]], 
         'k-', linewidth=2, label='Linha de Interpolação (Sniper)')

# Plota os Clones (Vizinhança) em cima da linha
plt.scatter(clones_finais[:, 0], clones_finais[:, 1], c='yellow', edgecolors='black', 
            s=20, alpha=0.8, zorder=3, label='Clones (Interpolação)')

# Fronteira Local da LogReg (Reta Tracejada cortando a rodovia)
Z_lr = logreg.predict(grid).reshape(xx.shape)
plt.contour(xx, yy, Z_lr, colors='black', linewidths=3, linestyles='dashed')

# Destaca os protagonistas da história
plt.scatter(instancia_alvo[:, 0], instancia_alvo[:, 1], color='lime', edgecolors='black', 
            s=300, marker='*', zorder=5, label='Paciente (Alvo)')
plt.scatter(inimigo_mais_proximo[0], inimigo_mais_proximo[1], color='red', edgecolors='black', 
            s=200, marker='X', zorder=5, label='Inimigo Mais Próximo')

plt.title("Aproximação Local 2D: Busca de Fronteira Direcionada", fontsize=14, fontweight='bold')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="best", framealpha=0.9)
plt.tight_layout()

print("[INFO] Abrindo o gráfico... Salve a imagem para a sua dissertação!")
plt.show()