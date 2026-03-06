import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. GERAÇÃO DOS DADOS E TREINAMENTO GLOBAL (A MLP)
# ==============================================================================
print("[INFO] Gerando dataset artificial 'Moons'...")
# Cria 500 pontos no formato de duas meias-luas entrelaçadas
X, y = make_moons(n_samples=500, noise=0.15, random_state=42)

print("[INFO] Treinando a MLP (Caixa-Preta) Global...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X, y)

# ==============================================================================
# 2. DEFINIÇÃO DA VIZINHANÇA LOCAL E TREINAMENTO DO SURROGATE (LogReg)
# ==============================================================================
# Escolhendo um ponto crítico (bem no meio da curva da fronteira)
instancia_alvo = np.array([[0.5, 0.0]]) 

print("[INFO] Gerando clones e treinando o Modelo Substituto Local (Regressão Logística)...")
# Cria 1000 clones ao redor da instância (Vizinhança)
ruido = np.random.normal(0, 0.25, size=(1000, 2))
clones = instancia_alvo + ruido

# O Oráculo (MLP) classifica os clones
y_oraculo = mlp.predict(clones)

# O Estudante (LogReg) aprende apenas com a vizinhança
logreg = LogisticRegression(C=1.0, solver='lbfgs')
logreg.fit(clones, y_oraculo)

# ==============================================================================
# 3. FUNÇÃO DE VISUALIZAÇÃO 2D
# ==============================================================================
def plot_2d():
    plt.figure(figsize=(10, 7))
    
    # Cria uma malha (grid) para pintar o fundo
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Fronteira Global da MLP (Curvas)
    Z_mlp = mlp.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z_mlp, alpha=0.3, cmap='coolwarm')
    
    # Plota os dados originais do dataset
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.6, label='Dados Originais')
    
    # Plota os Clones locais
    plt.scatter(clones[:, 0], clones[:, 1], c='gray', s=10, alpha=0.1, label='Clones (Vizinhança)')
    
    # Fronteira Local da LogReg (Reta)
    Z_lr = logreg.predict(grid).reshape(xx.shape)
    plt.contour(xx, yy, Z_lr, colors='black', linewidths=2, linestyles='dashed')
    
    # Destaca a instância que estamos explicando
    plt.scatter(instancia_alvo[:, 0], instancia_alvo[:, 1], color='lime', edgecolors='black', 
                s=200, marker='*', zorder=5, label='Instância Explicada')

    plt.title("Aproximação Local 2D: Reta da Regressão Logística vs Curva da MLP", fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. FUNÇÃO DE VISUALIZAÇÃO 3D (O que o professor sugeriu!)
# ==============================================================================
def plot_3d():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Foca a malha 3D apenas na vizinhança do ponto
    x_min, x_max = instancia_alvo[0][0] - 0.8, instancia_alvo[0][0] + 0.8
    y_min, y_max = instancia_alvo[0][1] - 0.8, instancia_alvo[0][1] + 0.8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Calcula as probabilidades (O Eixo Z)
    Z_mlp_prob = mlp.predict_proba(grid)[:, 1].reshape(xx.shape)
    Z_lr_prob = logreg.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Plota a Superfície da MLP (Montanha curva)
    surf_mlp = ax.plot_surface(xx, yy, Z_mlp_prob, cmap='Blues', alpha=0.7, 
                               linewidth=0, antialiased=True, label='MLP (Real)')
    
    # Plota a Superfície da Regressão Logística (Plano Reto/Vidro)
    surf_lr = ax.plot_surface(xx, yy, Z_lr_prob, color='orange', alpha=0.5, 
                              linewidth=0.5, edgecolors='k', antialiased=True, label='LogReg (Surrogate)')

    # Ajustes estéticos
    ax.set_title("Visão 3D: Superfície de Probabilidade (MLP vs Linear)", fontsize=14)
    ax.set_xlabel('Feature X')
    ax.set_ylabel('Feature Y')
    ax.set_zlabel('Probabilidade da Classe 1')
    
    # Pulo do gato para colocar legenda no 3D do matplotlib
    surf_mlp._facecolors2d = surf_mlp._facecolor3d
    surf_mlp._edgecolors2d = surf_mlp._edgecolor3d
    surf_lr._facecolors2d = surf_lr._facecolor3d
    surf_lr._edgecolors2d = surf_lr._edgecolor3d
    ax.legend()

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. EXECUÇÃO
# ==============================================================================
if __name__ == '__main__':
    print("[INFO] Abrindo Visão 2D... (Feche a janela para abrir a Visão 3D logo em seguida)")
    plot_2d()
    print("[INFO] Abrindo Visão 3D... (Use o mouse para girar o gráfico!)")
    plot_3d()
    print("[INFO] Finalizado.")