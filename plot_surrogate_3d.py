import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def gerar_grafico_3d_dissertacao():
    print("[1/6] Criando um universo 3D sintético...")
    X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, 
                               n_redundant=0, n_repeated=0, n_clusters_per_class=2, 
                               flip_y=0.05, random_state=42)

    print("[2/6] Treinando o Oráculo (MLP Caixa-Preta)...")
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', max_iter=1000, random_state=42)
    mlp.fit(X, y)

    print("[3/6] Escolhendo o 'Paciente Zero' perto da fronteira de decisão...")
    probas = mlp.predict_proba(X)[:, 1]
    indices_fronteira = np.where((probas > 0.4) & (probas < 0.6))[0]
    idx_alvo = indices_fronteira[0] if len(indices_fronteira) > 0 else 0
    instancia_alvo = X[idx_alvo]
    classe_original = mlp.predict([instancia_alvo])[0]

    print("[4/6] Aplicando a Técnica do Espelho (O Tubo Geométrico)...")
    # Acha o Vizinho Oposto
    preds_pool = mlp.predict(X)
    opostos_idx = np.where(preds_pool != classe_original)[0]
    X_opostos = X[opostos_idx]
    distancias = np.linalg.norm(X_opostos - instancia_alvo, axis=1)
    inimigo = X_opostos[np.argmin(distancias)]

    # Interpolação (O Tubo direcional)
    alphas = np.linspace(0, 1, 600)[:, np.newaxis]
    linha = instancia_alvo + alphas * (inimigo - instancia_alvo)
    ruido_tubo = np.random.normal(0, 0.15, size=linha.shape) # Espalhamento menor
    clones_espelho = linha + ruido_tubo

    print("[5/6] Aplicando a Amostragem Adversarial (A Caixa do Professor)...")
    bounds_min = clones_espelho.min(axis=0)
    bounds_max = clones_espelho.max(axis=0)
    
    clones_adv = np.zeros((150, 3))
    for i in range(150):
        # Fixa 1 ou 2 features, joga o resto pras bordas
        n_fixed = np.random.randint(1, 3)
        fixed_idx = np.random.choice(3, n_fixed, replace=False)
        clone = np.where(np.random.rand(3) > 0.5, bounds_max, bounds_min)
        clone[fixed_idx] = instancia_alvo[fixed_idx]
        clones_adv[i] = clone

    # Juntando tudo para o Surrogate Linear
    X_train_surrogate = np.vstack((clones_espelho, clones_adv))
    y_train_surrogate = mlp.predict(X_train_surrogate)
    lr = LogisticRegression(penalty='l2', C=1.0)
    lr.fit(X_train_surrogate, y_train_surrogate)

    print("[6/6] Renderizando a Mágica 3D...")
    delta = 1.5
    xmin, xmax = instancia_alvo[0] - delta, instancia_alvo[0] + delta
    ymin, ymax = instancia_alvo[1] - delta, instancia_alvo[1] + delta
    zmin, zmax = instancia_alvo[2] - delta, instancia_alvo[2] + delta

    grid_size = 30
    x_grid, y_grid, z_grid = np.mgrid[xmin:xmax:grid_size*1j, ymin:ymax:grid_size*1j, zmin:zmax:grid_size*1j]
    pontos_espaco = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    prob_espaco = mlp.predict_proba(pontos_espaco)[:, 1]

    X_plano, Y_plano = np.meshgrid(np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10))
    w1, w2, w3 = lr.coef_[0]
    b = lr.intercept_[0]
    w3_safe = w3 if abs(w3) > 1e-5 else 1e-5
    Z_plano = -(w1 * X_plano + w2 * Y_plano + b) / w3_safe

    fig = go.Figure()

    # 1. Fronteira Curva da MLP
    fig.add_trace(go.Isosurface(
        x=x_grid.flatten(), y=y_grid.flatten(), z=z_grid.flatten(),
        value=prob_espaco, isomin=0.48, isomax=0.52, surface_count=1,
        colorscale='Blues', opacity=0.3, showscale=False,
        name='Fronteira MLP (Curva)'
    ))

    # 2. Plano MINABRO
    fig.add_trace(go.Surface(
        x=X_plano, y=Y_plano, z=Z_plano,
        colorscale='Greys', opacity=0.6, showscale=False,
        name='Plano Substituto'
    ))

    # 3. Clones do Tubo (Espelho Padrão) - Azuis
    fig.add_trace(go.Scatter3d(
        x=clones_espelho[:, 0], y=clones_espelho[:, 1], z=clones_espelho[:, 2],
        mode='markers', marker=dict(size=2, color='deepskyblue', opacity=0.5),
        name='Amostragem Padrão (Tubo)'
    ))

    # 4. Clones do Pior Caso (A Caixa do Professor) - Vermelhos
    fig.add_trace(go.Scatter3d(
        x=clones_adv[:, 0], y=clones_adv[:, 1], z=clones_adv[:, 2],
        mode='markers', marker=dict(size=3, color='crimson', symbol='x'),
        name='Amostragem Adversarial (Quinas)'
    ))

    # 5. Instância Alvo e Inimigo
    fig.add_trace(go.Scatter3d(
        x=[instancia_alvo[0], inimigo[0]], y=[instancia_alvo[1], inimigo[1]], z=[instancia_alvo[2], inimigo[2]],
        mode='markers', marker=dict(size=[8, 6], color=['gold', 'magenta'], symbol=['diamond', 'circle'], line=dict(color='black', width=2)),
        name='Alvo e Inimigo'
    ))

    fig.update_layout(
        title="Visualização 3D: O Tubo do Espelho vs. A Caixa de Pior Caso (Adversarial)",
        scene=dict(xaxis_title='Feature X', yaxis_title='Feature Y', zaxis_title='Feature Z', zaxis=dict(range=[zmin, zmax])),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    nome_arquivo = 'plot_minabro_3d.html'
    fig.write_html(nome_arquivo)
    print(f"\n[SUCESSO] Gráfico gerado! Abra o arquivo '{nome_arquivo}' no navegador.")

if __name__ == "__main__":
    gerar_grafico_3d_dissertacao()