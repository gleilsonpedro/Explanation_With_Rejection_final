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
    preds_pool = mlp.predict(X)
    opostos_idx = np.where(preds_pool != classe_original)[0]
    X_opostos = X[opostos_idx]
    distancias = np.linalg.norm(X_opostos - instancia_alvo, axis=1)
    inimigo = X_opostos[np.argmin(distancias)]

    # Geramos clones do tubo (padrão) - quantidade reduzida para visualização
    alphas = np.linspace(0, 1, 40)[:, np.newaxis]
    linha = instancia_alvo + alphas * (inimigo - instancia_alvo)
    ruido_tubo = np.random.normal(0, 0.15, size=linha.shape)
    clones_espelho = linha + ruido_tubo   # shape (40, 3)

    print("[5/6] Aplicando a Amostragem Adversarial (A Caixa do Professor)...")
    bounds_min = clones_espelho.min(axis=0)
    bounds_max = clones_espelho.max(axis=0)
    
    n_adv = 10
    clones_adv = np.zeros((n_adv, 3))
    for i in range(n_adv):
        n_fixed = np.random.randint(1, 3)
        fixed_idx = np.random.choice(3, n_fixed, replace=False)
        clone = np.where(np.random.rand(3) > 0.5, bounds_max, bounds_min)
        clone[fixed_idx] = instancia_alvo[fixed_idx]
        clones_adv[i] = clone

    # Juntando tudo para treinar o Surrogate Linear
    X_train_surrogate = np.vstack((clones_espelho, clones_adv))
    y_train_surrogate = mlp.predict(X_train_surrogate)
    lr = LogisticRegression(penalty='l2', C=1.0)
    lr.fit(X_train_surrogate, y_train_surrogate)

    # --- Previsões dos clones (para colorir por classe) ---
    y_clones_espelho = lr.predict(clones_espelho)
    y_clones_adv = lr.predict(clones_adv)

    print("[6/6] Renderizando a Mágica 3D com cores melhoradas...")
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

    # 1. Fronteira da MLP (isosurface) - cinza suave para não competir
    fig.add_trace(go.Isosurface(
        x=x_grid.flatten(), y=y_grid.flatten(), z=z_grid.flatten(),
        value=prob_espaco, isomin=0.48, isomax=0.52, surface_count=1,
        colorscale='Gray', opacity=0.25, showscale=False,
        name='Fronteira MLP (curva)'
    ))

    # 2. Plano Substituto - bege/cinza claro com opacidade baixa
    fig.add_trace(go.Surface(
        x=X_plano, y=Y_plano, z=Z_plano,
        colorscale='Viridis', opacity=0.4, showscale=False,
        name='Plano Substituto (Regressão Linear)'
    ))

    # 3. Clones do Tubo (padrão) - coloridos pela classe predita
    #    Classe 0: laranja, Classe 1: azul turquesa
    for classe in [0, 1]:
        mascara = (y_clones_espelho == classe)
        if np.any(mascara):
            cor = 'red' if classe == 0 else 'blue'
            nome = f'Clones Tubo (classe {classe})'
            fig.add_trace(go.Scatter3d(
                x=clones_espelho[mascara, 0],
                y=clones_espelho[mascara, 1],
                z=clones_espelho[mascara, 2],
                mode='markers',
                marker=dict(size=3, color=cor, opacity=0.7, symbol='circle'),
                name=nome
            ))

    # 4. Clones Adversariais (quinas) - formato 'x' com borda preta, também coloridos por classe
    for classe in [0, 1]:
        mascara = (y_clones_adv == classe)
        if np.any(mascara):
            cor = 'red' if classe == 0 else 'blue'
            nome = f'Clones Adversariais (classe {classe})'
            fig.add_trace(go.Scatter3d(
                x=clones_adv[mascara, 0],
                y=clones_adv[mascara, 1],
                z=clones_adv[mascara, 2],
                mode='markers',
                marker=dict(size=5, color=cor, symbol='x', line=dict(color='black', width=1.5)),
                name=nome
            ))

    # 5. Instância Alvo (diamante dourado) e Inimigo (círculo magenta)
    fig.add_trace(go.Scatter3d(
        x=[instancia_alvo[0]], y=[instancia_alvo[1]], z=[instancia_alvo[2]],
        mode='markers', marker=dict(size=9, color='gold', symbol='diamond', line=dict(color='black', width=2)),
        name='Instância Alvo'
    ))
    fig.add_trace(go.Scatter3d(
        x=[inimigo[0]], y=[inimigo[1]], z=[inimigo[2]],
        mode='markers', marker=dict(size=7, color='magenta', symbol='circle', line=dict(color='black', width=1.5)),
        name='Vizinho Oposto (Inimigo)'
    ))

    # Ajustes finais da câmera e layout
    fig.update_layout(
        title="<b>Exploração Local do Surrogate Linear</b><br>Clones coloridos pela classe prevista",
        scene=dict(
            xaxis_title='Feature X',
            yaxis_title='Feature Y',
            zaxis_title='Feature Z',
            camera=dict(eye=dict(x=1.5, y=1.2, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        legend=dict(title="Legenda", x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    nome_arquivo = 'plot_minabro_cores_melhoradas.html'
    fig.write_html(nome_arquivo)
    print(f"\n[SUCESSO] Gráfico gerado! Abra o arquivo '{nome_arquivo}' no navegador.")
    print("Dica: Use o mouse para girar a cena. As cores diferenciam classe 0 (laranja) e classe 1 (teal). Os 'x' são amostras adversariais.")

if __name__ == "__main__":
    gerar_grafico_3d_dissertacao()