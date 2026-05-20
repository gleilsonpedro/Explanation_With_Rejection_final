import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def gerar_grafico_3d_dissertacao():
    print("[1/5] Criando um universo 3D sintético...")
    # Criamos um dataset não-linear de 3 dimensões (X, Y, Z)
    X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, 
                               n_redundant=0, n_repeated=0, n_clusters_per_class=2, 
                               flip_y=0.05, random_state=42)

    print("[2/5] Treinando o Oráculo (MLP Caixa-Preta)...")
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', max_iter=1000, random_state=42)
    mlp.fit(X, y)

    print("[3/5] Escolhendo o 'Paciente Zero' perto da fronteira de decisão...")
    # Vamos achar um ponto onde a MLP está meio na dúvida (perto da fronteira)
    probas = mlp.predict_proba(X)[:, 1]
    indices_fronteira = np.where((probas > 0.4) & (probas < 0.6))[0]
    idx_alvo = indices_fronteira[0] if len(indices_fronteira) > 0 else 0
    instancia_alvo = X[idx_alvo]
    classe_original = mlp.predict([instancia_alvo])[0]

    print("[4/5] Aplicando a Técnica do Espelho (MINABRO)...")
    # 1. Gerar vizinhança (clones) com ruído em 3D
    ruido = np.random.normal(0, 0.5, size=(1000, 3))
    clones = instancia_alvo + ruido
    y_clones = mlp.predict(clones)

    # 2. Treinar o Surrogate (Plano Linear)
    lr = LogisticRegression(penalty='l2', C=1.0)
    lr.fit(clones, y_clones)

    print("[5/5] Renderizando a Mágica 3D...")
    # --- PREPARANDO OS DADOS VISUAIS ---
    # Limites do "Zoom"
    delta = 1.5
    xmin, xmax = instancia_alvo[0] - delta, instancia_alvo[0] + delta
    ymin, ymax = instancia_alvo[1] - delta, instancia_alvo[1] + delta
    zmin, zmax = instancia_alvo[2] - delta, instancia_alvo[2] + delta

    # 1. Superfície da MLP (O Lençol Curvo)
    # Criamos uma nuvem de pontos densa no espaço para achar onde a probabilidade é exatamente 50%
    grid_size = 30
    x_grid, y_grid, z_grid = np.mgrid[xmin:xmax:grid_size*1j, ymin:ymax:grid_size*1j, zmin:zmax:grid_size*1j]
    pontos_espaco = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    prob_espaco = mlp.predict_proba(pontos_espaco)[:, 1]

    # 2. Superfície do MINABRO (A Prancheta Reta / Regressão Logística)
    # Equação do plano: w1*x + w2*y + w3*z + b = 0  => z = -(w1*x + w2*y + b) / w3
    X_plano, Y_plano = np.meshgrid(np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10))
    w1, w2, w3 = lr.coef_[0]
    b = lr.intercept_[0]
    w3_safe = w3 if abs(w3) > 1e-5 else 1e-5 # Evitar divisão por zero
    Z_plano = -(w1 * X_plano + w2 * Y_plano + b) / w3_safe

    # --- MONTANDO O GRÁFICO PLOTLY ---
    fig = go.Figure()

    # Adiciona a Fronteira Curva da MLP (Isosuperfície onde prob == 0.5)
    fig.add_trace(go.Isosurface(
        x=x_grid.flatten(), y=y_grid.flatten(), z=z_grid.flatten(),
        value=prob_espaco, isomin=0.48, isomax=0.52, surface_count=1,
        colorscale='Blues', opacity=0.4, showscale=False,
        name='Fronteira MLP (Curva)'
    ))

    # Adiciona o Plano do MINABRO (Surrogate)
    fig.add_trace(go.Surface(
        x=X_plano, y=Y_plano, z=Z_plano,
        colorscale='Reds', opacity=0.7, showscale=False,
        name='Plano MINABRO (Reto)'
    ))

    # Adiciona os Clones da Vizinhança
    mask_pos = y_clones == 1
    mask_neg = y_clones == 0
    fig.add_trace(go.Scatter3d(
        x=clones[mask_neg, 0], y=clones[mask_neg, 1], z=clones[mask_neg, 2],
        mode='markers', marker=dict(size=2, color='gray', opacity=0.3),
        name='Clones Classe 0'
    ))
    fig.add_trace(go.Scatter3d(
        x=clones[mask_pos, 0], y=clones[mask_pos, 1], z=clones[mask_pos, 2],
        mode='markers', marker=dict(size=2, color='lightblue', opacity=0.3),
        name='Clones Classe 1'
    ))

    # Adiciona a Instância Alvo (A grande estrela)
    fig.add_trace(go.Scatter3d(
        x=[instancia_alvo[0]], y=[instancia_alvo[1]], z=[instancia_alvo[2]],
        mode='markers', marker=dict(size=8, color='gold', symbol='diamond', line=dict(color='black', width=2)),
        name='Instância Original'
    ))

    # Estética do Gráfico
    fig.update_layout(
        title="Abdução Geométrica: Aproximação do Plano MINABRO sobre a Fronteira da MLP",
        scene=dict(
            xaxis_title='Feature X', yaxis_title='Feature Y', zaxis_title='Feature Z',
            zaxis=dict(range=[zmin, zmax])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Salva e abre no navegador
    nome_arquivo = 'plot_minabro_3d.html'
    fig.write_html(nome_arquivo)
    print(f"\n[SUCESSO] Gráfico gerado! Abra o arquivo '{nome_arquivo}' no seu navegador.")

if __name__ == "__main__":
    gerar_grafico_3d_dissertacao()