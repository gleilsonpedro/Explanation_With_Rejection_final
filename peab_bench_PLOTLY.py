import pandas as pd
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INPUT_DIR = 'results/benchmark/bench_csv'
OUTPUT_HTML_DIR = 'results/plots_interativos'

def carregar_tudo():
    arquivos = glob.glob(os.path.join(INPUT_DIR, "bench_*.csv"))
    lista = []
    for arq in arquivos:
        df = pd.read_csv(arq)
        df['Dataset'] = os.path.basename(arq).replace('bench_', '').replace('.csv', '').upper()
        if 'tipo_predicao' not in df.columns: df['tipo_predicao'] = 'N/A'
        lista.append(df)
    return pd.concat(lista, ignore_index=True)

def gerar_plots():
    df = carregar_tudo()
    if df is None: return
    os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)
    
    # ---------------------------------------------------------
    # PLOT 1: BOXPLOT DO GAP (A Verdade sobre o Erro)
    # Mostra a distribuição dos erros por classe.
    # ---------------------------------------------------------
    fig1 = px.box(df, x="tipo_predicao", y="GAP", color="tipo_predicao",
                  points="all", # Mostra todos os pontos
                  hover_data=["Dataset", "id"],
                  title="Distribuição do Erro de Minimalidade (GAP) por Classe",
                  labels={"GAP": "Features Excedentes (PEAB - Ótimo)", "tipo_predicao": "Classe"})
    fig1.update_layout(showlegend=False)
    fig1.write_html(f"{OUTPUT_HTML_DIR}/1_distribuicao_gap.html")
    print("Gráfico 1 gerado: Distribuição GAP")

    # ---------------------------------------------------------
    # PLOT 2: BARRAS COMPARATIVAS (Qualidade vs Gordura)
    # Mostra % de Perfeição e % de Erro Relativo lado a lado
    # ---------------------------------------------------------
    # Agrupando dados
    resumo = df.groupby('tipo_predicao').agg({
        'is_optimal': 'mean',
        'GAP': 'mean',
        'tamanho_OPTIMO': 'mean'
    }).reset_index()
    resumo['is_optimal'] = resumo['is_optimal'] * 100
    # Erro relativo recalculado na média
    resumo['erro_relativo'] = (resumo['GAP'] / resumo['tamanho_OPTIMO']) * 100

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Barra de Otimalidade (Queremos ALTO)
    fig2.add_trace(
        go.Bar(x=resumo['tipo_predicao'], y=resumo['is_optimal'], name="% Otimalidade (Perfeição)", 
               marker_color='rgb(46, 204, 113)', opacity=0.8),
        secondary_y=False,
    )
    
    # Linha de Erro Relativo (Queremos BAIXO)
    fig2.add_trace(
        go.Scatter(x=resumo['tipo_predicao'], y=resumo['erro_relativo'], name="% Erro Relativo (Gordura)",
                   mode='lines+markers+text', text=[f"{v:.1f}%" for v in resumo['erro_relativo']],
                   textposition="top center", marker_color='rgb(231, 76, 60)', line=dict(width=3)),
        secondary_y=True,
    )

    fig2.update_layout(title="Qualidade da Explicação: Perfeição vs Excesso")
    fig2.update_yaxes(title_text="% de Explicações Perfeitas", secondary_y=False, range=[0, 110])
    fig2.update_yaxes(title_text="% de Features Excedentes (Erro Relativo)", secondary_y=True)
    fig2.write_html(f"{OUTPUT_HTML_DIR}/2_qualidade_vs_erro.html")
    print("Gráfico 2 gerado: Qualidade vs Erro")

    # ---------------------------------------------------------
    # PLOT 3: SCATTER DE TEMPO (Davi vs Golias)
    # Mostra a relação Tamanho x Tempo em escala Log
    # ---------------------------------------------------------
    # Vamos fazer um "melt" para colocar PEAB e OPT na mesma coluna de tempo para plotar
    df_peab = df[['tamanho_PEAB', 'tempo_PEAB', 'Dataset', 'tipo_predicao']].copy()
    df_peab.columns = ['Tamanho', 'Tempo', 'Dataset', 'Classe']
    df_peab['Metodo'] = 'PEAB (Heurística)'

    df_opt = df[['tamanho_OPTIMO', 'tempo_OPTIMO', 'Dataset', 'tipo_predicao']].copy()
    df_opt.columns = ['Tamanho', 'Tempo', 'Dataset', 'Classe']
    df_opt['Metodo'] = 'MILP (Otimização)'
    
    df_total = pd.concat([df_peab, df_opt])

    fig3 = px.scatter(df_total, x="Tamanho", y="Tempo", color="Metodo", 
                      symbol="Classe", facet_col="Classe",
                      hover_data=["Dataset"],
                      log_y=True, # ESCALA LOGARÍTMICA É ESSENCIAL AQUI
                      title="Comparativo de Custo Computacional (Escala Log)",
                      labels={"Tempo": "Tempo (segundos) - Log", "Tamanho": "Tamanho da Explicação"})
    
    fig3.update_traces(marker=dict(size=6, opacity=0.7))
    fig3.write_html(f"{OUTPUT_HTML_DIR}/3_tempo_log.html")
    print("Gráfico 3 gerado: Tempo Log")

    print(f"\n[SUCESSO] Todos os gráficos interativos salvos em: {OUTPUT_HTML_DIR}/")

if __name__ == '__main__':
    gerar_plots()