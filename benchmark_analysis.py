import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração Visual
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6)})

BASE_DIR = 'results/benchmark'
OUTPUT_DIR = 'results/plots'

def carregar_arquivos():
    """Lista todos os benchmarks disponíveis."""
    padrao = os.path.join(BASE_DIR, "bench_*.csv")
    arquivos = glob.glob(padrao)
    arquivos.sort()
    return arquivos

def ler_dataset(caminho):
    """Lê o CSV e identifica o nome do dataset."""
    df = pd.read_csv(caminho)
    nome = os.path.basename(caminho).replace('bench_', '').replace('.csv', '')
    df['dataset'] = nome
    # Garante que a coluna existe para compatibilidade com CSVs antigos
    if 'tipo_predicao' not in df.columns:
        df['tipo_predicao'] = 'DESCONHECIDO'
    return df, nome

# --- 1. ANÁLISE GERAL (GAP E TEMPO TOTAL) ---
def plotar_analise_geral(df, nome_dataset):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"   > Gerando gráficos gerais para {nome_dataset}...")

    # Gráfico A: Distribuição do GAP (Histograma)
    plt.figure()
    ax = sns.countplot(x='GAP', data=df, palette='viridis')
    plt.title(f'Distribuição de Erro (GAP) - {nome_dataset}\n(0 = PEAB foi Ótimo)')
    plt.xlabel('Features Excedentes (PEAB - Ótimo)')
    plt.ylabel('Qtd. Instâncias')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome_dataset}_geral_gap.png"))
    plt.close()

    # Gráfico B: Scatter Plot Tempo vs Tamanho
    plt.figure()
    plt.scatter(df['tamanho_OPTIMO'], df['tempo_OPTIMO'], label='Otimização (Exata)', alpha=0.6, color='red', s=30)
    plt.scatter(df['tamanho_PEAB'], df['tempo_PEAB'], label='PEAB (Heurística)', alpha=0.6, color='blue', s=30)
    plt.xlabel('Tamanho da Explicação (|S|)')
    plt.ylabel('Tempo (s) - Log Scale')
    plt.title(f'Comparativo de Custo Computacional - {nome_dataset}')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome_dataset}_geral_tempo.png"))
    plt.close()

# --- 2. ANÁLISE FOCADA (REJEIÇÃO VS CLASSIFICAÇÃO) ---
def plotar_analise_rejeicao(df, nome_dataset):
    print(f"   > Gerando gráficos de Rejeição para {nome_dataset}...")
    
    if 'REJEITADA' not in df['tipo_predicao'].unique():
        print("     [AVISO] Sem instâncias rejeitadas. Pulando esta etapa.")
        return

    # Gráfico C: Boxplot do GAP por Tipo (Onde está o gargalo?)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='tipo_predicao', y='GAP', data=df, palette="Set2")
    plt.title(f'Gargalo de Minimalidade: Rejeitadas vs Classificadas ({nome_dataset})')
    plt.ylabel('GAP (Diferença para o Ótimo)')
    plt.xlabel('Tipo de Predição')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome_dataset}_rejeicao_gap_boxplot.png"))
    plt.close()

    # Gráfico D: Taxa de Otimalidade por Tipo
    resumo = df.groupby('tipo_predicao')['is_optimal'].mean() * 100
    resumo = resumo.reset_index()
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='tipo_predicao', y='is_optimal', data=resumo, palette="pastel")
    plt.ylim(0, 110)
    plt.axhline(100, color='green', linestyle='--', alpha=0.5)
    plt.title(f'Taxa de Otimalidade (%) por Tipo - {nome_dataset}')
    plt.ylabel('% de Casos Ótimos')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome_dataset}_rejeicao_otimalidade.png"))
    plt.close()

# --- 3. ANÁLISE GLOBAL (TODOS OS DATASETS) ---
def plotar_analise_global(arquivos):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n--- Processando Análise Global ---")
    
    lista_dfs = []
    for arq in arquivos:
        df, _ = ler_dataset(arq)
        lista_dfs.append(df)
    
    if not lista_dfs: return
    df_global = pd.concat(lista_dfs, ignore_index=True)

    # Gráfico E: GAP Médio Global (Rejeitada vs Classificada)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='dataset', y='GAP', hue='tipo_predicao', data=df_global, palette='coolwarm')
    plt.title('Diferença Média (GAP) por Dataset e Tipo')
    plt.ylabel('Média de Features a mais que o Ótimo')
    plt.xticks(rotation=45)
    plt.legend(title='Tipo')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "GLOBAL_gap_por_tipo.png"))
    plt.close()

    # Gráfico F: Speedup Global
    resumo = df_global.groupby('dataset').agg({
        'tempo_PEAB': 'mean',
        'tempo_OPTIMO': 'mean'
    }).reset_index()
    resumo['Speedup'] = resumo['tempo_OPTIMO'] / resumo['tempo_PEAB']

    plt.figure(figsize=(12, 6))
    sns.barplot(x='dataset', y='Speedup', data=resumo, palette='Reds_d')
    plt.title('Speedup Global (Quantas vezes PEAB é mais rápido)')
    plt.ylabel('Fator de Aceleração (x)')
    plt.yscale('log') # Log porque o speedup pode ser gigante
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "GLOBAL_speedup.png"))
    plt.close()
    
    print(f"-> Todos os gráficos globais salvos em {OUTPUT_DIR}/")

def menu():
    arquivos = carregar_arquivos()
    if not arquivos:
        print(f"ERRO: Nenhum arquivo CSV encontrado em {BASE_DIR}")
        print("Rode o 'benchmark_peab.py' primeiro.")
        return

    while True:
        print("\n" + "="*40)
        print("   MENU DE ANÁLISE GRÁFICA (BENCHMARK)")
        print("="*40)
        print("Arquivos disponíveis:")
        for i, arq in enumerate(arquivos):
            print(f"  {i+1}. {os.path.basename(arq)}")
        
        print("\n  99. ANÁLISE GLOBAL (Todos os Datasets)")
        print("  0. Sair")
        
        op = input("\nEscolha uma opção: ")
        
        if op == '0': 
            print("Encerrando.")
            break
        elif op == '99':
            plotar_analise_global(arquivos)
        else:
            try:
                idx = int(op) - 1
                if 0 <= idx < len(arquivos):
                    df, nome = ler_dataset(arquivos[idx])
                    print(f"\n--- Processando {nome} ---")
                    plotar_analise_geral(df, nome)
                    plotar_analise_rejeicao(df, nome)
                    print(f"-> Concluído! Verifique a pasta {OUTPUT_DIR}")
                else:
                    print("Opção inválida!")
            except ValueError:
                print("Entrada inválida!")

if __name__ == "__main__":
    menu()