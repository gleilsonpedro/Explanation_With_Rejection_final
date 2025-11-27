import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração de Estilo
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6)})

# Caminho onde os CSVs estão (ajuste se necessário)
BASE_DIR = 'results/benchmark'
OUTPUT_DIR = 'results/plots'

def carregar_arquivos():
    """Busca todos os arquivos bench_*.csv no diretório."""
    padrao = os.path.join(BASE_DIR, "bench_*.csv")
    arquivos = glob.glob(padrao)
    arquivos.sort()
    return arquivos

def ler_dataset(caminho):
    """Lê um CSV e extrai o nome do dataset do arquivo."""
    df = pd.read_csv(caminho)
    # Extrai nome do arquivo: bench_iris.csv -> iris
    nome_dataset = os.path.basename(caminho).replace('bench_', '').replace('.csv', '')
    df['dataset'] = nome_dataset
    return df, nome_dataset

def plotar_analise_individual(df, nome_dataset):
    """Gera gráficos para um único dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n--- Gerando gráficos para: {nome_dataset} ---")

    # 1. Distribuição do GAP (Qualidade)
    plt.figure()
    ax = sns.countplot(x='GAP', data=df, palette='viridis')
    plt.title(f'Distribuição do GAP - {nome_dataset}\n(0 = Otimalidade Perfeita)')
    plt.xlabel('Diferença de Tamanho (PEAB - Ótimo)')
    plt.ylabel('Quantidade de Instâncias')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome_dataset}_gap_hist.png"))
    plt.close()

    # 2. Comparativo de Tempo (Scatter)
    plt.figure()
    plt.scatter(df['tamanho_OPTIMO'], df['tempo_OPTIMO'], label='Otimização (Exata)', alpha=0.6, color='red')
    plt.scatter(df['tamanho_PEAB'], df['tempo_PEAB'], label='PEAB (Heurística)', alpha=0.6, color='blue')
    plt.xlabel('Tamanho da Explicação (|S|)')
    plt.ylabel('Tempo de Execução (s)')
    plt.title(f'Custo Computacional: PEAB vs Otimização - {nome_dataset}')
    plt.legend()
    plt.yscale('log') # Escala Log ajuda a ver a diferença se for grande
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome_dataset}_time_scatter.png"))
    plt.close()

    print(f"-> Gráficos salvos em {OUTPUT_DIR}/")

def plotar_analise_global(arquivos):
    """Gera gráficos comparativos entre todos os datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n--- Gerando Análise Global (Comparativa) ---")
    
    lista_dfs = []
    for arq in arquivos:
        df, _ = ler_dataset(arq)
        lista_dfs.append(df)
    
    df_global = pd.concat(lista_dfs, ignore_index=True)

    # Calcular métricas agregadas por dataset
    resumo = df_global.groupby('dataset').agg({
        'is_optimal': lambda x: np.mean(x) * 100,
        'tempo_PEAB': 'mean',
        'tempo_OPTIMO': 'mean',
        'GAP': 'mean'
    }).reset_index()
    
    resumo['Speedup'] = resumo['tempo_OPTIMO'] / resumo['tempo_PEAB']

    # 1. Taxa de Otimalidade por Dataset (Barplot)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='dataset', y='is_optimal', data=resumo, palette='Blues_d')
    plt.ylim(0, 105) # Margem para o 100%
    plt.axhline(100, color='green', linestyle='--', alpha=0.5)
    plt.title('Taxa de Otimalidade por Dataset (% de Casos onde GAP=0)')
    plt.ylabel('% de Instâncias Ótimas')
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "GLOBAL_otimalidade.png"))
    plt.close()

    # 2. Speedup Médio por Dataset (Barplot)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='dataset', y='Speedup', data=resumo, palette='Reds_d')
    plt.title('Speedup Médio (Quantas vezes PEAB é mais rápido que Otimização)')
    plt.ylabel('Speedup (x)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "GLOBAL_speedup.png"))
    plt.close()

    # 3. Boxplot de Tempos (Log Scale)
    # Reestruturar dados para o seaborn boxplot (melt)
    df_melt = df_global.melt(id_vars=['dataset'], value_vars=['tempo_PEAB', 'tempo_OPTIMO'], 
                             var_name='Metodo', value_name='Tempo')
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='dataset', y='Tempo', hue='Metodo', data=df_melt)
    plt.yscale('log')
    plt.title('Distribuição de Tempo de Execução (Escala Log)')
    plt.ylabel('Tempo (s) - Log Scale')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "GLOBAL_tempos_boxplot.png"))
    plt.close()

    print(f"-> Gráficos Globais salvos em {OUTPUT_DIR}/")

def menu():
    arquivos = carregar_arquivos()
    
    if not arquivos:
        print(f"ERRO: Nenhum arquivo CSV encontrado em {BASE_DIR}")
        print("Certifique-se de rodar o benchmark_peab.py primeiro.")
        return

    while True:
        print("\n" + "="*40)
        print("   MENU DE ANÁLISE GRÁFICA (BENCHMARK)")
        print("="*40)
        print("Arquivos disponíveis:")
        for i, arq in enumerate(arquivos):
            nome = os.path.basename(arq)
            print(f"  {i+1}. {nome}")
        
        print("\n  99. TODOS OS DATASETS (Análise Global)")
        print("  0. Sair")
        
        opcao = input("\nEscolha uma opção: ")
        
        if opcao == '0':
            print("Saindo...")
            break
        elif opcao == '99':
            plotar_analise_global(arquivos)
        else:
            try:
                idx = int(opcao) - 1
                if 0 <= idx < len(arquivos):
                    df, nome = ler_dataset(arquivos[idx])
                    plotar_analise_individual(df, nome)
                else:
                    print("Opção inválida!")
            except ValueError:
                print("Entrada inválida!")

if __name__ == "__main__":
    menu()