import pandas as pd
import glob
import os
import sys

# Configurações
INPUT_DIR = 'results/benchmark/bench_csv'
OUTPUT_FILE = 'results/benchmark/RELATORIO_GERAL_MESTRADO.txt'

def carregar_dados_globais():
    padrao = os.path.join(INPUT_DIR, "bench_*.csv")
    arquivos = glob.glob(padrao)
    
    if not arquivos:
        print(f"[ERRO] Nenhum arquivo encontrado em {INPUT_DIR}")
        return None

    lista_dfs = []
    print(f"--- Carregando {len(arquivos)} datasets ---")
    
    for arq in arquivos:
        df = pd.read_csv(arq)
        nome = os.path.basename(arq).replace('bench_', '').replace('.csv', '').upper()
        df['dataset'] = nome
        if 'tipo_predicao' not in df.columns: df['tipo_predicao'] = 'N/A'
        lista_dfs.append(df)
        
    return pd.concat(lista_dfs, ignore_index=True)

def interpretar_speedup(row):
    t_peab = row['tempo_PEAB'] if row['tempo_PEAB'] > 0 else 1e-9
    t_opt = row['tempo_OPTIMO']
    sp = t_opt / t_peab
    if sp >= 1.0: return f"{sp:.2f}x (PEAB Vence)"
    else: 
        inv = 1 / sp if sp > 0 else 0
        return f"{sp:.2f}x (OPT {inv:.1f}x mais rápido)"

def gerar_relatorio_final():
    df_global = carregar_dados_globais()
    if df_global is None: return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Recalcula erro relativo linha a linha para precisão
    # Evita divisão por zero trocando 0 por 1 (apenas para cálculo do erro, não afeta média de tamanho)
    df_global['erro_relativo'] = (df_global['GAP'] / df_global['tamanho_OPTIMO'].replace(0, 1)) * 100

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("RELATÓRIO CONSOLIDADO: PEAB vs OTIMIZAÇÃO (JUÍZO FINAL)\n")
        f.write("="*100 + "\n\n")
        
        # --- TABELA 1: POR DATASET ---
        f.write("1. DESEMPENHO POR DATASET (Ordenado por Gap)\n")
        f.write("-" * 100 + "\n")
        
        resumo_ds = df_global.groupby('dataset').agg({
            'id': 'count', 'is_optimal': 'mean', 'GAP': 'mean', 'erro_relativo': 'mean',
            'tempo_PEAB': 'mean', 'tempo_OPTIMO': 'mean'
        }).reset_index()
        
        resumo_ds['is_optimal'] = (resumo_ds['is_optimal'] * 100).map('{:.1f}%'.format)
        resumo_ds['erro_relativo'] = resumo_ds['erro_relativo'].map('{:.2f}%'.format)
        resumo_ds['Speedup'] = resumo_ds.apply(interpretar_speedup, axis=1)
        resumo_ds = resumo_ds.sort_values(by='GAP', ascending=False)
        
        # Seleção de colunas
        cols = ['dataset', 'id', 'is_optimal', 'GAP', 'erro_relativo', 'tempo_PEAB', 'tempo_OPTIMO', 'Speedup']
        cols_renomeadas = ['Dataset', 'Qtd', '% Ótimo', 'Gap Médio', 'Erro Rel.(%)', 'T. PEAB', 'T. OPT', 'Speedup']
        resumo_ds.columns = cols_renomeadas
        
        f.write(resumo_ds.to_string(index=False))
        f.write("\n\n")

        # --- TABELA 2: POR TIPO (COM TAMANHO MÉDIO PARA EXPLICAR O ERRO) ---
        f.write("-" * 100 + "\n")
        f.write("2. ANÁLISE DO GARGALO (MACRO)\n")
        f.write("-" * 100 + "\n")
        
        resumo_tipo = df_global.groupby('tipo_predicao').agg({
            'id': 'count', 
            'is_optimal': 'mean', 
            'GAP': 'mean', 
            'tamanho_OPTIMO': 'mean', # <--- ADICIONADO PARA CLAREZA
            'erro_relativo': 'mean',
            'tempo_PEAB': 'mean', 
            'tempo_OPTIMO': 'mean'
        }).reset_index()
        
        resumo_tipo['is_optimal'] = (resumo_tipo['is_optimal'] * 100).map('{:.2f}%'.format)
        resumo_tipo['erro_relativo'] = resumo_tipo['erro_relativo'].map('{:.2f}%'.format)
        resumo_tipo['Speedup'] = resumo_tipo.apply(interpretar_speedup, axis=1)
        
        # Renomeia para ficar claro
        resumo_tipo.columns = ['Tipo', 'Qtd', '% Ótimo', 'Gap Médio', 'Tam. Médio Ótimo', 'Erro Rel.(%)', 'T. PEAB', 'T. OPT', 'Speedup']
        
        f.write(resumo_tipo.to_string(index=False))
        f.write("\n\n")
        f.write("NOTA: 'Erro Rel.(%)' alto com 'Gap Médio' baixo indica que a explicação ótima é muito pequena.\n")
        f.write("      (Ex: Errar 1 feature numa explicação de tamanho 2 representa 50% de erro).\n")

    print(f"\n[SUCESSO] Relatório ajustado gerado em: {OUTPUT_FILE}")

if __name__ == '__main__':
    gerar_relatorio_final()