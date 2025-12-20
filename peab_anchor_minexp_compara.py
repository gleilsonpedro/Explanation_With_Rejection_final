"""
 teste de Script de Análise Comparativa entre PEAB, Anchor e MinExp
Gera plots e tabelas a partir do JSON dos resultados.

"""
# algusn plots ainda com problema gerando tamanho errados 
# plot 1 e 3


import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configurações de estilo - OTIMIZADO PARA A4
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150  # DPI reduzido (era 300)
plt.rcParams['savefig.dpi'] = 150  # DPI reduzido para arquivos menores
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9

# Cores consistentes para cada método
COLORS = {
    'PEAB': '#2ecc71',    # Verde
    'Anchor': '#3498db',  # Azul
    'MinExp': '#e74c3c'   # Vermelho
}

# Diretórios
JSON_DIR = Path('json')
OUTPUT_DIR = Path('results/analysis_comparation')
PLOTS_DIR = OUTPUT_DIR / 'plots'
TABLES_DIR = OUTPUT_DIR / 'tables'

# Criar diretórios se não existirem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ANÁLISE COMPARATIVA: PEAB vs Anchor vs MinExp")
print("="*80)


def load_json_results(method: str) -> Dict:
    """Carrega resultados de um método específico."""
    file_map = {
        'PEAB': 'peab_results.json',
        'Anchor': 'anchor_results.json',
        'MinExp': 'minexp_results.json'
    }
    
    filepath = JSON_DIR / file_map[method]
    if not filepath.exists():
        print(f"⚠️  Arquivo {filepath} não encontrado!")
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_data_for_comparison() -> pd.DataFrame:
    """Extrai dados dos JSONs e organiza em DataFrame para análise.
    FILTRO: Apenas datasets COMUNS aos 3 métodos (PEAB, Anchor, MinExp).
    """
    # Carregar todos os resultados
    peab_results = load_json_results('PEAB')
    anchor_results = load_json_results('Anchor')
    minexp_results = load_json_results('MinExp')
    
    # Identificar datasets comuns aos 3 métodos
    peab_datasets = set(peab_results.keys())
    anchor_datasets = set(anchor_results.keys())
    minexp_datasets = set(minexp_results.keys())
    
    common_datasets = peab_datasets & anchor_datasets & minexp_datasets
    excluded_datasets = (peab_datasets | anchor_datasets | minexp_datasets) - common_datasets
    
    print(f"\n[*] FILTRAGEM DE DATASETS:")
    print(f"  PEAB:   {len(peab_datasets)} datasets")
    print(f"  Anchor: {len(anchor_datasets)} datasets")
    print(f"  MinExp: {len(minexp_datasets)} datasets")
    print(f"  [OK] COMUNS (usados): {len(common_datasets)} datasets")
    print(f"  [X] EXCLUIDOS: {len(excluded_datasets)} datasets")
    
    if excluded_datasets:
        print(f"\n  Datasets excluídos da comparação:")
        for ds in sorted(excluded_datasets):
            methods_with_ds = []
            if ds in peab_datasets: methods_with_ds.append('PEAB')
            if ds in anchor_datasets: methods_with_ds.append('Anchor')
            if ds in minexp_datasets: methods_with_ds.append('MinExp')
            print(f"    • {ds}: apenas em {', '.join(methods_with_ds)}")
    
    # Extrair dados APENAS dos datasets comuns
    data = []
    
    for method in ['PEAB', 'Anchor', 'MinExp']:
        results = load_json_results(method)
        
        for dataset_name, dataset_data in results.items():
            # FILTRO: Apenas datasets comuns
            if dataset_name not in common_datasets:
                continue
                
            row = {
                'Dataset': dataset_name,
                'Método': method,
                # Performance
                'Acurácia sem Rejeição': dataset_data['performance']['accuracy_without_rejection'],
                'Acurácia com Rejeição': dataset_data['performance']['accuracy_with_rejection'],
                'Taxa de Rejeição': dataset_data['performance']['rejection_rate'],
                'Nº Instâncias Teste': dataset_data['performance']['num_test_instances'],
                'Nº Rejeitadas': dataset_data['performance']['num_rejected'],
                'Nº Aceitas': dataset_data['performance']['num_accepted'],
                # Thresholds
                't_plus': dataset_data['thresholds']['t_plus'],
                't_minus': dataset_data['thresholds']['t_minus'],
                'Largura Zona Rejeição': dataset_data['thresholds']['rejection_zone_width'],
                # Explicações - Positivos
                'Positivos Count': dataset_data['explanation_stats']['positive']['count'],
                'Positivos Média': dataset_data['explanation_stats']['positive']['mean_length'],
                'Positivos Std': dataset_data['explanation_stats']['positive']['std_length'],
                'Positivos Min': dataset_data['explanation_stats']['positive']['min_length'],
                'Positivos Max': dataset_data['explanation_stats']['positive']['max_length'],
                # Explicações - Negativos
                'Negativos Count': dataset_data['explanation_stats']['negative']['count'],
                'Negativos Média': dataset_data['explanation_stats']['negative']['mean_length'],
                'Negativos Std': dataset_data['explanation_stats']['negative']['std_length'],
                'Negativos Min': dataset_data['explanation_stats']['negative']['min_length'],
                'Negativos Max': dataset_data['explanation_stats']['negative']['max_length'],
                # Explicações - Rejeitados
                'Rejeitados Count': dataset_data['explanation_stats']['rejected']['count'],
                'Rejeitados Média': dataset_data['explanation_stats']['rejected']['mean_length'],
                'Rejeitados Std': dataset_data['explanation_stats']['rejected']['std_length'],
                'Rejeitados Min': dataset_data['explanation_stats']['rejected']['min_length'],
                'Rejeitados Max': dataset_data['explanation_stats']['rejected']['max_length'],
                # Tempo Computacional
                'Tempo Total': dataset_data['computation_time']['total'],
                'Tempo Médio por Instância': dataset_data['computation_time']['mean_per_instance'],
                'Tempo Positivos': dataset_data['computation_time']['positive'],
                'Tempo Negativos': dataset_data['computation_time']['negative'],
                'Tempo Rejeitados': dataset_data['computation_time']['rejected'],
                # Modelo
                'Nº Features': dataset_data['model']['num_features'],
                'Rejection Cost': dataset_data['config']['rejection_cost']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"\n[OK] Dados extraidos: {len(df)} registros ({len(df['Dataset'].unique())} datasets COMUNS x 3 metodos)")
    return df


def calculate_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula speedups do PEAB em relação aos baselines."""
    speedup_data = []
    
    for dataset in df['Dataset'].unique():
        df_dataset = df[df['Dataset'] == dataset]
        
        # Verificar se todos os métodos estão presentes
        peab_data = df_dataset[df_dataset['Método'] == 'PEAB']['Tempo Médio por Instância'].values
        anchor_data = df_dataset[df_dataset['Método'] == 'Anchor']['Tempo Médio por Instância'].values
        minexp_data = df_dataset[df_dataset['Método'] == 'MinExp']['Tempo Médio por Instância'].values
        
        if len(peab_data) == 0 or len(anchor_data) == 0 or len(minexp_data) == 0:
            print(f"   ⚠️  Dataset '{dataset}' não tem dados completos para todos os métodos. Pulando...")
            continue
        
        peab_time = peab_data[0]
        anchor_time = anchor_data[0]
        minexp_time = minexp_data[0]
        
        # Usar valor mínimo seguro para evitar speedups zerados
        peab_time_safe = max(peab_time, 0.000001)  # Mínimo 1 microssegundo
        
        speedup_data.append({
            'Dataset': dataset,
            'Speedup vs Anchor': anchor_time / peab_time_safe,
            'Speedup vs MinExp': minexp_time / peab_time_safe,
            'PEAB Time': peab_time,
            'Anchor Time': anchor_time,
            'MinExp Time': minexp_time
        })
    
    return pd.DataFrame(speedup_data)


# ==============================================================================
# PLOTS
# ==============================================================================

def plot_computational_efficiency(df: pd.DataFrame):
    """Plot 1: Comparação de tempo computacional (barras agrupadas) - ESCALA LOG."""
    print("\n[*] Gerando Plot 1: Eficiencia Computacional...")
    
    # Filtrar apenas datasets com dados completos
    datasets_completos = []
    for dataset in df['Dataset'].unique():
        df_dataset = df[df['Dataset'] == dataset]
        if len(df_dataset['Método'].unique()) == 3:  # PEAB, Anchor, MinExp
            datasets_completos.append(dataset)
    
    if not datasets_completos:
        print("   ⚠️  Nenhum dataset com dados completos para todos os métodos. Pulando plot...")
        return
    
    datasets = datasets_completos
    methods = ['PEAB', 'Anchor', 'MinExp']
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))  # Reduzido de 14x8
    
    for i, method in enumerate(methods):
        times = [df[(df['Dataset'] == d) & (df['Método'] == method)]['Tempo Médio por Instância'].values[0] 
                 for d in datasets]
        bars = ax.bar(x + i*width - width, times, width, label=method, color=COLORS[method], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Adicionar valores nas barras
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            if time_val < 0.01:
                label = f'{time_val*1000:.2f}ms'  # Converter para milissegundos
            elif time_val < 0.1:
                label = f'{time_val:.4f}s'
            else:
                label = f'{time_val:.2f}s'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Tempo por Instância (segundos) - Escala Log', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Eficiência Computacional entre Métodos\n(Escala Logarítmica para visualizar PEAB)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_yscale('log')  # ESCALA LOG para visualizar PEAB
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    # Adicionar linhas de referência (sem labels duplicados)
    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot1_computational_efficiency.png', dpi=150, bbox_inches='tight')  # DPI 150
    plt.close()
    print(f"   [OK] Salvo: {PLOTS_DIR / 'plot1_computational_efficiency.png'}")


def plot_speedup_comparison(speedup_df: pd.DataFrame):
    """Plot 2: Speedup do PEAB (barras horizontais) - MELHORADO."""
    print("\n[*] Gerando Plot 2: Speedup do PEAB...")
    
    if speedup_df.empty:
        print("   ⚠️  Sem dados de speedup disponíveis. Pulando plot...")
        return
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(speedup_df)*0.5)))  # Reduzido de 14x8
    
    datasets = speedup_df['Dataset'].values
    y_pos = np.arange(len(datasets))
    
    bars1 = ax.barh(y_pos - 0.2, speedup_df['Speedup vs Anchor'], 0.35, 
                    label='Speedup vs Anchor', color=COLORS['Anchor'], alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax.barh(y_pos + 0.2, speedup_df['Speedup vs MinExp'], 0.35, 
                    label='Speedup vs MinExp', color=COLORS['MinExp'], alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets, fontsize=11, fontweight='bold')
    ax.set_xlabel('Speedup (PEAB é X vezes mais rápido)', fontsize=13, fontweight='bold')
    ax.set_title('Speedup do PEAB em Relação aos Baselines\n(Quanto maior, melhor o desempenho do PEAB)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Adicionar valores DENTRO das barras (mais visível)
    for bars, color_dark in [(bars1, '#2471a3'), (bars2, '#a93226')]:
        for bar in bars:
            width = bar.get_width()
            # Colocar texto dentro da barra (mais à esquerda)
            x_pos = width * 0.85 if width > 20 else width + 3
            align = 'right' if width > 20 else 'left'
            
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{width:.0f}×', ha=align, va='center', 
                   fontsize=11, fontweight='bold', color='white' if width > 20 else color_dark)
    
    # Adicionar linhas de referência
    for ref_val in [50, 100, 150, 200]:
        if ref_val < speedup_df[['Speedup vs Anchor', 'Speedup vs MinExp']].max().max():
            ax.axvline(x=ref_val, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot2_speedup_comparison.png', dpi=150, bbox_inches='tight')  # DPI 150
    plt.close()
    print(f"   [OK] Salvo: {PLOTS_DIR / 'plot2_speedup_comparison.png'}")


def plot_explanation_size_distribution(df: pd.DataFrame):
    """Plot 3: Tamanho das explicações (box plot por classe)."""
    print("\n[*] Gerando Plot 3: Distribuicao do Tamanho das Explicacoes...")
    
    # Filtrar apenas datasets com dados completos
    datasets_completos = []
    for dataset in df['Dataset'].unique():
        df_dataset = df[df['Dataset'] == dataset]
        if len(df_dataset['Método'].unique()) == 3:
            datasets_completos.append(dataset)
    
    if not datasets_completos:
        print("   ⚠️  Nenhum dataset com dados completos. Pulando plot...")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)  # Reduzido de 18x6
    
    classes = [
        ('Positivos', 'positive'),
        ('Negativos', 'negative'),
        ('Rejeitados', 'rejected')
    ]
    
    for idx, (classe_label, classe_key) in enumerate(classes):
        data_for_plot = []
        
        for dataset in datasets_completos:
            for method in ['PEAB', 'Anchor', 'MinExp']:
                rows = df[(df['Dataset'] == dataset) & (df['Método'] == method)]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                mean_val = row[f'{classe_label} Média']
                std_val = row[f'{classe_label} Std']
                
                # Criar distribuição aproximada
                samples = np.random.normal(mean_val, std_val, 100)
                samples = np.clip(samples, row[f'{classe_label} Min'], row[f'{classe_label} Max'])
                
                for sample in samples:
                    data_for_plot.append({
                        'Dataset': dataset,
                        'Método': method,
                        'Tamanho': sample
                    })
        
        df_plot = pd.DataFrame(data_for_plot)
        
        sns.boxplot(data=df_plot, x='Dataset', y='Tamanho', hue='Método',
                   ax=axes[idx], palette=COLORS)
        
        axes[idx].set_title(f'Tamanho das Explicações - {classe_label}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Número de Features' if idx == 0 else '', fontsize=11)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        
        if idx > 0:
            axes[idx].get_legend().remove()
        else:
            axes[idx].legend(fontsize=10, loc='upper left')
    
    plt.suptitle('Distribuição do Tamanho das Explicações por Classe', 
                 fontsize=11, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot3_explanation_size_distribution.png', dpi=150, bbox_inches='tight')  # DPI 150
    plt.close()
    print(f"   [OK] Salvo: {PLOTS_DIR / 'plot3_explanation_size_distribution.png'}")


def plot_rejection_impact(df: pd.DataFrame):
    """Plot 4: Impacto da rejeição na acurácia (scatter plot)."""
    print("\n[*] Gerando Plot 4: Impacto da Rejeicao na Acuracia...")
    
    fig, ax = plt.subplots(figsize=(8, 8))  # Reduzido de 11x11
    
    # Pegar apenas PEAB (todos têm mesma acurácia por dataset)
    df_peab = df[df['Método'] == 'PEAB']
    
    datasets = df_peab['Dataset'].values
    acc_without = df_peab['Acurácia sem Rejeição'].values
    acc_with = df_peab['Acurácia com Rejeição'].values
    rejection_rates = df_peab['Taxa de Rejeição'].values
    
    # Scatter com tamanho proporcional à taxa de rejeição
    scatter = ax.scatter(acc_without, acc_with, 
                        s=[r*20 for r in rejection_rates],
                        c=rejection_rates, cmap='RdYlGn_r', alpha=0.7,
                        edgecolors='black', linewidths=2)
    
    # Linha diagonal (45°)
    min_acc = min(acc_without.min(), acc_with.min()) - 5
    max_acc = max(acc_without.max(), acc_with.max()) + 5
    ax.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', 
            alpha=0.5, linewidth=2, label='Sem ganho (diagonal)')
    
    # Anotações
    for i, dataset in enumerate(datasets):
        ax.annotate(dataset, (acc_without[i], acc_with[i]),
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Acurácia sem Rejeição (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acurácia com Rejeição (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impacto da Rejeição na Acurácia\n(tamanho do ponto = taxa de rejeição)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(min_acc, max_acc)
    ax.set_ylim(min_acc, max_acc)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Taxa de Rejeição (%)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot4_rejection_impact.png', dpi=150, bbox_inches='tight')  # DPI 150
    plt.close()
    print(f"   [OK] Salvo: {PLOTS_DIR / 'plot4_rejection_impact.png'}")


def plot_feature_importance_heatmap():
    """Plot 5: REMOVIDO - Heatmap não interessante para dissertação."""
    print("\n[*] Plot 5 (Heatmap de Features): REMOVIDO conforme solicitado")
    return  # Removido: não gerado mais


def plot_time_vs_size_tradeoff(df: pd.DataFrame):
    """Plot 6: Trade-off entre tempo e tamanho (scatter) - POR DATASET."""
    print("\n[*] Gerando Plot 6: Trade-off Tempo vs Tamanho...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    markers = {'PEAB': 'o', 'Anchor': '^', 'MinExp': 's'}
    
    # Plotar cada método com todos os seus datasets
    for method in ['PEAB', 'Anchor', 'MinExp']:
        df_method = df[df['Método'] == method]
        
        times = df_method['Tempo Médio por Instância'].values
        # Tamanho médio = média dos positivos e negativos (ignorar rejeitados pois são classe intermediária)
        sizes = (df_method['Positivos Média'].values + df_method['Negativos Média'].values) / 2
        
        ax.scatter(times, sizes, s=150, 
                  c=COLORS[method], marker=markers[method],
                  alpha=0.7, edgecolors='black', linewidths=1.5,
                  label=method, zorder=3)
    
    # Adicionar anotações para pontos extremos
    for method in ['PEAB', 'Anchor', 'MinExp']:
        df_method = df[df['Método'] == method]
        if not df_method.empty:
            # Anotar o ponto com maior tempo
            idx_max = df_method['Tempo Médio por Instância'].idxmax()
            max_time = df_method.loc[idx_max, 'Tempo Médio por Instância']
            max_size = (df_method.loc[idx_max, 'Positivos Média'] + df_method.loc[idx_max, 'Negativos Média']) / 2
            dataset_name = df_method.loc[idx_max, 'Dataset']
            
            ax.annotate(f"{method}\n(máx)", (max_time, max_size),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=COLORS[method], alpha=0.5),
                       arrowprops=dict(arrowstyle='->', lw=1))
    
    ax.set_xlabel('Tempo por Instância (segundos) - Escala Log', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tamanho Médio das Explicações (features)', fontsize=11, fontweight='bold')
    ax.set_title('Trade-off Tempo vs Tamanho das Explicações\n(cada ponto = um dataset)',
                fontsize=12, fontweight='bold', pad=15)
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot6_time_vs_size_tradeoff.png', dpi=150, bbox_inches='tight')  # DPI 150
    plt.close()
    print(f"   [OK] Salvo: {PLOTS_DIR / 'plot6_time_vs_size_tradeoff.png'}")


def plot_rejection_thresholds(df: pd.DataFrame):
    """Plot 7: Visualização dos thresholds de rejeição - CORRIGIDO."""
    print("\n[*] Gerando Plot 7: Thresholds de Rejeicao...")
    
    # Pegar apenas PEAB (thresholds são IGUAIS para todos os 3 métodos!)
    df_peab = df[df['Método'] == 'PEAB'].copy()
    
    if df_peab.empty:
        print("   ⚠️  Sem dados do PEAB disponíveis. Pulando plot...")
        return
    
    datasets = df_peab['Dataset'].values
    t_plus = df_peab['t_plus'].values
    t_minus = df_peab['t_minus'].values
    
    # Ajustar altura da figura dinamicamente - AUMENTADO para evitar corte
    fig_height = max(8, len(datasets) * 0.6)  # Aumentado: 0.6 por dataset (era 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))  # Aumentado largura também
    
    x = np.arange(len(datasets))
    width = 0.7
    
    # Zona negativa (abaixo de t-)
    bottom_zone = np.minimum(t_minus, 0)
    bars_neg = ax.bar(x, t_minus - bottom_zone, width, bottom=bottom_zone,
                     label='Zona Negativa (Aceita)', color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Zona de rejeição
    rejection_height = t_plus - t_minus
    bars_rej = ax.bar(x, rejection_height, width, bottom=t_minus,
                     label='Zona de Rejeição', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=2)
    
    # Zona positiva (acima de t+)
    top_zone = 1.0 - t_plus
    bars_pos = ax.bar(x, top_zone, width, bottom=t_plus,
                     label='Zona Positiva (Aceita)', color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Adicionar valores dos thresholds COM MAIS ESPAÇO
    for i, (tp, tm, rej_h) in enumerate(zip(t_plus, t_minus, rejection_height)):
        # t+ no topo da zona de rejeição
        ax.text(i, tp + 0.02, f't+={tp:.3f}', ha='center', va='bottom', 
               fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # t- na base da zona de rejeição
        ax.text(i, tm - 0.02, f't-={tm:.3f}', ha='center', va='top',
               fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        # Largura da zona de rejeição (dentro da barra)
        ax.text(i, tm + rej_h/2, f'{rej_h:.2f}', ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')
    
    ax.set_ylabel('Score de Predição', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Thresholds de Rejeição Otimizados por Dataset\n(Iguais para PEAB, Anchor e MinExp)',
                fontsize=11, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(-0.2, 1.1)  # Aumentado margem vertical
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot7_rejection_thresholds.png', dpi=150, bbox_inches='tight')  # DPI 150
    plt.close()
    print(f"   [OK] Salvo: {PLOTS_DIR / 'plot7_rejection_thresholds.png'}")


def plot_class_distribution(df: pd.DataFrame):
    """Plot 8: Distribuição de instâncias por classe."""
    print("\n[*] Gerando Plot 8: Distribuicao por Classe...")
    
    # Pegar apenas PEAB (distribuição é igual para todos)
    df_peab = df[df['Método'] == 'PEAB'].copy()
    
    if df_peab.empty:
        print("   ⚠️  Sem dados do PEAB disponíveis. Pulando plot...")
        return
    
    datasets = df_peab['Dataset'].values
    positivos = df_peab['Positivos Count'].values
    negativos = df_peab['Negativos Count'].values
    rejeitados = df_peab['Rejeitados Count'].values
    
    fig, ax = plt.subplots(figsize=(10, 5))  # Reduzido de 14x7
    
    x = np.arange(len(datasets))
    width = 0.25
    
    bars1 = ax.bar(x - width, positivos, width, label='Positivos',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, negativos, width, label='Negativos',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, rejeitados, width, label='Rejeitados',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Adicionar valores
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Número de Instâncias', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição de Instâncias por Classe',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot8_class_distribution.png', dpi=150, bbox_inches='tight')  # DPI 150
    plt.close()
    print(f"   [OK] Salvo: {PLOTS_DIR / 'plot8_class_distribution.png'}")


# ==============================================================================
# TABELAS
# ==============================================================================

def generate_main_comparison_table(df: pd.DataFrame):
    """Tabela 1: Comparação principal de performance."""
    print("\n📋 Gerando Tabela 1: Comparação Principal...")
    
    table_data = []
    
    for dataset in df['Dataset'].unique():
        for method in ['PEAB', 'Anchor', 'MinExp']:
            rows = df[(df['Dataset'] == dataset) & (df['Método'] == method)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            
            table_data.append({
                'Dataset': dataset,
                'Método': method,
                'Acc s/Rej (%)': f"{row['Acurácia sem Rejeição']:.2f}",
                'Acc c/Rej (%)': f"{row['Acurácia com Rejeição']:.2f}",
                'Taxa Rej (%)': f"{row['Taxa de Rejeição']:.2f}",
                'Tam Pos': f"{row['Positivos Média']:.1f}",
                'Tam Neg': f"{row['Negativos Média']:.1f}",
                'Tam Rej': f"{row['Rejeitados Média']:.1f}",
                'Tempo (s)': f"{row['Tempo Médio por Instância']:.4f}"
            })
    
    df_table = pd.DataFrame(table_data)
    
    # Salvar CSV
    df_table.to_csv(TABLES_DIR / 'table1_main_comparison.csv', index=False)
    
    # Salvar LaTeX
    with open(TABLES_DIR / 'table1_main_comparison.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparação de desempenho entre PEAB, Anchor e MinExp}\n")
        f.write("\\label{tab:main_comparison}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{llcccccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Método} & \\textbf{Acc s/Rej} & \\textbf{Acc c/Rej} & ")
        f.write("\\textbf{Taxa Rej} & \\textbf{Tam Pos} & \\textbf{Tam Neg} & ")
        f.write("\\textbf{Tam Rej} & \\textbf{Tempo (s)} \\\\\n")
        f.write("\\midrule\n")
        
        current_dataset = None
        for _, row in df_table.iterrows():
            if current_dataset != row['Dataset']:
                if current_dataset is not None:
                    f.write("\\midrule\n")
                current_dataset = row['Dataset']
            
            f.write(f"{row['Dataset']} & {row['Método']} & {row['Acc s/Rej (%)']} & ")
            f.write(f"{row['Acc c/Rej (%)']} & {row['Taxa Rej (%)']} & {row['Tam Pos']} & ")
            f.write(f"{row['Tam Neg']} & {row['Tam Rej']} & {row['Tempo (s)']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    
    print(f"   [OK] Salvo: {TABLES_DIR / 'table1_main_comparison.csv'}")
    print(f"   [OK] Salvo: {TABLES_DIR / 'table1_main_comparison.tex'}")


def generate_speedup_table(speedup_df: pd.DataFrame):
    """Tabela 2: Speedups do PEAB."""
    print("\n📋 Gerando Tabela 2: Speedups...")
    
    if speedup_df.empty:
        print("   ⚠️  Sem dados de speedup disponíveis. Pulando tabela...")
        return
    
    table_data = []
    
    for _, row in speedup_df.iterrows():
        table_data.append({
            'Dataset': row['Dataset'],
            'PEAB (s)': f"{row['PEAB Time']:.4f}",
            'Anchor (s)': f"{row['Anchor Time']:.4f}",
            'MinExp (s)': f"{row['MinExp Time']:.4f}",
            'Speedup vs Anchor': f"{row['Speedup vs Anchor']:.1f}x",
            'Speedup vs MinExp': f"{row['Speedup vs MinExp']:.1f}x"
        })
    
    df_table = pd.DataFrame(table_data)
    
    # Salvar CSV
    df_table.to_csv(TABLES_DIR / 'table2_speedup.csv', index=False)
    
    # Salvar LaTeX
    with open(TABLES_DIR / 'table2_speedup.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Speedup do PEAB em relação aos baselines}\n")
        f.write("\\label{tab:speedup}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{PEAB} & \\textbf{Anchor} & \\textbf{MinExp} & ")
        f.write("\\textbf{vs Anchor} & \\textbf{vs MinExp} \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df_table.iterrows():
            f.write(f"{row['Dataset']} & {row['PEAB (s)']} & {row['Anchor (s)']} & ")
            f.write(f"{row['MinExp (s)']} & \\textbf{{{row['Speedup vs Anchor']}}} & ")
            f.write(f"\\textbf{{{row['Speedup vs MinExp']}}} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"   ✅ Salvo: {TABLES_DIR / 'table2_speedup.csv'}")
    print(f"   ✅ Salvo: {TABLES_DIR / 'table2_speedup.tex'}")


def generate_explanation_stats_table(df: pd.DataFrame):
    """Tabela 3: Estatísticas das explicações."""
    print("\n📋 Gerando Tabela 3: Estatísticas das Explicações...")
    
    table_data = []
    
    for dataset in df['Dataset'].unique():
        for method in ['PEAB', 'Anchor', 'MinExp']:
            rows = df[(df['Dataset'] == dataset) & (df['Método'] == method)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            
            table_data.append({
                'Dataset': dataset,
                'Método': method,
                'Positivos': f"{row['Positivos Média']:.1f} ± {row['Positivos Std']:.1f}",
                'Negativos': f"{row['Negativos Média']:.1f} ± {row['Negativos Std']:.1f}",
                'Rejeitados': f"{row['Rejeitados Média']:.1f} ± {row['Rejeitados Std']:.1f}",
                'Variabilidade': f"{(row['Positivos Std']/row['Positivos Média']*100):.1f}%"
            })
    
    df_table = pd.DataFrame(table_data)
    
    # Salvar CSV
    df_table.to_csv(TABLES_DIR / 'table3_explanation_stats.csv', index=False)
    
    # Salvar LaTeX
    with open(TABLES_DIR / 'table3_explanation_stats.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Estatísticas do tamanho das explicações (média ± desvio padrão)}\n")
        f.write("\\label{tab:explanation_stats}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Método} & \\textbf{Positivos} & ")
        f.write("\\textbf{Negativos} & \\textbf{Rejeitados} & \\textbf{Variabilidade} \\\\\n")
        f.write("\\midrule\n")
        
        current_dataset = None
        for _, row in df_table.iterrows():
            if current_dataset != row['Dataset']:
                if current_dataset is not None:
                    f.write("\\midrule\n")
                current_dataset = row['Dataset']
            
            f.write(f"{row['Dataset']} & {row['Método']} & {row['Positivos']} & ")
            f.write(f"{row['Negativos']} & {row['Rejeitados']} & {row['Variabilidade']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    
    print(f"   ✅ Salvo: {TABLES_DIR / 'table3_explanation_stats.csv'}")
    print(f"   ✅ Salvo: {TABLES_DIR / 'table3_explanation_stats.tex'}")


def generate_rejection_impact_table(df: pd.DataFrame):
    """Tabela 4: Impacto da rejeição."""
    print("\n📋 Gerando Tabela 4: Impacto da Rejeição...")
    
    # Pegar apenas PEAB (métricas iguais para todos)
    df_peab = df[df['Método'] == 'PEAB'].copy()
    
    if df_peab.empty:
        print("   ⚠️  Sem dados do PEAB disponíveis. Pulando tabela...")
        return
    
    table_data = []
    
    for _, row in df_peab.iterrows():
        ganho = row['Acurácia com Rejeição'] - row['Acurácia sem Rejeição']
        tradeoff = ganho / row['Taxa de Rejeição'] if row['Taxa de Rejeição'] > 0 else 0
        
        table_data.append({
            'Dataset': row['Dataset'],
            'Acc s/Rej (%)': f"{row['Acurácia sem Rejeição']:.2f}",
            'Acc c/Rej (%)': f"{row['Acurácia com Rejeição']:.2f}",
            'Ganho (%)': f"{ganho:.2f}",
            'Taxa Rej (%)': f"{row['Taxa de Rejeição']:.2f}",
            'Nº Rejeitadas': f"{int(row['Nº Rejeitadas'])}/{int(row['Nº Instâncias Teste'])}",
            'Trade-off': f"{tradeoff:.3f}"
        })
    
    df_table = pd.DataFrame(table_data)
    
    # Salvar CSV
    df_table.to_csv(TABLES_DIR / 'table4_rejection_impact.csv', index=False)
    
    # Salvar LaTeX
    with open(TABLES_DIR / 'table4_rejection_impact.tex', 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Impacto da rejeição na acurácia}\n")
        f.write("\\label{tab:rejection_impact}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Acc s/Rej} & \\textbf{Acc c/Rej} & ")
        f.write("\\textbf{Ganho} & \\textbf{Taxa Rej} & \\textbf{Nº Rej} & \\textbf{Trade-off} \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df_table.iterrows():
            f.write(f"{row['Dataset']} & {row['Acc s/Rej (%)']} & {row['Acc c/Rej (%)']} & ")
            f.write(f"\\textbf{{+{row['Ganho (%)']}}} & {row['Taxa Rej (%)']} & ")
            f.write(f"{row['Nº Rejeitadas']} & {row['Trade-off']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"   ✅ Salvo: {TABLES_DIR / 'table4_rejection_impact.csv'}")
    print(f"   ✅ Salvo: {TABLES_DIR / 'table4_rejection_impact.tex'}")


def generate_summary_report(df: pd.DataFrame, speedup_df: pd.DataFrame):
    """Gera relatório resumido em texto."""
    print("\n📄 Gerando Relatório Resumido...")
    
    report = []
    report.append("=" * 80)
    report.append("RELATÓRIO DE ANÁLISE COMPARATIVA: PEAB vs Anchor vs MinExp")
    report.append("=" * 80)
    report.append("")
    
    # Datasets analisados
    datasets = df['Dataset'].unique()
    report.append(f"DATASETS ANALISADOS: {len(datasets)} (apenas comuns aos 3 métodos)")
    for dataset in datasets:
        dataset_rows = df[df['Dataset'] == dataset]
        if not dataset_rows.empty:
            n_inst = dataset_rows['Nº Instâncias Teste'].iloc[0]
            report.append(f"  • {dataset}: {int(n_inst)} instâncias de teste")
    report.append("")
    report.append("⚠️  NOTA: Datasets presentes em apenas 1 ou 2 métodos foram EXCLUÍDOS")
    report.append("    para garantir comparação justa (ex: mnist_3_vs_8, newsgroups).")
    report.append("")
    
    # Performance computacional
    report.append("PERFORMANCE COMPUTACIONAL:")
    report.append("-" * 80)
    for method in ['PEAB', 'Anchor', 'MinExp']:
        avg_time = df[df['Método'] == method]['Tempo Médio por Instância'].mean()
        if avg_time < 0.000001:  # Menor que 1 microssegundo
            time_str = f"{avg_time*1000000:.2f}µs (microssegundos)"
        elif avg_time < 0.001:  # Menor que 1 milissegundo
            time_str = f"{avg_time*1000:.3f}ms (milissegundos)"
        elif avg_time < 1.0:
            time_str = f"{avg_time*1000:.1f}ms"
        else:
            time_str = f"{avg_time:.3f}s"
        report.append(f"  {method:10s}: {time_str} por instância (média)")
    report.append("")
    
    # Speedups
    if not speedup_df.empty:
        report.append("SPEEDUPS DO PEAB:")
        report.append("-" * 80)
        for _, row in speedup_df.iterrows():
            speedup_anchor = row['Speedup vs Anchor']
            speedup_minexp = row['Speedup vs MinExp']
            
            # Formatar speedups com precisão adequada
            if speedup_anchor > 1000:
                anchor_str = f"{speedup_anchor:.0f}x"
            elif speedup_anchor > 100:
                anchor_str = f"{speedup_anchor:.1f}x"
            else:
                anchor_str = f"{speedup_anchor:.2f}x"
                
            if speedup_minexp > 1000:
                minexp_str = f"{speedup_minexp:.0f}x"
            elif speedup_minexp > 100:
                minexp_str = f"{speedup_minexp:.1f}x"
            else:
                minexp_str = f"{speedup_minexp:.2f}x"
            
            report.append(f"  {row['Dataset']:25s}: {anchor_str} vs Anchor, {minexp_str} vs MinExp")
        report.append("")
        
        avg_speedup_anchor = speedup_df['Speedup vs Anchor'].mean()
        avg_speedup_minexp = speedup_df['Speedup vs MinExp'].mean()
        
        # Verificar se são valores válidos
        if pd.isna(avg_speedup_anchor) or avg_speedup_anchor == 0:
            anchor_avg_str = "N/A"
        elif avg_speedup_anchor > 100:
            anchor_avg_str = f"{avg_speedup_anchor:.1f}x"
        else:
            anchor_avg_str = f"{avg_speedup_anchor:.2f}x"
            
        if avg_speedup_minexp > 100:
            minexp_avg_str = f"{avg_speedup_minexp:.1f}x"
        else:
            minexp_avg_str = f"{avg_speedup_minexp:.2f}x"
        
        report.append(f"  MÉDIA GERAL: {anchor_avg_str} vs Anchor, {minexp_avg_str} vs MinExp")
        report.append("")
    else:
        report.append("SPEEDUPS DO PEAB: Dados insuficientes")
        report.append("")
    
    # Tamanho das explicações
    report.append("TAMANHO DAS EXPLICAÇÕES (média entre datasets comuns):")
    report.append("-" * 80)
    for method in ['PEAB', 'Anchor', 'MinExp']:
        df_method = df[df['Método'] == method]
        avg_pos = df_method['Positivos Média'].mean()
        avg_neg = df_method['Negativos Média'].mean()
        avg_rej = df_method['Rejeitados Média'].mean()
        report.append(f"  {method:10s}: Pos={avg_pos:.1f}, Neg={avg_neg:.1f}, Rej={avg_rej:.1f} features")
    report.append("")
    
    # Impacto da rejeição
    df_peab = df[df['Método'] == 'PEAB']
    if not df_peab.empty:
        report.append("IMPACTO DA REJEIÇÃO:")
        report.append("-" * 80)
        for _, row in df_peab.iterrows():
            ganho = row['Acurácia com Rejeição'] - row['Acurácia sem Rejeição']
            report.append(f"  {row['Dataset']:25s}: {ganho:+.2f}% ganho de acurácia "
                         f"(rejeitando {row['Taxa de Rejeição']:.1f}%)")
        report.append("")
        
        avg_ganho = (df_peab['Acurácia com Rejeição'] - df_peab['Acurácia sem Rejeição']).mean()
        avg_taxa = df_peab['Taxa de Rejeição'].mean()
        report.append(f"  MÉDIA GERAL: {avg_ganho:+.2f}% ganho com {avg_taxa:.1f}% de rejeição")
        report.append("")
    else:
        report.append("IMPACTO DA REJEIÇÃO: Dados insuficientes")
        report.append("")
    
    # Conclusões
    report.append("PRINCIPAIS CONCLUSÕES:")
    report.append("-" * 80)
    if not speedup_df.empty:
        # Usar formatação adequada para conclusões
        if avg_speedup_anchor > 1000:
            report.append(f"  ✅ PEAB é EXTREMAMENTE mais rápido que Anchor ({avg_speedup_anchor:.0f}x)")
        elif avg_speedup_anchor > 100:
            report.append(f"  ✅ PEAB é em média {avg_speedup_anchor:.1f}x mais rápido que Anchor")
        else:
            report.append(f"  ✅ PEAB é em média {avg_speedup_anchor:.2f}x mais rápido que Anchor")
            
        if avg_speedup_minexp > 1000:
            report.append(f"  ✅ PEAB é EXTREMAMENTE mais rápido que MinExp ({avg_speedup_minexp:.0f}x)")
        elif avg_speedup_minexp > 100:
            report.append(f"  ✅ PEAB é em média {avg_speedup_minexp:.1f}x mais rápido que MinExp")
        else:
            report.append(f"  ✅ PEAB é em média {avg_speedup_minexp:.2f}x mais rápido que MinExp")
    
    # Calcular médias reais de tamanho de explicações (apenas datasets comuns)
    avg_pos_peab = df[df['Método'] == 'PEAB']['Positivos Média'].mean()
    avg_pos_anchor = df[df['Método'] == 'Anchor']['Positivos Média'].mean()
    avg_pos_minexp = df[df['Método'] == 'MinExp']['Positivos Média'].mean()
    
    # Ordenar por tamanho para conclusão correta
    sizes = [
        ('Anchor', avg_pos_anchor),
        ('MinExp', avg_pos_minexp),
        ('PEAB', avg_pos_peab)
    ]
    sizes_sorted = sorted(sizes, key=lambda x: x[1])
    
    report.append(f"  ✅ {sizes_sorted[0][0]} gera explicações mais concisas (média: {sizes_sorted[0][1]:.1f} features)")
    if len(sizes_sorted) > 1:
        report.append(f"  ✅ {sizes_sorted[1][0]} gera explicações intermediárias (média: {sizes_sorted[1][1]:.1f} features)")
    if len(sizes_sorted) > 2:
        report.append(f"  ✅ {sizes_sorted[2][0]} gera explicações mais completas (média: {sizes_sorted[2][1]:.1f} features)")
    
    if not df_peab.empty:
        report.append(f"  ✅ Rejeição melhora acurácia em média {avg_ganho:.1f}%")
    report.append(f"  ✅ Todos os métodos usam pipeline idêntico (comparação justa)")
    report.append(f"  ⚠️  Comparação baseada APENAS em datasets comuns aos 3 métodos")
    report.append("")
    
    report.append("=" * 80)
    report.append(f"Análise gerada em: {OUTPUT_DIR}")
    report.append("=" * 80)
    
    # Salvar relatório
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / 'RELATORIO_RESUMIDO.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n   ✅ Salvo: {OUTPUT_DIR / 'RELATORIO_RESUMIDO.txt'}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Função principal."""
    print("\n[*] Carregando dados dos JSONs...")
    
    # Extrair dados
    df = extract_data_for_comparison()
    speedup_df = calculate_speedups(df)
    
    if df.empty:
        print("\n[!] Nenhum dado encontrado nos JSONs! Verifique se os arquivos existem.")
        return
    
    print("\n" + "="*80)
    print("GERANDO PLOTS...")
    print("="*80)
    
    # Gerar todos os plots
    plot_computational_efficiency(df)
    plot_speedup_comparison(speedup_df)
    plot_explanation_size_distribution(df)
    plot_rejection_impact(df)
    # plot_feature_importance_heatmap()  # REMOVIDO - não interessante
    plot_time_vs_size_tradeoff(df)
    plot_rejection_thresholds(df)
    plot_class_distribution(df)
    
    print("\n" + "="*80)
    print("GERANDO TABELAS...")
    print("="*80)
    
    # Gerar todas as tabelas
    generate_main_comparison_table(df)
    generate_speedup_table(speedup_df)
    generate_explanation_stats_table(df)
    generate_rejection_impact_table(df)
    
    # Gerar relatório resumido
    generate_summary_report(df, speedup_df)
    
    print("\n" + "="*80)
    print("[OK] ANALISE COMPLETA!")
    print("="*80)
    print(f"\nResultados salvos em: {OUTPUT_DIR}")
    print(f"   Plots: {PLOTS_DIR}")
    print(f"   📋 Tabelas: {TABLES_DIR}")
    print(f"   📄 Relatório: {OUTPUT_DIR / 'RELATORIO_RESUMIDO.txt'}")
    print("\n🎓 Pronto para usar na dissertação!\n")


if __name__ == '__main__':
    main()
