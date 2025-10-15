import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from typing import Dict, Any

# Criar diretório para salvar os plots
os.makedirs("analysis_output", exist_ok=True)

# --- FUNÇÕES AUXILIARES ---

def load_data(filepath="comparative_results.json"):
    """Carrega os dados do arquivo JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERRO: O arquivo {filepath} não foi encontrado.")
        return None

def extract_metric(data, metric_path: str):
    """
    Extrai uma métrica específica do dicionário de dados usando um caminho.
    Adaptado para lidar com diferentes nomes de campos entre métodos.
    """
    results = {}
    for method, datasets in data.items():
        results[method] = {}
        for dataset_name, values in datasets.items():
            try:
                # Navega pelo caminho da métrica
                keys = metric_path.split('.')
                metric_value = values
                
                # Adaptação para diferentes nomes de campos
                for key in keys:
                    if key == "mean_length" and method == "anchor":
                        # Para o Anchor, "mean_length" se chama "media"
                        key = "media"
                    elif key == "count" and method == "anchor":
                        # Para o Anchor, "count" se chama "instancias"
                        key = "instancias"
                    elif key == "std_length" and method == "anchor":
                        # Para o Anchor, "std_length" se chama "std_dev"
                        key = "std_dev"
                    elif key == "min_length" and method == "anchor":
                        # Para o Anchor, "min_length" se chama "min"
                        key = "min"
                    elif key == "max_length" and method == "anchor":
                        # Para o Anchor, "max_length" se chama "max"
                        key = "max"
                    
                    metric_value = metric_value[key]
                
                # Converte para porcentagem se for um valor entre 0 e 1
                if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1 and "time" not in metric_path:
                    results[method][dataset_name] = metric_value * 100
                else:
                    results[method][dataset_name] = metric_value
                    
            except (KeyError, TypeError):
                results[method][dataset_name] = 0  # Assume 0 se a métrica não existir
    return results

def plot_metric_horizontally(metric_data, title, xlabel, filename):
    """
    Cria um gráfico de barras HORIZONTAL para uma determinada métrica, com anotações.
    """
    df = pd.DataFrame(metric_data).sort_index()
    
    # Remove métodos que não têm dados (todos zeros)
    df = df.loc[:, (df != 0).any(axis=0)]
    
    if df.empty:
        print(f"⚠️  Nenhum dado para plotar: {title}")
        return
    
    # Define uma paleta de cores para consistência
    sns.set_palette("Set2")
    colors = sns.color_palette("Set2", n_colors=len(df.columns))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind='barh', ax=ax, color=colors, width=0.8)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Dataset", fontsize=12)
    plt.legend(title='Método')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adiciona os valores no final de cada barra
    for bar in ax.patches:
        width = bar.get_width()
        if width > 0:  # Só adiciona texto se o valor for maior que zero
            x_pos = width + (plt.xlim()[1] * 0.01)
            y_pos = bar.get_y() + bar.get_height() / 2
            
            label = f"{width:.2f}"
            if '%' in xlabel:
                label += '%'
                
            ax.text(x_pos, y_pos, label, va='center', ha='left', fontsize=9)

    # Ajusta o limite do eixo x para dar espaço para os rótulos
    if df.values.max() > 0:
        plt.xlim(right=plt.xlim()[1] * 1.15)
    
    plt.tight_layout()
    
    # Salva a figura
    plt.savefig(f"analysis_output/{filename}", dpi=300, bbox_inches='tight')
    plt.close()  # Fecha a figura sem mostrar
    print(f"✅ Gráfico salvo como: analysis_output/{filename}")

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---

if __name__ == "__main__":
    data = load_data()
    
    if data:
        print("📊 Gerando gráficos...")
        
        # 1. Taxa de Rejeição
        rejection_rates = extract_metric(data, 'performance.rejection_rate')
        plot_metric_horizontally(rejection_rates, 
                               'Taxa de Rejeição por Dataset', 
                               'Rejeição (%)',
                               'rejection_rate.png')

        # 2. Acurácia COM Rejeição
        accuracy_with_rejection = extract_metric(data, 'performance.accuracy_with_rejection')
        plot_metric_horizontally(accuracy_with_rejection, 
                               'Acurácia (com Rejeição) por Dataset', 
                               'Acurácia (%)',
                               'accuracy_with_rejection.png')
        
        # 3. Acurácia SEM Rejeição
        accuracy_without_rejection = extract_metric(data, 'performance.accuracy_without_rejection')
        plot_metric_horizontally(accuracy_without_rejection, 
                               'Acurácia (sem Rejeição) por Dataset', 
                               'Acurácia (%)',
                               'accuracy_without_rejection.png')

        # 4. Tamanho Médio da Explicação - Classe Positiva
        explanation_length_pos = extract_metric(data, 'explanation_stats.positive.mean_length')
        plot_metric_horizontally(explanation_length_pos, 
                               'Tamanho Médio da Explicação (Classe Positiva)', 
                               'Nº Médio de Regras',
                               'explanation_length_positive.png')

        # 5. Tamanho Médio da Explicação - Classe Negativa
        explanation_length_neg = extract_metric(data, 'explanation_stats.negative.mean_length')
        plot_metric_horizontally(explanation_length_neg, 
                               'Tamanho Médio da Explicação (Classe Negativa)', 
                               'Nº Médio de Regras',
                               'explanation_length_negative.png')

        # 6. Tamanho Médio da Explicação - Classe Rejeitada
        explanation_length_rej = extract_metric(data, 'explanation_stats.rejected.mean_length')
        plot_metric_horizontally(explanation_length_rej, 
                               'Tamanho Médio da Explicação (Classe Rejeitada)', 
                               'Nº Médio de Regras',
                               'explanation_length_rejected.png')

        # 7. Tempo Médio de Computação por Instância
        computation_time = extract_metric(data, 'computation_time.mean_per_instance')
        plot_metric_horizontally(computation_time, 
                               'Tempo Médio de Explicação por Instância', 
                               'Tempo (segundos)',
                               'computation_time.png')

        # 8. Tempo Total de Computação
        total_time = extract_metric(data, 'computation_time.total')
        plot_metric_horizontally(total_time, 
                               'Tempo Total de Computação', 
                               'Tempo (segundos)',
                               'total_computation_time.png')
        
        print("🎉 Análise concluída! Verifique a pasta 'analysis_output'.")