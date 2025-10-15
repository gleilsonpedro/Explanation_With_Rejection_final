import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Configuração do estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

class PEABPlotGenerator:
    def __init__(self, results_file='comparative_results.json'):
        """
        Inicializa o gerador de plots com os resultados do PEAB
        
        Args:
            results_file (str): Caminho para o arquivo JSON com resultados
        """
        self.results_file = results_file
        self.load_results()
        
    def load_results(self):
        """Carrega os resultados do arquivo JSON"""
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        except FileNotFoundError:
            print(f"Arquivo {self.results_file} não encontrado")
            self.results = {}
    
    def plot_distribuicao_pontuacoes(self, decision_scores, titulo="Distribuição das Pontuações de Decisão e Zona de Rejeição (Dataset Pima)"):
        """
        Gráfico 1: Distribuição das Pontuações e Zona de Rejeição
        
        Args:
            decision_scores (array): Pontuações de decisão do modelo
            titulo (str): Título do gráfico
        """
        # Extrair limiares do JSON
        pima_results = self.results.get('peab', {}).get('pima_indians_diabetes', {})
        t_plus = pima_results.get('t_plus', 0.6)
        t_minus = pima_results.get('t_minus', 0.4)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Definir cores para as regiões
        cores = ['#2E86AB', '#F7EF99', '#A23B72']
        
        # Criar histograma com cores diferentes para cada região
        n, bins, patches = ax.hist(decision_scores, bins=30, alpha=0.7, 
                                  color='lightgray', edgecolor='black')
        
        # Colorir as barras baseado nas regiões
        for i, (bar, bin_edge) in enumerate(zip(patches, bins)):
            if bin_edge < t_minus:
                bar.set_facecolor(cores[0])  # Classe Negativa - Azul
            elif bin_edge < t_plus:
                bar.set_facecolor(cores[1])  # Zona de Rejeição - Amarelo
            else:
                bar.set_facecolor(cores[2])  # Classe Positiva - Vermelho
        
        # Adicionar linhas verticais para os limiares
        ax.axvline(x=t_minus, color='red', linestyle='--', linewidth=2, 
                  label=f't_minus = {t_minus:.3f}')
        ax.axvline(x=t_plus, color='blue', linestyle='--', linewidth=2, 
                  label=f't_plus = {t_plus:.3f}')
        
        # Configurações do gráfico
        ax.set_xlabel('Pontuação de Decisão', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        
        # Legenda personalizada
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cores[0], label='Classe Negativa (score < t_minus)'),
            Patch(facecolor=cores[1], label='Zona de Rejeição (t_minus ≤ score ≤ t_plus)'),
            Patch(facecolor=cores[2], label='Classe Positiva (score > t_plus)'),
            plt.Line2D([0], [0], color='red', linestyle='--', label=f't_minus = {t_minus:.3f}'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label=f't_plus = {t_plus:.3f}')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_comparacao_tamanho_explicacoes(self, titulo="Tamanho Médio da Explicação por Método (Dataset Pima)"):
        """
        Gráfico 2: Comparação do Tamanho Médio das Explicações
        
        Args:
            titulo (str): Título do gráfico
        """
        # Extrair dados do JSON
        pima_results = self.results.get('peab', {}).get('pima_indians_diabetes', {})
        anchor_results = self.results.get('anchor', {}).get('pima_indians_diabetes', {})
        minexp_results = self.results.get('minexp', {}).get('pima_indians_diabetes', {})
        
        # Preparar dados para o gráfico
        metodos = ['PEAB', 'Anchor', 'MinExp']
        categorias = ['Positivas', 'Negativas', 'Rejeitadas']
        
        # Dados de exemplo (substitua com seus dados reais)
        dados = {
            'PEAB': {
                'Positivas': pima_results.get('positive', {}).get('mean_length', 3.2),
                'Negativas': pima_results.get('negative', {}).get('mean_length', 2.8),
                'Rejeitadas': pima_results.get('rejected', {}).get('mean_length', 4.1)
            },
            'Anchor': {
                'Positivas': anchor_results.get('positive', {}).get('mean_length', 4.5),
                'Negativas': anchor_results.get('negative', {}).get('mean_length', 4.2),
                'Rejeitadas': anchor_results.get('rejected', {}).get('mean_length', 5.0)
            },
            'MinExp': {
                'Positivas': minexp_results.get('positive', {}).get('mean_length', 3.8),
                'Negativas': minexp_results.get('negative', {}).get('mean_length', 3.5),
                'Rejeitadas': minexp_results.get('rejected', {}).get('mean_length', 4.5)
            }
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurações para gráfico de barras agrupadas
        x = np.arange(len(categorias))
        largura = 0.25
        multiplicador = 0
        
        cores = ['#2E86AB', '#F7B801', '#A23B72']
        
        for i, metodo in enumerate(metodos):
            offset = largura * multiplicador
            valores = [dados[metodo][cat] for cat in categorias]
            rects = ax.bar(x + offset, valores, largura, label=metodo, color=cores[i])
            ax.bar_label(rects, padding=3, fmt='%.1f')
            multiplicador += 1
        
        # Configurações do gráfico
        ax.set_xlabel('Tipo de Classificação', fontsize=12)
        ax.set_ylabel('Tamanho Médio da Explicação (Nº de Features)', fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xticks(x + largura, categorias)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_top_features_peab(self, titulo="Top 10 Features Mais Frequentes nas Explicações do PEAB (Dataset Pima)"):
        """
        Gráfico 3: Frequência das Features Mais Relevantes no PEAB
        
        Args:
            titulo (str): Título do gráfico
        """
        # Extrair top features do JSON
        pima_results = self.results.get('peab', {}).get('pima_indians_diabetes', {})
        top_features = pima_results.get('top_features', [])
        
        # Se não houver dados, usar dados de exemplo
        if not top_features:
            top_features = [
                {'feature': 'Glucose', 'count': 145},
                {'feature': 'BMI', 'count': 132},
                {'feature': 'Age', 'count': 118},
                {'feature': 'DiabetesPedigree', 'count': 95},
                {'feature': 'Pregnancies', 'count': 87},
                {'feature': 'BloodPressure', 'count': 76},
                {'feature': 'SkinThickness', 'count': 65},
                {'feature': 'Insulin', 'count': 54},
                {'feature': 'DPF', 'count': 43},
                {'feature': 'Triceps', 'count': 32}
            ]
        
        # Ordenar por frequência e pegar top 10
        top_features_sorted = sorted(top_features, key=lambda x: x['count'], reverse=True)[:10]
        
        features = [item['feature'] for item in top_features_sorted]
        counts = [item['count'] for item in top_features_sorted]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Gráfico de barras horizontais
        bars = ax.barh(features, counts, color='#2E86AB', alpha=0.7)
        
        # Adicionar valores nas barras
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{count}', ha='left', va='center', fontsize=10)
        
        # Configurações do gráfico
        ax.set_xlabel('Frequência de Aparição nas Explicações', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Inverter ordem para ter a mais frequente no topo
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_otimizacao_instancia_rejeitada(self, dados_otimizacao=None, titulo="Etapas de Otimização da Explicação para uma Instância Rejeitada"):
        """
        Gráfico 4: Visualização do Processo de Otimização
        
        Args:
            dados_otimizacao (dict): Dados do processo de otimização
            titulo (str): Título do gráfico
        """
        # Dados de exemplo (substitua com dados reais do seu log)
        if dados_otimizacao is None:
            dados_otimizacao = {
                'caminho1': {
                    'inicial': 8,
                    'robusta': 12,
                    'final': 5
                },
                'caminho2': {
                    'inicial': 7,
                    'robusta': 11,
                    'final': 4
                }
            }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurações para gráfico de barras agrupadas
        categorias = ['Inicial', 'Robusta', 'Mínima']
        caminhos = ['Caminho 1 (Alvo: Classe Positiva)', 'Caminho 2 (Alvo: Classe Negativa)']
        
        x = np.arange(len(categorias))
        largura = 0.35
        
        # Valores para cada caminho
        valores_caminho1 = [
            dados_otimizacao['caminho1']['inicial'],
            dados_otimizacao['caminho1']['robusta'],
            dados_otimizacao['caminho1']['final']
        ]
        
        valores_caminho2 = [
            dados_otimizacao['caminho2']['inicial'],
            dados_otimizacao['caminho2']['robusta'],
            dados_otimizacao['caminho2']['final']
        ]
        
        # Plotar barras
        bars1 = ax.bar(x - largura/2, valores_caminho1, largura, 
                      label='Caminho 1 (Alvo: Classe Positiva)', color='#2E86AB')
        bars2 = ax.bar(x + largura/2, valores_caminho2, largura, 
                      label='Caminho 2 (Alvo: Classe Negativa)', color='#A23B72')
        
        # Adicionar valores nas barras
        ax.bar_label(bars1, padding=3)
        ax.bar_label(bars2, padding=3)
        
        # Configurações do gráfico
        ax.set_xlabel('Etapas de Otimização', fontsize=12)
        ax.set_ylabel('Tamanho da Explicação (Nº de Features)', fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xticks(x, categorias)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar linhas para mostrar o processo
        for i in range(len(valores_caminho1)-1):
            # Linha para caminho 1
            ax.plot([x[i] - largura/2, x[i+1] - largura/2], 
                   [valores_caminho1[i], valores_caminho1[i+1]], 
                   'o-', color='#1B5E7F', alpha=0.7)
            # Linha para caminho 2
            ax.plot([x[i] + largura/2, x[i+1] + largura/2], 
                   [valores_caminho2[i], valores_caminho2[i+1]], 
                   'o-', color='#7A2B5F', alpha=0.7)
        
        plt.tight_layout()
        return fig

# Exemplo de uso
def main():
    # Inicializar o gerador de plots
    plot_generator = PEABPlotGenerator('comparative_results.json')
    
    # Gerar dados de exemplo para as pontuações de decisão
    np.random.seed(42)
    decision_scores = np.random.normal(0.5, 0.2, 1000)
    
    # Gerar todos os gráficos
    print("Gerando gráficos...")
    
    # Gráfico 1: Distribuição das Pontuações
    fig1 = plot_generator.plot_distribuicao_pontuacoes(decision_scores)
    fig1.savefig('distribuicao_pontuacoes.png', dpi=300, bbox_inches='tight')
    
    # Gráfico 2: Comparação do Tamanho das Explicações
    fig2 = plot_generator.plot_comparacao_tamanho_explicacoes()
    fig2.savefig('comparacao_tamanho_explicacoes.png', dpi=300, bbox_inches='tight')
    
    # Gráfico 3: Top Features PEAB
    fig3 = plot_generator.plot_top_features_peab()
    fig3.savefig('top_features_peab.png', dpi=300, bbox_inches='tight')
    
    # Gráfico 4: Processo de Otimização
    fig4 = plot_generator.plot_otimizacao_instancia_rejeitada()
    fig4.savefig('processo_otimizacao.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    print("Gráficos gerados e salvos com sucesso!")

if __name__ == "__main__":
    main()