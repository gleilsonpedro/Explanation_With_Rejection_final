import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any

# Configurações de estilo com a paleta de cores "Set2"
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

class ComparativeAnalyzer:
    def __init__(self, results_file: str = "comparative_results.json"):
        self.results_file = results_file
        self.results = self._load_results()
        self.methods = [m for m in ["peab", "anchor", "MinExp"] if self.results.get(m) and self.results[m]]
        self.datasets = self._get_all_datasets()
        
        os.makedirs("analysis_output", exist_ok=True)
        os.makedirs("analysis_output/plots", exist_ok=True)
        os.makedirs("analysis_output/tables", exist_ok=True)

    def _load_results(self) -> Dict[str, Any]:
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERRO: O arquivo '{self.results_file}' não foi encontrado.")
            print("Por favor, certifique-se de que o script está na mesma pasta que o arquivo JSON.")
            exit()

    def _get_all_datasets(self) -> List[str]:
        datasets = set()
        for method in self.methods:
            datasets.update(self.results[method].keys())
        return sorted(list(datasets))

    def _get_stat(self, stats_dict: dict, generic_key: str, method: str) -> Any:
        if not stats_dict:
            return 0
        key_map = {
            'anchor': {"count": "instancias", "mean_length": "media", "min_length": "min", "max_length": "max", "std_length": "std_dev"},
            'peab': {"count": "count", "mean_length": "mean_length", "min_length": "min_length", "max_length": "max_length", "std_length": "std_length"}
        }
        key_map['mateus'] = key_map['peab']
        specific_key = key_map.get(method, {}).get(generic_key, generic_key)
        return stats_dict.get(specific_key, 0)

    def generate_all_analysis(self):
        print("Iniciando análise comparativa...")
        self._generate_summary_table()
        self._plot_metric_comparison()
        ### MUDANÇA: Chamando a nova função que gera 3 plots separados ###
        self._plot_explanation_length_separated_by_class()
        self._plot_computation_time()
        print(f"Análise concluída! Resultados salvos em '{os.path.abspath('analysis_output/')}'")

    def _generate_summary_table(self):
        table_data = []
        for method in self.methods:
            for dataset in self.datasets:
                if dataset in self.results[method]:
                    data = self.results[method][dataset]
                    stats_pos = data["explanation_stats"].get("positive", {})
                    stats_neg = data["explanation_stats"].get("negative", {})
                    stats_rej = data["explanation_stats"].get("rejected", {})
                    
                    # ### MUDANÇA: Renomeando PEAB para Exp_Abd ###
                    method_name = "Exp_Abd" if method.lower() == "peab" else method.upper()

                    row = {
                        "Method": method_name,
                        "Dataset": dataset,
                        "Acc (no rej)": data["performance"].get("accuracy_without_rejection", 0),
                        "Acc (with rej)": data["performance"].get("accuracy_with_rejection", 0),
                        "Rejection Rate": data["performance"].get("rejection_rate", 0),
                        "Mean Len (Pos)": self._get_stat(stats_pos, "mean_length", method),
                        "Mean Len (Neg)": self._get_stat(stats_neg, "mean_length", method),
                        "Mean Len (Rej)": self._get_stat(stats_rej, "mean_length", method),
                        "Total Time (s)": data["computation_time"].get("total", 0)
                    }
                    table_data.append(row)

        df = pd.DataFrame(table_data)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
        
        path = "analysis_output/tables/summary_table.csv"
        df.to_csv(path, index=False)
        print(f"Tabela resumo salva em {path}")

    def _plot_metric_comparison(self):
        df = pd.read_csv("analysis_output/tables/summary_table.csv")
        if df.empty: return

        metrics_to_plot = {
            "Acc (no rej)": "Acurácia sem Rejeição (%)",
            "Acc (with rej)": "Acurácia com Rejeição (%)",
            "Rejection Rate": "Taxa de Rejeição (%)"
        }

        for metric, title in metrics_to_plot.items():
            plt.figure(figsize=(12, 7))
            sns.barplot(data=df, x="Dataset", y=metric, hue="Method", errorbar=None)
            plt.title(title, fontsize=16)
            plt.ylabel(metric)
            plt.xlabel("Dataset")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"analysis_output/plots/{metric.replace(' ', '_').lower()}.png", dpi=300)
            plt.close()
            print(f"Gráfico '{title}' salvo.")

    ### NOVO MÉTODO DE PLOTAGEM QUE GERA 3 GRÁFICOS SEPARADOS ###
    def _plot_explanation_length_separated_by_class(self):
        """
        Gera três gráficos de barras separados para o tamanho médio das explicações,
        um para cada classe (Positiva, Negativa, Rejeitada).
        """
        df = pd.read_csv("analysis_output/tables/summary_table.csv")
        if df.empty: return

        class_map = {
            "Mean Len (Pos)": "Classe Positiva",
            "Mean Len (Neg)": "Classe Negativa",
            "Mean Len (Rej)": "Classe Rejeitada"
        }

        for y_col, title in class_map.items():
            plt.figure(figsize=(12, 7))
            
            # Filtra datasets onde a média de tamanho para a classe é maior que zero para não poluir o gráfico
            plot_df = df[df[y_col] > 0]

            sns.barplot(data=plot_df, x="Dataset", y=y_col, hue="Method", errorbar=None)
            
            plt.title(f"Tamanho Médio da Explicação ({title})", fontsize=16)
            plt.ylabel("Tamanho da Explicação")
            plt.xlabel("Dataset")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            filename = f"explanation_length_{title.replace(' ', '_').lower()}.png"
            plt.savefig(f"analysis_output/plots/{filename}", dpi=300)
            plt.close()
            print(f"Gráfico '{title}' salvo.")

    def _plot_computation_time(self):
        df = pd.read_csv("analysis_output/tables/summary_table.csv")
        if df.empty: return

        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x="Dataset", y="Total Time (s)", hue="Method", errorbar=None)
        plt.title("Tempo de Execução Total por Dataset", fontsize=16)
        plt.ylabel("Tempo Total (s) - Escala Logarítmica")
        plt.xlabel("Dataset")
        plt.yscale('log')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("analysis_output/plots/computation_time_total.png", dpi=300)
        plt.close()
        print("Gráfico 'Tempo de Execução' salvo.")


if __name__ == "__main__":
    analyzer = ComparativeAnalyzer()
    analyzer.generate_all_analysis()