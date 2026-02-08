"""
Script para gerar tabela LaTeX unificada de tempo de execução
com média e desvio padrão para instâncias classificadas e rejeitadas.
"""

import json
import numpy as np
from pathlib import Path


# Configuração de datasets
DATASETS = [
    {"name": "banknote", "display": "Banknote"},
    {"name": "vertebral_column", "display": "Vertebral Column"},
    {"name": "pima_indians_diabetes", "display": "Pima Indians"},
    {"name": "heart_disease", "display": "Heart Disease"},
    {"name": "creditcard", "display": "Credit Card"},
    {"name": "breast_cancer", "display": "Breast Cancer"},
    {"name": "covertype", "display": "Covertype"},
    {"name": "spambase", "display": "Spambase"},
    {"name": "sonar", "display": "Sonar"},
    {"name": "mnist_3_vs_8", "display": "MNIST (3 vs 8)", "minexp_name": "mnist", "anchor_name": "mnist"}
]

METODOS = {
    "peab": "MINABRO",
    "anchor": "Anchor",
    "minexp": "AbLinRO"
}


def carregar_json(metodo, dataset_name):
    """Carrega JSON de um método e dataset."""
    json_path = Path(f"json/{metodo}/{dataset_name}.json")
    if not json_path.exists():
        print(f"⚠️ Arquivo não encontrado: {json_path}")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erro ao ler {json_path}: {e}")
        return None


def calcular_tempo_por_tipo_com_std(data, metodo):
    """
    Calcula tempo médio e desvio padrão para classificadas e rejeitadas
    usando os dados de per_instance.
    
    Returns:
        tuple: (mean_classif, std_classif, mean_rej, std_rej) em ms
    """
    if data is None:
        return None, None, None, None
    
    try:
        if metodo in ["peab", "minexp"]:
            per_instance = data.get("per_instance", [])
            
            if not per_instance:
                print(f"⚠️ Sem dados per_instance para {metodo}")
                # Tentar usar dados agregados como fallback
                return calcular_tempo_agregado(data, metodo)
            
            # Separar tempos de classificadas e rejeitadas
            tempos_classif = []
            tempos_rej = []
            
            for inst in per_instance:
                tempo = inst.get("computation_time")
                rejeitada = inst.get("rejected", False)
                
                if tempo is not None:
                    # Converter para ms
                    tempo_ms = tempo * 1000
                    
                    if rejeitada:
                        tempos_rej.append(tempo_ms)
                    else:
                        tempos_classif.append(tempo_ms)
            
            # Calcular média e desvio padrão
            mean_classif = np.mean(tempos_classif) if tempos_classif else None
            std_classif = np.std(tempos_classif, ddof=1) if len(tempos_classif) > 1 else 0.0
            
            mean_rej = np.mean(tempos_rej) if tempos_rej else None
            std_rej = np.std(tempos_rej, ddof=1) if len(tempos_rej) > 1 else 0.0
            
            return mean_classif, std_classif, mean_rej, std_rej
        
        elif metodo == "anchor":
            # Anchor não tem per_instance, usar dados agregados sem std
            return calcular_tempo_agregado(data, metodo)
        
        elif metodo == "pulp":
            # PULP não tem per_instance, usar estatísticas agregadas
            stats = data.get("estatisticas_por_tipo", {})
            
            pos_time = stats.get("positiva", {}).get("tempo_medio")
            neg_time = stats.get("negativa", {}).get("tempo_medio")
            pos_count = stats.get("positiva", {}).get("instancias", 0)
            neg_count = stats.get("negativa", {}).get("instancias", 0)
            
            rej_time = stats.get("rejeitada", {}).get("tempo_medio")
            
            # Média ponderada para classificadas
            if pos_time and neg_time and (pos_count + neg_count) > 0:
                mean_classif = ((pos_time * pos_count + neg_time * neg_count) / 
                               (pos_count + neg_count)) * 1000
            else:
                mean_classif = None
            
            mean_rej = rej_time * 1000 if rej_time else None
            
            # PULP não tem std, retornar 0
            return mean_classif, 0.0, mean_rej, 0.0
            
    except Exception as e:
        print(f"❌ Erro ao calcular tempo para {metodo}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def calcular_tempo_agregado(data, metodo):
    """
    Calcula tempo usando dados agregados (sem desvio padrão).
    Fallback para quando per_instance não está disponível.
    """
    if data is None:
        return None, None, None, None
    
    try:
        comp_time = data.get("computation_time", {})
        stats = data.get("explanation_stats", {})
        
        pos_time = comp_time.get("positive")
        neg_time = comp_time.get("negative")
        pos_count = stats.get("positive", {}).get("count", 0)
        neg_count = stats.get("negative", {}).get("count", 0)
        rej_time = comp_time.get("rejected")
        
        # Média ponderada para classificadas
        if pos_time is not None and neg_time is not None and (pos_count + neg_count) > 0:
            mean_classif = ((pos_time * pos_count + neg_time * neg_count) / 
                           (pos_count + neg_count)) * 1000
        else:
            mean_classif = None
        
        mean_rej = rej_time * 1000 if rej_time is not None else None
        
        # Sem desvio padrão disponível
        return mean_classif, 0.0, mean_rej, 0.0
        
    except Exception as e:
        print(f"❌ Erro ao calcular tempo agregado: {e}")
        return None, None, None, None


def gerar_tabela_runtime_unificada():
    """Gera tabela LaTeX unificada com tempos médios e desvios padrão."""
    
    print("=" * 70)
    print("GERANDO TABELA UNIFICADA DE RUNTIME COM DESVIO PADRÃO")
    print("=" * 70)
    print()
    
    # Coletar dados
    dados = {}
    
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        display_name = dataset["display"]
        
        print(f"📊 Processando {display_name}...")
        
        dados[dataset_name] = {"display": display_name}
        
        for metodo_key, metodo_display in METODOS.items():
            # Ajustar nome do dataset para minexp e anchor se necessário
            nome_json = dataset.get(f"{metodo_key}_name", dataset_name)
            
            data = carregar_json(metodo_key, nome_json)
            mean_c, std_c, mean_r, std_r = calcular_tempo_por_tipo_com_std(data, metodo_key)
            
            dados[dataset_name][metodo_key] = {
                "classif_mean": mean_c,
                "classif_std": std_c,
                "rej_mean": mean_r,
                "rej_std": std_r
            }
            
            if mean_c is not None:
                print(f"  ✓ {metodo_display}: Classif={mean_c:.1f}±{std_c:.1f} ms, Rej={mean_r:.1f}±{std_r:.1f} ms")
            else:
                print(f"  ⚠️ {metodo_display}: Dados não disponíveis")
    
    print()
    print("=" * 70)
    print("GERANDO CÓDIGO LATEX")
    print("=" * 70)
    
    # Gerar LaTeX
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\caption{Average execution time (ms) for \\emph{classified} and \\emph{rejected} instances.}")
    latex.append("\\label{tab:runtime_unified}")
    latex.append("\\small")
    latex.append("\\setlength{\\tabcolsep}{4pt}")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}}")
    latex.append("& \\multicolumn{2}{c}{\\textbf{MINABRO}}")
    latex.append("& \\multicolumn{2}{c}{\\textbf{Anchor}}")
    latex.append("& \\multicolumn{2}{c}{\\textbf{AbLinRO}} \\\\")
    latex.append("\\cline{2-7}")
    latex.append(" & \\textbf{Clas.} & \\textbf{Rej.}")
    latex.append(" & \\textbf{Clas.} & \\textbf{Rej.}")
    latex.append(" & \\textbf{Clas.} & \\textbf{Rej.} \\\\")
    latex.append("\\hline")
    
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        display_name = dataset["display"]
        d = dados[dataset_name]
        
        # Formatar valores com desvio padrão quando disponível
        valores = []
        
        for metodo_key in ["peab", "anchor", "minexp"]:
            info = d[metodo_key]
            
            # Classificadas
            if info["classif_mean"] is not None:
                mean_c = info["classif_mean"]
                std_c = info["classif_std"]
                # Mostrar ± apenas se std > 0
                if std_c > 0:
                    str_c = f"{mean_c:.1f} $\\pm$ {std_c:.1f}"
                else:
                    str_c = f"{mean_c:.1f}"
            else:
                str_c = "N/A"
            
            # Rejeitadas
            if info["rej_mean"] is not None:
                mean_r = info["rej_mean"]
                std_r = info["rej_std"]
                # Mostrar ± apenas se std > 0
                if std_r > 0:
                    str_r = f"{mean_r:.1f} $\\pm$ {std_r:.1f}"
                else:
                    str_r = f"{mean_r:.1f}"
            else:
                str_r = "N/A"
            
            valores.extend([str_c, str_r])
        
        linha = f"{display_name} & {' & '.join(valores)} \\\\"
        latex.append(linha)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_code = "\n".join(latex)
    
    # Salvar arquivo
    output_path = Path("results/tabelas_latex/runtime_unified_with_std.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"\n✅ Tabela salva em: {output_path}")
    print()
    print("=" * 70)
    print("PRÉVIA DA TABELA")
    print("=" * 70)
    print(latex_code)
    print()
    
    return latex_code


if __name__ == "__main__":
    gerar_tabela_runtime_unificada()
