"""Script para gerar tabelas LaTeX para múltiplos datasets.

Gera tabelas comparando PEAB (MINABRO), PuLP, Anchors e MinExp (AbLinRO).
Inclui uma tabela unificada de tempo (classificadas vs rejeitadas) com média e desvio padrão.

Resultados salvos em results/tabelas_latex/mnist/
"""

import json
import os
from pathlib import Path
import numpy as np


# Datasets (ordem igual à tabela de tempo do artigo)
DATASETS = [
    {
        "name": "banknote",
        "display_name": "Banknote",
        "num_features": 4,
        "peab_name": "banknote",
        "pulp_name": "banknote",
        "anchor_name": "banknote",
        "minexp_name": "banknote",
    },
    {
        "name": "vertebral_column",
        "display_name": "Vertebral Column",
        "num_features": 6,
        "peab_name": "vertebral_column",
        "pulp_name": "vertebral_column",
        "anchor_name": "vertebral_column",
        "minexp_name": "vertebral_column",
    },
    {
        "name": "pima_indians_diabetes",
        "display_name": "Pima Indians",
        "num_features": 8,
        "peab_name": "pima_indians_diabetes",
        "pulp_name": "pima_indians_diabetes",
        "anchor_name": "pima_indians_diabetes",
        "minexp_name": "pima_indians_diabetes",
    },
    {
        "name": "heart_disease",
        "display_name": "Heart Disease",
        "num_features": 13,
        "peab_name": "heart_disease",
        "pulp_name": "heart_disease",
        "anchor_name": "heart_disease",
        "minexp_name": "heart_disease",
    },
    {
        "name": "creditcard",
        "display_name": "Credit Card",
        "num_features": 29,
        "peab_name": "creditcard",
        "pulp_name": "creditcard",
        "anchor_name": "creditcard",
        "minexp_name": "creditcard",
    },
    {
        "name": "breast_cancer",
        "display_name": "Breast Cancer",
        "num_features": 30,
        "peab_name": "breast_cancer",
        "pulp_name": "breast_cancer",
        "anchor_name": "breast_cancer",
        "minexp_name": "breast_cancer",
    },
    {
        "name": "covertype",
        "display_name": "Covertype",
        "num_features": 54,
        "peab_name": "covertype",
        "pulp_name": "covertype",
        "anchor_name": "covertype",
        "minexp_name": "covertype",
    },
    {
        "name": "spambase",
        "display_name": "Spambase",
        "num_features": 57,
        "peab_name": "spambase",
        "pulp_name": "spambase",
        "anchor_name": "spambase",
        "minexp_name": "spambase",
    },
    {
        "name": "sonar",
        "display_name": "Sonar",
        "num_features": 60,
        "peab_name": "sonar",
        "pulp_name": "sonar",
        "anchor_name": "sonar",
        "minexp_name": "sonar",
    },
    {
        "name": "mnist",
        "display_name": "MNIST (3 vs 8)",
        "num_features": 784,
        "peab_name": "mnist_3_vs_8",
        "pulp_name": "mnist_3_vs_8",
        "anchor_name": "mnist",
        "minexp_name": "mnist",
    },
]

METODOS = ["peab", "pulp", "anchor", "minexp"]
OUTPUT_DIR = "results/tabelas_latex/mnist"

def carregar_dados_json(dataset_config, metodo):
    """Carrega os dados de JSON para um dataset e método específicos."""
    dataset_name = dataset_config[f"{metodo}_name"]
    json_path = Path(f"json/{metodo}/{dataset_name}.json")
    
    if not json_path.exists():
        print(f"Aviso: {json_path} não encontrado")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler {json_path}: {e}")
        return None


def carregar_dados_validacao(dataset_config, metodo):
    """Carrega os dados de validação para um dataset e método específicos."""
    dataset_name = dataset_config[f"{metodo}_name"]
    
    # Tentar primeiro com o nome do dataset específico
    json_path = Path(f"json/validation/{metodo}_validation_{dataset_name}.json")
    
    if not json_path.exists():
        # Se não encontrar, tentar com o nome base
        json_path = Path(f"json/validation/{metodo}_validation_{dataset_config['name']}.json")
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler {json_path}: {e}")
        return None


def extrair_tempo_medio_ms(data, metodo):
    """Extrai o tempo médio por instância em milissegundos."""
    if data is None:
        return None
    
    try:
        if metodo in ["peab", "anchor", "minexp"]:
            tempo_s = data.get("computation_time", {}).get("mean_per_instance", None)
            return tempo_s * 1000 if tempo_s is not None else None
        elif metodo == "pulp":
            tempo_s = data.get("estatisticas_gerais", {}).get("tempo_medio_segundos", None)
            return tempo_s * 1000 if tempo_s is not None else None
    except Exception as e:
        print(f"Erro ao extrair tempo para {metodo}: {e}")
        return None


def extrair_tempo_por_tipo(data, metodo):
    """Extrai tempo médio separado por tipo: classificadas e rejeitadas."""
    if data is None:
        return None, None
    
    try:
        if metodo in ["peab", "anchor", "minexp"]:
            comp_time = data.get("computation_time", {})
            
            pos_time = comp_time.get("positive", None)
            neg_time = comp_time.get("negative", None)
            
            stats = data.get("explanation_stats", {})
            pos_count = stats.get("positive", {}).get("count", 0)
            neg_count = stats.get("negative", {}).get("count", 0)
            
            rej_time = comp_time.get("rejected", None)
            
            if pos_time is not None and neg_time is not None and (pos_count + neg_count) > 0:
                classif_time = (pos_time * pos_count + neg_time * neg_count) / (pos_count + neg_count)
                classif_time_ms = classif_time * 1000
            else:
                classif_time_ms = None
            
            rej_time_ms = rej_time * 1000 if rej_time is not None else None
            
            return classif_time_ms, rej_time_ms
            
        elif metodo == "pulp":
            stats = data.get("estatisticas_por_tipo", {})
            
            pos_time = stats.get("positiva", {}).get("tempo_medio", None)
            neg_time = stats.get("negativa", {}).get("tempo_medio", None)
            pos_count = stats.get("positiva", {}).get("instancias", 0)
            neg_count = stats.get("negativa", {}).get("instancias", 0)
            
            rej_time = stats.get("rejeitada", {}).get("tempo_medio", None)
            
            if pos_time is not None and neg_time is not None and (pos_count + neg_count) > 0:
                classif_time = (pos_time * pos_count + neg_time * neg_count) / (pos_count + neg_count)
                classif_time_ms = classif_time * 1000
            else:
                classif_time_ms = None
            
            rej_time_ms = rej_time * 1000 if rej_time is not None else None
            
            return classif_time_ms, rej_time_ms
    except Exception as e:
        print(f"Erro ao extrair tempo por tipo para {metodo}: {e}")
        return None, None


def extrair_tempo_por_tipo_media_std_ms(data, metodo):
    """Extrai (média, desvio padrão) em ms para instâncias classificadas e rejeitadas.

    Fonte preferida: tempos por instância nos JSONs.

    - PEAB/Anchor/MinExp: usa per_instance[*].computation_time (segundos) + per_instance[*].rejected
    - PuLP: usa explicacoes[*].tempo_segundos (segundos) + explicacoes[*].tipo_predicao

    Retorna:
      - (mean_class_ms, std_class_ms), (mean_rej_ms, std_rej_ms)
    """
    if data is None:
        return (None, None), (None, None)

    try:
        classif_s: list[float] = []
        rej_s: list[float] = []

        if metodo in ["peab", "anchor", "minexp"]:
            per_instance = data.get("per_instance", [])
            if not isinstance(per_instance, list):
                per_instance = []

            for pi in per_instance:
                if not isinstance(pi, dict):
                    continue
                t = pi.get("computation_time", None)
                if t is None:
                    continue
                try:
                    t_f = float(t)
                except Exception:
                    continue

                if bool(pi.get("rejected", False)):
                    rej_s.append(t_f)
                else:
                    classif_s.append(t_f)

            # Fallback: alguns JSONs (ex.: Anchor em certos datasets) podem ter
            # computation_time agregado, mas per_instance.computation_time = 0.0
            all_times = classif_s + rej_s
            if len(all_times) > 0 and float(np.max(np.abs(np.array(all_times, dtype=float)))) < 1e-12:
                mean_class_ms, mean_rej_ms = extrair_tempo_por_tipo(data, metodo)
                return (
                    (0.0 if mean_class_ms is None else float(mean_class_ms), 0.0),
                    (0.0 if mean_rej_ms is None else float(mean_rej_ms), 0.0),
                )

        elif metodo == "pulp":
            explicacoes = data.get("explicacoes", [])
            if not isinstance(explicacoes, list):
                explicacoes = []

            for ex in explicacoes:
                if not isinstance(ex, dict):
                    continue
                t = ex.get("tempo_segundos", None)
                if t is None:
                    continue
                try:
                    t_f = float(t)
                except Exception:
                    continue

                tipo = str(ex.get("tipo_predicao", "")).upper()
                if tipo == "REJEITADA":
                    rej_s.append(t_f)
                else:
                    classif_s.append(t_f)

        else:
            return (None, None), (None, None)

        def stats_ms(values_s: list[float]) -> tuple[float, float]:
            if len(values_s) == 0:
                return (0.0, 0.0)
            arr = np.array(values_s, dtype=float)
            mean_ms = float(arr.mean() * 1000.0)
            std_ms = float(arr.std(ddof=1) * 1000.0) if len(arr) > 1 else 0.0
            return (mean_ms, std_ms)

        return stats_ms(classif_s), stats_ms(rej_s)

    except Exception as e:
        print(f"Erro ao extrair tempo (média/std) para {metodo}: {e}")
        return (None, None), (None, None)


def gerar_tabela_runtime_unified():
    """Gera tabela unificada de tempo (ms) com média e desvio padrão.

    Formato alinhado ao modelo fornecido pelo usuário.
    """

    def fmt_cell(mean_std: tuple[float, float] | tuple[None, None]) -> str:
        mean, std = mean_std
        if mean is None or std is None:
            return "N/A"
        return f"{mean:.2f} $\\pm$ {std:.2f}"

    latex: list[str] = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\caption{Tempo médio de execução $\\pm$ desvio padrão (ms) para instâncias classificadas e rejeitadas.}")
    latex.append("\\label{tab:runtime_unified}")
    latex.append("\\small")
    latex.append("\\setlength{\\tabcolsep}{4pt}")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c}{\\textbf{MINABRO}} & \\multicolumn{2}{c}{\\textbf{Anchors}} & \\multicolumn{2}{c}{\\textbf{AbLinRO}} \\\\")
    latex.append("\\cline{2-7}")
    latex.append(" & \\textbf{Clas.} & \\textbf{Rej.} & \\textbf{Clas.} & \\textbf{Rej.} & \\textbf{Clas.} & \\textbf{Rej.} \\\\")
    latex.append("\\hline")

    for dataset_config in DATASETS:
        data_peab = carregar_dados_json(dataset_config, "peab")
        data_anchor = carregar_dados_json(dataset_config, "anchor")
        data_minexp = carregar_dados_json(dataset_config, "minexp")

        peab_class, peab_rej = extrair_tempo_por_tipo_media_std_ms(data_peab, "peab")
        anchor_class, anchor_rej = extrair_tempo_por_tipo_media_std_ms(data_anchor, "anchor")
        minexp_class, minexp_rej = extrair_tempo_por_tipo_media_std_ms(data_minexp, "minexp")

        latex.append(
            f"{dataset_config['display_name']} & "
            f"{fmt_cell(peab_class)} & {fmt_cell(peab_rej)} & "
            f"{fmt_cell(anchor_class)} & {fmt_cell(anchor_rej)} & "
            f"{fmt_cell(minexp_class)} & {fmt_cell(minexp_rej)} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex)


def extrair_contagens_explicacoes(data, metodo):
    """Extrai as contagens de explicações por tipo."""
    if data is None:
        return None, None, None
    
    try:
        if metodo in ["peab", "anchor", "minexp"]:
            stats = data.get("explanation_stats", {})
            pos = stats.get("positive", {}).get("count", 0)
            neg = stats.get("negative", {}).get("count", 0)
            rej = stats.get("rejected", {}).get("count", 0)
            return pos, neg, rej
        elif metodo == "pulp":
            stats = data.get("estatisticas_por_tipo", {})
            pos = stats.get("positiva", {}).get("instancias", 0)
            neg = stats.get("negativa", {}).get("instancias", 0)
            rej = stats.get("rejeitada", {}).get("instancias", 0)
            return pos, neg, rej
    except Exception as e:
        print(f"Erro ao extrair contagens para {metodo}: {e}")
        return None, None, None


def extrair_tamanho_medio_por_tipo(data, metodo):
    """Extrai tamanho médio de explicações por tipo."""
    if data is None:
        return None, None
    
    try:
        if metodo in ["peab", "anchor", "minexp"]:
            stats = data.get("explanation_stats", {})
            
            # Buscar mean_size ou mean_length (peab/anchor/minexp podem usar ambos)
            pos_size = stats.get("positive", {}).get("mean_size", stats.get("positive", {}).get("mean_length", 0))
            neg_size = stats.get("negative", {}).get("mean_size", stats.get("negative", {}).get("mean_length", 0))
            pos_count = stats.get("positive", {}).get("count", 0)
            neg_count = stats.get("negative", {}).get("count", 0)
            
            if (pos_count + neg_count) > 0:
                classif_size = (pos_size * pos_count + neg_size * neg_count) / (pos_count + neg_count)
            else:
                classif_size = None
            
            rej_size = stats.get("rejected", {}).get("mean_size", stats.get("rejected", {}).get("mean_length", 0)) if stats.get("rejected", {}).get("count", 0) > 0 else 0
            
            return classif_size, rej_size
            
        elif metodo == "pulp":
            stats = data.get("estatisticas_por_tipo", {})
            
            pos_size = stats.get("positiva", {}).get("tamanho_medio", 0)
            neg_size = stats.get("negativa", {}).get("tamanho_medio", 0)
            pos_count = stats.get("positiva", {}).get("instancias", 0)
            neg_count = stats.get("negativa", {}).get("instancias", 0)
            
            if (pos_count + neg_count) > 0:
                classif_size = (pos_size * pos_count + neg_size * neg_count) / (pos_count + neg_count)
            else:
                classif_size = None
            
            rej_size = stats.get("rejeitada", {}).get("tamanho_medio", 0) if stats.get("rejeitada", {}).get("instancias", 0) > 0 else 0
            
            return classif_size, rej_size
    except Exception as e:
        print(f"Erro ao extrair tamanhos para {metodo}: {e}")
        return None, None


def extrair_tamanho_medio_std_por_tipo(data, metodo):
    """Extrai tamanho médio E desvio padrão de explicações por tipo.
    
    Retorna:
        (mean_class, std_class), (mean_rej, std_rej)
    """
    if data is None:
        return (None, None), (None, None)
    
    try:
        if metodo in ["peab", "anchor", "minexp"]:
            stats = data.get("explanation_stats", {})
            
            # Positivas
            pos_mean = stats.get("positive", {}).get("mean_size", stats.get("positive", {}).get("mean_length", 0))
            pos_std = stats.get("positive", {}).get("std_size", stats.get("positive", {}).get("std_length", 0))
            pos_count = stats.get("positive", {}).get("count", 0)
            
            # Negativas
            neg_mean = stats.get("negative", {}).get("mean_size", stats.get("negative", {}).get("mean_length", 0))
            neg_std = stats.get("negative", {}).get("std_size", stats.get("negative", {}).get("std_length", 0))
            neg_count = stats.get("negative", {}).get("count", 0)
            
            # Classificadas: média ponderada
            if (pos_count + neg_count) > 0:
                classif_mean = (pos_mean * pos_count + neg_mean * neg_count) / (pos_count + neg_count)
                # Desvio padrão combinado (pooled standard deviation)
                classif_std = ((pos_std**2 * pos_count + neg_std**2 * neg_count) / (pos_count + neg_count)) ** 0.5
            else:
                classif_mean, classif_std = None, None
            
            # Rejeitadas
            rej_count = stats.get("rejected", {}).get("count", 0)
            if rej_count > 0:
                rej_mean = stats.get("rejected", {}).get("mean_size", stats.get("rejected", {}).get("mean_length", 0))
                rej_std = stats.get("rejected", {}).get("std_size", stats.get("rejected", {}).get("std_length", 0))
            else:
                rej_mean, rej_std = 0.0, 0.0
            
            return (classif_mean, classif_std), (rej_mean, rej_std)
            
        elif metodo == "pulp":
            stats = data.get("estatisticas_por_tipo", {})
            
            # Positivas
            pos_mean = stats.get("positiva", {}).get("tamanho_medio", 0)
            pos_std = stats.get("positiva", {}).get("desvio_padrao", 0)
            pos_count = stats.get("positiva", {}).get("instancias", 0)
            
            # Negativas
            neg_mean = stats.get("negativa", {}).get("tamanho_medio", 0)
            neg_std = stats.get("negativa", {}).get("desvio_padrao", 0)
            neg_count = stats.get("negativa", {}).get("instancias", 0)
            
            # Classificadas
            if (pos_count + neg_count) > 0:
                classif_mean = (pos_mean * pos_count + neg_mean * neg_count) / (pos_count + neg_count)
                classif_std = ((pos_std**2 * pos_count + neg_std**2 * neg_count) / (pos_count + neg_count)) ** 0.5
            else:
                classif_mean, classif_std = None, None
            
            # Rejeitadas
            rej_count = stats.get("rejeitada", {}).get("instancias", 0)
            if rej_count > 0:
                rej_mean = stats.get("rejeitada", {}).get("tamanho_medio", 0)
                rej_std = stats.get("rejeitada", {}).get("desvio_padrao", 0)
            else:
                rej_mean, rej_std = 0.0, 0.0
            
            return (classif_mean, classif_std), (rej_mean, rej_std)
    except Exception as e:
        print(f"Erro ao extrair tamanhos/std para {metodo}: {e}")
        return (None, None), (None, None)


def extrair_necessidade(data):
    """Extrai métricas de necessidade do JSON de validação."""
    if data is None:
        return None, None
    
    try:
        per_type = data.get("per_type_metrics", {})
        
        pos_nec = per_type.get("positive", {}).get("necessity", None)
        neg_nec = per_type.get("negative", {}).get("necessity", None)
        rej_nec = per_type.get("rejected", {}).get("necessity", None)
        
        pos_count = per_type.get("positive", {}).get("count", 0)
        neg_count = per_type.get("negative", {}).get("count", 0)
        
        if pos_nec is not None and neg_nec is not None and (pos_count + neg_count) > 0:
            classif_nec = (pos_nec * pos_count + neg_nec * neg_count) / (pos_count + neg_count)
        else:
            classif_nec = None
        
        return classif_nec, rej_nec
    except Exception as e:
        print(f"Erro ao extrair necessidade: {e}")
        return None, None


def calcular_features_redundantes(data):
    """Calcula o número médio de features redundantes."""
    if data is None:
        return None, None
    
    try:
        per_type = data.get("per_type_metrics", {})
        
        pos_size = per_type.get("positive", {}).get("mean_size", 0)
        neg_size = per_type.get("negative", {}).get("mean_size", 0)
        pos_nec = per_type.get("positive", {}).get("necessity", 0)
        neg_nec = per_type.get("negative", {}).get("necessity", 0)
        pos_count = per_type.get("positive", {}).get("count", 0)
        neg_count = per_type.get("negative", {}).get("count", 0)
        
        if (pos_count + neg_count) > 0:
            classif_size = (pos_size * pos_count + neg_size * neg_count) / (pos_count + neg_count)
            classif_nec = (pos_nec * pos_count + neg_nec * neg_count) / (pos_count + neg_count)
            classif_red = classif_size * (1 - classif_nec / 100)
        else:
            classif_red = None
        
        rej_size = per_type.get("rejected", {}).get("mean_size", 0)
        rej_nec = per_type.get("rejected", {}).get("necessity", 0)
        rej_red = rej_size * (1 - rej_nec / 100) if rej_size > 0 else None
        
        return classif_red, rej_red
    except Exception as e:
        print(f"Erro ao calcular redundância: {e}")
        return None, None


def gerar_tabela_caracteristicas():
    """Gera tabela com características dos datasets."""
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Características dos datasets ordenados por número de features.}")
    latex.append("\\label{tab:mnist_characteristics}")
    latex.append("\\begin{tabular}{lrrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Dataset} & \\textbf{Inst.} & \\textbf{Feat.} & \\textbf{Limiares ($t^-$, $t^+$)} & \\textbf{Largura} \\\\")
    latex.append("\\hline")
    
    for dataset_config in DATASETS:
        data = carregar_dados_json(dataset_config, "peab")
        
        if data:
            # Buscar num de instancias
            total_instances = data.get("total_instances", data.get("performance", {}).get("num_test_instances", "N/A"))
            
            # Buscar num de features
            num_features = data.get("num_features", data.get("model", {}).get("num_features", dataset_config["num_features"]))
            
            if "thresholds" in data:
                t_plus = data["thresholds"]["t_plus"]
                t_minus = data["thresholds"]["t_minus"]
                largura = t_plus - t_minus
                threshold_str = f"({t_minus:.2f}, {t_plus:.2f})"
                largura_str = f"{largura:.2f}"
            else:
                threshold_str = "N/A"
                largura_str = "N/A"
            
            linha = f"{dataset_config['display_name']} & {total_instances} & {num_features} & {threshold_str} & {largura_str} \\\\"
            latex.append(linha)
        else:
            linha = f"{dataset_config['display_name']} & N/A & {dataset_config['num_features']} & N/A & N/A \\\\"
            latex.append(linha)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def gerar_tabela_speedup_classificadas():
    """Gera tabela com comparação de tempos para instâncias CLASSIFICADAS."""
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Tempo médio de execução por instância CLASSIFICADA (ms) e \\emph{speedup} do PEAB em relação aos baselines.}")
    latex.append("\\label{tab:mnist_runtime_classified}")
    latex.append("\\begin{tabular}{lrrrrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Dataset} & \\textbf{PEAB} & \\textbf{PULP} & \\textbf{Anchor} & \\textbf{MinExp} & \\textbf{Speedup (Anchor)} & \\textbf{Speedup (MinExp)} \\\\")
    latex.append("\\hline")
    
    for dataset_config in DATASETS:
        dados_tempo = {}
        
        for metodo in METODOS:
            data = carregar_dados_json(dataset_config, metodo)
            tempo_classif, _ = extrair_tempo_por_tipo(data, metodo)
            dados_tempo[metodo] = tempo_classif
        
        peab_time = dados_tempo.get("peab")
        pulp_time = dados_tempo.get("pulp")
        anchor_time = dados_tempo.get("anchor")
        minexp_time = dados_tempo.get("minexp")
        
        peab_str = f"{peab_time:.1f}" if peab_time is not None else "N/A"
        pulp_str = f"{pulp_time:.1f}" if pulp_time is not None else "N/A"
        anchor_str = f"{anchor_time:.1f}" if anchor_time is not None else "N/A"
        minexp_str = f"{minexp_time:.1f}" if minexp_time is not None else "N/A"
        
        if peab_time and anchor_time:
            speedup_anchor = anchor_time / peab_time
            speedup_anchor_str = f"{speedup_anchor:.1f}$\\times$"
        else:
            speedup_anchor_str = "N/A"
        
        if peab_time and minexp_time:
            speedup_minexp = minexp_time / peab_time
            speedup_minexp_str = f"{speedup_minexp:.1f}$\\times$"
        else:
            speedup_minexp_str = "N/A"
        
        linha = f"{dataset_config['display_name']} & {peab_str} & {pulp_str} & {anchor_str} & {minexp_str} & {speedup_anchor_str} & {speedup_minexp_str} \\\\"
        latex.append(linha)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def gerar_tabela_speedup_rejeitadas():
    """Gera tabela com comparação de tempos para instâncias REJEITADAS."""
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Tempo médio de execução por instância REJEITADA (ms) e \\emph{speedup} do PEAB em relação aos baselines.}")
    latex.append("\\label{tab:mnist_runtime_rejected}")
    latex.append("\\begin{tabular}{lrrrrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Dataset} & \\textbf{PEAB} & \\textbf{PULP} & \\textbf{Anchor} & \\textbf{MinExp} & \\textbf{Speedup (Anchor)} & \\textbf{Speedup (MinExp)} \\\\")
    latex.append("\\hline")
    
    for dataset_config in DATASETS:
        dados_tempo = {}
        
        for metodo in METODOS:
            data = carregar_dados_json(dataset_config, metodo)
            _, tempo_rej = extrair_tempo_por_tipo(data, metodo)
            dados_tempo[metodo] = tempo_rej
        
        peab_time = dados_tempo.get("peab")
        pulp_time = dados_tempo.get("pulp")
        anchor_time = dados_tempo.get("anchor")
        minexp_time = dados_tempo.get("minexp")
        
        peab_str = f"{peab_time:.1f}" if peab_time is not None and peab_time > 0 else "0.0"
        pulp_str = f"{pulp_time:.1f}" if pulp_time is not None and pulp_time > 0 else "N/A"
        anchor_str = f"{anchor_time:.1f}" if anchor_time is not None and anchor_time > 0 else "0.0"
        minexp_str = f"{minexp_time:.1f}" if minexp_time is not None and minexp_time > 0 else "0.0"
        
        if peab_time and peab_time > 0 and anchor_time and anchor_time > 0:
            speedup_anchor = anchor_time / peab_time
            speedup_anchor_str = f"{speedup_anchor:.1f}$\\times$"
        else:
            speedup_anchor_str = "N/A"
        
        if peab_time and peab_time > 0 and minexp_time and minexp_time > 0:
            speedup_minexp = minexp_time / peab_time
            speedup_minexp_str = f"{speedup_minexp:.1f}$\\times$"
        else:
            speedup_minexp_str = "N/A"
        
        linha = f"{dataset_config['display_name']} & {peab_str} & {pulp_str} & {anchor_str} & {minexp_str} & {speedup_anchor_str} & {speedup_minexp_str} \\\\"
        latex.append(linha)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def gerar_tabela_explicacoes():
    """Gera tabela com tamanho médio das explicações E desvio padrão."""
    
    def fmt_cell(mean_std):
        mean, std = mean_std
        if mean is None or std is None:
            return "N/A"
        return f"{mean:.2f} $\\pm$ {std:.2f}"
    
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Tamanho médio $\\pm$ desvio padrão das explicações (número de features).}")
    latex.append("\\label{tab:mnist_explanation_size}")
    latex.append("\\small")
    latex.append("\\setlength{\\tabcolsep}{4pt}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c}{\\textbf{PEAB}} & \\multicolumn{2}{c}{\\textbf{Anchor}} & \\multicolumn{2}{c}{\\textbf{MinExp}} \\\\")
    latex.append("\\cline{2-7}")
    latex.append(" & \\textbf{Clas.} & \\textbf{Rej.} & \\textbf{Clas.} & \\textbf{Rej.} & \\textbf{Clas.} & \\textbf{Rej.} \\\\")
    latex.append("\\hline")
    
    for dataset_config in DATASETS:
        data_peab = carregar_dados_json(dataset_config, "peab")
        data_anchor = carregar_dados_json(dataset_config, "anchor")
        data_minexp = carregar_dados_json(dataset_config, "minexp")
        
        peab_class, peab_rej = extrair_tamanho_medio_std_por_tipo(data_peab, "peab")
        anchor_class, anchor_rej = extrair_tamanho_medio_std_por_tipo(data_anchor, "anchor")
        minexp_class, minexp_rej = extrair_tamanho_medio_std_por_tipo(data_minexp, "minexp")
        
        linha = (
            f"{dataset_config['display_name']} & "
            f"{fmt_cell(peab_class)} & {fmt_cell(peab_rej)} & "
            f"{fmt_cell(anchor_class)} & {fmt_cell(anchor_rej)} & "
            f"{fmt_cell(minexp_class)} & {fmt_cell(minexp_rej)} \\\\"
        )
        latex.append(linha)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def gerar_tabela_necessidade():
    """Gera tabela com percentual de features necessárias."""
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Percentual médio de features necessárias nas explicações.}")
    latex.append("\\label{tab:mnist_necessity}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c}{\\textbf{PEAB}} & \\multicolumn{2}{c}{\\textbf{PULP}} \\\\")
    latex.append("\\cline{2-5}")
    latex.append(" & \\textbf{Classif.} & \\textbf{Rejeit.} & \\textbf{Classif.} & \\textbf{Rejeit.} \\\\")
    latex.append("\\hline")
    
    for dataset_config in DATASETS:
        dados_necessidade = {}
        
        for metodo in ["peab", "pulp"]:
            data = carregar_dados_validacao(dataset_config, metodo)
            classif_nec, rej_nec = extrair_necessidade(data)
            dados_necessidade[metodo] = {"classificadas": classif_nec, "rejeitadas": rej_nec}
        
        peab_classif = dados_necessidade["peab"]["classificadas"]
        peab_rej = dados_necessidade["peab"]["rejeitadas"]
        pulp_classif = dados_necessidade["pulp"]["classificadas"]
        pulp_rej = dados_necessidade["pulp"]["rejeitadas"]
        
        peab_c_str = f"{peab_classif:.1f}\\%" if peab_classif is not None else "N/A"
        peab_r_str = f"{peab_rej:.1f}\\%" if peab_rej is not None else "N/A"
        pulp_c_str = f"{pulp_classif:.1f}\\%" if pulp_classif is not None else "N/A"
        pulp_r_str = f"{pulp_rej:.1f}\\%" if pulp_rej is not None else "N/A"
        
        linha = f"{dataset_config['display_name']} & {peab_c_str} & {peab_r_str} & {pulp_c_str} & {pulp_r_str} \\\\"
        latex.append(linha)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def gerar_tabela_redundancia():
    """Gera tabela com número médio de features redundantes."""
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Número médio de features redundantes por explicação.}")
    latex.append("\\label{tab:mnist_redundancy}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c}{\\textbf{PEAB}} & \\multicolumn{2}{c}{\\textbf{PULP}} \\\\")
    latex.append("\\cline{2-5}")
    latex.append(" & \\textbf{Classif.} & \\textbf{Rejeit.} & \\textbf{Classif.} & \\textbf{Rejeit.} \\\\")
    latex.append("\\hline")
    
    for dataset_config in DATASETS:
        dados_redundancia = {}
        
        for metodo in ["peab", "pulp"]:
            data = carregar_dados_validacao(dataset_config, metodo)
            classif_red, rej_red = calcular_features_redundantes(data)
            dados_redundancia[metodo] = {"classificadas": classif_red, "rejeitadas": rej_red}
        
        peab_classif = dados_redundancia["peab"]["classificadas"]
        peab_rej = dados_redundancia["peab"]["rejeitadas"]
        pulp_classif = dados_redundancia["pulp"]["classificadas"]
        pulp_rej = dados_redundancia["pulp"]["rejeitadas"]
        
        peab_c_str = f"{peab_classif:.2f}" if peab_classif is not None else "N/A"
        peab_r_str = f"{peab_rej:.2f}" if peab_rej is not None else "N/A"
        pulp_c_str = f"{pulp_classif:.2f}" if pulp_classif is not None else "N/A"
        pulp_r_str = f"{pulp_rej:.2f}" if pulp_rej is not None else "N/A"
        
        linha = f"{dataset_config['display_name']} & {peab_c_str} & {peab_r_str} & {pulp_c_str} & {pulp_r_str} \\\\"
        latex.append(linha)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    print("=" * 70)
    print("GERADOR DE TABELAS LATEX - MÚLTIPLOS DATASETS")
    print("Datasets: 10 (tabela unificada de tempo)")
    print("=" * 70)
    print()
    
    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Gerar tabelas
    print("Gerando tabela de características dos datasets...")
    tabela_caracteristicas = gerar_tabela_caracteristicas()
    with open(f"{OUTPUT_DIR}/mnist_caracteristicas.tex", 'w', encoding='utf-8') as f:
        f.write(tabela_caracteristicas)
    print(f"✓ Tabela salva em: {OUTPUT_DIR}/mnist_caracteristicas.tex")
    
    print("\nGerando tabela de tempo (unificada, média $\\pm$ desvio padrão)...")
    tabela_runtime = gerar_tabela_runtime_unified()
    with open(f"{OUTPUT_DIR}/mnist_runtime_unified.tex", 'w', encoding='utf-8') as f:
        f.write(tabela_runtime)
    print(f"✓ Tabela salva em: {OUTPUT_DIR}/mnist_runtime_unified.tex")
    
    print("\nGerando tabela de tamanho de explicações...")
    tabela_explicacoes = gerar_tabela_explicacoes()
    with open(f"{OUTPUT_DIR}/mnist_explicacoes.tex", 'w', encoding='utf-8') as f:
        f.write(tabela_explicacoes)
    print(f"✓ Tabela salva em: {OUTPUT_DIR}/mnist_explicacoes.tex")
    
    print("\nGerando tabela de necessidade de features...")
    tabela_necessidade = gerar_tabela_necessidade()
    with open(f"{OUTPUT_DIR}/mnist_necessidade.tex", 'w', encoding='utf-8') as f:
        f.write(tabela_necessidade)
    print(f"✓ Tabela salva em: {OUTPUT_DIR}/mnist_necessidade.tex")
    
    print("\nGerando tabela de features redundantes...")
    tabela_redundancia = gerar_tabela_redundancia()
    with open(f"{OUTPUT_DIR}/mnist_redundancia.tex", 'w', encoding='utf-8') as f:
        f.write(tabela_redundancia)
    print(f"✓ Tabela salva em: {OUTPUT_DIR}/mnist_redundancia.tex")
    
    # Gerar arquivo completo
    print("\nGerando arquivo completo...")
    completo = f"""% Tabelas geradas automaticamente para múltiplos datasets
% Datasets: 10 datasets (ver tabela de runtime unificada)
% Gerado em: {Path.cwd()}

{tabela_caracteristicas}

{tabela_runtime}

{tabela_explicacoes}

{tabela_necessidade}

{tabela_redundancia}
"""
    
    with open(f"{OUTPUT_DIR}/mnist_tabelas_completas.tex", 'w', encoding='utf-8') as f:
        f.write(completo)
    print(f"✓ Arquivo completo salvo em: {OUTPUT_DIR}/mnist_tabelas_completas.tex")
    
    print("\n" + "=" * 70)
    print("TABELAS GERADAS COM SUCESSO!")
    print("=" * 70)
    print(f"\nArquivos criados em: {Path(OUTPUT_DIR).resolve()}")
    print("- mnist_caracteristicas.tex")
    print("- mnist_runtime_unified.tex")
    print("- mnist_explicacoes.tex")
    print("- mnist_necessidade.tex")
    print("- mnist_redundancia.tex")
    print("- mnist_tabelas_completas.tex")
    print()
    
    # Exibir prévia
    print("=" * 70)
    print("PRÉVIA: TABELA DE CARACTERÍSTICAS")
    print("=" * 70)
    print(tabela_caracteristicas)
    print()
    
    print("=" * 70)
    print("PRÉVIA: TABELA DE TEMPO (UNIFICADA)")
    print("=" * 70)
    print(tabela_runtime)
    print()


if __name__ == "__main__":
    main()
