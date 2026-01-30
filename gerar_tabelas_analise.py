"""
Script para gerar tabelas LaTeX comparando os métodos PEAB, PULP, Anchor e MinExp.
Gera:
1. Tabela de speedup (comparação de tempos)
2. Tabela de quantidade de explicações (positivas, negativas, rejeitadas)

As tabelas são geradas apenas para os datasets comuns a todos os métodos.
Resultados salvos em results/tabelas_latex/
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Datasets comuns a todos os métodos
DATASETS_COMUNS = [
    "banknote",
    "breast_cancer", 
    "heart_disease",
    "pima_indians_diabetes",
    "sonar",
    "spambase",
    "vertebral_column"
]

# Mapeamento de nomes para exibição
DATASET_NAMES = {
    "banknote": "Banknote",
    "breast_cancer": "Breast Cancer",
    "heart_disease": "Heart Disease",
    "pima_indians_diabetes": "Pima Indians",
    "sonar": "Sonar",
    "spambase": "Spambase",
    "vertebral_column": "Vertebral Column"
}

METODOS = ["peab", "pulp", "anchor", "minexp"]

# Datasets comuns que possuem validação (PEAB e PULP)
DATASETS_COM_VALIDACAO = [
    "banknote",
    "breast_cancer",
    "heart_disease",
    "pima_indians_diabetes",
    "sonar",
    "spambase",
    "vertebral_column"
]


def carregar_dados_json(metodo, dataset):
    """Carrega os dados de um JSON específico."""
    json_path = Path(f"json/{metodo}/{dataset}.json")
    
    if not json_path.exists():
        print(f"Aviso: {json_path} não encontrado")
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
            # Tempo em segundos, converter para ms
            tempo_s = data.get("computation_time", {}).get("mean_per_instance", None)
            return tempo_s * 1000 if tempo_s is not None else None
        elif metodo == "pulp":
            # Tempo em segundos, converter para ms
            tempo_s = data.get("estatisticas_gerais", {}).get("tempo_medio_segundos", None)
            return tempo_s * 1000 if tempo_s is not None else None
    except Exception as e:
        print(f"Erro ao extrair tempo para {metodo}: {e}")
        return None

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

    return "\n".join(latex)


def gerar_tabela_necessidade():
    """Gera tabela LaTeX com percentual de features necessárias (PEAB vs PULP)."""
    
    # Coletar dados de necessidade
    dados_necessidade = {}
    for dataset in DATASETS_COM_VALIDACAO:
        dados_necessidade[dataset] = {}
        for metodo in ["peab", "pulp"]:
            data = carregar_dados_validacao(metodo, dataset)
            classif_nec, rej_nec = extrair_necessidade(data)
            dados_necessidade[dataset][metodo] = {
                "classificadas": classif_nec,
                "rejeitadas": rej_nec
            }
    
    # Gerar tabela LaTeX
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Percentual médio de features necessárias nas explicações.}")
    latex.append("\\label{tab:necessity}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c}{\\textbf{PEAB}} & \\multicolumn{2}{c}{\\textbf{PULP}} \\\\")
    latex.append("\\cline{2-5}")
    latex.append(" & \\textbf{Classif.} & \\textbf{Rejeit.} & \\textbf{Classif.} & \\textbf{Rejeit.} \\\\")
    latex.append("\\hline")
    
    # Acumuladores para médias
    peab_classif_values = []
    peab_rej_values = []
    pulp_classif_values = []
    pulp_rej_values = []
    
    for dataset in DATASETS_COM_VALIDACAO:
        nome = DATASET_NAMES[dataset]
        dados = dados_necessidade[dataset]
        
        # PEAB
        peab_classif = dados["peab"]["classificadas"]
        peab_rej = dados["peab"]["rejeitadas"]
        
        # PULP
        pulp_classif = dados["pulp"]["classificadas"]
        pulp_rej = dados["pulp"]["rejeitadas"]
        
        # Formatar valores
        str_peab_classif = f"{peab_classif:.1f}\\%" if peab_classif is not None else "N/A"
        str_peab_rej = f"{peab_rej:.1f}\\%" if peab_rej is not None else "N/A"
        str_pulp_classif = f"{pulp_classif:.1f}\\%" if pulp_classif is not None else "N/A"
        str_pulp_rej = f"{pulp_rej:.1f}\\%" if pulp_rej is not None else "N/A"
        
        # Acumular para médias
        if peab_classif is not None:
            peab_classif_values.append(peab_classif)
        if peab_rej is not None:
            peab_rej_values.append(peab_rej)
        if pulp_classif is not None:
            pulp_classif_values.append(pulp_classif)
        if pulp_rej is not None:
            pulp_rej_values.append(pulp_rej)
        
        linha = f"{nome} & {str_peab_classif} & {str_peab_rej} & {str_pulp_classif} & {str_pulp_rej} \\\\"
        latex.append(linha)
    
    # Adicionar linha de médias
    latex.append("\\hline")
    
    media_peab_classif = f"{np.mean(peab_classif_values):.1f}\\%" if peab_classif_values else "N/A"
    media_peab_rej = f"{np.mean(peab_rej_values):.1f}\\%" if peab_rej_values else "N/A"
    media_pulp_classif = f"{np.mean(pulp_classif_values):.1f}\\%" if pulp_classif_values else "N/A"
    media_pulp_rej = f"{np.mean(pulp_rej_values):.1f}\\%" if pulp_rej_values else "N/A"
    
    latex.append(f"\\textbf{{Média}} & {media_peab_classif} & {media_peab_rej} & {media_pulp_classif} & {media_pulp_rej} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def gerar_tabela_redundancia():
    """Gera tabela LaTeX com número médio de features redundantes (PEAB vs PULP)."""
    
    # Coletar dados de redundância
    dados_redundancia = {}
    for dataset in DATASETS_COM_VALIDACAO:
        dados_redundancia[dataset] = {}
        for metodo in ["peab", "pulp"]:
            data = carregar_dados_validacao(metodo, dataset)
            classif_red, rej_red = calcular_features_redundantes(data)
            dados_redundancia[dataset][metodo] = {
                "classificadas": classif_red,
                "rejeitadas": rej_red
            }
    
    # Gerar tabela LaTeX
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Número médio de features redundantes por explicação.}")
    latex.append("\\label{tab:redundancy}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c}{\\textbf{PEAB}} & \\multicolumn{2}{c}{\\textbf{PULP}} \\\\")
    latex.append("\\cline{2-5}")
    latex.append(" & \\textbf{Classif.} & \\textbf{Rejeit.} & \\textbf{Classif.} & \\textbf{Rejeit.} \\\\")
    latex.append("\\hline")
    
    # Acumuladores para médias
    peab_classif_values = []
    peab_rej_values = []
    pulp_classif_values = []
    pulp_rej_values = []
    
    for dataset in DATASETS_COM_VALIDACAO:
        nome = DATASET_NAMES[dataset]
        dados = dados_redundancia[dataset]
        
        # PEAB
        peab_classif = dados["peab"]["classificadas"]
        peab_rej = dados["peab"]["rejeitadas"]
        
        # PULP
        pulp_classif = dados["pulp"]["classificadas"]
        pulp_rej = dados["pulp"]["rejeitadas"]
        
        # Formatar valores
        str_peab_classif = f"{peab_classif:.2f}" if peab_classif is not None else "N/A"
        str_peab_rej = f"{peab_rej:.2f}" if peab_rej is not None else "N/A"
        str_pulp_classif = f"{pulp_classif:.2f}" if pulp_classif is not None else "N/A"
        str_pulp_rej = f"{pulp_rej:.2f}" if pulp_rej is not None else "N/A"
        
        # Acumular para médias
        if peab_classif is not None:
            peab_classif_values.append(peab_classif)
        if peab_rej is not None:
            peab_rej_values.append(peab_rej)
        if pulp_classif is not None:
            pulp_classif_values.append(pulp_classif)
        if pulp_rej is not None:
            pulp_rej_values.append(pulp_rej)
        
        linha = f"{nome} & {str_peab_classif} & {str_peab_rej} & {str_pulp_classif} & {str_pulp_rej} \\\\"
        latex.append(linha)
    
    # Adicionar linha de médias
    latex.append("\\hline")
    
    media_peab_classif = f"{np.mean(peab_classif_values):.2f}" if peab_classif_values else "N/A"
    media_peab_rej = f"{np.mean(peab_rej_values):.2f}" if peab_rej_values else "N/A"
    media_pulp_classif = f"{np.mean(pulp_classif_values):.2f}" if pulp_classif_values else "N/A"
    media_pulp_rej = f"{np.mean(pulp_rej_values):.2f}" if pulp_rej_values else "N/A"
    
    latex.append(f"\\textbf{{Média}} & {media_peab_classif} & {media_peab_rej} & {media_pulp_classif} & {media_pulp_rej} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def carregar_dados_validacao(metodo, dataset):
    """Carrega os dados de validação de um JSON específico."""
    json_path = Path(f"json/validation/{metodo}_validation_{dataset}.json")
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler {json_path}: {e}")
        return None


def extrair_necessidade(data):
    """Extrai métricas de necessidade do JSON de validação."""
    if data is None:
        return None
    
    try:
        per_type = data.get("per_type_metrics", {})
        
        # Separar classificadas (positive + negative) de rejeitadas
        pos_nec = per_type.get("positive", {}).get("necessity", None)
        neg_nec = per_type.get("negative", {}).get("necessity", None)
        rej_nec = per_type.get("rejected", {}).get("necessity", None)
        
        pos_count = per_type.get("positive", {}).get("count", 0)
        neg_count = per_type.get("negative", {}).get("count", 0)
        
        # Calcular média ponderada para classificadas
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
        global_metrics = data.get("global_metrics", {})
        per_type = data.get("per_type_metrics", {})
        
        # Tamanho médio e necessidade para classificadas
        pos_size = per_type.get("positive", {}).get("mean_size", 0)
        neg_size = per_type.get("negative", {}).get("mean_size", 0)
        pos_nec = per_type.get("positive", {}).get("necessity", 0)
        neg_nec = per_type.get("negative", {}).get("necessity", 0)
        pos_count = per_type.get("positive", {}).get("count", 0)
        neg_count = per_type.get("negative", {}).get("count", 0)
        
        # Média ponderada para classificadas
        if (pos_count + neg_count) > 0:
            classif_size = (pos_size * pos_count + neg_size * neg_count) / (pos_count + neg_count)
            classif_nec = (pos_nec * pos_count + neg_nec * neg_count) / (pos_count + neg_count)
            classif_red = classif_size * (1 - classif_nec / 100)
        else:
            classif_red = None
        
        # Para rejeitadas
        rej_size = per_type.get("rejected", {}).get("mean_size", 0)
        rej_nec = per_type.get("rejected", {}).get("necessity", 0)
        rej_red = rej_size * (1 - rej_nec / 100) if rej_size > 0 else None
        
        return classif_red, rej_red
    except Exception as e:
        print(f"Erro ao calcular redundância: {e}")
        return None, None


def extrair_tempo_por_tipo(data, metodo):
    """Extrai tempo médio separado por tipo: classificadas e rejeitadas."""
    if data is None:
        return None, None
    
    try:
        if metodo in ["peab", "anchor", "minexp"]:
            # Buscar tempos em computation_time
            comp_time = data.get("computation_time", {})
            
            # Tempo para classificadas (positivas + negativas)
            pos_time = comp_time.get("positive", None)
            neg_time = comp_time.get("negative", None)
            
            # Contar instâncias para ponderação
            stats = data.get("explanation_stats", {})
            pos_count = stats.get("positive", {}).get("count", 0)
            neg_count = stats.get("negative", {}).get("count", 0)
            
            # Tempo para rejeitadas
            rej_time = comp_time.get("rejected", None)
            
            # Calcular tempo médio ponderado para classificadas
            if pos_time is not None and neg_time is not None and (pos_count + neg_count) > 0:
                classif_time = (pos_time * pos_count + neg_time * neg_count) / (pos_count + neg_count)
                # Converter para ms
                classif_time_ms = classif_time * 1000
            else:
                classif_time_ms = None
            
            # Converter tempo rejeitadas para ms
            rej_time_ms = rej_time * 1000 if rej_time is not None else None
            
            return classif_time_ms, rej_time_ms
            
        elif metodo == "pulp":
            stats = data.get("estatisticas_por_tipo", {})
            
            # Tempo para classificadas (positivas + negativas)
            pos_time = stats.get("positiva", {}).get("tempo_medio", None)
            neg_time = stats.get("negativa", {}).get("tempo_medio", None)
            pos_count = stats.get("positiva", {}).get("instancias", 0)
            neg_count = stats.get("negativa", {}).get("instancias", 0)
            
            # Tempo para rejeitadas
            rej_time = stats.get("rejeitada", {}).get("tempo_medio", None)
            
            # Calcular tempo médio ponderado para classificadas
            if pos_time is not None and neg_time is not None and (pos_count + neg_count) > 0:
                classif_time = (pos_time * pos_count + neg_time * neg_count) / (pos_count + neg_count)
                # Converter para ms
                classif_time_ms = classif_time * 1000
            else:
                classif_time_ms = None
            
            # Converter tempo rejeitadas para ms
            rej_time_ms = rej_time * 1000 if rej_time is not None else None
            
            return classif_time_ms, rej_time_ms
    except Exception as e:
        print(f"Erro ao extrair tempo por tipo para {metodo}: {e}")
        return None, None


def gerar_tabela_speedup_classificadas():
    """Gera tabela LaTeX com comparação de tempos (speedup) para instâncias CLASSIFICADAS."""
    
    # Coletar dados de tempo para classificadas
    dados_tempo = {}
    for dataset in DATASETS_COMUNS:
        dados_tempo[dataset] = {}
        for metodo in METODOS:
            data = carregar_dados_json(metodo, dataset)
            tempo_classif, _ = extrair_tempo_por_tipo(data, metodo)
            dados_tempo[dataset][metodo] = tempo_classif
    
    # Gerar tabela LaTeX
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Tempo médio de execução por instância CLASSIFICADA (ms) e \\emph{speedup} do PEAB em relação aos baselines.}")
    latex.append("\\label{tab:runtime_classified}")
    latex.append("\\begin{tabular}{lrrrrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Dataset} & \\textbf{PEAB} & \\textbf{PULP} & \\textbf{Anchor} & \\textbf{MinExp} & \\textbf{Speedup (Anchor)} & \\textbf{Speedup (MinExp)} \\\\")
    latex.append("\\hline")
    
    speedups_pulp = []
    speedups_anchor = []
    speedups_minexp = []
    
    for dataset in DATASETS_COMUNS:
        nome = DATASET_NAMES[dataset]
        tempos = dados_tempo[dataset]
        
        # Valores de tempo em ms
        t_peab = tempos.get("peab")
        t_pulp = tempos.get("pulp")
        t_anchor = tempos.get("anchor")
        t_minexp = tempos.get("minexp")
        
        # Formatar tempos com 1 casa decimal
        str_peab = f"{t_peab:.1f}" if t_peab is not None else "N/A"
        str_pulp = f"{t_pulp:.1f}" if t_pulp is not None else "N/A"
        str_anchor = f"{t_anchor:.1f}" if t_anchor is not None else "N/A"
        str_minexp = f"{t_minexp:.1f}" if t_minexp is not None else "N/A"
        
        # Calcular speedups
        speedup_pulp = ""
        speedup_anchor = ""
        speedup_minexp = ""
        
        if t_peab is not None and t_pulp is not None and t_peab > 0:
            sp = t_pulp / t_peab
            speedup_pulp = f"{sp:.1f}$\\times$"
            speedups_pulp.append(sp)
        else:
            speedup_pulp = "N/A"
        
        if t_peab is not None and t_anchor is not None and t_peab > 0:
            sp = t_anchor / t_peab
            speedup_anchor = f"{sp:.1f}$\\times$"
            speedups_anchor.append(sp)
        else:
            speedup_anchor = "N/A"
        
        if t_peab is not None and t_minexp is not None and t_peab > 0:
            sp = t_minexp / t_peab
            speedup_minexp = f"{sp:.1f}$\\times$"
            speedups_minexp.append(sp)
        else:
            speedup_minexp = "N/A"
        
        linha = f"{nome} & {str_peab} & {str_pulp} & {str_anchor} & {str_minexp} & {speedup_anchor} & {speedup_minexp} \\\\"
        latex.append(linha)
    
    # Adicionar linha de média dos speedups
    latex.append("\\hline")
    
    media_anchor = f"{np.mean(speedups_anchor):.1f}$\\times$" if speedups_anchor else "N/A"
    media_minexp = f"{np.mean(speedups_minexp):.1f}$\\times$" if speedups_minexp else "N/A"
    
    latex.append(f"\\textbf{{Média}} & - & - & - & - & {media_anchor} & {media_minexp} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def gerar_tabela_speedup_rejeitadas():
    """Gera tabela LaTeX com comparação de tempos (speedup) para instâncias REJEITADAS."""
    
    # Coletar dados de tempo para rejeitadas
    dados_tempo = {}
    for dataset in DATASETS_COMUNS:
        dados_tempo[dataset] = {}
        for metodo in METODOS:
            data = carregar_dados_json(metodo, dataset)
            _, tempo_rej = extrair_tempo_por_tipo(data, metodo)
            dados_tempo[dataset][metodo] = tempo_rej
    
    # Gerar tabela LaTeX
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Tempo médio de execução por instância REJEITADA (ms) e \\emph{speedup} do PEAB em relação aos baselines.}")
    latex.append("\\label{tab:runtime_rejected}")
    latex.append("\\begin{tabular}{lrrrrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Dataset} & \\textbf{PEAB} & \\textbf{PULP} & \\textbf{Anchor} & \\textbf{MinExp} & \\textbf{Speedup (Anchor)} & \\textbf{Speedup (MinExp)} \\\\")
    latex.append("\\hline")
    
    speedups_pulp = []
    speedups_anchor = []
    speedups_minexp = []
    
    for dataset in DATASETS_COMUNS:
        nome = DATASET_NAMES[dataset]
        tempos = dados_tempo[dataset]
        
        # Valores de tempo em ms
        t_peab = tempos.get("peab")
        t_pulp = tempos.get("pulp")
        t_anchor = tempos.get("anchor")
        t_minexp = tempos.get("minexp")
        
        # Formatar tempos com 1 casa decimal
        str_peab = f"{t_peab:.1f}" if t_peab is not None else "N/A"
        str_pulp = f"{t_pulp:.1f}" if t_pulp is not None else "N/A"
        str_anchor = f"{t_anchor:.1f}" if t_anchor is not None else "N/A"
        str_minexp = f"{t_minexp:.1f}" if t_minexp is not None else "N/A"
        
        # Calcular speedups
        speedup_pulp = ""
        speedup_anchor = ""
        speedup_minexp = ""
        
        if t_peab is not None and t_pulp is not None and t_peab > 0:
            sp = t_pulp / t_peab
            speedup_pulp = f"{sp:.1f}$\\times$"
            speedups_pulp.append(sp)
        else:
            speedup_pulp = "N/A"
        
        if t_peab is not None and t_anchor is not None and t_peab > 0:
            sp = t_anchor / t_peab
            speedup_anchor = f"{sp:.1f}$\\times$"
            speedups_anchor.append(sp)
        else:
            speedup_anchor = "N/A"
        
        if t_peab is not None and t_minexp is not None and t_peab > 0:
            sp = t_minexp / t_peab
            speedup_minexp = f"{sp:.1f}$\\times$"
            speedups_minexp.append(sp)
        else:
            speedup_minexp = "N/A"
        
        linha = f"{nome} & {str_peab} & {str_pulp} & {str_anchor} & {str_minexp} & {speedup_anchor} & {speedup_minexp} \\\\"
        latex.append(linha)
    
    # Adicionar linha de média dos speedups
    latex.append("\\hline")
    
    media_anchor = f"{np.mean(speedups_anchor):.1f}$\\times$" if speedups_anchor else "N/A"
    media_minexp = f"{np.mean(speedups_minexp):.1f}$\\times$" if speedups_minexp else "N/A"
    
    latex.append(f"\\textbf{{Média}} & - & - & - & - & {media_anchor} & {media_minexp} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def extrair_tamanhos_explicacoes(data, metodo):
    """Extrai os tamanhos médios de explicações por tipo."""
    if data is None:
        return None, None, None
    
    try:
        if metodo in ["peab", "anchor", "minexp"]:
            stats = data.get("explanation_stats", {})
            pos = stats.get("positive", {}).get("mean_length", None)
            neg = stats.get("negative", {}).get("mean_length", None)
            rej = stats.get("rejected", {}).get("mean_length", None)
            return pos, neg, rej
        elif metodo == "pulp":
            stats = data.get("estatisticas_por_tipo", {})
            pos = stats.get("positiva", {}).get("tamanho_medio", None)
            neg = stats.get("negativa", {}).get("tamanho_medio", None)
            rej = stats.get("rejeitada", {}).get("tamanho_medio", None)
            return pos, neg, rej
    except Exception as e:
        print(f"Erro ao extrair tamanhos para {metodo}: {e}")
        return None, None, None


def gerar_tabela_explicacoes():
    """Gera tabela LaTeX com tamanho médio de explicações separando classificadas de rejeitadas."""
    
    # Coletar dados de tamanhos
    dados_tamanhos = {}
    for dataset in DATASETS_COMUNS:
        dados_tamanhos[dataset] = {}
        for metodo in METODOS:
            data = carregar_dados_json(metodo, dataset)
            pos, neg, rej = extrair_tamanhos_explicacoes(data, metodo)
            dados_tamanhos[dataset][metodo] = {"pos": pos, "neg": neg, "rej": rej}
    
    # Gerar tabela LaTeX
    latex = []
    latex.append("\\begin{table}[!t]")
    latex.append("\\centering")
    latex.append("\\caption{Tamanho médio das explicações (número de features): Classificadas vs Rejeitadas.}")
    latex.append("\\label{tab:explanation_size}")
    latex.append("\\begin{tabular}{lcccccccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c}{\\textbf{PEAB}} & \\multicolumn{2}{c}{\\textbf{PULP}} & \\multicolumn{2}{c}{\\textbf{Anchor}} & \\multicolumn{2}{c}{\\textbf{MinExp}} \\\\")
    latex.append("\\cline{2-9}")
    latex.append(" & \\textbf{Classif.} & \\textbf{Rejeit.} & \\textbf{Classif.} & \\textbf{Rejeit.} & \\textbf{Classif.} & \\textbf{Rejeit.} & \\textbf{Classif.} & \\textbf{Rejeit.} \\\\")
    latex.append("\\hline")
    
    # Acumuladores para médias
    medias_classif = {"peab": [], "pulp": [], "anchor": [], "minexp": []}
    medias_rej = {"peab": [], "pulp": [], "anchor": [], "minexp": []}
    
    for dataset in DATASETS_COMUNS:
        nome = DATASET_NAMES[dataset]
        dados = dados_tamanhos[dataset]
        
        # Para cada método, calcular média de classificadas e pegar rejeitadas
        valores_linha = []
        
        for metodo in METODOS:
            pos = dados[metodo]["pos"]
            neg = dados[metodo]["neg"]
            rej = dados[metodo]["rej"]
            
            # Obter contagens para média ponderada de classificadas
            data_orig = carregar_dados_json(metodo, dataset)
            pos_count, neg_count, rej_count = extrair_contagens_explicacoes(data_orig, metodo)
            
            # Média ponderada de classificadas (positivas + negativas)
            if (pos is not None and neg is not None and 
                pos_count is not None and neg_count is not None and
                (pos_count + neg_count) > 0):
                media_classif = (pos * pos_count + neg * neg_count) / (pos_count + neg_count)
                str_classif = f"{media_classif:.2f}"
                medias_classif[metodo].append(media_classif)
            else:
                str_classif = "N/A"
            
            # Rejeitadas
            if rej is not None:
                str_rej = f"{rej:.2f}"
                medias_rej[metodo].append(rej)
            else:
                str_rej = "N/A"
            
            valores_linha.extend([str_classif, str_rej])
        
        linha = f"{nome} & {' & '.join(valores_linha)} \\\\"
        latex.append(linha)
    
    # Adicionar linha de médias
    latex.append("\\hline")
    
    media_peab_c = f"{np.mean(medias_classif['peab']):.2f}" if medias_classif['peab'] else "N/A"
    media_peab_r = f"{np.mean(medias_rej['peab']):.2f}" if medias_rej['peab'] else "N/A"
    media_pulp_c = f"{np.mean(medias_classif['pulp']):.2f}" if medias_classif['pulp'] else "N/A"
    media_pulp_r = f"{np.mean(medias_rej['pulp']):.2f}" if medias_rej['pulp'] else "N/A"
    media_anchor_c = f"{np.mean(medias_classif['anchor']):.2f}" if medias_classif['anchor'] else "N/A"
    media_anchor_r = f"{np.mean(medias_rej['anchor']):.2f}" if medias_rej['anchor'] else "N/A"
    media_minexp_c = f"{np.mean(medias_classif['minexp']):.2f}" if medias_classif['minexp'] else "N/A"
    media_minexp_r = f"{np.mean(medias_rej['minexp']):.2f}" if medias_rej['minexp'] else "N/A"
    
    latex.append(f"\\textbf{{Média}} & {media_peab_c} & {media_peab_r} & {media_pulp_c} & {media_pulp_r} & {media_anchor_c} & {media_anchor_r} & {media_minexp_c} & {media_minexp_r} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def extrair_zona_rejeicao(data):
    """Extrai t+ e t- dos dados do PEAB."""
    if data is None:
        return None, None, None
    
    try:
        thresholds = data.get("thresholds", {})
        t_plus = thresholds.get("t_plus", None)
        t_minus = thresholds.get("t_minus", None)
        
        if t_plus is not None and t_minus is not None:
            largura = t_plus - t_minus
            return t_plus, t_minus, largura
        
        return None, None, None
    except Exception as e:
        print(f"Erro ao extrair zona de rejeição: {e}")
        return None, None, None


def gerar_grafico_zona_rejeicao(output_dir):
    """Gera gráfico de barras com a largura da zona de rejeição por dataset."""
    
    # Datasets selecionados conforme solicitação
    datasets_selecionados = [
        "banknote",
        "heart_disease", 
        "pima_indians_diabetes",
        "spambase",
        "vertebral_column"
    ]
    
    # Nomes para exibição
    nomes_display = {
        "banknote": "Banknote",
        "heart_disease": "Heart Disease",
        "pima_indians_diabetes": "Pima Indians",
        "spambase": "Spambase",
        "vertebral_column": "Vertebral Column"
    }
    
    # Coletar dados
    larguras = []
    nomes = []
    
    print("\nColetando dados da zona de rejeição...")
    for dataset in datasets_selecionados:
        data = carregar_dados_json("peab", dataset)
        t_plus, t_minus, largura = extrair_zona_rejeicao(data)
        
        if largura is not None:
            larguras.append(largura)
            nomes.append(nomes_display[dataset])
            print(f"  {nomes_display[dataset]}: t+ = {t_plus:.6f}, t- = {t_minus:.6f}, largura = {largura:.6f}")
        else:
            print(f"  {nomes_display[dataset]}: Dados não disponíveis")
    
    if not larguras:
        print("⚠ Nenhum dado disponível para gerar o gráfico.")
        return
    
    # Criar gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(nomes)), larguras, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Configurar eixos
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('t+ - t- (largura da zona de rejeição)', fontsize=12, fontweight='bold')
    plt.title('Largura da Zona de Rejeição por Dataset', fontsize=14, fontweight='bold')
    
    # Configurar labels do eixo X
    plt.xticks(range(len(nomes)), nomes, rotation=45, ha='right')
    
    # Adicionar valores sobre as barras
    for i, (nome, valor) in enumerate(zip(nomes, larguras)):
        plt.text(i, valor, f'{valor:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    grafico_file = output_dir / "rejection_zone.png"
    plt.savefig(grafico_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico da zona de rejeição salvo em: {grafico_file}")


def gerar_grafico_intervalos_horizontais(output_dir):
    """Gera gráfico de intervalos horizontais mostrando a zona de rejeição no contexto do espaço de decisão."""
    
    # Datasets selecionados conforme solicitação
    datasets_selecionados = [
        "banknote",
        "heart_disease", 
        "pima_indians_diabetes",
        "spambase",
        "vertebral_column"
    ]
    
    # Nomes para exibição
    nomes_display = {
        "banknote": "Banknote",
        "heart_disease": "Heart Disease",
        "pima_indians_diabetes": "Pima Indians",
        "spambase": "Spambase",
        "vertebral_column": "Vertebral Column"
    }
    
    # Cores para cada dataset
    cores = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Coletar dados
    dados = []
    
    print("\nColetando dados para gráfico de intervalos...")
    for dataset in datasets_selecionados:
        data = carregar_dados_json("peab", dataset)
        t_plus, t_minus, largura = extrair_zona_rejeicao(data)
        
        if t_plus is not None and t_minus is not None:
            dados.append({
                'nome': nomes_display[dataset],
                't_minus': t_minus,
                't_plus': t_plus,
                'largura': largura
            })
            print(f"  {nomes_display[dataset]}: [{t_minus:.4f}, {t_plus:.4f}]")
    
    if not dados:
        print("⚠ Nenhum dado disponível para gerar o gráfico de intervalos.")
        return
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotar intervalos horizontais (zonas de rejeição)
    for i, info in enumerate(dados):
        # Barra horizontal da zona de rejeição
        ax.barh(i, info['largura'], left=info['t_minus'], height=0.6, 
                color=cores[i], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Marcadores nos extremos
        ax.plot(info['t_minus'], i, 'o', color='darkred', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.plot(info['t_plus'], i, 'o', color='darkgreen', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        
        # Adicionar valores de t- e t+
        # Se a largura for pequena, posicionar os textos de forma alternada
        if info['largura'] < 0.15:  # Zona pequena
            # t- abaixo
            ax.text(info['t_minus'], i - 0.35, f"t-={info['t_minus']:.3f}", 
                    ha='center', va='top', fontsize=9, fontweight='bold')
            # t+ acima
            ax.text(info['t_plus'], i + 0.35, f"t+={info['t_plus']:.3f}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:  # Zona normal
            ax.text(info['t_minus'], i - 0.35, f"t-={info['t_minus']:.3f}", 
                    ha='center', va='top', fontsize=9, fontweight='bold')
            ax.text(info['t_plus'], i - 0.35, f"t+={info['t_plus']:.3f}", 
                    ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Linha vertical no limiar de decisão (0)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2.5, zorder=3)
    
    # Adicionar região sombreada para contexto
    y_min, y_max = -0.5, len(dados) - 0.5
    ax.axvspan(-10, 0, alpha=0.1, color='blue')
    ax.axvspan(0, 10, alpha=0.1, color='red')
    
    # Configurações dos eixos
    ax.set_yticks(range(len(dados)))
    ax.set_yticklabels([d['nome'] for d in dados], fontsize=11, fontweight='bold')
    ax.set_xlabel('Classifier Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_title('Rejection Zone in Classifier Decision Space', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Ajustar limites do eixo X para mostrar contexto
    all_values = [d['t_minus'] for d in dados] + [d['t_plus'] for d in dados]
    x_min, x_max = min(all_values), max(all_values)
    margin = (x_max - x_min) * 0.3
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(-0.5, len(dados) - 0.5)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    grafico_file = output_dir / "rejection_zone_intervals.png"
    plt.savefig(grafico_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico de intervalos da zona de rejeição salvo em: {grafico_file}")


def main():
    """Função principal que gera todas as tabelas."""
    
    # Criar diretório de saída se não existir
    output_dir = Path("results/tabelas_latex")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GERADOR DE TABELAS LATEX - COMPARAÇÃO DE MÉTODOS")
    print("="*70)
    print(f"\nDatasets comuns: {len(DATASETS_COMUNS)}")
    print(f"Métodos: {', '.join(METODOS).upper()}")
    print()
    
    # Gerar tabelas de speedup (classificadas e rejeitadas)
    print("Gerando tabela de speedup para instâncias CLASSIFICADAS...")
    tabela_speedup_classif = gerar_tabela_speedup_classificadas()
    
    # Salvar tabela de speedup classificadas
    speedup_classif_file = output_dir / "tabela_speedup_classificadas.tex"
    with open(speedup_classif_file, 'w', encoding='utf-8') as f:
        f.write(tabela_speedup_classif)
    print(f"✓ Tabela de speedup (classificadas) salva em: {speedup_classif_file}")
    
    print("\nGerando tabela de speedup para instâncias REJEITADAS...")
    tabela_speedup_rej = gerar_tabela_speedup_rejeitadas()
    
    # Salvar tabela de speedup rejeitadas
    speedup_rej_file = output_dir / "tabela_speedup_rejeitadas.tex"
    with open(speedup_rej_file, 'w', encoding='utf-8') as f:
        f.write(tabela_speedup_rej)
    print(f"✓ Tabela de speedup (rejeitadas) salva em: {speedup_rej_file}")
    
    # Gerar tabela de explicações
    print("\nGerando tabela de quantidade de explicações...")
    tabela_explicacoes = gerar_tabela_explicacoes()
    
    # Salvar tabela de explicações
    explicacoes_file = output_dir / "tabela_explicacoes.tex"
    with open(explicacoes_file, 'w', encoding='utf-8') as f:
        f.write(tabela_explicacoes)
    print(f"✓ Tabela de explicações salva em: {explicacoes_file}")
    
    # Gerar tabela de necessidade
    print("\nGerando tabela de necessidade de features...")
    tabela_necessidade = gerar_tabela_necessidade()
    
    # Salvar tabela de necessidade
    necessidade_file = output_dir / "tabela_necessidade.tex"
    with open(necessidade_file, 'w', encoding='utf-8') as f:
        f.write(tabela_necessidade)
    print(f"✓ Tabela de necessidade salva em: {necessidade_file}")
    
    # Gerar tabela de redundância
    print("\nGerando tabela de features redundantes...")
    tabela_redundancia = gerar_tabela_redundancia()
    
    # Salvar tabela de redundância
    redundancia_file = output_dir / "tabela_redundancia.tex"
    with open(redundancia_file, 'w', encoding='utf-8') as f:
        f.write(tabela_redundancia)
    print(f"✓ Tabela de redundância salva em: {redundancia_file}")
    
    # Gerar gráfico da zona de rejeição
    print("\nGerando gráfico da zona de rejeição...")
    gerar_grafico_zona_rejeicao(output_dir)
    
    # Gerar gráfico de intervalos horizontais
    print("\nGerando gráfico de intervalos horizontais...")
    gerar_grafico_intervalos_horizontais(output_dir)
    
    # Gerar arquivo completo com ambas as tabelas
    print("\nGerando arquivo completo...")
    completo_file = output_dir / "tabelas_completas.tex"
    with open(completo_file, 'w', encoding='utf-8') as f:
        f.write("% Tabela 1: Speedup Classificadas (Comparação de Tempos)\n")
        f.write(tabela_speedup_classif)
        f.write("\n\n")
        f.write("% Tabela 2: Speedup Rejeitadas (Comparação de Tempos)\n")
        f.write(tabela_speedup_rej)
        f.write("\n\n")
        f.write("% Tabela 3: Quantidade de Explicações\n")
        f.write(tabela_explicacoes)
        f.write("\n\n")
        f.write("% Tabela 4: Percentual de Features Necessárias\n")
        f.write(tabela_necessidade)
        f.write("\n\n")
        f.write("% Tabela 5: Features Redundantes Médias\n")
        f.write(tabela_redundancia)
    print(f"✓ Arquivo completo salvo em: {completo_file}")
    
    print("\n" + "="*70)
    print("TABELAS GERADAS COM SUCESSO!")
    print("="*70)
    print(f"\nArquivos criados em: {output_dir.absolute()}")
    print("- tabela_speedup_classificadas.tex")
    print("- tabela_speedup_rejeitadas.tex")
    print("- tabela_explicacoes.tex")
    print("- tabela_necessidade.tex")
    print("- tabela_redundancia.tex")
    print("- tabelas_completas.tex")
    print("- rejection_zone.png")
    print("- rejection_zone_intervals.png")
    print()
    
    # Mostrar prévia das tabelas de speedup
    print("\n" + "="*70)
    print("PRÉVIA: TABELA DE SPEEDUP (CLASSIFICADAS)")
    print("="*70)
    print(tabela_speedup_classif)
    
    print("\n" + "="*70)
    print("PRÉVIA: TABELA DE SPEEDUP (REJEITADAS)")
    print("="*70)
    print(tabela_speedup_rej)
    
    print("\n" + "="*70)
    print("PRÉVIA: TABELA DE EXPLICAÇÕES")
    print("="*70)
    print(tabela_explicacoes)
    
    print("\n" + "="*70)
    print("PRÉVIA: TABELA DE NECESSIDADE")
    print("="*70)
    print(tabela_necessidade)
    
    print("\n" + "="*70)
    print("PRÉVIA: TABELA DE REDUNDÂNCIA")
    print("="*70)
    print(tabela_redundancia)


if __name__ == "__main__":
    main()
