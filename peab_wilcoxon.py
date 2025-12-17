"""
================================================================================
PEAB - TESTES ESTATÍSTICOS DE SIGNIFICÂNCIA
================================================================================

Script para testar se diferenças entre PEAB e métodos baseline (MinExp, Anchor, 
PULP) são estatisticamente significativas.

Fundamentação Teórica:
- Wilcoxon Signed-Rank Test: Teste não-paramétrico para amostras pareadas
- Teste t-pareado: Teste paramétrico (assume normalidade)
- Correção de Bonferroni: Ajusta p-value para múltiplas comparações
- Tamanho de Efeito (Cohen's d): Mede magnitude da diferença

Referências:
- Demšar (2006): "Statistical Comparisons of Classifiers over Multiple Data Sets"
- Wilcoxon (1945): "Individual Comparisons by Ranking Methods"

Autor: Sistema XAI com Rejection Option
Data: 2025
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ==============================================================================
# CONSTANTES
# ==============================================================================
RESULTS_DIR = 'json'
METHODS = ['peab', 'minexp', 'anchor', 'pulp']
OUTPUT_DIR = 'results/statistical_tests'
ALPHA = 0.05  # Nível de significância

# ==============================================================================
# FUNÇÕES DE CARGA DE DADOS
# ==============================================================================

def carregar_resultados_metodo(method: str) -> Dict:
    """Carrega resultados JSON de um método."""
    filepath = os.path.join(RESULTS_DIR, f'{method}_results.json')
    if not os.path.exists(filepath):
        print(f"⚠️  Arquivo não encontrado: {filepath}")
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)

def extrair_metricas_por_dataset(method: str) -> Dict[str, Dict]:
    """
    Extrai métricas principais de cada dataset.
    
    Returns:
        Dict com formato: {dataset_name: {metrica: valor, ...}}
    """
    resultados = carregar_resultados_metodo(method)
    metricas_datasets = {}
    
    for dataset_name, data in resultados.items():
        if not isinstance(data, dict):
            continue
            
        metricas = {}
        
        # Métricas de performance
        if 'performance' in data:
            perf = data['performance']
            metricas['accuracy_with_rejection'] = perf.get('accuracy_with_rejection', None)
            metricas['accuracy_without_rejection'] = perf.get('accuracy_without_rejection', None)
            metricas['rejection_rate'] = perf.get('rejection_rate', None)
            metricas['num_test_instances'] = perf.get('num_test_instances', None)
        
        # Tamanho médio das explicações (PRINCIPAL MÉTRICA DE COMPARAÇÃO)
        tamanhos = []
        for tipo in ['positive', 'negative', 'rejected']:
            if 'explanation_stats' in data and tipo in data['explanation_stats']:
                stats_tipo = data['explanation_stats'][tipo]
                count = stats_tipo.get('count', 0)
                mean_length = stats_tipo.get('mean_length', 0)
                # Multiplicar para obter total de features
                if count > 0 and mean_length > 0:
                    tamanhos.extend([mean_length] * count)
        
        if tamanhos:
            metricas['mean_explanation_size'] = float(np.mean(tamanhos))
            metricas['std_explanation_size'] = float(np.std(tamanhos))
            metricas['median_explanation_size'] = float(np.median(tamanhos))
            metricas['all_sizes'] = tamanhos  # Para testes pareados
        
        # Tempo computacional
        if 'computation_time' in data:
            comp_time = data['computation_time']
            metricas['total_time'] = comp_time.get('total_explanation_time', None)
            metricas['mean_time_per_instance'] = comp_time.get('mean_time_per_instance', None)
        
        metricas_datasets[dataset_name] = metricas
    
    return metricas_datasets

# ==============================================================================
# TESTES ESTATÍSTICOS
# ==============================================================================

def wilcoxon_signed_rank_test(values1: List[float], values2: List[float]) -> Tuple[float, float]:
    """
    Teste de Wilcoxon para amostras pareadas.
    
    H0: As medianas das duas amostras são iguais
    H1: As medianas são diferentes
    
    Returns:
        statistic: Estatística de teste
        p_value: P-value (bilateral)
    """
    if len(values1) != len(values2):
        raise ValueError("Amostras devem ter mesmo tamanho")
    
    if len(values1) < 3:
        return np.nan, np.nan
    
    try:
        statistic, p_value = stats.wilcoxon(values1, values2, alternative='two-sided')
        return statistic, p_value
    except Exception as e:
        print(f"⚠️  Erro no teste de Wilcoxon: {e}")
        return np.nan, np.nan

def paired_t_test(values1: List[float], values2: List[float]) -> Tuple[float, float]:
    """
    Teste t-pareado para amostras dependentes.
    
    Assume normalidade das diferenças.
    
    Returns:
        statistic: Estatística t
        p_value: P-value (bilateral)
    """
    if len(values1) != len(values2):
        raise ValueError("Amostras devem ter mesmo tamanho")
    
    if len(values1) < 3:
        return np.nan, np.nan
    
    try:
        statistic, p_value = stats.ttest_rel(values1, values2)
        return statistic, p_value
    except Exception as e:
        print(f"⚠️  Erro no teste t-pareado: {e}")
        return np.nan, np.nan

def cohen_d(values1: List[float], values2: List[float]) -> float:
    """
    Calcula tamanho de efeito de Cohen (d).
    
    Interpretação:
    - |d| < 0.2: Efeito pequeno
    - 0.2 ≤ |d| < 0.5: Efeito médio
    - 0.5 ≤ |d| < 0.8: Efeito grande
    - |d| ≥ 0.8: Efeito muito grande
    
    Returns:
        cohen_d: Tamanho de efeito
    """
    mean_diff = np.mean(values1) - np.mean(values2)
    pooled_std = np.sqrt((np.std(values1, ddof=1)**2 + np.std(values2, ddof=1)**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std

def interpretar_cohen_d(d: float) -> str:
    """Interpreta o tamanho de efeito."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "pequeno"
    elif abs_d < 0.5:
        return "médio"
    elif abs_d < 0.8:
        return "grande"
    else:
        return "muito grande"

def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Correção de Bonferroni para múltiplas comparações.
    
    p_adjusted = min(p * n_tests, 1.0)
    """
    n_tests = len(p_values)
    return [min(p * n_tests, 1.0) for p in p_values]

# ==============================================================================
# COMPARAÇÕES ENTRE MÉTODOS
# ==============================================================================

def comparar_metodos_por_dataset(method1: str, method2: str, metrica: str = 'mean_explanation_size') -> pd.DataFrame:
    """
    Compara dois métodos em todos os datasets disponíveis.
    
    Args:
        method1: Nome do primeiro método (ex: 'peab')
        method2: Nome do segundo método (ex: 'minexp')
        metrica: Métrica a ser comparada
    
    Returns:
        DataFrame com resultados da comparação
    """
    metricas1 = extrair_metricas_por_dataset(method1)
    metricas2 = extrair_metricas_por_dataset(method2)
    
    # Datasets comuns
    datasets_comuns = set(metricas1.keys()) & set(metricas2.keys())
    
    if not datasets_comuns:
        print(f"⚠️  Nenhum dataset comum entre {method1} e {method2}")
        return pd.DataFrame()
    
    resultados = []
    
    for dataset in sorted(datasets_comuns):
        valor1 = metricas1[dataset].get(metrica)
        valor2 = metricas2[dataset].get(metrica)
        
        if valor1 is None or valor2 is None:
            continue
        
        diferenca = valor1 - valor2
        percent_diff = (diferenca / valor2) * 100 if valor2 != 0 else 0
        
        resultados.append({
            'dataset': dataset,
            f'{method1}': valor1,
            f'{method2}': valor2,
            'diferenca': diferenca,
            'percent_diff': percent_diff,
            'melhor': method1 if diferenca < valor2 else method2  # Menor é melhor
        })
    
    return pd.DataFrame(resultados)

def teste_significancia_global(method1: str, method2: str, metrica: str = 'mean_explanation_size') -> Dict:
    """
    Realiza testes estatísticos comparando dois métodos globalmente.
    
    Retorna dicionário com resultados dos testes.
    """
    df_comp = comparar_metodos_por_dataset(method1, method2, metrica)
    
    if df_comp.empty:
        return {}
    
    values1 = df_comp[method1].values
    values2 = df_comp[method2].values
    
    # Teste de Wilcoxon
    w_stat, w_pvalue = wilcoxon_signed_rank_test(values1, values2)
    
    # Teste t-pareado
    t_stat, t_pvalue = paired_t_test(values1, values2)
    
    # Tamanho de efeito
    effect_size = cohen_d(values1, values2)
    effect_interp = interpretar_cohen_d(effect_size)
    
    # Estatísticas descritivas
    mean_diff = np.mean(values1) - np.mean(values2)
    median_diff = np.median(values1) - np.median(values2)
    
    return {
        'method1': method1,
        'method2': method2,
        'metrica': metrica,
        'n_datasets': len(values1),
        'mean_method1': float(np.mean(values1)),
        'mean_method2': float(np.mean(values2)),
        'mean_diff': float(mean_diff),
        'median_method1': float(np.median(values1)),
        'median_method2': float(np.median(values2)),
        'median_diff': float(median_diff),
        'wilcoxon_statistic': float(w_stat) if not np.isnan(w_stat) else None,
        'wilcoxon_pvalue': float(w_pvalue) if not np.isnan(w_pvalue) else None,
        't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
        't_pvalue': float(t_pvalue) if not np.isnan(t_pvalue) else None,
        'cohen_d': float(effect_size),
        'effect_size_interpretation': effect_interp,
        'significativo_wilcoxon': bool(w_pvalue < ALPHA) if not np.isnan(w_pvalue) else None,
        'significativo_ttest': bool(t_pvalue < ALPHA) if not np.isnan(t_pvalue) else None
    }

# ==============================================================================
# COMPARAÇÕES MÚLTIPLAS
# ==============================================================================

def comparar_peab_vs_todos(metrica: str = 'mean_explanation_size') -> List[Dict]:
    """Compara PEAB contra todos os baselines."""
    baselines = [m for m in METHODS if m != 'peab']
    
    resultados = []
    for baseline in baselines:
        print(f"\n{'='*80}")
        print(f"Comparando PEAB vs {baseline.upper()}")
        print(f"{'='*80}")
        
        resultado = teste_significancia_global('peab', baseline, metrica)
        if resultado:
            resultados.append(resultado)
    
    return resultados

# ==============================================================================
# GERAÇÃO DE RELATÓRIO
# ==============================================================================

def gerar_relatorio_texto(resultados: List[Dict], output_file: str):
    """Gera relatório em texto com os resultados dos testes."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, output_file)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TESTES ESTATÍSTICOS DE SIGNIFICÂNCIA - PEAB vs BASELINES\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Nível de significância (α): {ALPHA}\n")
        f.write(f"Métrica analisada: Tamanho Médio das Explicações\n")
        f.write(f"Hipótese nula (H0): Não há diferença entre os métodos\n")
        f.write(f"Hipótese alternativa (H1): Há diferença significativa\n\n")
        
        for i, res in enumerate(resultados, 1):
            f.write("-"*80 + "\n")
            f.write(f"COMPARAÇÃO {i}: {res['method1'].upper()} vs {res['method2'].upper()}\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"Número de datasets: {res['n_datasets']}\n\n")
            
            # Estatísticas descritivas
            f.write("ESTATÍSTICAS DESCRITIVAS:\n")
            f.write(f"  {res['method1'].upper()}:\n")
            f.write(f"    - Média: {res['mean_method1']:.2f} features\n")
            f.write(f"    - Mediana: {res['median_method1']:.2f} features\n")
            f.write(f"  {res['method2'].upper()}:\n")
            f.write(f"    - Média: {res['mean_method2']:.2f} features\n")
            f.write(f"    - Mediana: {res['median_method2']:.2f} features\n")
            f.write(f"  DIFERENÇA:\n")
            f.write(f"    - Média: {res['mean_diff']:.2f} features ({res['mean_diff']/res['mean_method2']*100:+.1f}%)\n")
            f.write(f"    - Mediana: {res['median_diff']:.2f} features\n\n")
            
            # Testes de hipótese
            f.write("TESTES DE HIPÓTESE:\n")
            
            # Wilcoxon
            if res['wilcoxon_pvalue'] is not None:
                f.write(f"  1. Teste de Wilcoxon (não-paramétrico):\n")
                f.write(f"     - Estatística W: {res['wilcoxon_statistic']:.2f}\n")
                f.write(f"     - P-value: {res['wilcoxon_pvalue']:.6f}\n")
                if res['significativo_wilcoxon']:
                    f.write(f"     - Resultado: ✓ SIGNIFICATIVO (p < {ALPHA})\n")
                else:
                    f.write(f"     - Resultado: ✗ NÃO significativo (p ≥ {ALPHA})\n")
            
            # T-test
            if res['t_pvalue'] is not None:
                f.write(f"\n  2. Teste t-pareado (paramétrico):\n")
                f.write(f"     - Estatística t: {res['t_statistic']:.2f}\n")
                f.write(f"     - P-value: {res['t_pvalue']:.6f}\n")
                if res['significativo_ttest']:
                    f.write(f"     - Resultado: ✓ SIGNIFICATIVO (p < {ALPHA})\n")
                else:
                    f.write(f"     - Resultado: ✗ NÃO significativo (p ≥ {ALPHA})\n")
            
            # Tamanho de efeito
            f.write(f"\n  3. Tamanho de Efeito (Cohen's d):\n")
            f.write(f"     - d: {res['cohen_d']:.3f}\n")
            f.write(f"     - Interpretação: Efeito {res['effect_size_interpretation']}\n\n")
            
            # Conclusão
            f.write("CONCLUSÃO:\n")
            if res['significativo_wilcoxon']:
                direcao = "menor" if res['mean_diff'] < 0 else "maior"
                f.write(f"  ✓ {res['method1'].upper()} produz explicações significativamente {direcao}es\n")
                f.write(f"    que {res['method2'].upper()} (p = {res['wilcoxon_pvalue']:.6f}, d = {res['cohen_d']:.3f}).\n")
            else:
                f.write(f"  ✗ Não há diferença estatisticamente significativa entre os métodos\n")
                f.write(f"    ao nível de {ALPHA} (p = {res['wilcoxon_pvalue']:.6f}).\n")
            
            f.write("\n")
        
        # Resumo final
        f.write("="*80 + "\n")
        f.write("RESUMO GERAL\n")
        f.write("="*80 + "\n\n")
        
        sig_count = sum(1 for r in resultados if r.get('significativo_wilcoxon'))
        f.write(f"Comparações significativas: {sig_count}/{len(resultados)}\n\n")
        
        for res in resultados:
            status = "✓" if res.get('significativo_wilcoxon') else "✗"
            f.write(f"{status} {res['method1'].upper()} vs {res['method2'].upper()}: ")
            if res.get('significativo_wilcoxon'):
                f.write(f"p = {res['wilcoxon_pvalue']:.6f}, d = {res['cohen_d']:.3f}\n")
            else:
                f.write(f"p = {res['wilcoxon_pvalue']:.6f} (não significativo)\n")
    
    print(f"\n✓ Relatório salvo em: {filepath}")

def gerar_tabela_comparativa_latex(resultados: List[Dict]) -> str:
    """Gera tabela em LaTeX para o paper."""
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Testes Estatísticos de Significância - PEAB vs Baselines}\n"
    latex += "\\label{tab:statistical_tests}\n"
    latex += "\\begin{tabular}{lccccc}\n"
    latex += "\\hline\n"
    latex += "Comparação & Média $\\pm$ DP & Wilcoxon $p$ & Cohen's $d$ & Efeito & Sig. \\\\\n"
    latex += "\\hline\n"
    
    for res in resultados:
        method1_str = f"{res['mean_method1']:.1f}"
        method2_str = f"{res['mean_method2']:.1f}"
        p_str = f"{res['wilcoxon_pvalue']:.4f}" if res['wilcoxon_pvalue'] else "---"
        d_str = f"{res['cohen_d']:.2f}" if res['cohen_d'] else "---"
        sig_str = "✓" if res.get('significativo_wilcoxon') else "---"
        
        latex += f"PEAB vs {res['method2'].upper()} & "
        latex += f"{method1_str} vs {method2_str} & "
        latex += f"{p_str} & {d_str} & {res['effect_size_interpretation']} & {sig_str} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Função principal."""
    print("="*80)
    print("TESTES ESTATÍSTICOS DE SIGNIFICÂNCIA")
    print("="*80)
    print("\nCarregando resultados dos experimentos...\n")
    
    # Verificar disponibilidade dos métodos
    metodos_disponiveis = []
    for method in METHODS:
        filepath = os.path.join(RESULTS_DIR, f'{method}_results.json')
        if os.path.exists(filepath):
            metodos_disponiveis.append(method)
            print(f"✓ {method.upper()}: Encontrado")
        else:
            print(f"✗ {method.upper()}: Não encontrado")
    
    if 'peab' not in metodos_disponiveis:
        print("\n⚠️  ERRO: Resultados do PEAB não encontrados!")
        return
    
    # Comparar PEAB vs todos os baselines
    print("\n" + "="*80)
    print("EXECUTANDO TESTES ESTATÍSTICOS")
    print("="*80)
    
    resultados = comparar_peab_vs_todos()
    
    if not resultados:
        print("\n⚠️  Nenhuma comparação foi possível.")
        return
    
    # Gerar relatórios
    print("\n" + "="*80)
    print("GERANDO RELATÓRIOS")
    print("="*80)
    
    gerar_relatorio_texto(resultados, 'wilcoxon_test_report.txt')
    
    # Salvar resultados em JSON
    output_json = os.path.join(OUTPUT_DIR, 'wilcoxon_results.json')
    with open(output_json, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"✓ Resultados JSON salvos em: {output_json}")
    
    # Gerar tabela LaTeX
    latex_table = gerar_tabela_comparativa_latex(resultados)
    output_latex = os.path.join(OUTPUT_DIR, 'comparison_table.tex')
    with open(output_latex, 'w') as f:
        f.write(latex_table)
    print(f"✓ Tabela LaTeX salva em: {output_latex}")
    
    print("\n" + "="*80)
    print("CONCLUÍDO!")
    print("="*80)

if __name__ == '__main__':
    main()
