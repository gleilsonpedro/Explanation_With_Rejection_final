"""
================================================================================
PEAB - MÉTRICAS EXTRAS DE EXPLICABILIDADE
================================================================================

Script para calcular métricas complementares de qualidade das explicações,
além do tamanho médio (minimalidade).

Métricas Implementadas:
1. Consistência: Instâncias similares têm explicações similares?
2. Cobertura de Features: Diversidade de features nas explicações
3. Estabilidade: Variância do tamanho das explicações
4. Tempo Computacional: Eficiência do método
5. Taxa de Features Únicas: Quão específicas são as explicações

Fundamentação Teórica:
- Ribeiro et al. (2016): "Model-Agnostic Interpretability"
- Alvarez-Melis & Jaakkola (2018): "Robustness of Interpretability Methods"
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"

Autor: Sistema XAI com Rejection Option
Data: 2025
================================================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# CONSTANTES
# ==============================================================================
RESULTS_DIR = 'json'
METHODS = ['peab', 'minexp', 'anchor', 'pulp']
OUTPUT_DIR = 'results/extra_metrics'

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

def extrair_explicacoes_por_dataset(method: str, dataset: str) -> List[List[str]]:
    """
    Extrai lista de explicações (listas de features) de um dataset.
    
    Returns:
        Lista de explicações, onde cada explicação é uma lista de nomes de features
    """
    resultados = carregar_resultados_metodo(method)
    
    if dataset not in resultados:
        return []
    
    data = resultados[dataset]
    
    if 'test_instances' not in data:
        return []
    
    explicacoes = []
    for instance in data['test_instances']:
        if 'explanation' in instance:
            explicacoes.append(instance['explanation'])
    
    return explicacoes

# ==============================================================================
# MÉTRICA 1: CONSISTÊNCIA
# ==============================================================================

def calcular_similaridade_explicacoes(expl1: List[str], expl2: List[str]) -> float:
    """
    Calcula similaridade entre duas explicações usando Índice de Jaccard.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Returns:
        Float entre 0 (totalmente diferentes) e 1 (idênticas)
    """
    if not expl1 and not expl2:
        return 1.0
    if not expl1 or not expl2:
        return 0.0
    
    set1 = set(expl1)
    set2 = set(expl2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0

def consistencia_explicacoes(explicacoes: List[List[str]], n_samples: int = 100) -> Dict[str, float]:
    """
    Mede consistência das explicações através de similaridades pareadas.
    
    Fundamentação:
    - Explicações de instâncias da mesma classe deveriam ser similares
    - Alta variância indica inconsistência
    
    Args:
        explicacoes: Lista de explicações
        n_samples: Número de pares a amostrar (para eficiência)
    
    Returns:
        Dict com média, desvio padrão e mediana das similaridades
    """
    if len(explicacoes) < 2:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
    
    # Amostrar pares para evitar O(n²) em datasets grandes
    n_explicacoes = len(explicacoes)
    n_pares = min(n_samples, (n_explicacoes * (n_explicacoes - 1)) // 2)
    
    similaridades = []
    pares_testados = set()
    
    while len(similaridades) < n_pares:
        i = np.random.randint(0, n_explicacoes)
        j = np.random.randint(0, n_explicacoes)
        
        if i != j and (i, j) not in pares_testados and (j, i) not in pares_testados:
            sim = calcular_similaridade_explicacoes(explicacoes[i], explicacoes[j])
            similaridades.append(sim)
            pares_testados.add((i, j))
    
    return {
        'mean': float(np.mean(similaridades)),
        'std': float(np.std(similaridades)),
        'median': float(np.median(similaridades))
    }

# ==============================================================================
# MÉTRICA 2: COBERTURA DE FEATURES
# ==============================================================================

def cobertura_features(explicacoes: List[List[str]], total_features: int) -> Dict[str, float]:
    """
    Mede diversidade de features utilizadas nas explicações.
    
    Métricas:
    - Número de features únicas usadas
    - Percentual de features cobertas
    - Entropia da distribuição de features
    
    Args:
        explicacoes: Lista de explicações
        total_features: Número total de features disponíveis
    
    Returns:
        Dict com métricas de cobertura
    """
    if not explicacoes:
        return {'unique_features': 0, 'coverage_percent': 0.0, 'entropy': 0.0}
    
    # Contar frequência de cada feature
    feature_counts = Counter()
    for expl in explicacoes:
        feature_counts.update(expl)
    
    n_unique = len(feature_counts)
    coverage = (n_unique / total_features) * 100 if total_features > 0 else 0.0
    
    # Calcular entropia da distribuição
    total_count = sum(feature_counts.values())
    probs = [count / total_count for count in feature_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    # Features mais frequentes
    top_features = feature_counts.most_common(10)
    
    return {
        'unique_features': n_unique,
        'coverage_percent': float(coverage),
        'entropy': float(entropy),
        'top_features': top_features
    }

# ==============================================================================
# MÉTRICA 3: ESTABILIDADE
# ==============================================================================

def estabilidade_tamanho_explicacoes(explicacoes: List[List[str]]) -> Dict[str, float]:
    """
    Mede estabilidade através da variação no tamanho das explicações.
    
    Fundamentação:
    - Baixa variância indica estabilidade (explicações consistentes em tamanho)
    - Alta variância pode indicar sensibilidade excessiva à entrada
    
    Returns:
        Dict com estatísticas de tamanho
    """
    if not explicacoes:
        return {'mean': 0.0, 'std': 0.0, 'cv': 0.0, 'min': 0, 'max': 0}
    
    tamanhos = [len(expl) for expl in explicacoes]
    
    mean_size = np.mean(tamanhos)
    std_size = np.std(tamanhos)
    cv = (std_size / mean_size) if mean_size > 0 else 0.0  # Coeficiente de variação
    
    return {
        'mean': float(mean_size),
        'std': float(std_size),
        'cv': float(cv),  # CV < 0.3 indica estabilidade
        'min': int(np.min(tamanhos)),
        'max': int(np.max(tamanhos)),
        'range': int(np.max(tamanhos) - np.min(tamanhos))
    }

# ==============================================================================
# MÉTRICA 4: TEMPO COMPUTACIONAL
# ==============================================================================

def metricas_tempo_computacional(method: str, dataset: str) -> Dict[str, float]:
    """Extrai métricas de tempo computacional dos resultados."""
    resultados = carregar_resultados_metodo(method)
    
    if dataset not in resultados:
        return {}
    
    data = resultados[dataset]
    
    if 'computation_time' not in data:
        return {}
    
    comp_time = data['computation_time']
    
    return {
        'total_time': comp_time.get('total_explanation_time', 0.0),
        'mean_time_per_instance': comp_time.get('mean_time_per_instance', 0.0),
        'mean_time_positive': comp_time.get('mean_time_positive', 0.0),
        'mean_time_negative': comp_time.get('mean_time_negative', 0.0),
        'mean_time_rejected': comp_time.get('mean_time_rejected', 0.0)
    }

# ==============================================================================
# MÉTRICA 5: TAXA DE FEATURES ÚNICAS
# ==============================================================================

def taxa_features_unicas(explicacoes: List[List[str]]) -> float:
    """
    Mede quão específicas são as explicações.
    
    Taxa alta = Explicações usam features diferentes (específicas)
    Taxa baixa = Explicações usam sempre as mesmas features (genéricas)
    
    Returns:
        Percentual de features que aparecem em apenas uma explicação
    """
    if not explicacoes:
        return 0.0
    
    feature_counts = Counter()
    for expl in explicacoes:
        feature_counts.update(expl)
    
    features_unicas = sum(1 for count in feature_counts.values() if count == 1)
    total_features = len(feature_counts)
    
    return (features_unicas / total_features) * 100 if total_features > 0 else 0.0

# ==============================================================================
# ANÁLISE COMPLETA POR DATASET
# ==============================================================================

def analisar_dataset(method: str, dataset: str, total_features: int) -> Dict:
    """
    Realiza análise completa de métricas extras para um dataset.
    
    Args:
        method: Nome do método
        dataset: Nome do dataset
        total_features: Número total de features no dataset
    
    Returns:
        Dict com todas as métricas extras
    """
    print(f"  Analisando {dataset}...")
    
    # Carregar explicações
    explicacoes = extrair_explicacoes_por_dataset(method, dataset)
    
    if not explicacoes:
        print(f"    ⚠️  Nenhuma explicação encontrada")
        return {}
    
    # Calcular métricas
    metricas = {
        'dataset': dataset,
        'method': method,
        'num_instances': len(explicacoes),
        'consistencia': consistencia_explicacoes(explicacoes),
        'cobertura': cobertura_features(explicacoes, total_features),
        'estabilidade': estabilidade_tamanho_explicacoes(explicacoes),
        'tempo': metricas_tempo_computacional(method, dataset),
        'taxa_features_unicas': taxa_features_unicas(explicacoes)
    }
    
    return metricas

# ==============================================================================
# COMPARAÇÃO ENTRE MÉTODOS
# ==============================================================================

def comparar_metodos_metricas_extras(dataset: str, total_features: int) -> pd.DataFrame:
    """
    Compara todos os métodos em um dataset usando métricas extras.
    
    Returns:
        DataFrame com comparação lado-a-lado
    """
    resultados = []
    
    for method in METHODS:
        metricas = analisar_dataset(method, dataset, total_features)
        if metricas:
            # Flatten para DataFrame
            row = {
                'method': method,
                'num_instances': metricas['num_instances'],
                'consistency_mean': metricas['consistencia']['mean'],
                'consistency_std': metricas['consistencia']['std'],
                'unique_features': metricas['cobertura']['unique_features'],
                'coverage_percent': metricas['cobertura']['coverage_percent'],
                'entropy': metricas['cobertura']['entropy'],
                'stability_mean': metricas['estabilidade']['mean'],
                'stability_std': metricas['estabilidade']['std'],
                'stability_cv': metricas['estabilidade']['cv'],
                'mean_time': metricas['tempo'].get('mean_time_per_instance', 0.0),
                'unique_rate': metricas['taxa_features_unicas']
            }
            resultados.append(row)
    
    return pd.DataFrame(resultados)

# ==============================================================================
# GERAÇÃO DE RELATÓRIOS
# ==============================================================================

def gerar_relatorio_completo(dataset: str, total_features: int, output_file: str):
    """Gera relatório detalhado com todas as métricas extras."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, output_file)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"MÉTRICAS EXTRAS DE EXPLICABILIDADE - {dataset.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # Analisar cada método
        for method in METHODS:
            print(f"\nAnalisando {method.upper()}...")
            metricas = analisar_dataset(method, dataset, total_features)
            
            if not metricas:
                continue
            
            f.write("-"*80 + "\n")
            f.write(f"MÉTODO: {method.upper()}\n")
            f.write("-"*80 + "\n\n")
            
            # Consistência
            f.write("1. CONSISTÊNCIA DAS EXPLICAÇÕES:\n")
            cons = metricas['consistencia']
            f.write(f"   - Similaridade média (Jaccard): {cons['mean']:.3f}\n")
            f.write(f"   - Desvio padrão: {cons['std']:.3f}\n")
            f.write(f"   - Mediana: {cons['median']:.3f}\n")
            f.write(f"   → Interpretação: {'Alta consistência' if cons['mean'] > 0.5 else 'Baixa consistência'}\n\n")
            
            # Cobertura
            f.write("2. COBERTURA DE FEATURES:\n")
            cob = metricas['cobertura']
            f.write(f"   - Features únicas utilizadas: {cob['unique_features']}/{total_features}\n")
            f.write(f"   - Percentual de cobertura: {cob['coverage_percent']:.1f}%\n")
            f.write(f"   - Entropia da distribuição: {cob['entropy']:.2f} bits\n")
            f.write(f"   - Top 5 features mais frequentes:\n")
            for feat, count in cob['top_features'][:5]:
                f.write(f"     • {feat}: {count} vezes\n")
            f.write("\n")
            
            # Estabilidade
            f.write("3. ESTABILIDADE (Tamanho das Explicações):\n")
            est = metricas['estabilidade']
            f.write(f"   - Tamanho médio: {est['mean']:.2f} features\n")
            f.write(f"   - Desvio padrão: {est['std']:.2f}\n")
            f.write(f"   - Coeficiente de variação (CV): {est['cv']:.3f}\n")
            f.write(f"   - Range: [{est['min']}, {est['max']}]\n")
            f.write(f"   → Interpretação: {'Estável' if est['cv'] < 0.3 else 'Instável'} (CV < 0.3 = estável)\n\n")
            
            # Tempo
            f.write("4. TEMPO COMPUTACIONAL:\n")
            tempo = metricas['tempo']
            if tempo:
                f.write(f"   - Tempo total: {tempo.get('total_time', 0):.2f}s\n")
                f.write(f"   - Tempo médio por instância: {tempo.get('mean_time_per_instance', 0):.6f}s\n")
                f.write(f"   - Tempo médio (positivas): {tempo.get('mean_time_positive', 0):.6f}s\n")
                f.write(f"   - Tempo médio (negativas): {tempo.get('mean_time_negative', 0):.6f}s\n")
                if tempo.get('mean_time_rejected', 0) > 0:
                    f.write(f"   - Tempo médio (rejeitadas): {tempo.get('mean_time_rejected', 0):.6f}s\n")
            else:
                f.write(f"   - Dados não disponíveis\n")
            f.write("\n")
            
            # Taxa de features únicas
            f.write("5. ESPECIFICIDADE DAS EXPLICAÇÕES:\n")
            f.write(f"   - Taxa de features únicas: {metricas['taxa_features_unicas']:.1f}%\n")
            f.write(f"   → Interpretação: Explicações são ")
            if metricas['taxa_features_unicas'] > 50:
                f.write("altamente específicas\n")
            elif metricas['taxa_features_unicas'] > 20:
                f.write("moderadamente específicas\n")
            else:
                f.write("genéricas (usam features comuns)\n")
            f.write("\n\n")
        
        # Comparação resumida
        f.write("="*80 + "\n")
        f.write("COMPARAÇÃO RESUMIDA\n")
        f.write("="*80 + "\n\n")
        
        df_comp = comparar_metodos_metricas_extras(dataset, total_features)
        if not df_comp.empty:
            f.write(df_comp.to_string(index=False))
            f.write("\n\n")
            
            # Ranking
            f.write("-"*80 + "\n")
            f.write("RANKINGS (menor é melhor, exceto onde indicado):\n")
            f.write("-"*80 + "\n")
            
            # Tamanho médio
            df_sorted = df_comp.sort_values('stability_mean')
            f.write("\n1. Tamanho Médio (Minimalidade):\n")
            for i, row in enumerate(df_sorted.itertuples(), 1):
                f.write(f"   {i}. {row.method.upper()}: {row.stability_mean:.2f} features\n")
            
            # Consistência (maior é melhor)
            df_sorted = df_comp.sort_values('consistency_mean', ascending=False)
            f.write("\n2. Consistência (maior é melhor):\n")
            for i, row in enumerate(df_sorted.itertuples(), 1):
                f.write(f"   {i}. {row.method.upper()}: {row.consistency_mean:.3f}\n")
            
            # Estabilidade (menor CV é melhor)
            df_sorted = df_comp.sort_values('stability_cv')
            f.write("\n3. Estabilidade (menor CV é melhor):\n")
            for i, row in enumerate(df_sorted.itertuples(), 1):
                f.write(f"   {i}. {row.method.upper()}: CV = {row.stability_cv:.3f}\n")
            
            # Tempo (menor é melhor)
            df_sorted = df_comp.sort_values('mean_time')
            f.write("\n4. Eficiência Computacional (menor tempo é melhor):\n")
            for i, row in enumerate(df_sorted.itertuples(), 1):
                f.write(f"   {i}. {row.method.upper()}: {row.mean_time:.6f}s por instância\n")
    
    print(f"\n✓ Relatório salvo em: {filepath}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Função principal."""
    print("="*80)
    print("ANÁLISE DE MÉTRICAS EXTRAS DE EXPLICABILIDADE")
    print("="*80)
    print("\nEste script calcula métricas complementares de qualidade:")
    print("  1. Consistência (similaridade entre explicações)")
    print("  2. Cobertura de features (diversidade)")
    print("  3. Estabilidade (variância do tamanho)")
    print("  4. Tempo computacional (eficiência)")
    print("  5. Taxa de features únicas (especificidade)")
    print("\n" + "="*80)
    
    # Configuração por dataset
    datasets_config = {
        'breast_cancer': 30,
        'pima_indians_diabetes': 8,
        'vertebral_column': 6,
        'sonar': 60,
        'banknote': 4,
        'wine': 13
    }
    
    # Análise para cada dataset
    for dataset, n_features in datasets_config.items():
        print(f"\n{'='*80}")
        print(f"Analisando dataset: {dataset.upper()}")
        print(f"{'='*80}")
        
        try:
            gerar_relatorio_completo(dataset, n_features, f'extra_metrics_{dataset}.txt')
        except Exception as e:
            print(f"⚠️  Erro ao analisar {dataset}: {e}")
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)
    print(f"\nRelatórios salvos em: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
