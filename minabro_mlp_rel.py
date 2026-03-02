import os
import json
import numpy as np
from collections import Counter

def gerar_relatorio_do_json(json_filepath: str, output_dir: str = 'results/report/minabro_mlp'):
    """
    Lê o JSON de resultados gerado pelo MINABRO_MLP e constrói 
    um relatório de texto detalhado e bem formatado.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_name = data['config']['dataset_name']
    output_path = os.path.join(output_dir, f"report_{dataset_name}.txt")

    # Atalhos para os dicionários do JSON
    cfg = data['config']
    perf = data['performance']
    thresh = data['thresholds']
    exp_stats = data['explanation_stats']
    comp_time = data['computation_time']
    instancias = data['per_instance']

    # Extrair Top 10 Features
    todas_features = []
    for inst in instancias:
        todas_features.extend(inst['explanation'])
    top_features = Counter(todas_features).most_common(10)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("          RELATÓRIO DE ANÁLISE - MÉTODO MINABRO MLP COM REJEIÇÃO\n")
        f.write("="*80 + "\n\n")

        # 1. CONFIGURAÇÃO
        f.write("-" * 80 + "\n1. CONFIGURAÇÃO DO EXPERIMENTO\n" + "-" * 80 + "\n")
        f.write(f"  Dataset: {cfg['dataset_name']}\n")
        f.write(f"  Instâncias de teste: {perf['num_test_instances']}\n")
        f.write(f"  Features por instância: {data['model']['num_features']}\n")
        f.write(f"  Test size: {cfg['test_size']:.2%}\n")
        f.write(f"  Custo de rejeição (WR): {cfg['rejection_cost']:.4f}\n\n")

        # 2. THRESHOLDS
        f.write("-" * 80 + "\n2. THRESHOLDS DE REJEIÇÃO\n" + "-" * 80 + "\n")
        f.write(f"  t+ (limiar superior): {thresh['t_plus']:.6f}\n")
        f.write(f"  t- (limiar inferior): {thresh['t_minus']:.6f}\n")
        f.write(f"  Largura da zona de rejeição: {thresh['rejection_zone_width']:.6f}\n\n")

        # 3. DESEMPENHO
        f.write("-" * 80 + "\n3. DESEMPENHO DO MODELO\n" + "-" * 80 + "\n")
        f.write(f"  Acurácia sem rejeição: {perf['accuracy_without_rejection']:.2f}%\n")
        f.write(f"  Acurácia com rejeição: {perf['accuracy_with_rejection']:.2f}%\n")
        f.write(f"  Taxa de rejeição: {perf['rejection_rate']:.2f}%\n\n")

        # 4. ESTATÍSTICAS DAS EXPLICAÇÕES
        f.write("-" * 80 + "\n4. ESTATÍSTICAS DAS EXPLICAÇÕES\n" + "-" * 80 + "\n")
        for tipo_label, key in [('POSITIVAS', 'positive'), ('NEGATIVAS', 'negative'), ('REJEITADAS', 'rejected')]:
            stats = exp_stats[key]
            f.write(f"  {tipo_label}:\n")
            f.write(f"    Quantidade: {stats['count']}\n")
            f.write(f"    Tamanho médio: {stats['mean_length']:.2f} features\n")
            f.write(f"    Desvio padrão: {stats['std_length']:.2f}\n")
            f.write(f"    Mínimo: {stats['min_length']} features\n")
            f.write(f"    Máximo: {stats['max_length']} features\n\n")

        # 5. TEMPOS DE EXECUÇÃO
        f.write("-" * 80 + "\n5. TEMPOS DE EXECUÇÃO\n" + "-" * 80 + "\n")
        f.write(f"  Tempo total: {comp_time['total']:.4f}s\n")
        f.write(f"  Tempo médio por instância: {comp_time['mean_per_instance']:.6f}s\n\n")

        # 6. TOP 10 FEATURES
        f.write("-" * 80 + "\n6. TOP 10 FEATURES MAIS FREQUENTES NAS EXPLICAÇÕES\n" + "-" * 80 + "\n")
        if not top_features:
            f.write("  Nenhuma feature selecionada.\n")
        for feat, count in top_features:
            freq_pct = (count / perf['num_test_instances'] * 100)
            f.write(f"  {feat}: {count} ocorrências ({freq_pct:.1f}%)\n")
        f.write("\n")

    print(f"[RELATÓRIO] Documento formatado salvo em: {output_path}")