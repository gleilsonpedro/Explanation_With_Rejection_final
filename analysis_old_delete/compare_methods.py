# -*- coding: utf-8 -*-
"""
Gera um comparativo entre métodos (peab, anchor, MinExp) usando o JSON consolidado.
Métricas: acurácia com rejeição, taxa de rejeição, tempo médio/total, tamanhos médios das explicações por classe.
Saída: results/report/comparison/methods_comparison.csv
"""
import os
import json
import pandas as pd
from pathlib import Path

JSON_FILE = Path('json/comparative_results.json')
OUTPUT_DIR = Path('results/report/comparison')


def _get(d: dict, path: str, default=None):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _extract_stats_block(stats: dict):
    """Normaliza blocos de stats de explicação para um formato comum.
    Aceita dois formatos:
    - {count, min_length, mean_length, max_length, std_length}
    - {instancias, min, media, max, std_dev}
    Retorna: {'count':int,'min':float,'mean':float,'max':float,'std':float}
    """
    if not isinstance(stats, dict):
        return {'count': 0, 'min': 0.0, 'mean': 0.0, 'max': 0.0, 'std': 0.0}
    if 'mean_length' in stats:  # formato MinExp/peab_2
        return {
            'count': int(stats.get('count', 0)),
            'min': float(stats.get('min_length', 0.0)),
            'mean': float(stats.get('mean_length', 0.0)),
            'max': float(stats.get('max_length', 0.0)),
            'std': float(stats.get('std_length', 0.0)),
        }
    # formato Anchor (stats_pos/neg/rej)
    return {
        'count': int(stats.get('instancias', 0)),
        'min': float(stats.get('min', 0.0)),
        'mean': float(stats.get('media', 0.0)),
        'max': float(stats.get('max', 0.0)),
        'std': float(stats.get('std_dev', 0.0)),
    }


def _collect_for_method(method_name: str, datasets_block: dict):
    rows = []
    for dataset, data in datasets_block.items():
        perf = data.get('performance', {})
        comp = data.get('computation_time', {})
        expl = data.get('explanation_stats', {})
        pos = _extract_stats_block(expl.get('positive', {}))
        neg = _extract_stats_block(expl.get('negative', {}))
        rej = _extract_stats_block(expl.get('rejected', {}))
        rows.append({
            'dataset': dataset,
            'method': method_name,
            'accuracy_with_rejection': float(perf.get('accuracy_with_rejection', 0.0)),
            'rejection_rate': float(perf.get('rejection_rate', 0.0)),
            'time_total': float(comp.get('total', 0.0)),
            'time_mean': float(comp.get('mean_per_instance', 0.0)),
            'exp_len_pos_mean': pos['mean'],
            'exp_len_neg_mean': neg['mean'],
            'exp_len_rej_mean': rej['mean'],
        })
    return rows


def main():
    if not JSON_FILE.exists():
        print(f"Arquivo '{JSON_FILE}' não encontrado. Rode os métodos antes para popular o JSON.")
        return
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_rows = []
    for method in ['peab', 'anchor', 'MinExp']:
        block = data.get(method)
        if isinstance(block, dict) and block:
            all_rows.extend(_collect_for_method(method, block))

    if not all_rows:
        print("Nenhum dado de método encontrado no JSON.")
        return

    df = pd.DataFrame(all_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / 'methods_comparison.csv'
    df.sort_values(['dataset', 'method']).to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Comparativo salvo em: {out_csv}")

    # Também salvar um resumo por dataset (melhor método por acurácia)
    best_by_acc = df.sort_values(['dataset', 'accuracy_with_rejection'], ascending=[True, False])\
                    .groupby('dataset').head(1)
    out_best = OUTPUT_DIR / 'best_by_accuracy.csv'
    best_by_acc.to_csv(out_best, index=False, encoding='utf-8')
    print(f"Resumo (melhor por acurácia) salvo em: {out_best}")


if __name__ == '__main__':
    main()
