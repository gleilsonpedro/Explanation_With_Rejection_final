"""Gera relatórios detalhados a partir de json/comparative_results.json.

Cria três arquivos (se existirem dados):
  results/audit/report_peab.txt
  results/audit/report_minexp.txt
  results/audit/report_anchor.txt

Cada relatório contém para cada dataset:
  - Configuração (dataset_name, par MNIST, test_size, rejection_cost)
  - Thresholds e largura da zona (t_minus, t_plus, delta)
  - Métricas de desempenho (acurácias, taxa de rejeição)
  - Estatísticas das explicações (positive/negative/rejected)
  - Tempo computacional (total, médio geral e por classe)
  - Top features globais (Top N)
  - Exemplos de instâncias (1 positiva, 1 negativa, 1 rejeitada se existirem)

Uso rápido:
  python audit/detailed_json.py
Opções:
  --json caminho_do_json
  --top-features N
  --datasets lista separada por vírgula
  --methods peab,minexp,anchor (filtra quais gerar)
  --no-samples (não incluir exemplos por instância)

"""
import os
import json
import argparse
from typing import Dict, Any, List, Tuple
from datetime import datetime

DEFAULT_JSON = "json/comparative_results.json"
OUTPUT_DIR = "results/audit"
METHOD_MAP = {"peab": "PEAB", "minexp": "MinExp", "anchor": "Anchor"}
REVERSE_METHOD_MAP = {v: k for k, v in METHOD_MAP.items()}

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm_method_blocks(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    blocks = {}
    for raw_key, label in METHOD_MAP.items():
        if raw_key in data:
            blocks[label] = data[raw_key]
    # Compatibilidade com execuções antigas (chaves capitalizadas)
    for label in METHOD_MAP.values():
        if label in data and label not in blocks:
            blocks[label] = data[label]
    return blocks

def get_config_section(d: Dict[str, Any]) -> Dict[str, Any]:
    cfg = d.get("config", {})
    return {
        "dataset_name": cfg.get("dataset_name"),
        "mnist_digit_pair": cfg.get("mnist_digit_pair"),
        "test_size": cfg.get("test_size"),
        "rejection_cost": cfg.get("rejection_cost")
    }

def get_thresholds(d: Dict[str, Any]) -> Dict[str, Any]:
    th = d.get("thresholds", {})
    t_plus = th.get("t_plus")
    t_minus = th.get("t_minus")
    width = (t_plus - t_minus) if (isinstance(t_plus, (int,float)) and isinstance(t_minus, (int,float))) else None
    return {"t_plus": t_plus, "t_minus": t_minus, "width": width}

def get_performance(d: Dict[str, Any]) -> Dict[str, Any]:
    perf = d.get("performance", {})
    return {
        "accuracy_without_rejection": perf.get("accuracy_without_rejection"),
        "accuracy_with_rejection": perf.get("accuracy_with_rejection"),
        "rejection_rate": perf.get("rejection_rate")
    }

def get_explanation_stats(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stats = d.get("explanation_stats", {})
    out = {}
    for k in ["positive", "negative", "rejected"]:
        block = stats.get(k, {})
        out[k] = {
            "count": block.get("count", 0),
            "min_length": block.get("min_length", 0),
            "mean_length": block.get("mean_length", 0.0),
            "max_length": block.get("max_length", 0),
            "std_length": block.get("std_length", 0.0)
        }
    return out

def get_computation_time(d: Dict[str, Any]) -> Dict[str, Any]:
    ct = d.get("computation_time", {})
    return {
        "total": ct.get("total"),
        "mean_per_instance": ct.get("mean_per_instance"),
        "positive": ct.get("positive"),
        "negative": ct.get("negative"),
        "rejected": ct.get("rejected")
    }

def get_top_features(d: Dict[str, Any], top_n: int) -> List[Dict[str, Any]]:
    feats = d.get("top_features", [])
    return feats[:top_n] if isinstance(feats, list) else []

def pick_samples(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    per = d.get("per_instance", [])
    if not isinstance(per, list):
        return {}
    sample = {"positive": None, "negative": None, "rejected": None}
    for inst in per:
        if inst.get("rejected"):
            if sample["rejected"] is None:
                sample["rejected"] = inst
        else:
            yp = inst.get("y_pred")
            if yp == 1 and sample["positive"] is None:
                sample["positive"] = inst
            if yp == 0 and sample["negative"] is None:
                sample["negative"] = inst
        if all(sample.values()):
            break
    return sample

def fmt_float(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"

def build_dataset_section(name: str, blob: Dict[str, Any], top_n: int, include_samples: bool) -> str:
    cfg = get_config_section(blob)
    th = get_thresholds(blob)
    perf = get_performance(blob)
    stats = get_explanation_stats(blob)
    comp = get_computation_time(blob)
    feats = get_top_features(blob, top_n)
    samples = pick_samples(blob) if include_samples else {}
    # Distribuição de instâncias (positivas/negativas/rejeitadas)
    per_list = blob.get('per_instance', [])
    pos_count = neg_count = rej_count = 0
    if isinstance(per_list, list):
        for inst in per_list:
            if inst.get('rejected'):
                rej_count += 1
            else:
                yp = inst.get('y_pred')
                if yp == 1:
                    pos_count += 1
                elif yp == 0:
                    neg_count += 1
    # Total de instâncias de teste (preferir y_test se presente)
    y_test = blob.get('data', {}).get('y_test') if isinstance(blob.get('data'), dict) else None
    if isinstance(y_test, list):
        total_inst = len(y_test)
    else:
        total_inst = len(per_list) if isinstance(per_list, list) else None

    lines = []
    lines.append(f"== Dataset: {name} ==")
    lines.append("[Config]")
    lines.append(f"  - Nome: {cfg.get('dataset_name')}")
    dpair = cfg.get('mnist_digit_pair')
    if dpair:
        lines.append(f"  - Par MNIST: {dpair}")
    lines.append(f"  - Test Size: {fmt_float(cfg.get('test_size'), 2)}")
    lines.append(f"  - Rejection Cost (wr): {fmt_float(cfg.get('rejection_cost'), 2)}")
    
    # Hiperparâmetros do modelo (se disponíveis)
    model_params = blob.get('model', {}).get('params', {})
    if model_params:
        lines.append(f"  - Hiperparâmetros Modelo: {model_params}")
    elif 'C' in blob.get('config', {}): # Fallback para config antiga
        lines.append(f"  - C (Regularização): {blob['config'].get('C')}")

    if 'subsample_size' in blob.get('config', {}):
        lines.append(f"  - Subsample Size: {fmt_float(blob['config'].get('subsample_size'), 2)}")
    if total_inst is not None:
        lines.append(f"  - Total Instâncias Teste: {total_inst}")

    lines.append("[Thresholds]")
    lines.append(f"  - t_minus: {fmt_float(th.get('t_minus'))}")
    lines.append(f"  - t_plus : {fmt_float(th.get('t_plus'))}")
    lines.append(f"  - Largura: {fmt_float(th.get('width'))}")

    lines.append("[Performance]")
    lines.append(f"  - Acurácia sem rejeição: {fmt_float(perf.get('accuracy_without_rejection'))}%")
    lines.append(f"  - Acurácia com rejeição: {fmt_float(perf.get('accuracy_with_rejection'))}%")
    lines.append(f"  - Taxa de rejeição: {fmt_float(perf.get('rejection_rate'))}%")

    lines.append("[Explicações]")
    for cat in ["positive", "negative", "rejected"]:
        s = stats.get(cat, {})
        lines.append(f"  - {cat.capitalize()}: count={s.get('count')} mean={fmt_float(s.get('mean_length'))} std={fmt_float(s.get('std_length'))} min={s.get('min_length')} max={s.get('max_length')}")

    lines.append("[Distribuição de Instâncias]")
    lines.append(f"  - Positivas (aceitas): {pos_count}")
    lines.append(f"  - Negativas (aceitas): {neg_count}")
    lines.append(f"  - Rejeitadas: {rej_count}")

    lines.append("[Tempo Computacional]")
    lines.append(f"  - Total: {fmt_float(comp.get('total'))} s")
    lines.append(f"  - Média/Instância: {fmt_float(comp.get('mean_per_instance'))} s")
    lines.append(f"  - Positivas: {fmt_float(comp.get('positive'))} s | Negativas: {fmt_float(comp.get('negative'))} s | Rejeitadas: {fmt_float(comp.get('rejected'))} s")

    lines.append("[Top Features]")
    if feats:
        for f in feats:
            # Aceita vários formatos (feature/frequency/weight)
            feat_name = f.get('feature') or f.get('name') or str(f)
            freq = f.get('frequency') or f.get('count')
            delta = f.get('mean_delta') or f.get('delta')
            extra = []
            if freq is not None:
                extra.append(f"freq={freq}")
            if delta is not None:
                extra.append(f"delta={fmt_float(delta)}")
            lines.append(f"  - {feat_name} ({', '.join(extra) if extra else 'info indisponível'})")
    else:
        lines.append("  - Nenhuma feature registrada")

    if include_samples:
        lines.append("[Exemplos de Instâncias]")
        for cat in ["positive", "negative", "rejected"]:
            inst = samples.get(cat)
            if not inst:
                lines.append(f"  - {cat}: N/A")
                continue
            exp = inst.get('explanation', [])
            lines.append(f"  - {cat}: score={fmt_float(inst.get('decision_score'))} y_true={inst.get('y_true')} y_pred={inst.get('y_pred')} tamanho_exp={len(exp)}")
    lines.append("")
    return "\n".join(lines)

def build_report(method_label: str, datasets: Dict[str, Any], top_n: int, include_samples: bool) -> str:
    lines = []
    lines.append(f"==== RELATÓRIO {method_label.upper()} ====")
    lines.append(f"Gerado em: {datetime.now().isoformat()}")
    lines.append("")
    for ds_name, blob in sorted(datasets.items()):
        try:
            lines.append(build_dataset_section(ds_name, blob, top_n, include_samples))
        except Exception as e:
            lines.append(f"== Dataset: {ds_name} ==\nFalha ao processar: {e}\n")
    return "\n".join(lines)

def parse_args():
    p = argparse.ArgumentParser(description="Gera relatórios detalhados por método a partir do JSON consolidado.")
    p.add_argument("--json", default=DEFAULT_JSON, help="Caminho do comparative_results.json")
    p.add_argument("--top-features", type=int, default=10, help="Quantidade de top features por dataset")
    p.add_argument("--datasets", type=str, default=None, help="Filtro de datasets (lista separada por vírgula)")
    p.add_argument("--methods", type=str, default=None, help="Filtro de métodos (peab,minexp,anchor)")
    p.add_argument("--no-samples", action="store_true", help="Não incluir exemplos de instâncias")
    return p.parse_args()

def main():
    args = parse_args()
    data = load_json(args.json)
    blocks = norm_method_blocks(data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_filter = None
    if args.datasets:
        ds_filter = set(x.strip() for x in args.datasets.split(',') if x.strip())
    method_filter = None
    if args.methods:
        method_filter = set(REVERSE_METHOD_MAP.get(x.strip().capitalize(), x.strip().lower()) for x in args.methods.split(',') if x.strip())

    for raw_key, label in METHOD_MAP.items():
        if method_filter and raw_key not in method_filter:
            continue
        if label not in blocks:
            continue
        datasets = blocks[label]
        if ds_filter:
            datasets = {k: v for k, v in datasets.items() if k in ds_filter}
        report = build_report(label, datasets, args.top_features, not args.no_samples)
        out_path = os.path.join(OUTPUT_DIR, f"report_{raw_key}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Gerado: {out_path}")

if __name__ == "__main__":
    main()
