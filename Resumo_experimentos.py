import os
import json
from datetime import datetime
from typing import Dict, Any, List

try:
    # Preferimos usar o loader tolerante do projeto
    from utils.results_handler import load_results
except Exception:
    load_results = None

DATASETS = [
    "breast_cancer",
    "mnist",
    "pima_indians_diabetes",
    "sonar",
    "vertebral_column",
    "wine",
]

# Normalização de chaves de métodos
METHOD_ALIASES = {
    "peab": "PEAB",
    "anchor": "Anchor",
    "minexp": "MinExp",
}


def _load_json_results() -> Dict[str, Any]:
    if load_results is not None:
        return load_results()
    path = os.path.join("json", "comparative_results.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


essential_metrics = [
    ("accuracy_without_rejection", "Acurácia sem rejeição (%)"),
    ("accuracy_with_rejection", "Acurácia com rejeição (%)"),
    ("rejection_rate", "Taxa de rejeição (%)"),
]


def _get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _fmt_float(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _format_duration(seconds: float) -> str:
    """Formata segundos em uma string legível (h, m, s)."""
    if seconds is None:
        return "N/A"
    
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    if sec > 0 or not parts:
        parts.append(f"{int(sec)}s")
        
    return " ".join(parts)


def build_summary_text(results: Dict[str, Any]) -> str:
    # Mapear métodos presentes, normalizando para títulos
    present_methods = {}
    for k, v in results.items():
        key_l = str(k).lower()
        if key_l in METHOD_ALIASES:
            present_methods[METHOD_ALIASES[key_l]] = v
    # Alguns runs antigos podem ter salvo 'MinExp' com M maiúsculo
    if "MinExp" in results and "MinExp" not in present_methods:
        present_methods["MinExp"] = results["MinExp"]

    lines = []
    lines.append("==== Resumo Comparativo de Resultados ====")
    lines.append("")
    lines.append("Métodos encontrados no JSON: " + ", ".join(sorted(present_methods.keys())) if present_methods else "Nenhum método encontrado.")
    lines.append("")

    # Cobertura por dataset
    lines.append("-- Cobertura por Dataset --")
    for ds in DATASETS:
        flags = []
        for method in ["PEAB", "MinExp", "Anchor"]:
            has = method in present_methods and ds in (present_methods[method] or {})
            flags.append(f"{method}:{'OK' if has else 'MISSING'}")
        lines.append(f"{ds}: " + ", ".join(flags))
    lines.append("")

    # Resumo por dataset
    for ds in DATASETS:
        lines.append(f"== Dataset: {ds} ==")
        for method in ["PEAB", "MinExp", "Anchor"]:
            blob = (present_methods.get(method) or {}).get(ds)
            lines.append(f"  > {method}")
            if not blob:
                lines.append("    MISSING")
                continue
            # Performance
            perf = blob.get("performance", {})
            for key, label in essential_metrics:
                val = perf.get(key)
                lines.append(f"    - {label}: {_fmt_float(val, 4)}")
            # Tamanhos das explicações (aceita dois formatos: novo e legado)
            def norm_stats(s: Dict[str, Any]):
                if not isinstance(s, dict):
                    return {"count": 0, "mean_length": 0.0, "std_length": 0.0}
                if "count" in s or "mean_length" in s:
                    return {
                        "count": s.get("count", 0),
                        "mean_length": s.get("mean_length", 0.0),
                        "std_length": s.get("std_length", 0.0)
                    }
                # Formato legado: instancias/min/media/max/std_dev
                return {
                    "count": s.get("instancias", 0),
                    "mean_length": s.get("media", 0.0),
                    "std_length": s.get("std_dev", 0.0)
                }

            exps = blob.get("explanation_stats", {}) or {}
            for part, title in [("positive", "Explicação (positivas)"), ("negative", "Explicação (negativas)"), ("rejected", "Explicação (rejeitadas)")]:
                s = norm_stats(exps.get(part, {}) or {})
                lines.append(f"    - {title}: n={int(s['count'])}, mean={_fmt_float(s['mean_length'], 2)} ± {_fmt_float(s['std_length'], 2)}")
            # Tempo
            ct = blob.get("computation_time", {})
            total_time_secs = ct.get('total')
            lines.append(f"    - Tempo total: {_format_duration(total_time_secs)}")
            lines.append(f"    - Tempo médio/instância (s): {_fmt_float(ct.get('mean_per_instance'), 4)}")
            lines.append(f"    - Tempo (pos/neg/rej) (s): {_fmt_float(ct.get('positive'), 4)} / {_fmt_float(ct.get('negative'), 4)} / {_fmt_float(ct.get('rejected'), 4)}")
            
            # Top features
            all_feats = blob.get("top_features") or []
            top_5 = all_feats[:5]
            
            if top_5:
                txt = ", ".join([f"{t.get('feature','?')}({t.get('count','?')})" for t in top_5])
                lines.append(f"    - Top 5 features: {txt}")
            
            lines.append("")
        lines.append("")

    # Destaques simples por dataset
    lines.append("-- Destaques por Dataset (maior acurácia com rejeição) --")
    for ds in DATASETS:
        best = None
        for method in ["PEAB", "MinExp", "Anchor"]:
            blob = (present_methods.get(method) or {}).get(ds)
            if not blob:
                continue
            acc = _get(blob, ["performance", "accuracy_with_rejection"])
            try:
                acc = float(acc)
            except Exception:
                continue
            if best is None or acc > best[1]:
                best = (method, acc)
        if best:
            lines.append(f"{ds}: {best[0]} ({_fmt_float(best[1], 2)}%)")
        else:
            lines.append(f"{ds}: sem dados")

    return "\n".join(lines)


def main():
    results = _load_json_results()
    txt = build_summary_text(results)
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%d_%m_%H_%M")
    out_path = os.path.join("results", f"resumo_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(out_path)


if __name__ == "__main__":
    main()
