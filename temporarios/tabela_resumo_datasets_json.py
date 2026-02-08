import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Row:
    dataset: str
    total_instancias_dataset: int
    total_treino: int
    total_teste_full: int
    subsample: float | None
    total_teste_subsample: int
    instancias_explicadas: int
    num_features: int | None
    fonte_json: str


def _round_int(x: float) -> int:
    return int(round(x))


def _get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_peab_json(rel_path: str) -> dict[str, Any]:
    p = ROOT / rel_path
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_row(dataset_label: str, rel_path: str) -> Row:
    data = load_peab_json(rel_path)

    test_size = _get(data, ["config", "test_size"], None)
    subsample = _get(data, ["config", "subsample_size"], None)
    n_test_sub = _get(data, ["performance", "num_test_instances"], None)
    n_features = _get(data, ["model", "num_features"], None)

    per_instance = data.get("per_instance")
    n_explained = len(per_instance) if isinstance(per_instance, list) else None

    if test_size is None or n_test_sub is None:
        raise ValueError(f"JSON sem test_size/num_test_instances: {rel_path}")

    # ✅ Interpretação correta do pipeline deste projeto:
    # - train/test split é feito no dataset completo
    # - subsample_size (quando existe) é aplicado APENAS no TESTE
    subsample_val = None if subsample is None else float(subsample)
    if subsample_val is None or subsample_val >= 1.0:
        n_test_full = int(n_test_sub)
    else:
        n_test_full = _round_int(int(n_test_sub) / subsample_val)

    n_total_dataset = _round_int(n_test_full / float(test_size))
    n_train = max(int(n_total_dataset) - int(n_test_full), 0)

    if n_explained is None:
        n_explained = int(n_test_sub)

    return Row(
        dataset=dataset_label,
        total_instancias_dataset=int(n_total_dataset),
        total_treino=int(n_train),
        total_teste_full=int(n_test_full),
        subsample=subsample_val,
        total_teste_subsample=int(n_test_sub),
        instancias_explicadas=int(n_explained),
        num_features=int(n_features) if n_features is not None else None,
        fonte_json=rel_path.replace("\\", "/"),
    )


def to_markdown(rows: list[Row]) -> str:
    def fmt_sub(s: float | None) -> str:
        if s is None:
            return "-"
        return f"{s:g}"

    blocks: list[str] = []
    for r in rows:
        subsample = r.subsample

        blocks.append(
            "\n".join(
                [
                    f"*{r.dataset}*",
                    f"- instancias: {r.total_instancias_dataset}",
                    f"- treino: {r.total_treino}",
                    f"- teste: {r.total_teste_full}",
                    f"- subsample: {fmt_sub(subsample)}",
                    f"- instancias explicadas: {r.instancias_explicadas}",
                    f"- features: {r.num_features if r.num_features is not None else '-'}",
                ]
            )
        )

    return "\n\n".join(blocks) + "\n"


def to_csv(rows: list[Row]) -> str:
    cols = [
        "dataset",
        "total_instancias_dataset",
        "total_treino",
        "total_teste_full",
        "subsample",
        "total_teste_subsample",
        "instancias_explicadas",
        "num_features",
        "fonte_json",
    ]
    out = [";".join(cols)]
    for r in rows:
        out.append(
            ";".join(
                [
                    r.dataset,
                    str(r.total_instancias_dataset),
                    str(r.total_treino),
                    str(r.total_teste_full),
                    "" if r.subsample is None else f"{r.subsample:g}",
                    str(r.total_teste_subsample),
                    str(r.instancias_explicadas),
                    "" if r.num_features is None else str(r.num_features),
                    r.fonte_json,
                ]
            )
        )
    return "\n".join(out) + "\n"


def main() -> None:
    # Fonte principal: JSONs com explicações (per_instance)
    targets = [
        ("covertype", "json/peab/covertype.json"),
        ("mnist", "json/peab/mnist_3_vs_8.json"),
        ("creditcard", "json/peab/creditcard.json"),
    ]

    rows = [build_row(label, path) for label, path in targets]

    md = to_markdown(rows)
    csv = to_csv(rows)

    out_md = ROOT / "temporarios" / "tabela_resumo_datasets.md"
    out_csv = ROOT / "temporarios" / "tabela_resumo_datasets.csv"

    out_md.write_text(md, encoding="utf-8")
    out_csv.write_text(csv, encoding="utf-8")

    print(md)
    print(f"[OK] Markdown: {out_md}")
    print(f"[OK] CSV: {out_csv}")


if __name__ == "__main__":
    main()
