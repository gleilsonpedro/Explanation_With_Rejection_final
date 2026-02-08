import json
from pathlib import Path

DATASETS = [
    ("banknote", "banknote", "banknote", "banknote"),
    ("vertebral_column", "vertebral_column", "vertebral_column", "vertebral_column"),
    ("pima_indians_diabetes", "pima_indians_diabetes", "pima_indians_diabetes", "pima_indians_diabetes"),
    ("heart_disease", "heart_disease", "heart_disease", "heart_disease"),
    ("creditcard", "creditcard", "creditcard", "creditcard"),
    ("breast_cancer", "breast_cancer", "breast_cancer", "breast_cancer"),
    ("covertype", "covertype", "covertype", "covertype"),
    ("spambase", "spambase", "spambase", "spambase"),
    ("sonar", "sonar", "sonar", "sonar"),
    ("mnist", "mnist_3_vs_8", "mnist", "mnist"),
]

METHODS = ["peab", "anchor", "minexp"]


def summarize(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "err": str(exc)}

    per_instance = data.get("per_instance")
    if not isinstance(per_instance, list):
        return {"ok": True, "has_per_instance": False}

    times = []
    for pi in per_instance:
        if not isinstance(pi, dict):
            continue
        if "computation_time" not in pi:
            continue
        try:
            times.append(float(pi["computation_time"]))
        except Exception:
            continue

    if not times:
        return {"ok": True, "has_per_instance": True, "n": len(per_instance), "has_times": False}

    nonzero = sum(1 for t in times if abs(t) > 1e-12)
    return {
        "ok": True,
        "has_per_instance": True,
        "has_times": True,
        "n": len(times),
        "nonzero": nonzero,
        "allzero": nonzero == 0,
        "min": min(times),
        "max": max(times),
    }


def main():
    rows = []
    for base, peab_name, anchor_name, minexp_name in DATASETS:
        names = {"peab": peab_name, "anchor": anchor_name, "minexp": minexp_name}
        for method in METHODS:
            p = Path("json") / method / f"{names[method]}.json"
            if not p.exists():
                rows.append((base, method, "MISSING_FILE", "", ""))
                continue
            s = summarize(p)
            if not s.get("ok"):
                rows.append((base, method, "ERROR", s.get("err", "")[:70], ""))
                continue

            if not s.get("has_per_instance"):
                rows.append((base, method, "NO_per_instance", "", ""))
            elif not s.get("has_times"):
                rows.append((base, method, f"NO_times(n={s.get('n', 0)})", "", ""))
            elif s.get("allzero"):
                rows.append((base, method, f"ALL_ZERO(n={s['n']})", f"min={s['min']}", f"max={s['max']}"))
            else:
                rows.append(
                    (
                        base,
                        method,
                        f"OK(nonzero={s['nonzero']}/{s['n']})",
                        f"min={s['min']:.6g}",
                        f"max={s['max']:.6g}",
                    )
                )

    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    w3 = max(len(r[2]) for r in rows)

    print("CHECK per_instance.computation_time (nonzero?)\n")
    for r in rows:
        print(f"{r[0]:<{w1}}  {r[1]:<{w2}}  {r[2]:<{w3}}  {r[3]:<20}  {r[4]}")

    bad = [r for r in rows if r[2].startswith(("ALL_ZERO", "NO_times", "NO_per_instance", "MISSING_FILE", "ERROR"))]
    print("\nResumo:")
    print(f"- Total checks: {len(rows)}")
    print(f"- Com problema: {len(bad)}")
    if bad:
        print("- Problemas encontrados (até 15):")
        for r in bad[:15]:
            print("  ", r)


if __name__ == "__main__":
    main()
