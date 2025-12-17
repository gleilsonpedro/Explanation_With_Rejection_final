import json
from peab_validation import validar_metodo


DATASETS = [
    'banknote',
    'breast_cancer',
    'pima_indians_diabetes',
    'sonar',
    'vertebral_column',
    'wine',
    'spambase',
    'heart_disease',
    'creditcard',
    'mnist_3_vs_8',
]


def main():
    rows = []
    for ds in DATASETS:
        try:
            r = validar_metodo('peab', ds, verbose=False)
            if not r:
                print(f"SKIP {ds}: no result")
                continue
            g = r['global_metrics']
            pt = r['per_type_metrics']
            row = {
                'dataset': ds,
                'fidelity_overall': g['fidelity_overall'],
                'necessity_overall': g['necessity_overall'],
                'necessity_positive': pt['positive']['necessity'],
                'necessity_negative': pt['negative']['necessity'],
                'necessity_rejected': pt['rejected']['necessity'],
            }
            rows.append(row)
            print(f"{ds:22s} | Fid:{g['fidelity_overall']:.2f}% | NecG:{g['necessity_overall']:.2f}% | Nec(+):{pt['positive']['necessity']:.2f}% | Nec(-):{pt['negative']['necessity']:.2f}% | Nec(R):{pt['rejected']['necessity']:.2f}%")
        except Exception as e:
            print(f"ERR {ds}: {e}")

    # Sort by necessity overall desc, then fidelity desc
    rows.sort(key=lambda r: (r['necessity_overall'], r['fidelity_overall']), reverse=True)
    top6 = rows[:6]

    print("\nTOP 6 by Necessity Overall then Fidelity:")
    for i, r in enumerate(top6, 1):
        print(f"{i}. {r['dataset']}  | Fid:{r['fidelity_overall']:.2f}% | NecG:{r['necessity_overall']:.2f}%")

    print("\nJSON:")
    print(json.dumps({'all': rows, 'top6': top6}, indent=2))


if __name__ == '__main__':
    main()
