import json
import numpy as np

datasets = ['banknote', 'vertebral_column', 'pima_indians_diabetes', 'heart_disease', 'mnist']

for ds in datasets:
    try:
        with open(f'json/minexp/{ds}.json') as f:
            d = json.load(f)
        pi = d.get('per_instance', [])
        if pi:
            times = [p.get('computation_time', 0) for p in pi if isinstance(p, dict)]
            if times:
                print(f'{ds}: n={len(times)}, mean={np.mean(times):.3f}s, min={np.min(times):.3f}s, max={np.max(times):.3f}s')
    except Exception as e:
        print(f'{ds}: Error - {e}')
