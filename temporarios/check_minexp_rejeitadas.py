import json
from pathlib import Path

datasets = [
    "vertebral_column",
    "pima_indians_diabetes", 
    "heart_disease",
    "creditcard",
    "breast_cancer",
    "sonar",
    "mnist"
]

print("CHECK MinExp - Rejeitadas com std=0.00\n")

for ds in datasets:
    json_path = Path(f"json/minexp/{ds}.json")
    if not json_path.exists():
        print(f"{ds:25s} - JSON não existe")
        continue
    
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
        per_instance = data.get('per_instance', [])
        
        rejeitadas = [p for p in per_instance if p.get('rejected', False)]
        n_rej = len(rejeitadas)
        
        if n_rej == 0:
            print(f"{ds:25s} - 0 rejeitadas")
            continue
        
        tempos = [p.get('computation_time', 0.0) for p in rejeitadas]
        
        # Verificar se todos são iguais (std seria 0)
        todos_iguais = len(set(tempos)) == 1
        
        print(f"{ds:25s} - {n_rej:3d} rejeitadas | Tempos únicos: {len(set(tempos)):3d} | Todos iguais: {todos_iguais} | 1º tempo: {tempos[0]:.6f} | Último: {tempos[-1]:.6f}")
        
    except Exception as e:
        print(f"{ds:25s} - ERRO: {e}")
