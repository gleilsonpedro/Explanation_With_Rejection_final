import joblib
from pathlib import Path

cache_path = Path('cache/cache_cumulativo.pkl')
if not cache_path.exists():
    print('Arquivo cache/cache_cumulativo.pkl não encontrado!')
    exit(1)

cache = joblib.load(cache_path)
print(f"Datasets no cache: {list(cache.keys())}\n")
for ds, dados in cache.items():
    print(f"--- {ds} ---")
    if isinstance(dados, dict):
        print("Chaves:", list(dados.keys()))
        if 'error_on_cache_build' in dados:
            print("ERRO CAPTURADO:", dados['error_on_cache_build'])
    else:
        print("Valor não é um dicionário! Tipo:", type(dados))
    print()
