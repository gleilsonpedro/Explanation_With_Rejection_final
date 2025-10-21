import joblib

# Carrega o cache
cache = joblib.load('auxiliary_files/cache_cumulativo.pkl')

# Mostra informações do cache
print('Chaves disponíveis:', list(cache.keys()))
for dataset, data in cache.items():
    print(f'\nDataset {dataset}:')
    print('Chaves:', list(data.keys()))
    print('Tipos:', {k: type(v).__name__ for k, v in data.items()})