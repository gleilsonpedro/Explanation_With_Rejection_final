import json
from pathlib import Path
import numpy as np
import sys
sys.path.append('.')
from utils.shared_training import get_shared_pipeline

# Vamos verificar os valores de max_abs para cada dataset
datasets = ['breast_cancer', 'spambase']

print("=" * 90)
print("INVESTIGAÇÃO: Por que spambase quebrou após remover normalização?")
print("=" * 90)

for dataset_name in datasets:
    print(f"\n{'='*90}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*90}")
    
    # Carrega pipeline
    pipe, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)
    
    # Pega scaler e modelo
    scaler = pipe.named_steps['scaler']
    model = pipe.named_steps['model']
    
    # Calcula max_abs
    X_train_scaled = scaler.transform(X_train)
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Calcula scores
    scores_train = X_train_scaled.dot(coefs) + intercept
    max_abs = max(abs(scores_train.min()), abs(scores_train.max()))
    
    print(f"\n📊 Estatísticas do Score (RAW):")
    print(f"   Min score: {scores_train.min():.6f}")
    print(f"   Max score: {scores_train.max():.6f}")
    print(f"   Max abs: {max_abs:.6f}")
    
    print(f"\n🎯 Thresholds (RAW):")
    print(f"   t+: {t_plus:.6f}")
    print(f"   t-: {t_minus:.6f}")
    
    print(f"\n🔍 Thresholds NORMALIZADOS (divididos por max_abs):")
    print(f"   t+ / max_abs: {t_plus / max_abs:.6f}")
    print(f"   t- / max_abs: {t_minus / max_abs:.6f}")
    
    print(f"\n💡 Comparação de magnitudes:")
    print(f"   |t+| / max_abs = {abs(t_plus) / max_abs:.4f} ({abs(t_plus) / max_abs * 100:.1f}% da escala)")
    print(f"   |t-| / max_abs = {abs(t_minus) / max_abs:.4f} ({abs(t_minus) / max_abs * 100:.1f}% da escala)")
    
    # Verifica se os thresholds estão muito próximos de 0 em relação à escala
    if max(abs(t_plus), abs(t_minus)) / max_abs < 0.1:
        print(f"\n   ⚠️ ALERTA: Thresholds são MUITO PEQUENOS em relação à escala!")
        print(f"   Isso pode causar problemas numéricos no solver.")
    
    # Calcula um score de teste normalizado
    scores_test = scaler.transform(X_test).dot(coefs) + intercept
    print(f"\n📈 Scores de Teste (RAW):")
    print(f"   Min: {scores_test.min():.6f}")
    print(f"   Max: {scores_test.max():.6f}")
    
    # Carrega JSON do PEAB para comparar
    peab_file = Path(f"json/peab/{dataset_name}.json")
    if peab_file.exists():
        with open(peab_file, 'r') as f:
            peab = json.load(f)
        
        peab_t_plus = peab['thresholds']['t_plus']
        peab_t_minus = peab['thresholds']['t_minus']
        
        print(f"\n✅ Verificação com PEAB JSON:")
        print(f"   PEAB t+: {peab_t_plus:.6f}")
        print(f"   PEAB t-: {peab_t_minus:.6f}")
        print(f"   Match: {abs(peab_t_plus - t_plus) < 1e-6 and abs(peab_t_minus - t_minus) < 1e-6}")

print("\n" + "=" * 90)
print("CONCLUSÃO")
print("=" * 90)
print("\nA normalização pode ser necessária quando:")
print("1. Os scores têm magnitude muito diferente dos thresholds")
print("2. O max_abs é muito maior que os thresholds")
print("3. Há problemas numéricos no solver devido à escala")
print("\nPrecisamos decidir: usar normalização sempre, nunca, ou condicionalmente?")
