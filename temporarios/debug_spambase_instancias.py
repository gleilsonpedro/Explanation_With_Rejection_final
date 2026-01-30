"""
Debug do spambase: Vamos executar PEAB e PULP na mesma instância
para ver o que está acontecendo.
"""
import sys
sys.path.append('.')

from utils.shared_training import get_shared_pipeline
import numpy as np

# Carrega dados
print("=" * 90)
print("DEBUG SPAMBASE: Comparação instância por instância")
print("=" * 90)

dataset_name = 'spambase'
pipe, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(dataset_name)

feature_names = meta['feature_names']
c_rej = meta['rejection_cost']

print(f"\n📊 Dataset: {dataset_name}")
print(f"   Features: {len(feature_names)}")
print(f"   Test instances: {len(X_test)}")
print(f"   Thresholds: t+={t_plus:.6f}, t-={t_minus:.6f}")

# Pega o scaler e modelo
scaler = pipe.named_steps['scaler']
model = pipe.named_steps['model']

# Normaliza dados
X_test_scaled = scaler.transform(X_test)
coefs = model.coef_[0]
intercept = model.intercept_[0]

# Calcula scores
scores = X_test_scaled.dot(coefs) + intercept

# Encontra instâncias classificadas como negativas (as que têm o problema)
neg_indices = np.where((scores <= t_minus) & (scores <= t_plus))[0]

print(f"\n🔍 Instâncias classificadas como NEGATIVA: {len(neg_indices)}")

# Vamos analisar as primeiras 5
print("\n" + "=" * 90)
print("ANÁLISE DAS PRIMEIRAS 5 INSTÂNCIAS NEGATIVAS")
print("=" * 90)

from peab import explain_instance_minimal

for i, idx in enumerate(neg_indices[:5]):
    instance = X_test[idx]
    score = scores[idx]
    
    print(f"\n{'='*90}")
    print(f"Instância {i+1} (índice {idx})")
    print(f"{'='*90}")
    print(f"Score: {score:.6f} (threshold: {t_minus:.6f})")
    
    # Executa PEAB
    explanation_peab, _, _ = explain_instance_minimal(
        instance=instance,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        threshold_plus=t_plus,
        threshold_minus=t_minus,
        epsilon=1e-5
    )
    
    print(f"\n✅ PEAB: {len(explanation_peab)} features")
    print(f"   Features: {sorted([int(f.split('_')[1]) for f in explanation_peab])[:10]}{'...' if len(explanation_peab) > 10 else ''}")
    
    # Agora vamos simular o PULP manualmente
    import pulp
    
    # Normaliza instância
    vals_scaled = scaler.transform([instance])[0]
    
    # Bounds do scaler
    min_scaled = scaler.data_min_
    max_scaled = scaler.data_max_
    
    # Determina estado (negativa)
    estado = 0  # NEGATIVA
    
    # Problema de otimização
    prob = pulp.LpProblem("Debug", pulp.LpMinimize)
    z = [pulp.LpVariable(f"z_{j}", cat='Binary') for j in range(len(coefs))]
    
    # Objetivo
    prob += pulp.lpSum(z)
    
    # Worst-case
    base_worst_max = intercept
    termos_max = []
    
    for j, w in enumerate(coefs):
        v_worst_max = max_scaled[j] if w > 0 else min_scaled[j]
        contrib_worst_max = v_worst_max * w
        contrib_real = vals_scaled[j] * w
        
        base_worst_max += contrib_worst_max
        termos_max.append(z[j] * (contrib_real - contrib_worst_max))
    
    # Expressão sem normalização (corrigida)
    expr_max = base_worst_max + pulp.lpSum(termos_max)
    
    # Restrição para NEGATIVA
    EPSILON = 1e-5
    prob += expr_max <= t_minus + EPSILON
    
    # Resolve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        features_pulp = [j for j in range(len(z)) if pulp.value(z[j]) > 0.5]
        print(f"\n✅ PULP: {len(features_pulp)} features")
        print(f"   Features: {sorted(features_pulp)[:10]}{'...' if len(features_pulp) > 10 else ''}")
        
        # Verifica a restrição
        score_worst = base_worst_max
        for j in features_pulp:
            w = coefs[j]
            v_worst_max = max_scaled[j] if w > 0 else min_scaled[j]
            contrib_worst = v_worst_max * w
            contrib_real = vals_scaled[j] * w
            score_worst += (contrib_real - contrib_worst)
        
        print(f"\n🔍 Verificação:")
        print(f"   Score worst-case: {score_worst:.6f}")
        print(f"   Threshold t-: {t_minus:.6f}")
        print(f"   Margem: {score_worst - t_minus:.6f}")
        print(f"   Restrição satisfeita: {score_worst <= t_minus + EPSILON}")
        
        # Diferença
        diff = len(features_pulp) - len(explanation_peab)
        print(f"\n⚠️ DIFERENÇA: {diff:+d} features")
        if diff > 0:
            print(f"   PULP está usando {diff} features A MAIS que PEAB")
    else:
        print(f"\n❌ PULP: Falhou ({pulp.LpStatus[prob.status]})")

print("\n" + "=" * 90)
