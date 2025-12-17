"""
Teste direto de uma instância positiva do vertebral_column
para entender por que minimalidade = 0%
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\gleilsonpedro\OneDrive\Área de Trabalho\PYTHON\MESTRADO\XAI\Explanation_With_Rejection_final')

from utils.shared_training import get_shared_pipeline

# Carregar dados
pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline('vertebral_column')

# Pegar primeira instância positiva (y_pred=1)
positivas_idx = [i for i, pred in enumerate(pipeline.predict(X_test)) if pred == 1]
print(f"Total de positivas: {len(positivas_idx)}")

idx = positivas_idx[0]
inst = X_test.iloc[idx].values
y_pred = pipeline.predict(X_test.iloc[[idx]])[0]
score_orig = pipeline.decision_function(X_test.iloc[[idx]])[0]

print(f"\nInstancia {idx}:")
print(f"  Y_pred: {y_pred}")
print(f"  Score original: {score_orig:.4f}")
print(f"  Features: {X_test.columns.tolist()}")
print(f"  Valores: {inst}")

# Obter coeficientes
print(f"\nEtapas do pipeline: {list(pipeline.named_steps.keys())}")
logreg = None
for key in pipeline.named_steps.keys():
    step = pipeline.named_steps[key]
    if hasattr(step, 'coef_'):
        logreg = step
        print(f"Modelo encontrado: {key}")
        break

if logreg is None:
    print("ERRO: Modelo não encontrado!")
    sys.exit(1)

coefs = logreg.coef_[0]
print(f"\nCoeficientes:")
for i, feat in enumerate(X_test.columns):
    print(f"  {feat}: {coefs[i]:.4f}")

# Simular remover primeira feature e colocar outras no worst-case
features_explicacao = ['lumbar_lordosis_angle', 'pelvic_radius', 'pelvic_tilt']  # Exemplo típico
print(f"\nExplicacao: {features_explicacao}")

# Testar remover 'lumbar_lordosis_angle'
print(f"\n--- Testando sem lumbar_lordosis_angle ---")
expl_sem_feat = ['pelvic_radius', 'pelvic_tilt']

# Criar perturbação adversarial
n_pert = 10
pert = np.tile(inst, (n_pert, 1))

# Features não explicativas: todas exceto expl_sem_feat
features_fixas_idx = [X_test.columns.get_loc(f) for f in expl_sem_feat]
features_perturbar_idx = [i for i in range(len(X_test.columns)) if i not in features_fixas_idx]

print(f"Features fixas (da explicação sem lumbar): {expl_sem_feat}")
print(f"Features a perturbar (worst-case): {[X_test.columns[i] for i in features_perturbar_idx]}")

# Aplicar worst-case
for feat_idx in features_perturbar_idx:
    coef = coefs[feat_idx]
    feat_name = X_test.columns[feat_idx]
    
    # Obter min/max
    feat_min = X_train.iloc[:, feat_idx].min()
    feat_max = X_train.iloc[:, feat_idx].max()
    
    # Para positiva (y_pred=1), adversário quer DIMINUIR score
    if coef > 0:
        valor_worst = feat_min  # Coef positivo → diminuir feature
    else:
        valor_worst = feat_max  # Coef negativo → aumentar feature (diminui score)
    
    pert[:, feat_idx] = valor_worst
    print(f"  {feat_name}: coef={coef:.4f}, worst={valor_worst:.4f}")

# Testar predições
preds = pipeline.predict(pert)
acertos = np.sum(preds == y_pred)
fidelity = acertos / n_pert

print(f"\nResultado:")
print(f"  Acertos: {acertos}/{n_pert}")
print(f"  Fidelity: {fidelity*100:.1f}%")
print(f"  Feature redundante? {fidelity > 0.95}")

if fidelity > 0.95:
    print(f"\n  => lumbar_lordosis_angle marcada como REDUNDANTE")
    print(f"     (explicação funciona sem ela mesmo com worst-case!)")
