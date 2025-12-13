"""
Debug de uma instância rejeitada específica
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data.datasets import carregar_dataset

# Carregar dataset
X, y, _ = carregar_dataset('pima_indians_diabetes')

# Preparar dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar modelo
pipeline = make_pipeline(StandardScaler(), SVC(C=10, gamma=0.001, kernel='rbf', random_state=42))
pipeline.fit(X_train.values, y_train.values)

# Calcular decision scores
decision_scores = pipeline.decision_function(X_test.values)

# Carregar JSON
with open('json/peab/pima_indians_diabetes.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

t_plus = data['thresholds']['t_plus']
t_minus = data['thresholds']['t_minus']
max_abs = data['thresholds']['normalization']['max_abs']

print(f"Thresholds: t_minus={t_minus:.4f}, t_plus={t_plus:.4f}")
print(f"max_abs: {max_abs:.4f}\n")

# Calcular normalização
mean_score = np.mean(decision_scores)
std_score = np.std(decision_scores)
print(f"Mean score: {mean_score:.4f}")
print(f"Std score: {std_score:.4f}\n")

# Pegar primeira rejeitada
rejeitadas = [x for x in data['per_instance'] if x['rejected']]
print(f"Total rejeitadas: {len(rejeitadas)}\n")

exp = rejeitadas[0]
idx = int(exp['id'])
explicacao_names = exp['explanation']
score_norm_salvo = exp['decision_score_normalized']

print(f"Instância {idx}:")
print(f"  Score normalizado salvo: {score_norm_salvo:.4f}")
print(f"  Explicação: {explicacao_names}")
print(f"  Tamanho explicação: {len(explicacao_names)}")
print(f"  Está na zona? {t_minus <= score_norm_salvo <= t_plus}\n")

# Converter nomes para índices
feature_names = list(X_train.columns)
explicacao_indices = [feature_names.index(name) for name in explicacao_names if name in feature_names]

print(f"Features explicadas (índices): {explicacao_indices}")
print(f"Features NÃO explicadas: {[i for i in range(len(feature_names)) if i not in explicacao_indices]}\n")

# Pegar instância original
x_original = X_test.loc[idx].values
score_original = pipeline.decision_function([x_original])[0]
score_z_original = (score_original - mean_score) / std_score
score_norm_original = score_z_original / max_abs

print(f"Instância original:")
print(f"  Score raw: {score_original:.4f}")
print(f"  Score z: {score_z_original:.4f}")
print(f"  Score norm: {score_norm_original:.4f}")
print(f"  Está na zona? {t_minus <= score_norm_original <= t_plus}\n")

# Testar 10 perturbações
print("Testando 10 perturbações:")
for i in range(10):
    x_pert = x_original.copy()
    
    # Perturbar apenas features NÃO explicadas
    for feat_idx in range(len(x_original)):
        if feat_idx not in explicacao_indices:
            feat_min = X_train.values[:, feat_idx].min()
            feat_max = X_train.values[:, feat_idx].max()
            x_pert[feat_idx] = np.random.uniform(feat_min, feat_max)
    
    # Calcular score normalizado
    score_raw = pipeline.decision_function([x_pert])[0]
    score_z = (score_raw - mean_score) / std_score
    score_norm = score_z / max_abs
    
    na_zona = t_minus <= score_norm <= t_plus
    print(f"  Pert {i+1}: score_norm={score_norm:7.4f}  na_zona={na_zona}")
