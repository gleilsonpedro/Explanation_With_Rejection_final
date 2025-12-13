"""
Valida fidelidade especificamente para Pima Indians Diabetes - PEAB
"""
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

# Calcular thresholds
decision_scores = pipeline.decision_function(X_test.values)

# Carregar thresholds do JSON
with open('json/peab/pima_indians_diabetes.json', 'r', encoding='utf-8') as f:
    data_temp = json.load(f)
t_plus = data_temp['thresholds']['t_plus']
t_minus = data_temp['thresholds']['t_minus']

print(f"Thresholds: t_minus={t_minus:.4f}, t_plus={t_plus:.4f}")

# Carregar resultados PEAB
with open('json/peab/pima_indians_diabetes.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Criar normalizador
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train.values)

# Validar cada instância
resultados = {'aceitas': {'total': 0, 'fidelidade': []},
              'rejeitadas': {'total': 0, 'fidelidade': []}}

for exp in data['per_instance']:
    idx = int(exp['id'])
    is_rejected = exp['rejected']
    explicacao_names = exp['explanation']  # Nomes das features
    score_norm = exp['decision_score_normalized']
    
    # Converter nomes para índices
    feature_names = list(X_train.columns)
    explicacao_indices = [feature_names.index(name) for name in explicacao_names if name in feature_names]
    
    # Pegar instância
    x_original = X_test.loc[idx].values
    
    # Gerar 1000 perturbações
    n_perturb = 1000
    scores = []
    
    for _ in range(n_perturb):
        x_pert = x_original.copy()
        
        # Perturbar features NÃO explicadas
        for feat_idx in range(len(x_original)):
            if feat_idx not in explicacao_indices:
                # Amostragem uniforme da feature
                feat_min = X_train.values[:, feat_idx].min()
                feat_max = X_train.values[:, feat_idx].max()
                x_pert[feat_idx] = np.random.uniform(feat_min, feat_max)
        
        # Calcular score normalizado (mesmo método do PEAB)
        score_raw = pipeline.decision_function([x_pert])[0]
        max_abs = data['thresholds']['normalization']['max_abs']
        score_pert_norm = score_raw / max_abs
        
        scores.append(score_pert_norm)
    
    scores = np.array(scores)
    
    # Calcular fidelidade
    if is_rejected:
        # Rejeitadas: devem permanecer na zona
        acertos = np.sum((scores >= t_minus) & (scores <= t_plus))
        resultados['rejeitadas']['total'] += 1
        resultados['rejeitadas']['fidelidade'].append(acertos / n_perturb)
    else:
        # Aceitas: devem permanecer na mesma classe
        if score_norm > t_plus:  # Classe 1
            acertos = np.sum(scores > t_plus)
        else:  # Classe 0
            acertos = np.sum(scores < t_minus)
        
        resultados['aceitas']['total'] += 1
        resultados['aceitas']['fidelidade'].append(acertos / n_perturb)

# Mostrar resultados
print(f"\n{'='*80}")
print(f"RESULTADOS DA VALIDAÇÃO - PEAB - PIMA INDIANS DIABETES")
print(f"{'='*80}")

# Aceitas
aceitas = resultados['aceitas']
if aceitas['total'] > 0:
    media_aceitas = np.mean(aceitas['fidelidade']) * 100
    std_aceitas = np.std(aceitas['fidelidade']) * 100
    print(f"\n📊 PREDIÇÕES ACEITAS ({aceitas['total']} instâncias):")
    print(f"   Fidelidade média: {media_aceitas:.2f}% (±{std_aceitas:.2f}%)")

# Rejeitadas
rejeitadas = resultados['rejeitadas']
if rejeitadas['total'] > 0:
    media_rejeitadas = np.mean(rejeitadas['fidelidade']) * 100
    std_rejeitadas = np.std(rejeitadas['fidelidade']) * 100
    print(f"\n📊 PREDIÇÕES REJEITADAS ({rejeitadas['total']} instâncias):")
    print(f"   Fidelidade média: {media_rejeitadas:.2f}% (±{std_rejeitadas:.2f}%)")

# Total
todas_fidelidades = aceitas['fidelidade'] + rejeitadas['fidelidade']
media_total = np.mean(todas_fidelidades) * 100
std_total = np.std(todas_fidelidades) * 100
print(f"\n📊 TOTAL ({aceitas['total'] + rejeitadas['total']} instâncias):")
print(f"   Fidelidade média: {media_total:.2f}% (±{std_total:.2f}%)")

print(f"\n{'='*80}")
