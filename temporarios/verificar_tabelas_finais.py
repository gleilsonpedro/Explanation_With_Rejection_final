import json
import os

datasets = ['breast_cancer', 'pima_indians_diabetes', 'sonar', 'vertebral_column', 
            'banknote', 'heart_disease', 'spambase']

print("=" * 90)
print("VERIFICAÇÃO FINAL: Dados nas Tabelas vs JSONs")
print("=" * 90)

# Valores esperados na tabela de explicações (linha PEAB Classif.)
valores_tabela_peab_classif = {
    'banknote': 2.87,
    'breast_cancer': 1.69,
    'heart_disease': 7.06,
    'pima_indians_diabetes': 4.17,
    'sonar': 41.89,
    'spambase': 28.10,
    'vertebral_column': 4.34
}

valores_tabela_pulp_classif = {
    'banknote': 2.87,
    'breast_cancer': 1.69,
    'heart_disease': 7.12,
    'pima_indians_diabetes': 4.17,
    'sonar': 42.28,
    'spambase': 28.16,
    'vertebral_column': 4.34
}

print("\n✅ VERIFICAÇÃO: Tamanhos médios de explicações CLASSIFICADAS")
print("-" * 90)

for dataset in datasets:
    print(f"\n📊 {dataset.upper()}")
    
    # Ler JSONs
    peab_path = f'json/peab/{dataset}.json'
    pulp_path = f'json/pulp/{dataset}.json'
    
    if os.path.exists(peab_path) and os.path.exists(pulp_path):
        with open(peab_path, 'r') as f:
            peab_data = json.load(f)
        with open(pulp_path, 'r') as f:
            pulp_data = json.load(f)
        
        # Calcular tamanhos médios das classificadas (positivas + negativas)
        peab_stats = peab_data.get('explanation_stats', {})
        peab_pos = peab_stats.get('positive', {}).get('mean_length', 0)
        peab_neg = peab_stats.get('negative', {}).get('mean_length', 0)
        peab_pos_count = peab_stats.get('positive', {}).get('count', 0)
        peab_neg_count = peab_stats.get('negative', {}).get('count', 0)
        
        if (peab_pos_count + peab_neg_count) > 0:
            peab_classif_avg = (peab_pos * peab_pos_count + peab_neg * peab_neg_count) / (peab_pos_count + peab_neg_count)
        else:
            peab_classif_avg = 0
        
        pulp_stats = pulp_data.get('estatisticas_por_tipo', {})
        pulp_pos_avg = pulp_stats.get('positiva', {}).get('tamanho_medio', 0)
        pulp_neg_avg = pulp_stats.get('negativa', {}).get('tamanho_medio', 0)
        pulp_pos_count = pulp_stats.get('positiva', {}).get('instancias', 0)
        pulp_neg_count = pulp_stats.get('negativa', {}).get('instancias', 0)
        
        if (pulp_pos_count + pulp_neg_count) > 0:
            pulp_classif_avg = (pulp_pos_avg * pulp_pos_count + pulp_neg_avg * pulp_neg_count) / (pulp_pos_count + pulp_neg_count)
        else:
            pulp_classif_avg = 0
        
        # Comparar com valores da tabela
        peab_tabela = valores_tabela_peab_classif[dataset]
        pulp_tabela = valores_tabela_pulp_classif[dataset]
        
        peab_ok = abs(peab_classif_avg - peab_tabela) < 0.01
        pulp_ok = abs(pulp_classif_avg - pulp_tabela) < 0.01
        
        print(f"   PEAB: JSON={peab_classif_avg:.2f}, Tabela={peab_tabela:.2f} {'✅' if peab_ok else '❌'}")
        print(f"   PULP: JSON={pulp_classif_avg:.2f}, Tabela={pulp_tabela:.2f} {'✅' if pulp_ok else '❌'}")

print("\n" + "=" * 90)
print("✅ Verificação concluída!")
print("=" * 90)
