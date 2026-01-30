import json
import os

datasets = ['breast_cancer', 'pima_indians_diabetes', 'sonar', 'vertebral_column', 
            'banknote', 'heart_disease', 'spambase']

print("=" * 90)
print("VERIFICAÇÃO DOS ARQUIVOS JSON")
print("=" * 90)

for dataset in datasets:
    print(f"\n📁 Dataset: {dataset.upper()}")
    print("-" * 90)
    
    # Verificar PEAB
    peab_path = f'json/peab/{dataset}.json'
    if os.path.exists(peab_path):
        with open(peab_path, 'r') as f:
            peab_data = json.load(f)
        print(f"✅ PEAB JSON encontrado")
        print(f"   - Total explicações: {len(peab_data.get('explicacoes', []))}")
        if 'tempos' in peab_data:
            print(f"   - Tempos registrados: {len(peab_data['tempos'])}")
    else:
        print(f"❌ PEAB JSON não encontrado")
    
    # Verificar PULP
    pulp_path = f'json/pulp/{dataset}.json'
    if os.path.exists(pulp_path):
        with open(pulp_path, 'r') as f:
            pulp_data = json.load(f)
        print(f"✅ PULP JSON encontrado")
        print(f"   - Total explicações: {len(pulp_data.get('explicacoes', []))}")
        if 'tempos' in pulp_data:
            print(f"   - Tempos registrados: {len(pulp_data['tempos'])}")
        
        # Verificar se tem estrutura de tipo (classificada_positiva, etc)
        first_tempo = pulp_data['tempos'][0] if pulp_data.get('tempos') else {}
        if 'tipo' in first_tempo:
            print(f"   - Campo 'tipo' presente: ✅")
        else:
            print(f"   - Campo 'tipo' AUSENTE: ❌")
    else:
        print(f"❌ PULP JSON não encontrado")

print("\n" + "=" * 90)
print("RESUMO")
print("=" * 90)
print(f"✅ Verificação dos JSONs concluída")
print(f"   - Total datasets: {len(datasets)}")
print(f"   - Todos devem ter JSONs PEAB e PULP com campo 'tipo' nos tempos")
