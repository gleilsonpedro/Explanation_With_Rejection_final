"""
Script de teste para verificar se as configurações do PEAB estão sincronizadas
"""
import sys

print("="*80)
print("TESTE DE SINCRONIZAÇÃO DE CONFIGURAÇÕES")
print("="*80)

# Testar import direto do peab
print("\n1. Importando de peab.py...")
from peab import DATASET_CONFIG as PEAB_CONFIG
print("✅ Import de peab.py bem-sucedido")

# Testar import via shared_training
print("\n2. Importando via shared_training.py...")
from utils.shared_training import DATASET_CONFIG as SHARED_CONFIG
print("✅ Import via shared_training.py bem-sucedido")

# Testar import via pulp_experiment
print("\n3. Importando via pulp_experiment.py...")
from pulp_experiment import DATASET_CONFIG as PULP_CONFIG
print("✅ Import via pulp_experiment.py bem-sucedido")

print("\n" + "="*80)
print("COMPARAÇÃO DE CONFIGURAÇÕES - MNIST")
print("="*80)

datasets_to_check = ['mnist', 'rcv1', 'newsgroups']

for dataset in datasets_to_check:
    if dataset in PEAB_CONFIG:
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-"*80)
        
        peab_subsample = PEAB_CONFIG[dataset].get('subsample_size', 'N/A')
        shared_subsample = SHARED_CONFIG[dataset].get('subsample_size', 'N/A')
        pulp_subsample = PULP_CONFIG[dataset].get('subsample_size', 'N/A')
        
        print(f"PEAB subsample_size:   {peab_subsample}")
        print(f"Shared subsample_size: {shared_subsample}")
        print(f"PuLP subsample_size:   {pulp_subsample}")
        
        if peab_subsample == shared_subsample == pulp_subsample:
            print("✅ CONFIGURAÇÕES SINCRONIZADAS!")
        else:
            print("❌ CONFIGURAÇÕES DESINCRONIZADAS!")
            print("⚠️  Possível problema: imports ainda usando peab_original.py")

print("\n" + "="*80)
print("TESTE CONCLUÍDO")
print("="*80)
