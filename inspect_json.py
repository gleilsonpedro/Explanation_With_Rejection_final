import json

# Carregar JSON
with open('json/comparative_results.json', 'r') as f:
    data = json.load(f)

mnist = data['peab']['mnist']

print("="*80)
print("INSPEÇÃO DA ESTRUTURA DO JSON - MNIST")
print("="*80)

print(f"\nTotal de instâncias: {len(mnist['per_instance'])}")
print(f"\nTotal de features no X_test: {len(mnist['data']['X_test'])}")
print(f"Número de valores em cada feature: {len(mnist['data']['X_test']['pixel1'])}")

print("\n" + "="*80)
print("PRIMEIRAS 5 INSTÂNCIAS:")
print("="*80)
for i, inst in enumerate(mnist['per_instance'][:5]):
    print(f"idx_sequencial={i:3d} | id={inst['id']:>6} | y_true={inst['y_true']} | y_pred={inst['y_pred']} | rejected={inst['rejected']}")

print("\n" + "="*80)
print("ÚLTIMAS 5 INSTÂNCIAS:")
print("="*80)
for i, inst in enumerate(mnist['per_instance'][-5:], len(mnist['per_instance'])-5):
    print(f"idx_sequencial={i:3d} | id={inst['id']:>6} | y_true={inst['y_true']} | y_pred={inst['y_pred']} | rejected={inst['rejected']}")

print("\n" + "="*80)
print("VERIFICAÇÕES:")
print("="*80)
print(f"✓ Model tem campo 'rejection'? {'rejection' in mnist.get('model', {})}")
print(f"✓ Config tem 'rejection_cost'? {'rejection_cost' in mnist.get('config', {})}")
print(f"✓ Dataset name: {mnist['config']['dataset_name']}")
print(f"✓ Classes: {mnist['data']['class_names']}")

print("\n" + "="*80)
print("TESTE DE CORRESPONDÊNCIA:")
print("="*80)
print("Se idx_sequencial == id, então podemos usar o id diretamente")
print("Se idx_sequencial != id, então DEVEMOS usar idx_sequencial\n")

# Testar se há correspondência
correspondencia = all(i == int(inst['id']) for i, inst in enumerate(mnist['per_instance']))
print(f"Correspondência perfeita (idx == id)? {correspondencia}")

if not correspondencia:
    print("\n⚠️  IMPORTANTE: Os IDs NÃO correspondem aos índices sequenciais!")
    print("   Isso significa que:")
    print("   - inst['id'] = ID original do MNIST (ex: 45336, 67200)")
    print("   - idx sequencial = posição no array X_test (0, 1, 2, ...)")
    print("\n   ✅ Solução atual (usar idx sequencial) está CORRETA!")
else:
    print("\n✓ Os IDs correspondem aos índices - qualquer abordagem funciona")

print("\n" + "="*80)
print("RECOMENDAÇÃO FINAL:")
print("="*80)
if not correspondencia:
    print("❌ NÃO alterar para _get_instance_by_id - continuará quebrando!")
    print("✅ MANTER _get_instance_vector com idx sequencial")
else:
    print("✓ Pode usar qualquer abordagem")
