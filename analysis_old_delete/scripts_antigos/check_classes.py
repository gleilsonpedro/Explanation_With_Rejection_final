import json

data = json.load(open('json/minexp/mnist.json', 'r'))

print('=== MAPEAMENTO DE CLASSES NO MNIST ===')
print('class_names:', data['model']['class_names'])
print('')
print('✅ RESPOSTA:')
print(f"  class_names[0] (y=0, NEGATIVA): {data['model']['class_names'][0]}")
print(f"  class_names[1] (y=1, POSITIVA): {data['model']['class_names'][1]}")
print('')
print('Portanto:')
print(f"  • {data['model']['class_names'][0]} = NEGATIVA (y=0)")
print(f"  • {data['model']['class_names'][1]} = POSITIVA (y=1)")
print('')
print('=== EXEMPLOS ===')

for i in range(min(5, len(data['per_instance']))):
    inst = data['per_instance'][i]
    true_class = data['model']['class_names'][inst['y_true']]
    pred_class = data['model']['class_names'][inst['y_pred']] if inst['y_pred'] in [0, 1] else 'REJEITADA'
    print(f"Instância {i}: verdade={true_class} (y={inst['y_true']}), "
          f"predição={pred_class} (y={inst['y_pred']}), "
          f"score={inst['decision_score']:.4f}, "
          f"rejected={inst['rejected']}")
