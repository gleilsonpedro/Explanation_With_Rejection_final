import json

d = json.load(open('json/pulp/pima_indians_diabetes.json','r',encoding='utf-8'))
exp = d['explicacoes'][0]
print('Keys:', list(exp.keys()))
print('tipo_predicao:', exp.get('tipo_predicao'))

rej = [e for e in d['explicacoes'] if 'REJEIT' in e.get('tipo_predicao','')]
print(f'Rejeitadas: {len(rej)}')
if rej:
    print('Exemplo rejeitada:', rej[0])
