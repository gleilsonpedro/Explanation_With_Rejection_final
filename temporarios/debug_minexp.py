import json

# Testar creditcard
data = json.load(open('json/minexp/creditcard.json'))
per_inst = data.get('per_instance', [])
print(f'Creditcard - Total per_instance: {len(per_inst)}')
if per_inst:
    print(f'First instance: {per_inst[0]}')
    
# Testar covertype
data2 = json.load(open('json/minexp/covertype.json'))
per_inst2 = data2.get('per_instance', [])
print(f'\nCovertype - Total per_instance: {len(per_inst2)}')
if per_inst2:
    print(f'First instance: {per_inst2[0]}')
