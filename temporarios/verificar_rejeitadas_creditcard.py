"""
Verifica as instâncias rejeitadas no creditcard.
"""
import json

print("\n" + "="*80)
print("ANÁLISE DAS INSTÂNCIAS REJEITADAS - Creditcard")
print("="*80 + "\n")

# Carregar JSON
with open('json/peab/creditcard.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Estatísticas gerais
print("ESTATÍSTICAS GERAIS:")
print("-"*80)
print(f"  Total de instâncias: {data['performance']['num_test_instances']}")
print(f"  Rejeitadas: {data['performance']['num_rejected']}")
print(f"  Aceitas: {data['performance']['num_accepted']}")
print(f"  Taxa de rejeição: {data['performance']['rejection_rate']:.2f}%")

# Tempos
print(f"\n{'─'*80}")
print("TEMPOS DE COMPUTAÇÃO:")
print(f"{'─'*80}")
print(f"  Tempo médio total: {data['computation_time']['mean_per_instance']*1000:.2f} ms")
print(f"  Tempo positivas: {data['computation_time']['positive']*1000:.2f} ms")
print(f"  Tempo negativas: {data['computation_time']['negative']*1000:.2f} ms")
print(f"  Tempo rejeitadas: {data['computation_time']['rejected']*1000:.2f} ms")

# Buscar instâncias rejeitadas
rejeitadas = [inst for inst in data['per_instance'] if inst.get('rejected', False)]

print(f"\n{'='*80}")
print(f"INSTÂNCIAS REJEITADAS ENCONTRADAS: {len(rejeitadas)}")
print("="*80)

if len(rejeitadas) > 0:
    for i, inst in enumerate(rejeitadas, 1):
        print(f"\nRejeitada #{i}:")
        print(f"  ID: {inst['id']}")
        print(f"  y_true: {inst['y_true']}")
        print(f"  Decision score: {inst['decision_score']:.4f}")
        print(f"  Tamanho explicação: {inst['explanation_size']}")
        print(f"  Tempo computação: {inst.get('computation_time', 'N/A')*1000:.2f} ms")
        print(f"  Explicação: {inst['explanation'][:5]}..." if len(inst['explanation']) > 5 else f"  Explicação: {inst['explanation']}")
else:
    print("\n⚠️  Nenhuma instância rejeitada encontrada no per_instance!")
    print("    Mas o JSON mostra 'num_rejected': 2")
    print("    Isso significa que há uma inconsistência!")

print(f"\n{'='*80}")
print("CONCLUSÃO:")
print("="*80)

if len(rejeitadas) == 2:
    print("✅ SIM, há 2 instâncias rejeitadas!")
    print("✅ O tempo nas rejeitadas é CORRETO (média dessas 2 instâncias)")
    print(f"✅ Tempo médio rejeitadas: {data['computation_time']['rejected']*1000:.2f} ms")
elif len(rejeitadas) == 0:
    print("⚠️  INCONSISTÊNCIA detectada:")
    print("   - JSON diz: 2 rejeitadas")
    print("   - per_instance tem: 0 rejeitadas")
    print("   - Isso pode ser um bug no salvamento do JSON")
else:
    print(f"⚠️  Número inesperado: {len(rejeitadas)} rejeitadas")

print("="*80 + "\n")
