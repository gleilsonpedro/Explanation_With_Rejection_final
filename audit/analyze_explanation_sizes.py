"""
Análise das distribuições de tamanho de explicações por dataset e tipo de predição.
Compara PEAB e PuLP para entender anomalias.
"""
import json
from pathlib import Path
from typing import Dict, List
from collections import Counter

def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset(dataset_name: str):
    """Analisa as distribuições de tamanhos de explicações para um dataset."""
    
    peab_path = Path(f'json/peab/{dataset_name}.json')
    pulp_path = Path(f'json/pulp/{dataset_name}.json')
    
    if not peab_path.exists() or not pulp_path.exists():
        print(f"❌ Arquivos não encontrados para {dataset_name}")
        return
    
    peab_data = load_json(peab_path)
    pulp_data = load_json(pulp_path)
    
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISE: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Estatísticas agregadas
    print("\n📈 ESTATÍSTICAS AGREGADAS (PEAB):")
    for tipo in ['positive', 'negative', 'rejected']:
        stats = peab_data['explanation_stats'].get(tipo, {})
        if stats.get('count', 0) > 0:
            print(f"  {tipo.upper():10s}: "
                  f"Qtd={stats['count']:3d} | "
                  f"Média={stats['mean_length']:.2f} | "
                  f"Min={stats['min_length']:2d} | "
                  f"Max={stats['max_length']:2d}")
    
    print("\n📈 ESTATÍSTICAS AGREGADAS (PuLP):")
    for tipo_key, tipo_label in [('positiva', 'POSITIVE'), ('negativa', 'NEGATIVE'), ('rejeitada', 'REJECTED')]:
        stats = pulp_data['estatisticas_por_tipo'].get(tipo_key, {})
        if stats.get('instancias', 0) > 0:
            # Calcular min/max do PuLP manualmente
            tamanhos = []
            for expl in pulp_data['explicacoes']:
                if expl['tipo_predicao'] == tipo_key.upper():
                    tamanhos.append(expl['tamanho'])
            
            if tamanhos:
                print(f"  {tipo_label:10s}: "
                      f"Qtd={stats['instancias']:3d} | "
                      f"Média={stats['tamanho_medio']:.2f} | "
                      f"Min={min(tamanhos):2d} | "
                      f"Max={max(tamanhos):2d}")
    
    # Análise detalhada das negativas no PuLP (problema principal)
    print("\n🔍 DISTRIBUIÇÃO DETALHADA DAS NEGATIVAS (PuLP):")
    negativas_pulp = [e for e in pulp_data['explicacoes'] if e['tipo_predicao'] == 'NEGATIVA']
    
    if negativas_pulp:
        tamanhos_counter = Counter([e['tamanho'] for e in negativas_pulp])
        total_neg = len(negativas_pulp)
        
        print(f"  Total de negativas: {total_neg}")
        for tam in sorted(tamanhos_counter.keys()):
            count = tamanhos_counter[tam]
            pct = (count / total_neg) * 100
            bar = '█' * int(pct / 2)
            print(f"    {tam} features: {count:3d} instâncias ({pct:5.1f}%) {bar}")
        
        # Verificar se há muitas negativas com 5 ou 6 features
        grandes = sum(1 for e in negativas_pulp if e['tamanho'] >= 5)
        pct_grandes = (grandes / total_neg) * 100
        print(f"\n  ⚠️  Negativas com ≥5 features: {grandes}/{total_neg} ({pct_grandes:.1f}%)")
    
    # Análise das rejeitadas
    print("\n🔍 DISTRIBUIÇÃO DETALHADA DAS REJEITADAS (PuLP):")
    rejeitadas_pulp = [e for e in pulp_data['explicacoes'] if e['tipo_predicao'] == 'REJEITADA']
    
    if rejeitadas_pulp:
        tamanhos_counter = Counter([e['tamanho'] for e in rejeitadas_pulp])
        total_rej = len(rejeitadas_pulp)
        
        print(f"  Total de rejeitadas: {total_rej}")
        for tam in sorted(tamanhos_counter.keys()):
            count = tamanhos_counter[tam]
            pct = (count / total_rej) * 100
            bar = '█' * int(pct / 2)
            print(f"    {tam} features: {count:3d} instâncias ({pct:5.1f}%) {bar}")
    
    # Comparação direta Negativas vs Rejeitadas
    if negativas_pulp and rejeitadas_pulp:
        media_neg = sum(e['tamanho'] for e in negativas_pulp) / len(negativas_pulp)
        media_rej = sum(e['tamanho'] for e in rejeitadas_pulp) / len(rejeitadas_pulp)
        
        print(f"\n📊 COMPARAÇÃO NEGATIVAS vs REJEITADAS:")
        print(f"  Média Negativas: {media_neg:.2f} features")
        print(f"  Média Rejeitadas: {media_rej:.2f} features")
        print(f"  Diferença: {media_rej - media_neg:+.2f} features")
        
        if media_neg > media_rej * 0.9:
            print(f"  ⚠️  ANOMALIA: Negativas são quase do mesmo tamanho que rejeitadas!")
        elif media_neg < media_rej * 0.7:
            print(f"  ✅ NORMAL: Negativas são menores que rejeitadas (como esperado)")
        else:
            print(f"  ⚠️  BORDERLINE: Negativas estão próximas das rejeitadas")

if __name__ == "__main__":
    # Datasets para análise
    datasets = [
        'pima_indians_diabetes',
        'vertebral_column',
        'breast_cancer',
        'wine',
        'sonar'
    ]
    
    for dataset in datasets:
        try:
            analyze_dataset(dataset)
        except Exception as e:
            print(f"\n❌ Erro ao analisar {dataset}: {e}")
    
    print(f"\n{'='*80}")
    print("✅ Análise concluída!")
    print(f"{'='*80}\n")
