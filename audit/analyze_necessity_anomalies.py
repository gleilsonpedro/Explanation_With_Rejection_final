"""
Script para analisar a minimalidade das explicações nos JSONs de validação.
Investiga por que banknote/vertebral_column têm minimalidade invertida.
"""
import json
from pathlib import Path
from typing import Dict, List
from collections import Counter

def load_validation_json(dataset_name: str) -> dict:
    """Carrega JSON de validação PEAB."""
    path = Path(f'json/validation/peab_validation_{dataset_name}.json')
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_necessity_anomalies(dataset_name: str):
    """Analisa anomalias na minimalidade por tipo de predição."""
    
    data = load_validation_json(dataset_name)
    if not data:
        print(f"❌ Arquivo não encontrado para {dataset_name}")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISE DE MINIMALIDADE: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Métricas globais
    global_metrics = data.get('global_metrics', {})
    print(f"\n📈 Métricas Globais:")
    print(f"  Fidelidade Geral: {global_metrics.get('fidelity_overall', 0):.2f}%")
    print(f"  Minimalidade Geral: {global_metrics.get('necessity_overall', 'N/A')}")
    
    # Analisar por instância
    instances = data.get('per_instance_results', [])
    
    if not instances:
        print("⚠️  Sem dados de instâncias")
        return
    
    # Agrupar por tipo
    by_type = {'positive': [], 'negative': [], 'rejected': []}
    
    for inst in instances:
        tipo = inst.get('prediction_type', 'unknown')
        if tipo in by_type:
            by_type[tipo].append(inst)
    
    # Analisar cada tipo
    print(f"\n🔍 ANÁLISE DETALHADA POR TIPO:")
    
    for tipo, instancias in by_type.items():
        if not instancias:
            continue
        
        print(f"\n  {tipo.upper()}:")
        print(f"    Total de instâncias: {len(instancias)}")
        
        # Análise de necessity
        necessity_scores = []
        redundant_counts = []
        explanation_sizes = []
        
        for inst in instancias:
            necessity_data = inst.get('necessity_test', {})
            necessity_score = necessity_data.get('necessity_score', 100.0)
            redundant = necessity_data.get('redundant_features', [])
            size = inst.get('explanation_size', 0)
            
            necessity_scores.append(necessity_score)
            redundant_counts.append(len(redundant))
            explanation_sizes.append(size)
        
        avg_necessity = sum(necessity_scores) / len(necessity_scores) if necessity_scores else 0
        avg_redundant = sum(redundant_counts) / len(redundant_counts) if redundant_counts else 0
        avg_size = sum(explanation_sizes) / len(explanation_sizes) if explanation_sizes else 0
        
        print(f"    Minimalidade média: {avg_necessity:.2f}%")
        print(f"    Features redundantes (média): {avg_redundant:.2f}")
        print(f"    Tamanho explicação (média): {avg_size:.2f}")
        
        # Distribuição de necessity scores
        necessity_counter = Counter([round(ns, 0) for ns in necessity_scores])
        print(f"\n    Distribuição de Minimalidade:")
        for score in sorted(necessity_counter.keys(), reverse=True):
            count = necessity_counter[score]
            pct = (count / len(necessity_scores)) * 100
            print(f"      {int(score):3d}%: {count:3d} instâncias ({pct:5.1f}%)")
        
        # Casos extremos
        min_necessity = min(necessity_scores) if necessity_scores else 0
        max_necessity = max(necessity_scores) if necessity_scores else 0
        
        print(f"\n    Min minimalidade: {min_necessity:.2f}%")
        print(f"    Max minimalidade: {max_necessity:.2f}%")
        
        # Instâncias com minimalidade muito baixa (< 50%)
        low_minimal = [i for i, ns in enumerate(necessity_scores) if ns < 50]
        if low_minimal:
            print(f"\n    ⚠️  {len(low_minimal)} instâncias com minimalidade < 50%")
            # Mostrar algumas
            for idx in low_minimal[:3]:
                inst = instancias[idx]
                print(f"        Instância {inst.get('instance_index')}: "
                      f"Minimalidade={necessity_scores[idx]:.1f}%, "
                      f"Tamanho={explanation_sizes[idx]}, "
                      f"Redundantes={redundant_counts[idx]}")
        
        # Instâncias com minimalidade muito alta (100%)
        perfect_minimal = [i for i, ns in enumerate(necessity_scores) if ns >= 99.9]
        if perfect_minimal:
            print(f"\n    ✅ {len(perfect_minimal)} instâncias com minimalidade 100%")

if __name__ == "__main__":
    # Datasets com anomalias
    datasets = [
        'pima_indians_diabetes',  # Normal
        'vertebral_column',        # Anomalia (positivas 0%, negativas 99%)
        'breast_cancer'            # Para comparação
    ]
    
    for dataset in datasets:
        try:
            analyze_necessity_anomalies(dataset)
        except Exception as e:
            print(f"\n❌ Erro ao analisar {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ Análise concluída!")
    print(f"{'='*80}\n")
