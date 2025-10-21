import json
import os
from typing import Dict, Any

RESULTS_FILE = "json/comparative_results.json"

def load_results() -> Dict[str, Any]:
    """Carrega os resultados existentes ou retorna estrutura vazia"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "peab": {},
        "anchor": {},
        "mateus": {}
    }

def save_results(data: Dict[str, Any]) -> None:
    """Salva os resultados no arquivo JSON"""
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_method_results(method: str, dataset: str, results: Dict[str, Any]) -> None:
    """
    Atualiza os resultados de um método específico para um dataset
    """
    all_results = load_results()
    
    # Garante que a chave do método exista
    if method not in all_results:
        all_results[method] = {}
        
    all_results[method][dataset] = results
    save_results(all_results)
    print(f"Resultados salvos com sucesso no JSON para: {method} - {dataset}")