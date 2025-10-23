import json
import os
from typing import Dict, Any
import numpy as np
from datetime import datetime

RESULTS_FILE = "json/comparative_results.json"

def load_results() -> Dict[str, Any]:
    """Carrega os resultados existentes ou retorna estrutura vazia com tolerância a arquivo corrompido."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            # Faz backup do arquivo corrompido e retorna estrutura limpa
            try:
                base_dir = os.path.dirname(RESULTS_FILE) or '.'
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                bak_path = os.path.join(base_dir, f"comparative_results_corrompido_{ts}.json.bak")
                os.replace(RESULTS_FILE, bak_path)
                print(f"Aviso: JSON corrompido detectado. Backup criado em: {bak_path}")
            except Exception:
                pass
    return {
        "peab": {},
        "anchor": {},
        "mateus": {}
    }

def _to_builtin(obj):
    """Converte objetos numpy/pandas para tipos nativos serializáveis em JSON."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(v) for v in obj]
    return obj

def save_results(data: Dict[str, Any]) -> None:
    """Salva os resultados no arquivo JSON, garantindo tipos serializáveis e gravação atômica."""
    serializable = _to_builtin(data)
    base_dir = os.path.dirname(RESULTS_FILE) or '.'
    os.makedirs(base_dir, exist_ok=True)
    tmp_path = os.path.join(base_dir, '._comparative_results.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, RESULTS_FILE)

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