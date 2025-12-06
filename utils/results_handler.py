import json
import os
from typing import Dict, Any
import numpy as np
from datetime import datetime

RESULTS_FILE = "json/comparative_results.json"

# Arquivos individuais por método
PEAB_RESULTS_FILE = "json/peab_results.json"
ANCHOR_RESULTS_FILE = "json/anchor_results.json"
MINEXP_RESULTS_FILE = "json/minexp_results.json"

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
    Atualiza os resultados de um método específico para um dataset.
    Agora salva em arquivos separados (peab_results.json, anchor_results.json, minexp_results.json)
    """
    # Mapear método para arquivo específico
    method_files = {
        'peab': PEAB_RESULTS_FILE,
        'anchor': ANCHOR_RESULTS_FILE,
        'minexp': MINEXP_RESULTS_FILE,
        'mateus': MINEXP_RESULTS_FILE  # Alias para compatibilidade
    }
    
    results_file = method_files.get(method.lower(), RESULTS_FILE)
    
    # Carregar resultados existentes do arquivo específico
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                method_results = json.load(f)
        except Exception:
            method_results = {}
    else:
        method_results = {}
    
    # Atualizar dataset específico
    method_results[dataset] = results
    
    # Salvar no arquivo específico do método
    serializable = _to_builtin(method_results)
    base_dir = os.path.dirname(results_file) or '.'
    os.makedirs(base_dir, exist_ok=True)
    tmp_path = os.path.join(base_dir, f'._{method}_results.tmp')
    
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, results_file)
    
    print(f"✅ Resultados salvos: {results_file} - {dataset}")