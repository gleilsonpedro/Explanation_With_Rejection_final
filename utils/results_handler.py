import json
import os
from typing import Dict, Any
import numpy as np
from datetime import datetime

RESULTS_FILE = "json/comparative_results.json"

# ═══════════════════════════════════════════════════════════════════════════
# NOVA ESTRUTURA: JSONs separados por experimento e dataset
# ═══════════════════════════════════════════════════════════════════════════
# Estrutura:
#   json/
#   ├── peab/
#   │   ├── breast_cancer.json
#   │   ├── mnist_3_vs_8.json
#   │   └── wine.json
#   ├── pulp/
#   │   ├── breast_cancer.json
#   │   └── ...
#   ├── anchor/
#   └── minexp/
#
# Vantagens:
#   - Arquivos menores (1 dataset por arquivo)
#   - Fácil deletar resultados antigos
#   - Git ignora arquivos grandes individualmente
#   - Carrega apenas dados necessários
# ═══════════════════════════════════════════════════════════════════════════

JSON_BASE_DIR = "json"

# Arquivos antigos (manter para compatibilidade temporária)
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
    
    NOVA ESTRUTURA: Salva em json/{method}/{dataset}.json
    
    Exemplos:
        update_method_results('peab', 'breast_cancer', {...})
        → Salva em: json/peab/breast_cancer.json
        
        update_method_results('anchor', 'mnist_3_vs_8', {...})
        → Salva em: json/anchor/mnist_3_vs_8.json
    
    Args:
        method: Nome do método ('peab', 'pulp', 'anchor', 'minexp')
        dataset: Nome do dataset
        results: Dicionário com resultados do experimento
    """
    # Normalizar nome do método
    method_normalized = method.lower()
    if method_normalized == 'mateus':
        method_normalized = 'minexp'  # Alias para compatibilidade
    
    # Criar diretório do método se não existir
    method_dir = os.path.join(JSON_BASE_DIR, method_normalized)
    os.makedirs(method_dir, exist_ok=True)
    
    # Caminho do arquivo JSON para este dataset específico
    json_file = os.path.join(method_dir, f"{dataset}.json")
    
    # Converter para tipos serializáveis
    serializable = _to_builtin(results)
    
    # Salvar com gravação atômica (via arquivo temporário)
    tmp_file = os.path.join(method_dir, f".{dataset}.tmp")
    
    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, json_file)
        
        print(f"✅ Resultados salvos: {json_file}")
    
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
        # Limpar arquivo temporário se existir
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except:
                pass


def load_method_results(method: str, dataset: str) -> Dict[str, Any]:
    """
    Carrega os resultados de um método específico para um dataset.
    
    Args:
        method: Nome do método ('peab', 'pulp', 'anchor', 'minexp')
        dataset: Nome do dataset
    
    Returns:
        Dicionário com os resultados ou {} se não encontrado
    """
    method_normalized = method.lower()
    if method_normalized == 'mateus':
        method_normalized = 'minexp'
    
    json_file = os.path.join(JSON_BASE_DIR, method_normalized, f"{dataset}.json")
    
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Erro ao carregar {json_file}: {e}")
            return {}
    
    return {}


def list_available_datasets(method: str) -> list:
    """
    Lista todos os datasets disponíveis para um método.
    
    Args:
        method: Nome do método ('peab', 'pulp', 'anchor', 'minexp')
    
    Returns:
        Lista de nomes de datasets
    """
    method_normalized = method.lower()
    if method_normalized == 'mateus':
        method_normalized = 'minexp'
    
    method_dir = os.path.join(JSON_BASE_DIR, method_normalized)
    
    if not os.path.exists(method_dir):
        return []
    
    datasets = []
    for filename in os.listdir(method_dir):
        if filename.endswith('.json') and not filename.startswith('.'):
            dataset_name = filename[:-5]  # Remove .json
            datasets.append(dataset_name)
    
    return sorted(datasets)