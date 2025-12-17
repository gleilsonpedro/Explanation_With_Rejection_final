import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from peab import executar_experimento_para_dataset

if __name__ == "__main__":
    executar_experimento_para_dataset("pima_indians_diabetes")
