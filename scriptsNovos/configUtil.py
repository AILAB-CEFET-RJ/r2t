"""
configUtil.py     

Utilitário mínimo para carregar arquivos de configuração (config_ambiente.py,
config_experimentos.py) como módulos Python comuns, a partir de um caminho
arbitrário -- não precisam estar no sys.path nem ter nome de arquivo fixo.

Usado por montarAmbiente.py e rodarExperimentos.py.
"""

import importlib.util
from pathlib import Path


def carregar_modulo(caminho, nome_modulo: str = "config"):
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {caminho}")
    spec = importlib.util.spec_from_file_location(nome_modulo, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo