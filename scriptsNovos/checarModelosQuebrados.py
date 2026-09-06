"""
checarModelosQuebrados.py

Varre modelos/by_dataset/<dataset_id>/<modelo>_treinado/ procurando
diretorios que existem mas nao contem os arquivos de pesos esperados
(model.safetensors ou pytorch_model.bin) -- sinal de um save interrompido
(crash, OOM, kill) que o cache antigo aceitaria erroneamente.

Uso:
    python checarModelosQuebrados.py                 # so lista
    python checarModelosQuebrados.py --remover        # apaga as quebradas
"""

import argparse
import shutil
from pathlib import Path

PESOS_ESPERADOS = ("model.safetensors", "pytorch_model.bin")


def modelo_valido(diretorio: Path) -> bool:
    return any((diretorio / nome).exists() for nome in PESOS_ESPERADOS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelos_dir", default="modelos")
    p.add_argument("--remover", action="store_true")
    args = p.parse_args()

    base = Path(args.modelos_dir) / "by_dataset"
    if not base.exists():
        print(f"{base} nao existe.")
        return

    quebrados = []
    for dataset_dir in base.iterdir():
        if not dataset_dir.is_dir():
            continue
        for modelo_dir in dataset_dir.iterdir():
            if modelo_dir.is_dir() and modelo_dir.name.endswith("_treinado"):
                if not modelo_valido(modelo_dir):
                    quebrados.append(modelo_dir)

    if not quebrados:
        print("Nenhum modelo quebrado encontrado.")
        return

    print(f"{len(quebrados)} diretorio(s) quebrado(s):")
    for d in quebrados:
        print(" -", d)

    if args.remover:
        for d in quebrados:
            shutil.rmtree(d)
        print(f"\n{len(quebrados)} diretorio(s) removido(s). Rode o pipeline de novo para retreinar.")
    else:
        print("\nRode novamente com --remover para apagar e forcar retreino.")


if __name__ == "__main__":
    main()