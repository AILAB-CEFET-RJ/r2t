"""
criarDiretorios.py

Cria a estrutura de pastas em data/ e move para lá os CSVs de teste
gerados por splitDados.py (split/teste/*.csv), além do CSV de temas.

--seeds é OBRIGATÓRIO e DEVE ser exatamente o mesmo conjunto passado a
splitDados.py na mesma rodada -- não há mais fallback hardcoded aqui.
Quando chamado por montarAmbiente.py, as seeds resolvidas do config de
ambiente são passadas explicitamente aos dois scripts, garantindo que
nunca fiquem dessincronizadas.

Seguro para rodar mais de uma vez: se a origem já foi movida numa
execução anterior mas o destino já existe, não é tratado como erro (ver
mover_um_arquivo).

Sempre execute splitDados.py ANTES deste script (com as mesmas seeds).
"""

import argparse
import os
import shutil

BASE_DIR = "data"

# Kinds de split que usam seed (um arquivo por seed). "zeroshot" fica de
# fora: é gerado uma única vez, sem seed.
SPLIT_KINDS_COM_SEED = ["baseline", "stratified", "minority"]

DIRETORIOS = [
    os.path.join(BASE_DIR, "appeals", "clean", "embeddings"),
    os.path.join(BASE_DIR, "appeals", "clean", "resumos"),
    os.path.join(BASE_DIR, "appeals", "clean", "texto"),
    os.path.join(BASE_DIR, "appeals", "notClean", "embeddings"),
    os.path.join(BASE_DIR, "appeals", "notClean", "resumos"),
    os.path.join(BASE_DIR, "appeals", "notClean", "texto"),
    os.path.join(BASE_DIR, "temas", "clean", "embeddings"),
    os.path.join(BASE_DIR, "temas", "clean", "texto"),
    os.path.join(BASE_DIR, "temas", "notClean", "embeddings"),
    os.path.join(BASE_DIR, "temas", "notClean", "texto"),
]


def criar_diretorios():
    for directory in DIRETORIOS:
        os.makedirs(directory, exist_ok=True)
    print("Estrutura de diretórios criada com sucesso!")


def mover_um_arquivo(origem: str, destino: str) -> None:
    """
    Três casos, para não confundir reexecução normal com erro de verdade:

      1) origem existe -> move (comportamento de sempre).
      2) origem NÃO existe, mas destino já existe -> reexecução normal
         (esse arquivo já foi movido numa rodada anterior); não é erro.
      3) nem origem nem destino existem -> aviso real: essa seed/arquivo
         nunca foi gerado por splitDados.py.
    """
    if os.path.exists(origem):
        shutil.move(origem, destino)
        print(f"{origem} movido para {destino}")
    elif os.path.exists(destino):
        print(f"[OK] {origem} já estava em {destino} (execução anterior).")
    else:
        print(f"[AVISO] {origem} não encontrado (e {destino} também não existe).")


def mover_arquivos(seeds):
    # Os splits (baseline, stratified, minority: um por seed; zeroshot: único)
    # são gerados por splitDados.py diretamente em split/treino/ e
    # split/teste/. Aqui só os arquivos de TESTE são movidos para data/ --
    # os de treino continuam em split/treino/ e são referenciados de lá
    # diretamente por rodarExperimentos.py (appeals_treino).
    arquivos_para_mover = [
        (
            "special_appeal.csv",
            os.path.join(BASE_DIR, "appeals", "notClean", "texto", "special_appeal.csv"),
        ),
        (
            "temas_repetitivos.csv",
            os.path.join(BASE_DIR, "temas", "notClean", "texto", "temas_repetitivos.csv"),
        ),
        (
            "split/teste/special_appeal_zeroshot.csv",
            os.path.join(BASE_DIR, "appeals", "notClean", "texto", "special_appeal_zeroshot.csv"),
        ),
    ]

    # baseline / stratified / minority: um arquivo de teste por seed
    for kind in SPLIT_KINDS_COM_SEED:
        for seed in seeds:
            nome = f"special_appeal_{kind}_{seed}.csv"
            arquivos_para_mover.append((
                os.path.join("split", "teste", nome),
                os.path.join(BASE_DIR, "appeals", "notClean", "texto", nome),
            ))

    for origem, destino in arquivos_para_mover:
        mover_um_arquivo(origem, destino)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Cria a estrutura data/ e move para lá os CSVs de teste + temas."
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, required=True,
        help=(
            "Seeds de baseline/stratified/minority a localizar em split/teste/. "
            "DEVE bater com as seeds usadas ao rodar splitDados.py. Obrigatório."
        ),
    )
    args = p.parse_args(argv)

    criar_diretorios()
    mover_arquivos(args.seeds)


if __name__ == "__main__":
    main()