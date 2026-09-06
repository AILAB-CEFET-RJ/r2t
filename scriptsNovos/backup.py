"""
backupModelosResultados.py

Move os diretórios de modelos treinados e de resultados atuais para uma
pasta de backup identificada por timestamp, e garante que essa pasta de
backup fique fora do controle de versão (.gitignore).

Por que mover (e não copiar)
-----------------------------
O objetivo é "resetar" o estado de treino/resultados sem perder o
histórico -- por exemplo, antes de rodar novamente `rodarExperimentos.py`
com a correção de max_seq_length em treinarModelos.py, para comparar os
resultados antigos com os novos sem misturar os dois.

O que é movido (se existir)
-----------------------------
  - modelos/          -> modelos treinados (incl. by_dataset/ e symlinks)
  - resultados/        -> experimentos*.csv e tempos_arquivos.csv

Uso:
    python backupModelosResultados.py
    python backupModelosResultados.py --destino_base backups
    python backupModelosResultados.py --dry_run
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

# =====================================================
# CONFIGURAÇÃO
# =====================================================

# Diretórios de origem que serão movidos, se existirem.
ORIGENS_PADRAO = ["modelos", "resultados"]

DESTINO_BASE_PADRAO = "."           # onde a pasta backup-<timestamp> é criada
GITIGNORE_PADRAO = ".gitignore"
PADRAO_GITIGNORE = "backup-*/"      # cobre qualquer backup-<timestamp> futuro


# =====================================================
# FUNÇÕES
# =====================================================

def gerar_nome_backup(destino_base: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return destino_base / f"backup-{timestamp}"


def mover_diretorios(origens: list, destino: Path, dry_run: bool) -> list:
    """Move cada diretório de `origens` (se existir) para dentro de `destino`.
    Retorna a lista dos que foram efetivamente movidos."""
    movidos = []

    for nome in origens:
        origem = Path(nome)
        if not origem.exists():
            print(f"[PULAR] '{origem}' não existe, nada a mover.")
            continue

        alvo = destino / origem.name
        print(f"[MOVER] {origem}  ->  {alvo}")
        if not dry_run:
            destino.mkdir(parents=True, exist_ok=True)
            shutil.move(str(origem), str(alvo))
        movidos.append(nome)

    return movidos


def garantir_gitignore(gitignore_path: Path, padrao: str, dry_run: bool) -> None:
    """Adiciona `padrao` ao .gitignore se ainda não estiver presente."""
    linhas_existentes = []
    if gitignore_path.exists():
        linhas_existentes = gitignore_path.read_text(encoding="utf-8").splitlines()

    if padrao in linhas_existentes:
        print(f"[OK] '{padrao}' já está em {gitignore_path}.")
        return

    print(f"[GITIGNORE] Adicionando '{padrao}' em {gitignore_path}.")
    if dry_run:
        return

    with open(gitignore_path, "a", encoding="utf-8") as f:
        # garante quebra de linha antes, caso o arquivo não termine com \n
        if linhas_existentes and linhas_existentes[-1] != "":
            f.write("\n")
        f.write(f"# pastas de backup geradas por backupModelosResultados.py\n")
        f.write(f"{padrao}\n")


# =====================================================
# CLI / MAIN
# =====================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Move modelos/ e resultados/ para uma pasta backup-<timestamp> "
                     "e garante que ela esteja no .gitignore."
    )
    p.add_argument(
        "--origens", nargs="+", default=ORIGENS_PADRAO,
        help=f"Diretórios a mover, se existirem. Padrão: {ORIGENS_PADRAO}",
    )
    p.add_argument(
        "--destino_base", default=DESTINO_BASE_PADRAO,
        help="Onde criar a pasta backup-<timestamp>. Padrão: diretório atual.",
    )
    p.add_argument(
        "--gitignore", default=GITIGNORE_PADRAO,
        help=f"Caminho do .gitignore a atualizar. Padrão: {GITIGNORE_PADRAO}",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Só mostra o que seria feito, sem mover nada nem editar o .gitignore.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    destino_base = Path(args.destino_base)
    destino = gerar_nome_backup(destino_base)
    gitignore_path = Path(args.gitignore)

    print(f"Pasta de backup: {destino}\n")

    movidos = mover_diretorios(args.origens, destino, args.dry_run)

    if not movidos:
        print("\nNada foi encontrado para mover -- nenhuma pasta de backup foi criada.")
    else:
        print(f"\n{len(movidos)} diretório(s) movido(s) para {destino}"
              f"{' (SIMULADO)' if args.dry_run else ''}.")

    garantir_gitignore(gitignore_path, PADRAO_GITIGNORE, args.dry_run)

    if args.dry_run:
        print("\n--dry_run ativo: nenhuma alteração real foi feita.")


if __name__ == "__main__":
    main()