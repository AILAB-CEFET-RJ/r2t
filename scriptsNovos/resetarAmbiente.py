"""
resetarAmbiente.py

Reseta o ambiente do pipeline GLARE em dois movimentos:

  1) DADOS-FONTE -> src/
     Localiza os dois arquivos de dados PRINCIPAIS -- o corpus original,
     cru, com o texto completo (não são cache, são a fonte de tudo):
       - special_appeal.csv       (recursos especiais, notClean, completo)
       - temas_repetitivos.csv    (temas repetitivos, notClean, completo)
     Procura primeiro na raiz do projeto (estado antes de rodar
     montarAmbiente.py) e, se não achar, dentro de
     data/appeals/notClean/texto/ e data/temas/notClean/texto/ (onde
     criarDiretorios.py os deixa depois de rodar). Move o que encontrar
     para src/.

  2) TUDO O MAIS -> backup-<timestamp>/
     Move as pastas geradas pelo pipeline (splits, embeddings, resumos,
     modelos treinados, resultados, tabelas LaTeX) para uma pasta nova
     backup-<timestamp>/, preservando a estrutura original dentro dela.

Nada é apagado: o que não é dado-fonte vira backup, não lixo. Depois de
rodar, basta chamar montarAmbiente.py de novo para reconstruir tudo a
partir dos dois arquivos em src/.

Uso:
    python resetarAmbiente.py
    python resetarAmbiente.py --dry_run
    python resetarAmbiente.py --pastas_cache split data modelos resultados tabelas_latex outra_pasta
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PASTAS_CACHE_PADRAO = ["split", "data", "modelos", "resultados", "tabelas_latex"]


def localizar_arquivo_fonte(root: Path, nome: str, alternativas: List[Path]) -> Optional[Path]:
    """Procura 'nome' na raiz primeiro, depois nos caminhos alternativos (relativos à raiz)."""
    candidatos = [root / nome] + [root / alt for alt in alternativas]
    for c in candidatos:
        if c.is_file():
            return c
    return None


def mover(origem: Path, destino: Path, dry_run: bool, descricao: str) -> None:
    if dry_run:
        print(f"  [DRY RUN] moveria {origem} -> {destino}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(origem), str(destino))
    print(f"  [OK] {descricao}: {origem} -> {destino}")


def gerar_pasta_backup(root: Path) -> Path:
    """Nome único backup-<timestamp>[_N] -- nunca sobrescreve um backup anterior."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidato = root / f"backup-{ts}"
    sufixo = 1
    while candidato.exists():
        candidato = root / f"backup-{ts}_{sufixo}"
        sufixo += 1
    return candidato


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Reseta o ambiente: extrai os 2 arquivos de dados-fonte para "
            "src/ e move o resto (cache/artefatos gerados pelo pipeline) "
            "para backup-<timestamp>/."
        )
    )
    p.add_argument("--root", type=Path, default=Path("."),
                    help="Raiz do projeto (onde ficam split/, data/, modelos/, etc.)")
    p.add_argument("--src_dir", type=Path, default=Path("."),
                    help="Pasta de destino dos dois arquivos-fonte.")
    p.add_argument("--appeals_filename", default="special_appeal.csv",
                    help="Nome do arquivo-fonte de recursos especiais.")
    p.add_argument("--temas_filename", default="temas_repetitivos.csv",
                    help="Nome do arquivo-fonte de temas repetitivos.")
    p.add_argument("--pastas_cache", nargs="+", default=PASTAS_CACHE_PADRAO,
                    help="Pastas na raiz tratadas como cache/artefato gerado, movidas para o backup.")
    p.add_argument("--dry_run", action="store_true",
                    help="Só mostra o plano, não move nada de fato.")
    args = p.parse_args(argv)

    root = args.root.resolve()
    src_dir = args.src_dir if args.src_dir.is_absolute() else root / args.src_dir

    print(f"{'=' * 60}\nRESET DO AMBIENTE -- {root}\n{'=' * 60}")

    # -------------------------------------------------
    # 1) Dados-fonte -> src/
    # -------------------------------------------------
    fontes = {
        args.appeals_filename: [
            Path("data") / "appeals" / "notClean" / "texto" / args.appeals_filename,
        ],
        args.temas_filename: [
            Path("data") / "temas" / "notClean" / "texto" / args.temas_filename,
        ],
    }

    print("\n>> Localizando dados-fonte (não são cache)...")
    for nome, alternativas in fontes.items():
        origem = localizar_arquivo_fonte(root, nome, alternativas)
        destino = src_dir / nome

        if origem is None:
            print(f"  [AVISO] {nome} não encontrado (nem na raiz, nem em data/) -- pulando.")
            continue

        if destino.exists():
            print(
                f"  [AVISO] {destino} já existe -- NÃO sobrescrevendo. "
                f"{origem} permanece onde está; mova manualmente se for o caso."
            )
            continue

        mover(origem, destino, args.dry_run, "dado-fonte")

    # -------------------------------------------------
    # 2) Tudo o mais -> backup-<timestamp>/
    # -------------------------------------------------
    pasta_backup = gerar_pasta_backup(root)
    print(f"\n>> Pasta de backup: {pasta_backup}")

    algo_movido = False
    for nome_pasta in args.pastas_cache:
        origem = root / nome_pasta
        if not origem.exists():
            continue
        destino = pasta_backup / nome_pasta
        mover(origem, destino, args.dry_run, "cache/artefato gerado")
        algo_movido = True

    if not algo_movido:
        print("  Nenhuma pasta de cache encontrada -- nada a fazer aqui.")
    elif args.dry_run:
        print("\n--dry_run ativo: nada foi movido de fato.")
    else:
        print(f"\nBackup concluído em: {pasta_backup}")

    print(f"\n{'=' * 60}\nReset concluído. Rode montarAmbiente.py para reconstruir "
          f"tudo a partir de src/.\n{'=' * 60}")


if __name__ == "__main__":
    main()