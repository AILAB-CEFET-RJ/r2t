"""
prepararTreinoVariantes.py

Gera variantes pré-processadas dos splits de TREINO (split/treino/*.csv),
para permitir treinar modelos '_treinado' com o MESMO tipo de
pré-processamento que será usado no teste (limpeza / resumo) -- evitando o
descasamento treino/teste (modelo fine-tunado em texto cru, avaliado sobre
texto limpo ou resumido, produzindo queda de recall).

Os conjuntos de TESTE (data/appeals/.../texto/*.csv) NUNCA são alterados
por este script -- apenas CSVs de treino, usados só pelo fine-tuning.

Para cada dataset de treino (kind + seed, ou zeroshot), até 6 variantes
existem conceitualmente; este script gera as 5 que não são o cru original
(que já existe em split/treino/):

    raw                          -- split/treino/special_appeal_<kind>_<seed>.csv (já existe)
    clean                        -- texto limpo (limparTexto.py)
    resumo_lexrank                -- resumo LexRank do texto cru
    resumo_guided_lexrank         -- resumo Guided LexRank do texto cru (usa temas notClean)
    clean_resumo_lexrank          -- resumo LexRank do texto já limpo
    clean_resumo_guided_lexrank   -- resumo Guided LexRank do texto já limpo (usa temas clean)

Saída em: split/treino_variantes/<kind>_<seed>/
    clean.csv
    raw/resumo_<estrategia>_<tamanho>.csv
    clean/resumo_<estrategia>_<tamanho>.csv

Cada estratégia tem o SEU PRÓPRIO tamanho de resumo (ex: guided_lexrank
com 15 sentenças, lexrank com 60) -- não há uma única lista de estratégias
cruzada com um tamanho único. Ver --resumos.

'clean' é SEMPRE gerado (mesmo sem nenhuma estratégia de resumo pedida em
--resumos), incondicionalmente -- é insumo direto dos passos de resumo
"clean_resumo_*" e é barato de gerar (só regex + stopwords, sem GPU).

Também garante que data/temas/clean/texto/temas_repetitivos.csv existe
(gerado via limparTexto.py se necessário) -- as variantes 'clean' precisam
parear appeals limpos com temas igualmente limpos no fine-tuning (ver
temas_para_treino em experimento.py).

Idempotente: reexecuções pulam (com log [CACHE]) qualquer arquivo que já
existe -- seguro rodar de novo ao adicionar seeds/estratégias novas.

Uso:
    python prepararTreinoVariantes.py --seeds 42
    python prepararTreinoVariantes.py --kinds baseline stratified --seeds 42
    python prepararTreinoVariantes.py --seeds 42 --resumos guided_lexrank:15 lexrank:60
"""

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable

KINDS_COM_SEED = ["baseline", "stratified", "minority"]

# Estratégias de resumo que não precisam do CSV de temas (ver
# strategies_needing_temas em gerarResumos.py).
ESTRATEGIAS_SEM_TEMAS = {"lexrank"}


def parse_resumos(pares: list) -> dict:
    """Converte ['guided_lexrank:15', 'lexrank:60'] em {'guided_lexrank': 15, 'lexrank': 60}."""
    resumos = {}
    for par in pares:
        if ":" not in par:
            raise ValueError(
                f"--resumos espera pares 'estrategia:tamanho' (ex: 'lexrank:60'); recebido: '{par}'"
            )
        estrategia, tamanho = par.split(":", 1)
        try:
            resumos[estrategia] = int(tamanho)
        except ValueError:
            raise ValueError(f"Tamanho inválido em '{par}': '{tamanho}' não é um inteiro.")
    return resumos


def run(cmd: list) -> None:
    cmd = [str(c) for c in cmd]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def pular_ou_gerar(output_path: Path, descricao: str, gerar_func) -> None:
    if output_path.exists():
        print(f"  [CACHE] {descricao} já existe: {output_path}")
        return
    print(f"  [GERANDO] {descricao} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gerar_func()


def garantir_temas_clean(temas_notclean: Path, temas_clean: Path, scripts_dir: Path) -> None:
    def gerar():
        run([
            PYTHON, str(scripts_dir / "limparTexto.py"),
            "--input", str(temas_notclean),
            "--output", str(temas_clean),
            "--text_column", "theme_text",
        ])
    pular_ou_gerar(temas_clean, "temas limpos (compartilhado entre datasets)", gerar)


def preparar_variantes_de(
    appeals_treino_raw: Path,
    temas_notclean: Path,
    temas_clean: Path,
    output_root: Path,
    resumos_treino: dict,
    scripts_dir: Path,
) -> None:
    nome = appeals_treino_raw.stem
    if nome.startswith("special_appeal_"):
        nome = nome[len("special_appeal_"):]

    base_dir = output_root / nome
    clean_csv = base_dir / "clean.csv"

    print(f"\n{'=' * 60}\n{nome}\n{'=' * 60}")

    # ---- 1) versão limpa do texto de treino ----
    def gerar_clean():
        run([
            PYTHON, str(scripts_dir / "limparTexto.py"),
            "--input", str(appeals_treino_raw),
            "--output", str(clean_csv),
            "--text_column", "special_appeal_text",
        ])
    pular_ou_gerar(clean_csv, "treino limpo", gerar_clean)

    # ---- 2) resumos a partir do texto CRU ----
    raw_out_dir = base_dir / "raw"
    for estrategia, tamanho in resumos_treino.items():
        output = raw_out_dir / f"resumo_{estrategia}_{tamanho}.csv"

        def gerar(estrategia=estrategia, tamanho=tamanho):
            cmd = [
                PYTHON, str(scripts_dir / "gerarResumos.py"),
                "--input", str(appeals_treino_raw),
                "--output_dir", str(raw_out_dir),
                "--size", str(tamanho),
                "--strategy", estrategia,
            ]
            if estrategia not in ESTRATEGIAS_SEM_TEMAS:
                cmd += ["--temas", str(temas_notclean)]
            run(cmd)
        pular_ou_gerar(output, f"resumo {estrategia} tamanho {tamanho} (texto cru)", gerar)

    # ---- 3) resumos a partir do texto LIMPO ----
    clean_out_dir = base_dir / "clean"
    for estrategia, tamanho in resumos_treino.items():
        output = clean_out_dir / f"resumo_{estrategia}_{tamanho}.csv"

        def gerar(estrategia=estrategia, tamanho=tamanho):
            cmd = [
                PYTHON, str(scripts_dir / "gerarResumos.py"),
                "--input", str(clean_csv),
                "--output_dir", str(clean_out_dir),
                "--size", str(tamanho),
                "--strategy", estrategia,
            ]
            if estrategia not in ESTRATEGIAS_SEM_TEMAS:
                cmd += ["--temas", str(temas_clean)]
            run(cmd)
        pular_ou_gerar(output, f"resumo {estrategia} tamanho {tamanho} (texto limpo)", gerar)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Gera variantes pré-processadas (clean/resumo) dos splits de treino."
    )
    p.add_argument("--kinds", nargs="+", default=KINDS_COM_SEED + ["zeroshot"],
                    help="Kinds a processar (baseline, stratified, minority, zeroshot).")
    p.add_argument("--seeds", nargs="+", type=int, required=True,
                    help="Seeds a processar (ignorado para zeroshot, que não tem seed). Obrigatório.")
    p.add_argument("--resumos", nargs="+", default=[],
                    help=(
                        "Pares 'estrategia:tamanho' de resumo a preparar para o treino "
                        "(ex: --resumos guided_lexrank:15 lexrank:60). Cada estratégia "
                        "usa o tamanho especificado; se vazio, nenhuma variante de "
                        "resumo é gerada (só 'raw' e 'clean')."
                    ))
    p.add_argument("--split_treino_dir", type=Path, default=Path("split/treino"))
    p.add_argument("--output_root", type=Path, default=Path("split/treino_variantes"))
    p.add_argument("--base_dir", type=Path, default=Path("data"))
    p.add_argument("--scripts_dir", type=Path, default=Path("."))
    args = p.parse_args(argv)

    resumos_treino = parse_resumos(args.resumos)

    temas_notclean = args.base_dir / "temas" / "notClean" / "texto" / "temas_repetitivos.csv"
    temas_clean = args.base_dir / "temas" / "clean" / "texto" / "temas_repetitivos.csv"

    # Necessário incondicionalmente: a variante 'clean' (mesmo sem resumo)
    # já precisa parear com temas limpos no fine-tuning.
    garantir_temas_clean(temas_notclean, temas_clean, args.scripts_dir)

    arquivos = []
    for kind in args.kinds:
        if kind == "zeroshot":
            arquivos.append(args.split_treino_dir / "special_appeal_zeroshot.csv")
        elif kind in KINDS_COM_SEED:
            for seed in args.seeds:
                arquivos.append(args.split_treino_dir / f"special_appeal_{kind}_{seed}.csv")
        else:
            raise ValueError(f"kind desconhecido: {kind}")

    for appeals_treino_raw in arquivos:
        if not appeals_treino_raw.exists():
            print(f"[AVISO] {appeals_treino_raw} não encontrado -- pulando "
                  f"(rode montarAmbiente.py antes).")
            continue
        preparar_variantes_de(
            appeals_treino_raw, temas_notclean, temas_clean,
            args.output_root, resumos_treino,
            args.scripts_dir,
        )

    print(f"\n{'=' * 60}\nVariantes de treino preparadas em: {args.output_root}\n{'=' * 60}")


if __name__ == "__main__":
    main()