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

Cache de resumo por CORPUS COMPLETO (evita resumir o mesmo appeal N vezes)
---------------------------------------------------------------------------
Um mesmo appeal aparece no treino de várias combinações (kind x seed) --
com 5 seeds, por exemplo, a MAIORIA dos appeals acaba caindo no treino de
quase todas elas. Resumir (LexRank/Guided LexRank) é a etapa cara do
pipeline (usa GPU); resumir por split, então, resumia o mesmo appeal
várias vezes à toa.

Para evitar isso: em vez de rodar gerarResumos.py sobre cada split de
treino, este script agora resume o CORPUS COMPLETO de appeals (cru e
limpo) UMA ÚNICA VEZ por (estrategia, tamanho) -- ver
garantir_resumo_completo -- e depois, para cada split, apenas SELECIONA
as linhas correspondentes desse resumo já pronto, usando o mapa
appeal_id -> posição original gerado por splitDados.py em
split/ids/treino/<nome>.csv (ver gerar_variante_de_split_via_cache).

Isso só funciona porque gerarResumos.py processa uma linha de entrada
para cada linha de saída, na MESMA ORDEM, sem filtrar nada -- ou seja, a
linha i do resumo do corpus completo é o resumo do appeal cujo
appeal_id (posição no special_appeal.csv original) é i. Não alterar esse
invariante em gerarResumos.py sem revisar este script também.

Uso:
    python prepararTreinoVariantes.py --seeds 42
    python prepararTreinoVariantes.py --kinds baseline stratified --seeds 42
    python prepararTreinoVariantes.py --seeds 42 --resumos guided_lexrank:15 lexrank:60
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ioUtil import salvar_csv_atomico

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


def garantir_appeals_completo_clean(
    appeals_notclean_completo: Path, appeals_clean_completo: Path, scripts_dir: Path
) -> None:
    """
    Análogo a garantir_temas_clean, mas para o CORPUS COMPLETO de appeals
    (não um split) -- é o insumo de garantir_resumo_completo para a
    variante 'clean_resumo_*'. Só é chamado quando há alguma estratégia
    de resumo pedida (ver main).
    """
    def gerar():
        run([
            PYTHON, str(scripts_dir / "limparTexto.py"),
            "--input", str(appeals_notclean_completo),
            "--output", str(appeals_clean_completo),
            "--text_column", "special_appeal_text",
        ])
    pular_ou_gerar(appeals_clean_completo, "appeals completos limpos (compartilhado entre datasets)", gerar)


def caminho_resumo_completo(base_dir: Path, clean_flag: str, estrategia: str, tamanho: int) -> Path:
    """Caminho canônico do resumo do CORPUS COMPLETO para (clean_flag, estrategia, tamanho)."""
    return base_dir / "appeals" / clean_flag / "resumos" / "full" / f"resumo_{estrategia}_{tamanho}.csv"


def garantir_resumo_completo(
    appeals_notclean_completo: Path,
    appeals_clean_completo: Path,
    temas_notclean: Path,
    temas_clean: Path,
    resumos_treino: dict,
    base_dir: Path,
    scripts_dir: Path,
) -> None:
    """
    Gera, uma ÚNICA vez por (estrategia, tamanho), o resumo do CORPUS
    COMPLETO de appeals -- a partir do texto cru e a partir do texto
    limpo. Ver docstring do módulo para o porquê disso substituir resumir
    por split.

    Não precisa de nenhum mapa de appeal_id aqui -- é só rodar
    gerarResumos.py sobre o corpus inteiro, do mesmo jeito que já era
    feito por split, só que uma vez só.
    """
    for estrategia, tamanho in resumos_treino.items():
        # ---- a partir do texto CRU ----
        out_raw = caminho_resumo_completo(base_dir, "notClean", estrategia, tamanho)

        def gerar_raw(estrategia=estrategia, tamanho=tamanho, out_raw=out_raw):
            cmd = [
                PYTHON, str(scripts_dir / "gerarResumos.py"),
                "--input", str(appeals_notclean_completo),
                "--output_dir", str(out_raw.parent),
                "--size", str(tamanho),
                "--strategy", estrategia,
            ]
            if estrategia not in ESTRATEGIAS_SEM_TEMAS:
                cmd += ["--temas", str(temas_notclean)]
            run(cmd)
        pular_ou_gerar(out_raw, f"resumo COMPLETO {estrategia} tamanho {tamanho} (cru)", gerar_raw)

        # ---- a partir do texto LIMPO ----
        out_clean = caminho_resumo_completo(base_dir, "clean", estrategia, tamanho)

        def gerar_clean(estrategia=estrategia, tamanho=tamanho, out_clean=out_clean):
            cmd = [
                PYTHON, str(scripts_dir / "gerarResumos.py"),
                "--input", str(appeals_clean_completo),
                "--output_dir", str(out_clean.parent),
                "--size", str(tamanho),
                "--strategy", estrategia,
            ]
            if estrategia not in ESTRATEGIAS_SEM_TEMAS:
                cmd += ["--temas", str(temas_clean)]
            run(cmd)
        pular_ou_gerar(out_clean, f"resumo COMPLETO {estrategia} tamanho {tamanho} (limpo)", gerar_clean)


def gerar_variante_de_split_via_cache(
    ids_path: Path,
    resumo_completo_path: Path,
    split_original_path: Path,
    output_path: Path,
) -> None:
    """
    Monta o resumo de UM split de treino a partir do resumo do corpus
    COMPLETO já calculado (garantir_resumo_completo), usando o mapa
    appeal_id -> posição original (split/ids/treino/<nome>.csv, gerado
    por splitDados.py). Puro `iloc` posicional -- sem GPU, sem chamar
    gerarResumos.py.

    Faz uma checagem de sanidade: o theme_id de cada linha pescada no
    cache tem que bater com o theme_id da mesma posição no split
    original. Se não bater, o mapeamento está errado (ou o cache foi
    gerado a partir de um corpus diferente do que gerou os splits) -- e
    aborta em vez de gravar um arquivo de treino silenciosamente
    corrompido.
    """
    ids_df = pd.read_csv(ids_path)
    cache_df = pd.read_csv(resumo_completo_path, encoding="latin1")

    max_id = int(ids_df["appeal_id"].max())
    if max_id >= len(cache_df):
        raise RuntimeError(
            f"appeal_id {max_id} (de {ids_path}) está fora do range do cache "
            f"{resumo_completo_path} ({len(cache_df)} linhas) -- o cache está "
            f"desatualizado ou foi gerado a partir de um corpus diferente."
        )

    variante = cache_df.iloc[ids_df["appeal_id"].values].reset_index(drop=True)

    original = pd.read_csv(split_original_path, encoding="latin1")
    if not (variante["theme_id"].values == original["theme_id"].values).all():
        raise RuntimeError(
            f"Descasamento entre o resumo pescado via appeal_id e o split original "
            f"{split_original_path} -- não vou gravar um arquivo de treino "
            f"corrompido. Verifique se {ids_path} e {resumo_completo_path} "
            f"correspondem de fato ao mesmo corpus/estratégia."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    salvar_csv_atomico(variante, output_path, index=False, encoding="latin1")


def preparar_variantes_de(
    appeals_treino_raw: Path,
    output_root: Path,
    resumos_treino: dict,
    scripts_dir: Path,
    ids_treino_dir: Path,
    base_dir: Path,
) -> None:
    nome = appeals_treino_raw.stem
    if nome.startswith("special_appeal_"):
        nome = nome[len("special_appeal_"):]

    base_dir_out = output_root / nome
    clean_csv = base_dir_out / "clean.csv"

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

    if not resumos_treino:
        return

    ids_path = ids_treino_dir / appeals_treino_raw.name
    if not ids_path.is_file():
        print(
            f"  [AVISO] {ids_path} não encontrado -- pulando variantes de resumo "
            f"para {nome} (rode splitDados.py/montarAmbiente.py para gerar os "
            f"mapas de appeal_id antes)."
        )
        return

    # ---- 2) resumos a partir do texto CRU, via cache do corpus completo ----
    raw_out_dir = base_dir_out / "raw"
    for estrategia, tamanho in resumos_treino.items():
        output = raw_out_dir / f"resumo_{estrategia}_{tamanho}.csv"
        cache_raw = caminho_resumo_completo(base_dir, "notClean", estrategia, tamanho)

        def gerar(ids_path=ids_path, cache_raw=cache_raw, output=output):
            gerar_variante_de_split_via_cache(ids_path, cache_raw, appeals_treino_raw, output)
        pular_ou_gerar(output, f"resumo {estrategia} tamanho {tamanho} (texto cru, via cache)", gerar)

    # ---- 3) resumos a partir do texto LIMPO, via cache do corpus completo ----
    clean_out_dir = base_dir_out / "clean"
    for estrategia, tamanho in resumos_treino.items():
        output = clean_out_dir / f"resumo_{estrategia}_{tamanho}.csv"
        cache_clean = caminho_resumo_completo(base_dir, "clean", estrategia, tamanho)

        def gerar(ids_path=ids_path, cache_clean=cache_clean, output=output):
            gerar_variante_de_split_via_cache(ids_path, cache_clean, clean_csv, output)
        pular_ou_gerar(output, f"resumo {estrategia} tamanho {tamanho} (texto limpo, via cache)", gerar)


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

    # Mapas appeal_id -> posição original, gerados por splitDados.py junto
    # com cada split de treino (split/ids/treino/<nome>.csv). É por eles
    # que os resumos do corpus completo (abaixo) são "recortados" por
    # split, sem recalcular nada.
    ids_treino_dir = args.split_treino_dir.parent / "ids" / "treino"

    # Corpus completo (não por split) -- insumo do cache de resumo. Por
    # essa altura do pipeline (montarAmbiente.py roda splitDados.py e
    # criarDiretorios.py antes deste script), o corpus cru já está em
    # data/appeals/notClean/texto/special_appeal.csv (criarDiretorios.py
    # já o moveu para lá).
    appeals_notclean_completo = args.base_dir / "appeals" / "notClean" / "texto" / "special_appeal.csv"
    appeals_clean_completo = args.base_dir / "appeals" / "clean" / "texto" / "special_appeal.csv"

    if resumos_treino:
        if not appeals_notclean_completo.is_file():
            raise FileNotFoundError(
                f"{appeals_notclean_completo} não encontrado -- rode splitDados.py e "
                f"criarDiretorios.py antes (via montarAmbiente.py) para que o corpus "
                f"completo esteja disponível."
            )
        garantir_appeals_completo_clean(appeals_notclean_completo, appeals_clean_completo, args.scripts_dir)
        garantir_resumo_completo(
            appeals_notclean_completo, appeals_clean_completo,
            temas_notclean, temas_clean, resumos_treino,
            args.base_dir, args.scripts_dir,
        )

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
            appeals_treino_raw,
            args.output_root, resumos_treino,
            args.scripts_dir, ids_treino_dir, args.base_dir,
        )

    print(f"\n{'=' * 60}\nVariantes de treino preparadas em: {args.output_root}\n{'=' * 60}")


if __name__ == "__main__":
    main()