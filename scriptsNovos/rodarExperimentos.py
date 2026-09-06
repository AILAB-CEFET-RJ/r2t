"""
rodarExperimentos.py

Orquestra a execução de múltiplos experimentos (experimento.py) cobrindo
uma grade de configurações.

Modelo _treinado
-----------------
Se MODELO_VALORES contiver strings terminadas em '_treinado'
(ex: "distiluseMultilingual_treinado"), o campo "appeals_treino" de cada
dataset é obrigatório. experimento.py cuidará de chamar treinarModelos.py
antes do experimento, com custo zero se o modelo já estiver em cache.

Eixos dependentes
-----------------
Cada variável é um "eixo". Eixos com condição de relevância só varrem seus
valores quando a condição é satisfeita — evitando combinações redundantes:
  • resumo_tamanho / resumo_estrategia: só variam quando usar_resumo=True
  • begin_point: só varia quando usar_clean=True
  • modelo: só varia quando metodo_similaridade=COS

Uso:
    python rodarExperimentos.py             # roda tudo pendente
    python rodarExperimentos.py --dry_run   # mostra o plano, não executa
    python rodarExperimentos.py --forcar    # ignora resultados já existentes
"""

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from configUtil import carregar_modulo

PYTHON = sys.executable
EXPERIMENTO_SCRIPT = "experimento.py"
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_EXPERIMENTOS_PADRAO = SCRIPT_DIR / "config_experimentos.py"


# =====================================================
# CONFIGURAÇÃO — agora vem de um arquivo externo (ver --config)
# =====================================================
# Os valores que antes ficavam hardcoded aqui (DATASETS, USAR_CLEAN_VALORES,
# MODELO_VALORES, TREINO_VARIANTE_VALORES etc.) agora vêm de um módulo
# Python carregado dinamicamente por --config (default: config_experimentos.py,
# neste mesmo diretório). Ver exemplo em config_experimentos.py.
#
# O config de experimentos DEVE referenciar, por sua vez, um config de
# AMBIENTE (campo CONFIG_AMBIENTE, obrigatório -- sem default) -- de onde vem
# RESUMOS_TREINO, necessário para resolver_appeals_treino saber com que
# tamanho cada estratégia de resumo de TREINO foi gerada.

KINDS_COM_SEED = ["baseline", "stratified", "minority"]

_TEMAS_NOTCLEAN = "data/temas/notClean/texto/temas_repetitivos.csv"


def _gerar_datasets(kinds: list, seeds: list) -> list:
    """
    Gera a lista de datasets (kind x seed, + zeroshot se presente em
    `kinds`) a partir de KINDS/SEEDS do config de experimentos.
    """
    datasets = []
    for kind in kinds:
        if kind == "zeroshot":
            continue
        if kind not in KINDS_COM_SEED:
            raise ValueError(
                f"KINDS contém '{kind}' desconhecido -- esperado um de "
                f"{KINDS_COM_SEED + ['zeroshot']}."
            )
        for seed in seeds:
            datasets.append({
                "nome": f"{kind}_teste_{seed}",
                "appeals": f"data/appeals/notClean/texto/special_appeal_{kind}_{seed}.csv",
                "temas": _TEMAS_NOTCLEAN,
                "appeals_treino": f"split/treino/special_appeal_{kind}_{seed}.csv",
            })
    if "zeroshot" in kinds:
        datasets.append({
            "nome": "zeroshot_teste",
            "appeals": "data/appeals/notClean/texto/special_appeal_zeroshot.csv",
            "temas": _TEMAS_NOTCLEAN,
            "appeals_treino": "split/treino/special_appeal_zeroshot.csv",
        })
    return datasets


def _filtrar_datasets_existentes(datasets: list) -> list:
    """
    Remove (com aviso) datasets cujo arquivo de TESTE não existe no disco
    -- por exemplo, uma seed listada em SEEDS no config de experimentos
    mas que nunca foi gerada por montarAmbiente.py. Não aborta a
    execução inteira: só aquele dataset fica de fora da grade.
    """
    validos = []
    for ds in datasets:
        if not Path(ds["appeals"]).is_file():
            print(
                f"[AVISO] dataset '{ds['nome']}': {ds['appeals']} não encontrado "
                f"-- pulando (rode montarAmbiente.py com essa seed/kind antes)."
            )
            continue
        validos.append(ds)
    return validos


TREINO_VARIANTES_DIR = "split/treino_variantes"

# Mapeia cada variante de resumo para (estrategia, subpasta). 'raw' e
# 'clean' são tratadas à parte em resolver_appeals_treino, pois não
# dependem de estratégia/tamanho de resumo.
MAPA_VARIANTE_RESUMO = {
    "resumo_lexrank": ("lexrank", "raw"),
    "resumo_guided_lexrank": ("guided_lexrank", "raw"),
    "clean_resumo_lexrank": ("lexrank", "clean"),
    "clean_resumo_guided_lexrank": ("guided_lexrank", "clean"),
}


def _pasta_variante(appeals_treino_raw: str) -> str:
    """'split/treino/special_appeal_baseline_42.csv' -> 'baseline_42'."""
    nome = Path(appeals_treino_raw).stem
    prefixo = "special_appeal_"
    return nome[len(prefixo):] if nome.startswith(prefixo) else nome


def resolver_appeals_treino(ds: dict, variante: str, resumos_treino: dict) -> str:
    """
    Dado um dataset (com seu appeals_treino cru), a variante desejada, e o
    dict RESUMOS_TREINO (estrategia -> tamanho) do config de ambiente,
    devolve o caminho do CSV de treino correspondente.

    'raw' devolve o próprio ds['appeals_treino']; as demais apontam para
    os arquivos gerados por prepararTreinoVariantes.py em
    TREINO_VARIANTES_DIR, usando o tamanho configurado para cada
    estratégia (ex: guided_lexrank:15, lexrank:60 -- tamanhos podem ser
    diferentes por estratégia).
    """
    if variante == "raw":
        return ds["appeals_treino"]

    pasta = _pasta_variante(ds["appeals_treino"])
    base = f"{TREINO_VARIANTES_DIR}/{pasta}"

    if variante == "clean":
        return f"{base}/clean.csv"

    if variante not in MAPA_VARIANTE_RESUMO:
        raise ValueError(f"treino_variante desconhecida: '{variante}'")

    estrategia, subpasta = MAPA_VARIANTE_RESUMO[variante]
    if estrategia not in resumos_treino:
        raise ValueError(
            f"Variante '{variante}' precisa de RESUMOS_TREINO['{estrategia}'] "
            f"definido no config de ambiente, mas ela não está lá. "
            f"Config de ambiente atual: {resumos_treino}"
        )
    tamanho = resumos_treino[estrategia]
    return f"{base}/{subpasta}/resumo_{estrategia}_{tamanho}.csv"


BASE_DIR_PADRAO = "data"
SCRIPTS_DIR_PADRAO = "."
MODELOS_DIR_PADRAO = "modelos"
RESULTADOS_CSV_PADRAO = "resultados/experimentos4.csv"
TEMPOS_CSV_PADRAO = "resultados/tempos_arquivos.csv"


# =====================================================
# VALIDAÇÃO ANTECIPADA
# =====================================================

def validar_config(cfg, resumos_treino: dict) -> None:
    """
    Verifica, antes de gerar qualquer combinação, se a configuração é
    internamente consistente.
    """
    usa_treinado = any(m.endswith("_treinado") for m in cfg.MODELO_VALORES)
    usa_cos = "COS" in cfg.METODO_SIMILARIDADE_VALORES

    if usa_treinado and usa_cos:
        for ds in cfg.DATASETS:
            if not ds.get("appeals_treino"):
                raise ValueError(
                    f"Dataset '{ds['nome']}' não tem 'appeals_treino' definido, "
                    f"mas MODELO_VALORES contém modelos '_treinado'. "
                    f"Adicione 'appeals_treino' ao dataset ou remova os modelos _treinado."
                )

        for variante in cfg.TREINO_VARIANTE_VALORES:
            if variante in ("raw", "clean"):
                continue
            if variante not in MAPA_VARIANTE_RESUMO:
                raise ValueError(
                    f"TREINO_VARIANTE_VALORES contém variante desconhecida: '{variante}'."
                )
            estrategia, _ = MAPA_VARIANTE_RESUMO[variante]
            if estrategia not in resumos_treino:
                raise ValueError(
                    f"TREINO_VARIANTE_VALORES contém '{variante}', mas o config de "
                    f"ambiente não define RESUMOS_TREINO['{estrategia}']. "
                    f"Adicione a estratégia lá (e gere os arquivos com "
                    f"prepararTreinoVariantes.py), ou remova a variante daqui."
                )


# =====================================================
# MOTOR DE COMBINAÇÕES COM EIXOS DEPENDENTES
# =====================================================

@dataclass
class Eixo:
    nome: str
    valores: list
    relevante_se: Optional[Callable[[dict], bool]] = None  # None = sempre relevante

    def ativo(self, combo_parcial: dict) -> bool:
        return True if self.relevante_se is None else self.relevante_se(combo_parcial)


def montar_eixos(cfg) -> List[Eixo]:
    """
    A ORDEM IMPORTA: um eixo dependente deve aparecer depois dos eixos
    dos quais ele depende.
    """
    return [
        Eixo("dataset", cfg.DATASETS),
        Eixo("usar_clean", cfg.USAR_CLEAN_VALORES),
        Eixo("begin_point", cfg.BEGIN_POINT_VALORES,
             relevante_se=lambda c: c["usar_clean"] is True),
        Eixo("usar_resumo", cfg.USAR_RESUMO_VALORES),
        Eixo("resumo_tamanho", cfg.RESUMO_TAMANHO_VALORES,
             relevante_se=lambda c: c["usar_resumo"] is True),
        Eixo("resumo_estrategia", cfg.RESUMO_ESTRATEGIA_VALORES,
             relevante_se=lambda c: c["usar_resumo"] is True),
        Eixo("metodo_similaridade", cfg.METODO_SIMILARIDADE_VALORES),
        Eixo("modelo", cfg.MODELO_VALORES,
             relevante_se=lambda c: c["metodo_similaridade"] == "COS"),
        Eixo("treino_variante", cfg.TREINO_VARIANTE_VALORES,
             relevante_se=lambda c: (
                 c["metodo_similaridade"] == "COS"
                 and c["modelo"] is not None
                 and c["modelo"].endswith("_treinado")
             )),
        Eixo("k", cfg.K_VALORES),
    ]


def gerar_combinacoes(eixos: List[Eixo]) -> List[dict]:
    combos = [{}]
    for eixo in eixos:
        novos = []
        for combo in combos:
            if eixo.ativo(combo):
                for valor in eixo.valores:
                    novos.append({**combo, eixo.nome: valor})
            else:
                novos.append({**combo, eixo.nome: None})
        combos = novos
    return combos


def assinatura(combo: dict) -> tuple:
    """Chave canônica para deduplicação e retomada."""
    d = dict(combo)
    ds = combo.get("dataset")
    d["dataset"] = ds["nome"] if isinstance(ds, dict) else ds
    return tuple(sorted(d.items(), key=lambda kv: kv[0]))


def deduplicar(combos: List[dict]) -> List[dict]:
    vistos, unicos = set(), []
    for c in combos:
        s = assinatura(c)
        if s not in vistos:
            vistos.add(s)
            unicos.append(c)
    return unicos


# =====================================================
# RETOMADA — pula combinações já no CSV de resultados
# =====================================================

CAMPOS_ESPERADOS = {
    "dataset", "usar_clean", "begin_point", "usar_resumo",
    "resumo_tamanho", "resumo_estrategia", "metodo_similaridade",
    "modelo", "treino_variante", "k",
}


def _ou_none(v: str):
    return None if v in ("", "None") else v


def carregar_assinaturas_existentes(resultados_csv: Path) -> set:
    if not resultados_csv.exists():
        return set()
    with open(resultados_csv, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        if not CAMPOS_ESPERADOS.issubset(set(leitor.fieldnames or [])):
            print(
                f"[AVISO] {resultados_csv} não tem o schema esperado "
                f"(versão antiga?). Retomada automática desativada."
            )
            return set()
        existentes = set()
        for linha in leitor:
            combo = {
                "dataset": linha["dataset"],
                "usar_clean": linha["usar_clean"] == "True",
                "begin_point": _ou_none(linha["begin_point"]),
                "usar_resumo": linha["usar_resumo"] == "True",
                "resumo_tamanho": (
                    int(linha["resumo_tamanho"])
                    if _ou_none(linha["resumo_tamanho"]) is not None else None
                ),
                "resumo_estrategia": _ou_none(linha["resumo_estrategia"]),
                "metodo_similaridade": linha["metodo_similaridade"],
                "modelo": _ou_none(linha["modelo"]),
                "treino_variante": _ou_none(linha["treino_variante"]) or "raw",
                "k": int(linha["k"]),
            }
            existentes.add(assinatura(combo))
    return existentes


# =====================================================
# MONTAR COMANDO CLI
# =====================================================

def montar_cli(combo: dict, args, resumos_treino: dict) -> List[str]:
    ds = combo["dataset"]
    cli = [
        PYTHON, str(Path(args.scripts_dir) / EXPERIMENTO_SCRIPT),
        "--dataset_nome", ds["nome"],
        "--appeals", ds["appeals"],
        "--temas", ds["temas"],
        "--usar_clean", str(combo["usar_clean"]),
        "--usar_resumo", str(combo["usar_resumo"]),
        "--metodo_similaridade", combo["metodo_similaridade"],
        "--k", str(combo["k"]),
        "--base_dir", args.base_dir,
        "--scripts_dir", args.scripts_dir,
        "--modelos_dir", args.modelos_dir,
        "--resultados_csv", args.resultados_csv,
        "--tempos_csv", args.tempos_csv,
    ]
    if combo["begin_point"] is not None:
        cli += ["--begin_point", combo["begin_point"]]
    if combo["usar_resumo"]:
        cli += ["--resumo_tamanho", str(combo["resumo_tamanho"])]
        cli += ["--resumo_estrategia", combo["resumo_estrategia"]]
    if combo["metodo_similaridade"] == "COS":
        cli += ["--modelo", combo["modelo"]]
        # Se o modelo é _treinado, resolve o appeals_treino conforme a
        # variante de pré-processamento sorteada para o treino (ver
        # resolver_appeals_treino / prepararTreinoVariantes.py). Os
        # appeals/temas de TESTE (acima) não mudam com a variante.
        if combo["modelo"] and combo["modelo"].endswith("_treinado"):
            variante = combo.get("treino_variante") or "raw"
            cli += ["--appeals_treino", resolver_appeals_treino(ds, variante, resumos_treino)]
            cli += ["--treino_variante", variante]
    return [str(x) for x in cli]


def descrever(combo: dict) -> str:
    ds = combo["dataset"]
    nome_ds = ds["nome"] if isinstance(ds, dict) else ds
    partes = [f"dataset={nome_ds}", f"clean={combo['usar_clean']}"]
    if combo["begin_point"]:
        partes.append(f"begin_point={combo['begin_point']}")
    partes.append(f"resumo={combo['usar_resumo']}")
    if combo["usar_resumo"]:
        partes.append(f"estrategia={combo['resumo_estrategia']}")
        partes.append(f"tamanho={combo['resumo_tamanho']}")
    partes.append(f"sim={combo['metodo_similaridade']}")
    if combo["modelo"]:
        partes.append(f"modelo={combo['modelo']}")
        if combo["modelo"].endswith("_treinado"):
            partes.append(f"treino_variante={combo.get('treino_variante') or 'raw'}")
    partes.append(f"k={combo['k']}")
    return " | ".join(partes)


# =====================================================
# MAIN
# =====================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Roda a grade de experimentos GLARE.")
    p.add_argument(
        "--config", type=Path, default=CONFIG_EXPERIMENTOS_PADRAO,
        help=(
            "Arquivo de config de experimentos (módulo Python com KINDS, SEEDS, "
            "TREINO_VARIANTE_VALORES e as demais listas de eixos). "
            f"Default: {CONFIG_EXPERIMENTOS_PADRAO.name}."
        ),
    )
    p.add_argument("--base_dir", default=BASE_DIR_PADRAO)
    p.add_argument("--scripts_dir", default=SCRIPTS_DIR_PADRAO)
    p.add_argument("--modelos_dir", default=MODELOS_DIR_PADRAO)
    p.add_argument("--resultados_csv", default=RESULTADOS_CSV_PADRAO)
    p.add_argument("--tempos_csv", default=TEMPOS_CSV_PADRAO)
    p.add_argument("--dry_run", action="store_true",
                    help="Só mostra o plano, não executa nada.")
    p.add_argument("--forcar", action="store_true",
                    help="Roda mesmo combinações já presentes no CSV de resultados.")
    args = p.parse_args(argv)

    cfg = carregar_modulo(args.config, "config_experimentos")

    # O config de experimentos DEVE referenciar (por caminho) o config de
    # ambiente com o qual os splits/variantes de treino foram gerados --
    # sem isso não há como resolver_appeals_treino saber os tamanhos de
    # resumo usados no treino. Caminho relativo é resolvido em relação à
    # pasta do config_experimentos, não do diretório de trabalho atual.
    if not getattr(cfg, "CONFIG_AMBIENTE", None):
        p.error(
            f"{args.config} precisa definir CONFIG_AMBIENTE (caminho para o "
            f"config de ambiente usado para gerar os splits, ex: "
            f"CONFIG_AMBIENTE = \"config_ambiente.py\")."
        )
    caminho_ambiente = Path(cfg.CONFIG_AMBIENTE)
    if not caminho_ambiente.is_absolute():
        caminho_ambiente = args.config.resolve().parent / caminho_ambiente
    cfg_ambiente = carregar_modulo(caminho_ambiente, "config_ambiente")
    resumos_treino = getattr(cfg_ambiente, "RESUMOS_TREINO", {})

    print(f"Config de experimentos: {args.config}")
    print(f"Config de ambiente:     {caminho_ambiente}")
    print(f"  KINDS = {cfg.KINDS}")
    print(f"  SEEDS = {cfg.SEEDS}")
    print(f"  RESUMOS_TREINO (do ambiente) = {resumos_treino}\n")

    cfg.DATASETS = _filtrar_datasets_existentes(_gerar_datasets(cfg.KINDS, cfg.SEEDS))
    if not cfg.DATASETS:
        p.error(
            "Nenhum dataset válido encontrado para o KINDS/SEEDS pedido -- "
            "rode montarAmbiente.py primeiro (com as mesmas seeds)."
        )

    # validação antecipada
    validar_config(cfg, resumos_treino)

    eixos = montar_eixos(cfg)
    combos = deduplicar(gerar_combinacoes(eixos))
    # Eixos inativos (ex: treino_variante quando o modelo não é _treinado)
    # ganham None em gerar_combinacoes; normaliza para "raw", o mesmo
    # default gravado por experimento.py no CSV de resultados.
    for c in combos:
        if c.get("treino_variante") is None:
            c["treino_variante"] = "raw"
    existentes = (
        set() if args.forcar
        else carregar_assinaturas_existentes(Path(args.resultados_csv))
    )
    pendentes = [c for c in combos if assinatura(c) not in existentes]

    print(f"Total de combinações únicas planejadas : {len(combos)}")
    print(f"Já presentes no CSV de resultados      : {len(combos) - len(pendentes)}")
    print(f"A executar agora                       : {len(pendentes)}\n")

    for i, combo in enumerate(combos, 1):
        ja_existe = assinatura(combo) in existentes and not args.forcar
        status = "PULAR (já existe)" if ja_existe else "RODAR"
        print(f"[{i:03d}/{len(combos)}] {status:18s} {descrever(combo)}")

    if args.dry_run:
        print("\n--dry_run ativo: nenhum experimento executado.")
        return pendentes

    falhas = []
    for i, combo in enumerate(pendentes, 1):
        print(f"\n{'=' * 70}\n"
              f"Experimento {i}/{len(pendentes)}: {descrever(combo)}\n"
              f"{'=' * 70}")
        cli = montar_cli(combo, args, resumos_treino)
        try:
            subprocess.run(cli, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERRO] {descrever(combo)} -> código {e.returncode}")
            falhas.append(combo)

    print(f"\nConcluído. {len(pendentes) - len(falhas)}/{len(pendentes)} com sucesso.")
    if falhas:
        print(f"{len(falhas)} falharam:")
        for c in falhas:
            print("  -", descrever(c))

    return pendentes


if __name__ == "__main__":
    main()