"""
experimento.py

Script que executa UM experimento de recuperação de temas repetitivos
(GLARE) a partir de parâmetros recebidos via linha de comando.

Modelos treinados (_treinado)
------------------------------
Se --modelo terminar em '_treinado' (ex.: "distiluseMultilingual_treinado"),
o script garante automaticamente que o modelo foi treinado para o dataset
de TREINO indicado em --appeals_treino antes de executar o experimento.
Internamente usa o caminho dataset-específico
(modelos/by_dataset/<id>/<modelo>_treinado/) para nomear o arquivo de
cache de embeddings, evitando reuso de embeddings de outro dataset.
O campo "modelo" gravado no CSV continua mostrando o nome legível
("distiluseMultilingual_treinado").

Uso — modelo não-treinado:
    python experimento.py \\
        --dataset_nome baseline_teste \\
        --appeals data/appeals/notClean/texto/special_appeal_baseline.csv \\
        --temas data/temas/notClean/texto/temas_repetitivos.csv \\
        --metodo_similaridade BM25

Uso — modelo treinado:
    python experimento.py \\
        --dataset_nome baseline_teste \\
        --appeals data/appeals/notClean/texto/special_appeal_baseline.csv \\
        --temas data/temas/notClean/texto/temas_repetitivos.csv \\
        --metodo_similaridade COS \\
        --modelo distiluseMultilingual_treinado \\
        --appeals_treino split/treino/special_appeal_baseline.csv
"""

import argparse
import csv
import json
import string
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

PYTHON = sys.executable


# =====================================================
# UTIL
# =====================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("true", "1", "yes", "sim", "y", "t"):
        return True
    if v in ("false", "0", "no", "nao", "não", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v!r}")


def sanitize(nome: str) -> str:
    return nome.replace("/", "_").replace("-", "_").replace("\\", "_")


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def run_script(cmd: list, scripts_dir: Path) -> None:
    """Executa um script auxiliar do pipeline como subprocesso."""
    cmd = [str(c) for c in cmd]
    cmd[1] = str(scripts_dir / cmd[1])
    subprocess.run(cmd, check=True)


# =====================================================
# PERSISTÊNCIA DE TEMPOS
# =====================================================

def carregar_tempos(tempos_csv: Path) -> dict:
    tempos = {}
    if tempos_csv.exists():
        with open(tempos_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    try:
                        tempos[row[0]] = float(row[1])
                    except ValueError:
                        pass
    return tempos


def salvar_tempo(tempos_csv: Path, path: Path, tempo: float) -> None:
    tempos_csv.parent.mkdir(parents=True, exist_ok=True)
    tempos = carregar_tempos(tempos_csv)
    tempos[str(path)] = tempo
    with open(tempos_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arquivo", "tempo_s"])
        for k, v in tempos.items():
            writer.writerow([k, v])


def executar_ou_carregar(output_path: Path, tempos_csv: Path, gerar_func) -> float:
    tempos = carregar_tempos(tempos_csv)
    key = str(output_path)

    if output_path.exists():
        tempo = tempos.get(key, 0.0)
        print(f"[CACHE] {output_path}  ({tempo:.2f}s)")
        return tempo

    print(f"[GERANDO] {output_path}")
    t0 = time.time()
    gerar_func()
    tempo = round(time.time() - t0, 4)
    salvar_tempo(tempos_csv, output_path, tempo)
    print(f"[CONCLUÍDO] {output_path}  ({tempo:.2f}s)")
    return tempo


# =====================================================
# PASSO 0 — GARANTIR MODELO TREINADO (quando necessário)
# =====================================================

def temas_para_treino(args) -> Path:
    """
    Os temas são um corpus de referência ÚNICO (as 190 classes), não são
    particionados em treino/teste -- só existem em duas versões fixas:
    data/temas/notClean/texto/temas_repetitivos.csv e
    data/temas/clean/texto/temas_repetitivos.csv.

    Para parear corretamente no fine-tuning, usamos a versão de temas que
    combina com o texto de appeals de TREINO usado (args.appeals_treino):
    se a variante de treino (args.treino_variante) envolve texto limpo
    ('clean' ou 'clean_resumo_*'), usamos os temas limpos; caso contrário
    (variante 'raw' ou 'resumo_*' sobre texto cru), os temas notClean --
    o mesmo padrão clean/notClean usado nos appeals de teste.
    """
    usa_clean = args.treino_variante.startswith("clean")
    subpasta = "clean" if usa_clean else "notClean"
    return args.base_dir / "temas" / subpasta / "texto" / "temas_repetitivos.csv"


def step_garantir_modelo_treinado(args, scripts_dir: Path) -> None:
    """
    Se args.modelo termina em '_treinado', chama treinarModelos.py para
    garantir que o modelo está treinado no dataset indicado em
    args.appeals_treino, pareado com o CSV de temas correspondente
    (ver temas_para_treino).

    Preenche args._modelo_emb com o identificador relativo ao modelos_dir,
    por exemplo 'by_dataset/abc123/distiluseMultilingual_treinado'.
    Esse valor é usado internamente para:
      - nomear o arquivo de cache de embeddings (único por dataset de treino)
      - passar como --model ao gerarEmbedding.py

    Se o modelo NÃO for do tipo _treinado, args._modelo_emb == args.modelo
    e nenhuma chamada extra é feita.
    """
    if not args.modelo or not args.modelo.endswith("_treinado"):
        args._modelo_emb = args.modelo
        return

    nome_base = args.modelo[: -len("_treinado")]
    temas_csv_treino = temas_para_treino(args)

    cmd = [
        PYTHON, str(scripts_dir / "treinarModelos.py"),
        "--appeals_csv", str(args.appeals_treino),
        "--temas_csv", str(temas_csv_treino),
        "--modelos", nome_base,
        "--modelos_dir", str(args.modelos_dir),
    ]

    print(f"\n>> GARANTINDO MODELO TREINADO: {nome_base} "
          f"(dataset treino: {args.appeals_treino.name} | "
          f"variante: {args.treino_variante} | temas: {temas_csv_treino})")

    # stderr herda do processo pai (progress bars do tqdm aparecem normalmente).
    # stdout é capturado para interceptar a linha MODELO_CAMINHO:.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    caminho_modelo_str = None
    prefix = f"MODELO_CAMINHO:{nome_base}="
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.strip().startswith(prefix):
            caminho_modelo_str = line.strip()[len(prefix):]
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"treinarModelos.py falhou (código {proc.returncode}) "
            f"ao treinar '{nome_base}'."
        )
    if caminho_modelo_str is None:
        raise RuntimeError(
            f"treinarModelos.py não imprimiu 'MODELO_CAMINHO:{nome_base}=...'.\n"
            f"Verifique se treinarModelos.py está atualizado."
        )

    # Converte o caminho absoluto/relativo para um identificador relativo
    # ao modelos_dir, ex: "by_dataset/abc123/distiluseMultilingual_treinado"
    caminho_completo = Path(caminho_modelo_str)
    try:
        relativo = caminho_completo.resolve().relative_to(
            args.modelos_dir.resolve()
        )
        args._modelo_emb = str(relativo)
    except ValueError:
        # fallback: usa o caminho completo como string (ainda é único)
        args._modelo_emb = str(caminho_completo)

    print(f"[MODELO TREINADO] identificador para embeddings: {args._modelo_emb}")


# =====================================================
# PASSO 1 — CLEAN
# =====================================================

def step_clean(args, scripts_dir: Path) -> tuple:
    """Retorna (appeals_path, temas_path, tempo_total_s)."""
    if not args.usar_clean:
        return args.appeals, args.temas, 0.0

    print(">> CLEAN")
    bp_suffix = f"_{args.begin_point}" if args.begin_point else ""
    appeals_clean = args.base_dir / f"appeals/clean/texto/{args.appeals.stem}{bp_suffix}.csv"
    temas_clean = args.base_dir / "temas/clean/texto/temas_repetitivos.csv"
    appeals_clean.parent.mkdir(parents=True, exist_ok=True)
    temas_clean.parent.mkdir(parents=True, exist_ok=True)

    def gerar_appeals():
        cmd = [
            PYTHON, "limparTexto.py",
            "--input", str(args.appeals),
            "--output", str(appeals_clean),
            "--text_column", "special_appeal_text",
        ]
        if args.begin_point:
            cmd += ["--begin_point", args.begin_point]
        run_script(cmd, scripts_dir)

    def gerar_temas():
        run_script([
            PYTHON, "limparTexto.py",
            "--input", str(args.temas),
            "--output", str(temas_clean),
            "--text_column", "theme_text",
        ], scripts_dir)

    t = executar_ou_carregar(appeals_clean, args.tempos_csv, gerar_appeals)
    t += executar_ou_carregar(temas_clean, args.tempos_csv, gerar_temas)
    return appeals_clean, temas_clean, t


# =====================================================
# PASSO 2 — RESUMO
# =====================================================

def step_resumo(args, appeals_path: Path, temas_path: Path, scripts_dir: Path) -> tuple:
    """Retorna (appeals_path, tempo_s)."""
    if not args.usar_resumo:
        return appeals_path, 0.0

    print(">> RESUMO")
    clean_flag = "clean" if args.usar_clean else "notClean"
    out_dir = args.base_dir / "appeals" / clean_flag / "resumos" / args.dataset_nome
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"resumo_{args.resumo_estrategia}_{args.resumo_tamanho}.csv"

    def gerar():
        run_script([
            PYTHON, "gerarResumos.py",
            "--input", str(appeals_path),
            "--temas", str(temas_path),
            "--output_dir", str(out_dir),
            "--size", str(args.resumo_tamanho),
            "--strategy", args.resumo_estrategia,
        ], scripts_dir)

    t = executar_ou_carregar(output, args.tempos_csv, gerar)
    return output, t


# =====================================================
# PASSO 3 — EMBEDDINGS
# =====================================================

def step_embeddings(args, input_path: Path, tipo: str, scripts_dir: Path) -> tuple:
    """
    Retorna (emb_path, tempo_s).

    Usa args._modelo_emb (não args.modelo) para nomear o arquivo de cache
    e para o argumento --model do gerarEmbedding.py.
    Para modelos _treinado, _modelo_emb é o caminho dataset-específico
    (ex: by_dataset/abc123/distiluseMultilingual_treinado), garantindo que
    embeddings de datasets de treino diferentes nunca sejam confundidas.
    """
    modelo_emb = args._modelo_emb          # identificador interno (único por dataset)
    modelo_sanit = sanitize(modelo_emb)    # string segura para usar em nomes de arquivo

    pasta = "appeals" if tipo == "appeals" else "temas"
    clean_flag = "clean" if args.usar_clean else "notClean"
    out_dir = args.base_dir / pasta / clean_flag / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    nome = input_path.stem
    output = out_dir / f"embedding_{modelo_sanit}__{nome}.csv"

    def gerar():
        run_script([
            PYTHON, "gerarEmbedding.py",
            "--input", str(input_path),
            "--model", modelo_emb,
            "--output_dir", str(out_dir),
        ], scripts_dir)

    t = executar_ou_carregar(output, args.tempos_csv, gerar)
    return output, t


# =====================================================
# MÉTRICAS
# =====================================================

def calcular_metricas(similaridades, labels, k: int = 6):
    rankings = np.argsort(-similaridades, axis=1)[:, :k]
    recall = np.mean([labels[i] in rankings[i] for i in range(len(labels))])

    ap, ndcg = [], []
    for i in range(len(labels)):
        ranking = rankings[i]
        if labels[i] in ranking:
            pos = int(np.where(ranking == labels[i])[0][0]) + 1
            ap.append(1.0 / pos)
            ndcg.append(1.0 / np.log2(pos + 1))
        else:
            ap.append(0.0)
            ndcg.append(0.0)

    map_k = float(np.mean(ap))
    ndcg_k = float(np.mean(ndcg))
    f1 = 2 * (map_k * recall) / (map_k + recall) if (map_k + recall) > 0 else 0.0
    return float(recall), map_k, ndcg_k, f1


# =====================================================
# RESULTADOS
# =====================================================

CAMPOS_RESULTADOS = [
    "timestamp", "dataset", "modelo", "treino_variante", "usar_clean", "begin_point",
    "usar_resumo", "resumo_tamanho", "resumo_estrategia",
    "metodo_similaridade", "k",
    "recall_at_k", "map_at_k", "ndcg_at_k", "f1_score", "tempo",
]


def salvar_resultado(args, recall, map_k, ndcg_k, f1, tf) -> dict:
    args.resultados_csv.parent.mkdir(parents=True, exist_ok=True)
    arquivo_existe = args.resultados_csv.exists()

    linha = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset_nome,
        # args.modelo é o nome legível (ex: "distiluseMultilingual_treinado")
        "modelo": args.modelo,
        "treino_variante": args.treino_variante,
        "usar_clean": args.usar_clean,
        "begin_point": args.begin_point,
        "usar_resumo": args.usar_resumo,
        "resumo_tamanho": args.resumo_tamanho if args.usar_resumo else None,
        "resumo_estrategia": args.resumo_estrategia if args.usar_resumo else None,
        "metodo_similaridade": args.metodo_similaridade,
        "k": args.k,
        "recall_at_k": round(recall, 6),
        "map_at_k": round(map_k, 6),
        "ndcg_at_k": round(ndcg_k, 6),
        "f1_score": round(f1, 6),
        "tempo": round(tf, 6),
    }

    with open(args.resultados_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_RESULTADOS)
        if not arquivo_existe:
            writer.writeheader()
        writer.writerow(linha)

    return linha


def imprimir_resultados(linha: dict) -> None:
    print("\n=== RESULTADOS ===")
    print(f"Dataset     : {linha['dataset']}")
    print(f"Método      : {linha['metodo_similaridade']}")
    print(f"Modelo      : {linha['modelo']}")
    print(f"Clean       : {linha['usar_clean']}  | begin_point: {linha['begin_point']}")
    print(f"Resumo      : {linha['usar_resumo']} "
          f"(estrategia={linha['resumo_estrategia']}, tamanho={linha['resumo_tamanho']})")
    print(f"\nRecall@{linha['k']}  : {linha['recall_at_k']:.4f}")
    print(f"MAP@{linha['k']}     : {linha['map_at_k']:.4f}")
    print(f"NDCG@{linha['k']}    : {linha['ndcg_at_k']:.4f}")
    print(f"F1-score  : {linha['f1_score']:.4f}")
    print(f"\nResultado salvo em: {linha}")


# =====================================================
# EXECUTAR EXPERIMENTO
# =====================================================

def executar_experimento(args) -> dict:
    scripts_dir = args.scripts_dir
    tf = 0.0

    # Passo 0: garantir modelo treinado (se aplicável) e definir _modelo_emb
    step_garantir_modelo_treinado(args, scripts_dir)

    # Passo 1: limpeza
    appeals_path, temas_path, t = step_clean(args, scripts_dir)
    tf += t

    # Passo 2: resumo
    appeals_path, t = step_resumo(args, appeals_path, temas_path, scripts_dir)
    tf += t

    df_app = pd.read_csv(appeals_path, encoding="latin1")
    df_tema = pd.read_csv(temas_path, encoding="latin1")

    map_idx = {tid: i for i, tid in enumerate(df_tema["theme_id"])}
    labels = np.array([map_idx[x] for x in df_app["theme_id"]])

    # Passo 3: similaridade
    if args.metodo_similaridade == "COS":
        emb_app_path, t = step_embeddings(args, appeals_path, "appeals", scripts_dir)
        tf += t
        emb_tema_path, t = step_embeddings(args, temas_path, "temas", scripts_dir)
        tf += t

        emb_app = pd.read_csv(emb_app_path).iloc[:, 1:].values
        emb_tema = pd.read_csv(emb_tema_path).iloc[:, 1:].values

        t0 = time.time()
        sim = cosine_similarity(emb_app, emb_tema)
        tf += time.time() - t0

    elif args.metodo_similaridade == "BM25":
        textos_app = df_app.iloc[:, 1].tolist()
        t0 = time.time()
        temas_tok = [remove_punctuation(t_).split() for t_ in df_tema["theme_text"]]
        bm25 = BM25Okapi(temas_tok)
        sim = np.array([
            bm25.get_scores(remove_punctuation(t_).split())
            for t_ in textos_app
        ])
        tf += time.time() - t0

    else:
        raise ValueError(f"metodo_similaridade inválido: '{args.metodo_similaridade}'")

    recall, map_k, ndcg_k, f1 = calcular_metricas(sim, labels, args.k)
    linha = salvar_resultado(args, recall, map_k, ndcg_k, f1, tf)
    imprimir_resultados(linha)
    return linha


# =====================================================
# CLI
# =====================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Executa um experimento GLARE com uma combinação de parâmetros."
    )

    # --- dataset de teste ---
    p.add_argument("--dataset_nome", required=True)
    p.add_argument("--appeals", required=True, type=Path,
                    help="CSV de appeals de TESTE.")
    p.add_argument("--temas", required=True, type=Path,
                    help="CSV de temas.")

    # --- pré-processamento ---
    p.add_argument("--usar_clean", type=str2bool, default=True)
    p.add_argument("--begin_point", default=None)

    # --- resumo ---
    p.add_argument("--usar_resumo", type=str2bool, default=False)
    p.add_argument("--resumo_tamanho", type=int, default=None)
    p.add_argument("--resumo_estrategia", default=None,
                    choices=["lexrank", "guided_lexrank", "guided_lexrank_opt",
                             "personalized_pagerank", "bayesian_dirichlet"])

    # --- similaridade / modelo ---
    p.add_argument("--metodo_similaridade", required=True, choices=["COS", "BM25"])
    p.add_argument("--modelo", default=None,
                    help=(
                        "Nome do modelo de embedding. Para modelos base, use a chave "
                        "de modelos.py (ex: 'distiluseMultilingual'). Para modelos "
                        "fine-tunados, adicione o sufixo '_treinado' "
                        "(ex: 'distiluseMultilingual_treinado'). "
                        "Obrigatório quando --metodo_similaridade=COS."
                    ))
    p.add_argument("--appeals_treino", type=Path, default=None,
                    help=(
                        "CSV de appeals de TREINO, usado para garantir que o modelo "
                        "foi treinado no dataset correto. Obrigatório quando --modelo "
                        "termina em '_treinado'."
                    ))
    p.add_argument("--treino_variante", default="raw",
                    choices=[
                        "raw", "clean",
                        "resumo_lexrank", "resumo_guided_lexrank",
                        "clean_resumo_lexrank", "clean_resumo_guided_lexrank",
                    ],
                    help=(
                        "Qual pré-processamento foi aplicado ao --appeals_treino antes "
                        "do fine-tuning (ver prepararTreinoVariantes.py). Usado para "
                        "escolher os temas corretos (clean/notClean) no pareamento e "
                        "para registro no CSV de resultados. Só relevante quando "
                        "--modelo termina em '_treinado'; ignorado nos demais casos."
                    ))
    p.add_argument("--modelos_dir", type=Path, default=Path("modelos"),
                    help="Diretório raiz dos modelos treinados. Padrão: modelos/")

    # --- outros ---
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--base_dir", type=Path, default=Path("data"))
    p.add_argument("--scripts_dir", type=Path, default=Path("."),
                    help="Diretório com limparTexto.py, gerarResumos.py, gerarEmbedding.py.")
    p.add_argument("--resultados_csv", type=Path, default=Path("resultados/experimentos.csv"))
    p.add_argument("--tempos_csv", type=Path, default=Path("resultados/tempos_arquivos.csv"))

    args = p.parse_args(argv)

    # ---------------------------------------------------------------
    # Normalização: zera parâmetros que não fazem sentido para a
    # combinação pedida, para que o CSV de resultados seja limpo.
    # ---------------------------------------------------------------

    # resumo
    if not args.usar_resumo:
        if args.resumo_tamanho is not None or args.resumo_estrategia is not None:
            print("[AVISO] usar_resumo=False: ignorando resumo_tamanho/resumo_estrategia.")
        args.resumo_tamanho = None
        args.resumo_estrategia = None
    else:
        if args.resumo_tamanho is None or args.resumo_estrategia is None:
            p.error("--usar_resumo true requer --resumo_tamanho e --resumo_estrategia.")

    # begin_point
    if not args.usar_clean and args.begin_point is not None:
        print("[AVISO] usar_clean=False: ignorando begin_point.")
        args.begin_point = None

    # modelo / método
    if args.metodo_similaridade == "BM25":
        if args.modelo is not None:
            print("[AVISO] metodo_similaridade=BM25: ignorando --modelo (BM25 não usa embeddings).")
        args.modelo = None
        args.appeals_treino = None   # irrelevante sem modelo
        args.treino_variante = "raw"
    elif args.metodo_similaridade == "COS":
        if args.modelo is None:
            p.error("--metodo_similaridade COS requer --modelo.")
        if args.modelo.endswith("_treinado") and args.appeals_treino is None:
            p.error(
                f"--modelo '{args.modelo}' termina em '_treinado', portanto "
                f"--appeals_treino é obrigatório para garantir que o modelo "
                f"foi treinado no dataset correto."
            )
        if not args.modelo.endswith("_treinado") and args.appeals_treino is not None:
            print("[AVISO] --appeals_treino ignorado: --modelo não é do tipo _treinado.")
            args.appeals_treino = None
        if not args.modelo.endswith("_treinado") and args.treino_variante != "raw":
            print("[AVISO] --treino_variante ignorado: --modelo não é do tipo _treinado.")
            args.treino_variante = "raw"

    # _modelo_emb será preenchido em step_garantir_modelo_treinado (no runtime)
    args._modelo_emb = args.modelo

    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"\n{'=' * 60}\nEXPERIMENTO: {args.dataset_nome}\n{'=' * 60}")
    linha = executar_experimento(args)
    print("\nRESULTADO_JSON: " + json.dumps(linha, ensure_ascii=False))


if __name__ == "__main__":
    main()