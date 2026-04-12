import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import string
import os
import csv
from datetime import datetime

# =====================================================
# CONFIGURAÇÃO GLOBAL
# =====================================================
RESULTADOS_CSV = "resultados/experimentos.csv"

PYTHON = sys.executable

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"

USAR_CLEAN = True
USAR_RESUMO = True

RESUMO_TAMANHO = 10
RESUMO_ESTRATEGIA = "personalized_pagerank" #"lexrank"

METODO_SIMILARIDADE = "BM25"  # "COS" ou "BM25"

BASE = Path("data")

APPEALS_ORIG = BASE / "appeals/notClean/texto/special_appeal_teste.csv"
TEMAS_ORIG   = BASE / "temas/notClean/texto/temas_repetitivos.csv"

APPEALS_CLEAN = BASE / "appeals/clean/texto/special_appeal_teste.csv"
TEMAS_CLEAN   = BASE / "temas/clean/texto/temas_repetitivos.csv"

K = 6

# =====================================================
# UTIL
# =====================================================

def run(cmd):
    subprocess.run([PYTHON] + cmd[1:], check=True)

def sanitize(nome):
    return nome.replace("/", "_").replace("-", "_")

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def executar_ou_carregar(output_path: Path, gerar_func):

    if output_path.exists():
        print(f"[CACHE] Usando existente: {output_path}")
        return output_path

    print(f"[GERANDO] {output_path}")
    gerar_func()

    return output_path

# =====================================================
# PASSO 1 — CLEAN
# =====================================================

def step_clean():
    if not USAR_CLEAN:
        return APPEALS_ORIG, TEMAS_ORIG

    print(">> CLEAN")

    APPEALS_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    TEMAS_CLEAN.parent.mkdir(parents=True, exist_ok=True)

    def gerar_appeals():
        run([
            PYTHON, "limparTexto.py",
            "--input", str(APPEALS_ORIG),
            "--output", str(APPEALS_CLEAN),
            "--text_column", "special_appeal_text"
        ])

    def gerar_temas():
        run([
            PYTHON, "limparTexto.py",
            "--input", str(TEMAS_ORIG),
            "--output", str(TEMAS_CLEAN),
            "--text_column", "theme_text"
        ])

    executar_ou_carregar(APPEALS_CLEAN, gerar_appeals)
    executar_ou_carregar(TEMAS_CLEAN, gerar_temas)

    return APPEALS_CLEAN, TEMAS_CLEAN

# =====================================================
# EMBEDDINGS
# =====================================================

def step_embeddings(input_path, tipo):

    modelo_sanit = sanitize(MODELO)

    pasta = "appeals" if tipo == "appeals" else "temas"
    clean_flag = "clean" if USAR_CLEAN else "notClean"

    out_dir = BASE / pasta / clean_flag / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    nome = input_path.stem
    output = out_dir / f"embedding_{modelo_sanit}__{nome}.csv"

    def gerar():
        run([
            PYTHON, "gerarEmbedding.py",
            "--input", str(input_path),
            "--model", MODELO,
            "--output_dir", str(out_dir)
        ])

    return executar_ou_carregar(output, gerar)

# =====================================================
# RESUMO
# =====================================================

def step_resumo(input_path, temas_path):

    if not USAR_RESUMO:
        return input_path

    print(">> RESUMO")

    clean_flag = "clean" if USAR_CLEAN else "notClean"
    out_dir = BASE / "appeals" / clean_flag / "resumos"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = out_dir / f"resumo_{RESUMO_ESTRATEGIA}_{RESUMO_TAMANHO}.csv"

    def gerar():
        run([
            PYTHON, "gerarResumos.py",
            "--input", str(input_path),
            "--temas", str(temas_path),
            "--output_dir", str(out_dir),
            "--size", str(RESUMO_TAMANHO),
            "--strategy", RESUMO_ESTRATEGIA,
            "--model", MODELO
        ])

    return executar_ou_carregar(output, gerar)

# =====================================================
# MÉTRICAS
# =====================================================

def calcular_metricas(similaridades, labels, k=6):

    rankings = np.argsort(-similaridades, axis=1)[:, :k]

    recall = np.mean([labels[i] in rankings[i] for i in range(len(labels))])

    ap = []
    ndcg = []

    for i in range(len(labels)):
        ranking = rankings[i]
        if labels[i] in ranking:
            pos = np.where(ranking == labels[i])[0][0] + 1
            ap.append(1/pos)
            ndcg.append(1/np.log2(pos+1))
        else:
            ap.append(0)
            ndcg.append(0)

    map_k = np.mean(ap)
    ndcg_k = np.mean(ndcg)

    f1 = 2*(map_k*recall)/(map_k+recall) if (map_k+recall)>0 else 0

    return recall, map_k, ndcg_k, f1

# =====================================================
# PRINT FORMATADO
# =====================================================

def print_resultados(recall, map_k, ndcg_k, f1, tf):

    print("\n=== RESULTADOS ===")
    print(f"Método: {METODO_SIMILARIDADE}")
    print(f"Clean: {USAR_CLEAN}")
    print(f"Resumo: {USAR_RESUMO}")

    print("\nRecall@6 : {:.4f} → % de vezes que o tema correto aparece no top-6".format(recall))
    print("MAP@6    : {:.4f} → qualidade do ranking (posição do correto)".format(map_k))
    print("NDCG@6   : {:.4f} → ranking com peso logarítmico".format(ndcg_k))
    print("F1-score : {:.4f} → equilíbrio entre recall e ranking".format(f1))

    # =====================================================
    # SALVAR NO CSV
    # =====================================================

    os.makedirs(os.path.dirname(RESULTADOS_CSV), exist_ok=True)
    arquivo_existe = os.path.isfile(RESULTADOS_CSV)

    with open(RESULTADOS_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Cabeçalho (somente na primeira vez)
        if not arquivo_existe:
            writer.writerow([
                "timestamp",
                "MODELO",
                "CLEAN",
                "RESUMO",
                "RESUMO_TAMANHO",
                "RESUMO_ESTRATEGIA",
                "METODO_SIMILARIDADE",
                "recall@6",
                "map@6",
                "ndcg@6",
                "f1_score",
                "tempo"
            ])

        # Linha do experimento
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            MODELO,
            USAR_CLEAN,
            USAR_RESUMO,
            RESUMO_TAMANHO if USAR_RESUMO else None,
            RESUMO_ESTRATEGIA if USAR_RESUMO else None,
            METODO_SIMILARIDADE,
            round(recall, 6),
            round(map_k, 6),
            round(ndcg_k, 6),
            round(f1, 6),
            round(tf, 6)
        ])

    print(f"\nResultado salvo em: {RESULTADOS_CSV}")

# =====================================================
# MAIN
# =====================================================

def main():

    t0 = time.time()

    appeals_path, temas_path = step_clean()

    # =====================
    # RESUMO 
    # =====================
    if USAR_RESUMO:
        appeals_path = step_resumo(appeals_path, temas_path)

    df_app = pd.read_csv(appeals_path)
    df_tema = pd.read_csv(temas_path)

    # =====================
    # COSINE
    # =====================
    if METODO_SIMILARIDADE == "COS":

        emb_app_path = step_embeddings(appeals_path, "appeals")
        emb_tema_path = step_embeddings(temas_path, "temas")

        emb_app = pd.read_csv(emb_app_path).iloc[:,1:].values
        emb_tema = pd.read_csv(emb_tema_path).iloc[:,1:].values

        map_idx = {t:i for i,t in enumerate(df_tema["theme_id"])}
        labels = np.array([map_idx[x] for x in df_app["theme_id"]])

        sim = cosine_similarity(emb_app, emb_tema)

        tf = time.time() - t0
        recall, map_k, ndcg_k, f1 = calcular_metricas(sim, labels, K)

        print_resultados(recall, map_k, ndcg_k, f1, tf)

    # =====================
    # BM25
    # =====================
    elif METODO_SIMILARIDADE == "BM25":

        temas_tok = [remove_punctuation(t).split() for t in df_tema["theme_text"]]
        bm25 = BM25Okapi(temas_tok)

        textos_app = df_app.iloc[:,1].tolist()

        sim = np.array([bm25.get_scores(t.split()) for t in textos_app])

        map_idx = {t:i for i,t in enumerate(df_tema["theme_id"])}
        labels = np.array([map_idx[x] for x in df_app["theme_id"]])

        tf = time.time() - t0
        recall, map_k, ndcg_k, f1 = calcular_metricas(sim, labels, K)

        print_resultados(recall, map_k, ndcg_k, f1, tf)

    else:
        raise ValueError("Método inválido: use COS ou BM25")

    print("\nTempo total:", round(time.time()-t0, 2), "segundos")


if __name__ == "__main__":
    main()