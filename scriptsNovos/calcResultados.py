import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import string
import itertools

# =====================================================
# CONFIGURAÇÕES
# =====================================================

BASE = Path("data")
K = 6

RESULTADOS_CSV = "resultados_avaliacao_distiluse-base-multilingual-cased-v1.csv"

# =====================================================
# MÉTRICAS (EXATAMENTE COMO TEMPLATE)
# =====================================================

def calcular_metricas(similaridades, labels_corretos, k=6):

    n_recursos = similaridades.shape[0]
    rankings = np.argsort(-similaridades, axis=1)[:, :k]

    acertos = 0
    for i in range(n_recursos):
        if labels_corretos[i] in rankings[i]:
            acertos += 1

    recall_at_k = acertos / n_recursos

    ap_scores = []
    for i in range(n_recursos):
        tema_correto = labels_corretos[i]
        ranking = rankings[i]

        if tema_correto in ranking:
            posicao = np.where(ranking == tema_correto)[0][0] + 1
            precision_at_pos = 1.0 / posicao
            ap_scores.append(precision_at_pos)
        else:
            ap_scores.append(0.0)

    map_at_k = np.mean(ap_scores)

    ndcg_scores = []
    for i in range(n_recursos):
        tema_correto = labels_corretos[i]
        ranking = rankings[i]

        if tema_correto in ranking:
            posicao = np.where(ranking == tema_correto)[0][0] + 1
            dcg = 1.0 / np.log2(posicao + 1)
            idcg = 1.0 / np.log2(2)
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    ndcg_at_k = np.mean(ndcg_scores)

    precision = map_at_k
    if (precision + recall_at_k) > 0:
        f1_score = 2 * (precision * recall_at_k) / (precision + recall_at_k)
    else:
        f1_score = 0.0

    return {
        "recall@6": recall_at_k,
        "map@6": map_at_k,
        "ndcg@6": ndcg_at_k,
        "f1_score": f1_score
    }

# =====================================================
# BM25 (IGUAL TEMPLATE)
# =====================================================

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def calcular_similaridade_bm25(topics, themes):

    cleaned_themes = [remove_punctuation(t) for t in themes]
    tokenized_themes = [t.split(" ") for t in cleaned_themes]

    bm25 = BM25Okapi(tokenized_themes)

    n_topics = len(topics)
    n_themes = len(themes)

    similaridade_matrix = np.zeros((n_topics, n_themes), dtype=float)

    for i, topic in enumerate(topics):
        cleaned_topic = topic.replace("-", "")
        tokenized_topic = cleaned_topic.split(" ")
        scores = bm25.get_scores(tokenized_topic)
        similaridade_matrix[i, :] = scores

    return similaridade_matrix

# =====================================================
# UTIL
# =====================================================

def carregar_embeddings(path):

    df = pd.read_csv(path)
    ids = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype(float)

    return ids, embeddings

def construir_labels(df_recursos, df_temas):

    theme_id_to_idx = {tid: idx for idx, tid in enumerate(df_temas['theme_id'])}
    labels = np.array([theme_id_to_idx[tid] for tid in df_recursos['theme_id']])

    return labels

# =====================================================
# PASSO 1 — COSINE COM EMBEDDINGS
# =====================================================

def avaliar_cosine():

    resultados = []

    for clean_flag in ["clean", "notClean"]:

        print(f"\nAvaliando {clean_flag}")

        appeals_embed_dir = BASE / "appeals" / clean_flag / "embeddings"
        temas_embed_dir   = BASE / "temas"   / clean_flag / "embeddings"

        # =============================
        # Indexar temas por modelo
        # =============================
        temas_por_modelo = {}

        for tema_file in temas_embed_dir.glob("embedding_*.csv"):

            if "__" not in tema_file.name:
                print(f"Formato inválido (tema): {tema_file.name}")
                continue

            modelo = tema_file.name.split("__")[0]  # ex: embedding_modelo
            temas_por_modelo[modelo] = tema_file

        if not temas_por_modelo:
            print("Nenhum tema indexado.")
            continue

        # =============================
        # Percorrer appeals
        # =============================
        encontrou_algum = False

        for appeal_file in appeals_embed_dir.glob("embedding_*.csv"):

            if "__" not in appeal_file.name:
                print(f"Formato inválido (appeal): {appeal_file.name}")
                continue

            modelo = appeal_file.name.split("__")[0]

            if modelo not in temas_por_modelo:
                continue

            encontrou_algum = True
            tema_file = temas_por_modelo[modelo]

            print(f"Comparando:\n   {appeal_file.name}\n   {tema_file.name}")

            tema_ids, tema_embeddings = carregar_embeddings(tema_file)
            appeal_ids, appeal_embeddings = carregar_embeddings(appeal_file)

            if appeal_embeddings.shape[1] != tema_embeddings.shape[1]:
                print(f"Dimensão incompatível: {appeal_file.name}")
                continue

            similaridades = cosine_similarity(
                appeal_embeddings,
                tema_embeddings
            )

            df_recursos = pd.DataFrame({"theme_id": appeal_ids})
            df_temas    = pd.DataFrame({"theme_id": tema_ids})
            labels = construir_labels(df_recursos, df_temas)

            metricas = calcular_metricas(similaridades, labels, k=K)

            resultados.append({
                "similaridade": "COS",
                "clean_flag": clean_flag,
                "modelo": modelo.replace("embedding_", ""),
                "arquivo_temas": tema_file.name,
                "arquivo_appeals": appeal_file.name,
                **metricas
            })

        if not encontrou_algum:
            print("Nenhum par compatível encontrado.")

    return resultados

# =====================================================
# PASSO 2 — BM25
# =====================================================

def avaliar_bm25():

    resultados = []

    for clean_flag in ["clean", "notClean"]:

        temas_texto_path = BASE / "temas" / clean_flag / "texto" / "temas_repetitivos.csv"
        df_temas = pd.read_csv(temas_texto_path)

        temas_textos = df_temas["theme_text"].tolist()

        # Texto original
        appeals_texto_path = BASE / "appeals" / clean_flag / "texto" / "special_appeal.csv"
        df_recursos_texto = pd.read_csv(appeals_texto_path)

        # Resumos
        resumo_dir = BASE / "appeals" / clean_flag / "resumos"

        arquivos_appeals = [(appeals_texto_path, df_recursos_texto)]

        for resumo_file in resumo_dir.glob("resumo_*.csv"):
            df_resumo = pd.read_csv(resumo_file)
            arquivos_appeals.append((resumo_file, df_resumo))

        for path_appeal, df_recursos in arquivos_appeals:

            textos_appeals = df_recursos.iloc[:, 1].tolist()

            similaridades = calcular_similaridade_bm25(
                textos_appeals,
                temas_textos
            )

            labels = construir_labels(df_recursos, df_temas)

            metricas = calcular_metricas(similaridades, labels, k=K)

            resultados.append({
                "similaridade": "BM25",
                "clean_flag": clean_flag,
                "arquivo_temas": temas_texto_path.name,
                "arquivo_appeals": path_appeal.name,
                **metricas
            })

    return resultados

# =====================================================
# MAIN
# =====================================================

def main():

    print("Avaliando COS...")
    resultados_cos = avaliar_cosine()

    print("Avaliando BM25...")
    #resultados_bm25 = avaliar_bm25()

    resultados_totais = resultados_cos #+ resultados_bm25

    df_resultados = pd.DataFrame(resultados_totais)
    df_resultados.to_csv(RESULTADOS_CSV, index=False)

    print("\nAvaliação concluída.")
    print(f"Resultados salvos em: {RESULTADOS_CSV}")

if __name__ == "__main__":
    main()
