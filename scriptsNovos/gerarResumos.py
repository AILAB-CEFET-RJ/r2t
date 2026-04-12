import argparse
import os
import pandas as pd
import numpy as np
import nltk
import bm25s

from sentence_transformers import SentenceTransformer, util
from LexRank import degree_centrality_scores
from rank_bm25 import BM25Okapi


from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize
import numpy as np

# =====================================================
# CONFIGURAÇÕES
# =====================================================
LEXRANK_THRESHOLD = 0.3
GUIDED_LEXRANK_THRESHOLD = 0.05

# Pesos Guided LexRank
ALPHA = 1.0
BETA = 3.0

# =====================================================
# LEXRANK CLÁSSICO
# =====================================================

def lexrank_summary(sentences, model, summary_size, threshold):
    if len(sentences) <= summary_size:
        return " ".join(sentences)

    embeddings = model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

    scores = degree_centrality_scores(
        cos_scores,
        threshold=threshold
    )

    ranked_indices = np.argsort(-scores)[:summary_size]
    return " ".join(sentences[i].strip() for i in ranked_indices)

# =====================================================
# GUIDED LEXRANK
# =====================================================

def guided_lexrank_summary(
    sentences,
    model,
    bm25,
    summary_size,
    threshold,
    alpha,
    beta
):
    if len(sentences) <= summary_size:
        return " ".join(sentences)

    # --- LexRank ---
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

    lexrank_scores = degree_centrality_scores(
        cos_scores,
        threshold=threshold
    )

    lr_min, lr_max = lexrank_scores.min(), lexrank_scores.max()
    if lr_max > lr_min:
        lexrank_norm = (lexrank_scores - lr_min) / (lr_max - lr_min)
    else:
        lexrank_norm = np.zeros_like(lexrank_scores)

    # --- BM25 ---
    bm25_scores = []
    for s in sentences:
        query = s.split() 
        doc_scores = bm25.get_scores(query)
        bm25_scores.append(doc_scores.max())

    bm25_scores = np.array(bm25_scores)
    bm_min, bm_max = bm25_scores.min(), bm25_scores.max()

    if bm_max > bm_min:
        bm25_norm = (bm25_scores - bm_min) / (bm_max - bm_min)
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # --- Score final ---
    final_scores = alpha * lexrank_norm + beta * bm25_norm
    ranked_indices = np.argsort(-final_scores)[:summary_size]

    return " ".join(sentences[i].strip() for i in ranked_indices)

def guided_lexrank_summary_opt(
    sentences,
    model,
    retriever,
    summary_size,
    threshold,
    alpha,
    beta
):
    """
    Guided LexRank usando BM25 da biblioteca bm25s.
    Cada sentença é pontuada pelo score máximo entre todos os temas.
    """
    if len(sentences) <= summary_size:
        return " ".join(sentences)

    # --- LexRank ---
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

    lexrank_scores = degree_centrality_scores(
        cos_scores,
        threshold=threshold
    )

    # Normalização LexRank
    lr_min, lr_max = lexrank_scores.min(), lexrank_scores.max()
    if lr_max > lr_min:
        lexrank_norm = (lexrank_scores - lr_min) / (lr_max - lr_min)
    else:
        lexrank_norm = np.zeros_like(lexrank_scores)

    # --- BM25 ---
    query_tokens = bm25s.tokenize(sentences)
    results, scores = retriever.retrieve(query_tokens, k=len(retriever.corpus))
    bm25_scores = np.max(scores, axis=1)

    bm25_scores = np.array(bm25_scores)
    bm_min, bm_max = bm25_scores.min(), bm25_scores.max()

    # Normalização BM25
    if bm_max > bm_min:
        bm25_norm = (bm25_scores - bm_min) / (bm_max - bm_min)
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # --- Score final ---
    final_scores = alpha * lexrank_norm + beta * bm25_norm
    ranked_indices = np.argsort(-final_scores)[:summary_size]

    return " ".join(sentences[i].strip() for i in ranked_indices)

def personalized_pagerank_summary(
    sentences,
    model,
    bm25,
    summary_size,
    alpha
):
    if len(sentences) <= summary_size:
        return " ".join(sentences)

    # =====================================================
    # 1. EMBEDDINGS
    # =====================================================
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # similaridade via embeddings
    cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # =====================================================
    # 2. MATRIZ ESTOCÁSTICA
    # =====================================================
    P = csr_matrix(cos_scores)
    P = normalize(P, norm='l1', axis=1)

    # =====================================================
    # 3. VETOR DE PERSONALIZAÇÃO (BM25)
    # =====================================================
    bm25_scores = []
    for s in sentences:
        query = s.split()
        doc_scores = bm25.get_scores(query)
        bm25_scores.append(doc_scores.max())

    bm25_scores = np.array(bm25_scores)

    # normalização (igual seu guided lexrank)
    bm_min, bm_max = bm25_scores.min(), bm25_scores.max()

    if bm_max > bm_min:
        pref_vector = (bm25_scores - bm_min) / (bm_max - bm_min)
    else:
        pref_vector = np.zeros_like(bm25_scores)

    pref_vector = pref_vector / (pref_vector.sum() + 1e-12)

    # =====================================================
    # 4. PERSONALIZED PAGERANK (SISTEMA LINEAR)
    # =====================================================
    n = len(sentences)

    A = eye(n) - (1 - alpha) * P.T
    b = alpha * pref_vector

    pr = spsolve(A, b)

    # =====================================================
    # 5. RANKING FINAL
    # =====================================================
    ranked_indices = np.argsort(-pr)[:summary_size]

    return " ".join(sentences[i].strip() for i in ranked_indices)

# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Geração de resumos com LexRank ou Guided LexRank"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV de entrada no formato theme_id,special_appeal_text"
    )

    parser.add_argument(
        "--temas",
        required=False,
        help="CSV de temas no formato theme_id,theme_text"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Diretório onde os resumos serão salvos"
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Tamanho do resumo (número de sentenças)"
    )

    parser.add_argument(
        "--strategy",
        choices=["lexrank", "guided_lexrank", "guided_lexrank_opt", "personalized_pagerank"],
        required=True,
        help="Estratégia de resumo"
    )

    parser.add_argument(
        "--model",
        default = 'distiluse-base-multilingual-cased-v1',
        required=False,
        help="modelo para o lex rank"
    )

    args = parser.parse_args()

    if args.strategy == "guided_lexrank" and args.temas is None:
        raise ValueError("Para Guided LexRank, o CSV de temas é obrigatório (--temas)")

    # -------------------------------------------------
    # Criar diretório de saída
    # -------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(
        args.output_dir,
        f"resumo_{args.strategy}_{args.size}.csv"
    )

    # -------------------------------------------------
    # Setup NLTK
    # -------------------------------------------------
    nltk.download("punkt", quiet=True)

    # -------------------------------------------------
    # Leitura dos dados
    # -------------------------------------------------
    df = pd.read_csv(args.input)
    temas = pd.read_csv(args.temas)

    if not {"theme_id", "special_appeal_text"}.issubset(df.columns):
        raise ValueError(
            "CSV deve conter as colunas: theme_id, special_appeal_text"
        )

    # -------------------------------------------------
    # Modelo
    # -------------------------------------------------
    MODEL_NAME = args.model
    model = SentenceTransformer(MODEL_NAME)

    # -------------------------------------------------
    # BM25 global (apenas Guided)
    # -------------------------------------------------
    bm25 = None
    if args.strategy == "guided_lexrank" or args.strategy == "personalized_pagerank":
        themes = [row['theme_text'] for _, row in temas.iterrows()]
        tokenized_themes = [theme.split() for theme in themes] 
        bm25 = BM25Okapi(tokenized_themes)
    
    if args.strategy == "guided_lexrank_opt":
        themes = [row['theme_text'] for _, row in temas.iterrows()]
        retriever = bm25s.BM25(corpus=themes)
        retriever.index(bm25s.tokenize(themes))

    # -------------------------------------------------
    # Processamento
    # -------------------------------------------------
    resultados = []

    for idx, row in df.iterrows():
        theme_id = row["theme_id"]
        text = str(row["special_appeal_text"])

        sentences = nltk.sent_tokenize(text, language="portuguese")

        if args.strategy == "lexrank":
            resumo = lexrank_summary(
                sentences,
                model,
                args.size,
                LEXRANK_THRESHOLD
            )
        elif args.strategy == "guided_lexrank":
            resumo = guided_lexrank_summary(
                sentences,
                model,
                bm25,
                args.size,
                GUIDED_LEXRANK_THRESHOLD,
                ALPHA,
                BETA
            )
        elif args.strategy == "guided_lexrank_opt":
            resumo = guided_lexrank_summary_opt(
                sentences,
                model,
                retriever,
                args.size,
                GUIDED_LEXRANK_THRESHOLD,
                ALPHA,
                BETA
            )
        elif args.strategy == "personalized_pagerank":
            resumo = personalized_pagerank_summary(
                sentences,
                model,
                bm25,
                args.size,
                0.15
            )

        resultados.append({
            "theme_id": theme_id,
            "special_appeal_text": resumo
        })

        if (idx + 1) % 100 == 0:
            print(f"Processados {idx + 1}/{len(df)}")

    # -------------------------------------------------
    # Salvar CSV
    # -------------------------------------------------
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(output_csv, index=False)

    print(f"\nResumos salvos em: {output_csv}")

if __name__ == "__main__":
    main()
