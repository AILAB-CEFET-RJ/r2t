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
# BAYESIAN DIRICHLET SUMMARY
# =====================================================

def bayesian_dirichlet_summary(
    sentences,
    model,
    bm25,
    summary_size,
    alpha_prior=10.0,
    sim_threshold=0.0
):
    """
    Sumarização extrativa via inferência bayesiana com prior Dirichlet.

    Modelo generativo
    -----------------
    Interpreta cada par de sentenças (i, j) acima do threshold como um
    "voto" que j emite para i, com peso proporcional à similaridade semântica.
    O vetor de votos acumulados c é tratado como a realização de uma
    Multinomial(N, θ), onde θ_i é a importância latente da sentença i.

    Especificação bayesiana
    -----------------------
      Prior:      θ ~ Dirichlet(α · p)
      Likelihood: c | θ ~ Multinomial(N, θ)
      Posterior:  θ | c ~ Dirichlet(α · p + c)   ← conjugação exata

    onde:
      p  = vetor de personalização derivado do BM25 em relação aos temas
           (normalizado no simplex → interpretação probabilística direta)
      α  = concentração do prior (peso relativo da personalização vs dados)
      c  = contagens de votos recebidos por cada sentença, derivadas da
           matriz de similaridade com threshold para esparsidade

    Estimador pontual
    -----------------
      E[θ_i | c] = (α·p_i + c_i) / Σ_j (α·p_j + c_j)

    Parâmetros
    ----------
    sentences     : lista de sentenças do documento
    model         : SentenceTransformer para codificar as sentenças
    bm25          : instância de BM25Okapi indexada sobre os temas
    summary_size  : número de sentenças a retornar
    alpha_prior   : concentração do prior Dirichlet (padrão: 10.0)
                    valores altos → prior domina; baixos → dados dominam
    sim_threshold : limiar mínimo de similaridade para computar voto
                    (padrão: 0.0, sem filtro; use ~0.1 para esparsidade)
    """
    if len(sentences) <= summary_size:
        return " ".join(sentences)

    # --------------------------------------------------
    # 1. PRIOR: vetor de personalização via BM25
    #    p_i = relevância da sentença i em relação aos temas,
    #    normalizada no simplex para interpretação probabilística.
    # --------------------------------------------------
    bm25_scores = np.array([
        bm25.get_scores(s.split()).max()
        for s in sentences
    ], dtype=float)

    # Garante não-negatividade antes de normalizar no simplex
    bm25_scores = np.clip(bm25_scores, 0.0, None)
    total = bm25_scores.sum()
    if total > 1e-12:
        prior_p = bm25_scores / total
    else:
        # Prior uniforme quando BM25 não discrimina
        prior_p = np.ones(len(sentences)) / len(sentences)

    # --------------------------------------------------
    # 2. VEROSSIMILHANÇA: contagens de votos (evidência)
    #    Cada par (j → i) com sim(j, i) > threshold contribui
    #    com sim(j, i) como contagem fracionária para c_i.
    #    Isso é a estatística suficiente da multinomial.
    # --------------------------------------------------
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # Zera diagonal (sentença não vota em si mesma) e aplica threshold
    np.fill_diagonal(cos_scores, 0.0)
    cos_scores[cos_scores < sim_threshold] = 0.0

    # c_i = soma dos votos recebidos por sentença i de todas as outras
    # (soma sobre colunas da transposta = soma sobre linhas para cada i)
    vote_counts = cos_scores.sum(axis=0)  # shape: (n,)

    # --------------------------------------------------
    # 3. ATUALIZAÇÃO BAYESIANA (conjugação Dirichlet-Multinomial)
    #    posterior_params = α·p + c
    #    θ | c ~ Dirichlet(posterior_params)
    # --------------------------------------------------
    posterior_params = alpha_prior * prior_p + vote_counts

    # --------------------------------------------------
    # 4. ESTIMADOR PONTUAL: esperança da Dirichlet
    #    E[θ_i] = posterior_params_i / Σ_j posterior_params_j
    # --------------------------------------------------
    theta = posterior_params / posterior_params.sum()

    # --------------------------------------------------
    # 5. RANKING E SELEÇÃO
    # --------------------------------------------------
    ranked_indices = np.argsort(-theta)[:summary_size]
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
        choices=[
            "lexrank",
            "guided_lexrank",
            "guided_lexrank_opt",
            "personalized_pagerank",
            "bayesian_dirichlet",
        ],
        required=True,
        help="Estratégia de resumo"
    )

    parser.add_argument(
        "--model",
        default='distiluse-base-multilingual-cased-v1',
        required=False,
        help="modelo para o lex rank"
    )

    parser.add_argument(
        "--alpha_prior",
        type=float,
        default=10.0,
        required=False,
        help=(
            "Concentração do prior Dirichlet para bayesian_dirichlet. "
            "Valores altos dão mais peso à personalização BM25; "
            "valores baixos deixam a estrutura do corpus dominar. (padrão: 10.0)"
        )
    )

    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.0,
        required=False,
        help=(
            "Threshold mínimo de similaridade para contabilizar votos "
            "no bayesian_dirichlet. Use ~0.1 para esparsidade. (padrão: 0.0)"
        )
    )

    args = parser.parse_args()

    # Valida dependência de --temas
    strategies_needing_temas = {
        "guided_lexrank",
        "guided_lexrank_opt",
        "personalized_pagerank",
        "bayesian_dirichlet",
    }
    if args.strategy in strategies_needing_temas and args.temas is None:
        raise ValueError(
            f"Para a estratégia '{args.strategy}', "
            "o CSV de temas é obrigatório (--temas)"
        )

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
    df = pd.read_csv(args.input, encoding='latin1')

    if args.temas is not None:
        temas = pd.read_csv(args.temas, encoding='latin1')
    else:
        temas = None

    if not {"theme_id", "special_appeal_text"}.issubset(df.columns):
        raise ValueError(
            "CSV deve conter as colunas: theme_id, special_appeal_text"
        )

    # -------------------------------------------------
    # Modelo
    # -------------------------------------------------
    MODEL_NAME = args.model
    model = SentenceTransformer(MODEL_NAME)

    if args.temas is not None:
        # -------------------------------------------------
        # BM25 global
        # -------------------------------------------------
        bm25 = None
        retriever = None

        if args.strategy in {"guided_lexrank", "personalized_pagerank", "bayesian_dirichlet"}:
            themes = [row['theme_text'] for _, row in temas.iterrows()]
            tokenized_themes = [theme.split(" ") for theme in themes]
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
        elif args.strategy == "bayesian_dirichlet":
            resumo = bayesian_dirichlet_summary(
                sentences,
                model,
                bm25,
                args.size,
                alpha_prior=args.alpha_prior,
                sim_threshold=args.sim_threshold
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