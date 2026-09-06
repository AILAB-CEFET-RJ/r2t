import argparse
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os

from ioUtil import salvar_csv_atomico

# ========= FUNÇÕES DO PIPELINE ========= #

def extract_relevant_text(doc, marker):
    """
    Trunca o texto a partir da primeira ocorrência de ' <marker>'.
    Idêntico ao createEmbedding.py do pipeline original.
    Retorna o texto em minúsculas (o clean_text também faz lowercase,
    portanto não há perda de informação).
    """
    doc_lower = str(doc).lower()
    match = re.search(fr' {marker}[^\n]*', doc_lower)
    if match:
        return doc_lower[match.start():]
    return doc_lower


def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update([
        "nº", "cep", "telefone", "rua", "avenida", "endereço", "fax", "fones",
        "egrégia", "egrégio", "eg", "e.g.",
        "copy", "reg", "trade", "ldquo", "rdquo", "lsquo", "rsquo", "bull",
        "middot", "sdot", "ndash", "mdash", "cent", "pound", "euro", "ne",
        "frac12", "frac14", "frac34", "deg", "larr", "rarr", "uarr", "darr",
        "egrave", "eacute", "ccedil", "hellip"
    ])

    tokens = word_tokenize(text, language="portuguese")
    tokens_cleaned = [t for t in tokens if t not in stop_words]

    detokenizer = TreebankWordDetokenizer()
    return detokenizer.detokenize(tokens_cleaned)


def clean_text(doc):
    match_patterns = [
        r',\s*,',
        r'\bpágina\s+(\d+)\s+(\d+)\b',
        r'\bpágina\s+(\d+)\s+de\s+(\d+)\b',
        r'\?',
        r'\b_+(?:\d+|[a-zA-Z]+)?\b',
        r'https?://\S+',
        r'www\.\S+',
        r'\S+@\S+',
        r'^\d{3}.\d{3}.\d{3}-\d{2}$',
        r'^\d{2}\.\d{3}\.\d{3}\/\d{4}\-\d{2}$',
        r'\d{2}/\d{2}/\d{4}[ ,]',
        r'\bprocuradoria regional (federal|da união) da \d+[ªa] região\b',
        r'\btribunal regional federal( da) \d+[ªa] região\b',
        r'\badvocacia( -)geral da união\b',
        r'\b(excelentíssimo|senhor|vice-presidente|desembargador|\(a\))\b',
        r'\bprocuradoria[ -]geral federal\b',
        r'\bescritório de advocacia\b',
        r'\b(superior) tribunal de justiça\b',
        r'\bsupremo tribunal federal\b',
        r'\bfones\b',
        r'\bfax\b'
    ]

    final_doc = str(doc).lower()
    for pattern in match_patterns:
        final_doc = re.sub(pattern, '', final_doc)

    return remove_stopwords(final_doc)


# ========= SCRIPT ========= #

def main():
    parser = argparse.ArgumentParser(
        description="Limpeza de texto jurídico (stopwords + regex)"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV de entrada"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="CSV de saída"
    )

    parser.add_argument(
        "--text_column",
        required=True,
        help="Nome da coluna de texto a ser limpa"
    )

    parser.add_argument(
        "--begin_point",
        required=False,
        default=None,
        help=(
            "Palavra-chave que marca o início do trecho relevante. "
            "Quando fornecida, o texto é truncado a partir da primeira "
            "ocorrência de ' <begin_point>' (com espaço antes). "
            "Corresponde ao parâmetro --begin_point do pipeline original."
        )
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Setup NLTK
    # -------------------------------------------------
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # -------------------------------------------------
    # Leitura
    # -------------------------------------------------
    df = pd.read_csv(args.input, encoding='latin1')

    if args.text_column not in df.columns:
        raise ValueError(
            f"Coluna '{args.text_column}' não encontrada no CSV"
        )

    # -------------------------------------------------
    # Pré-corte (begin_point) — deve vir ANTES do clean_text,
    # exatamente como no createEmbedding.py original.
    # -------------------------------------------------
    df_out = df.copy()

    if args.begin_point:
        print(f"Aplicando begin_point='{args.begin_point}'")
        df_out[args.text_column] = df_out[args.text_column].apply(
            lambda x: extract_relevant_text(x, args.begin_point)
        )

    # -------------------------------------------------
    # Limpeza (regex + stopwords)
    # -------------------------------------------------
    df_out[args.text_column] = df_out[args.text_column].astype(str).apply(clean_text)

    # -------------------------------------------------
    # Salvar
    # -------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    salvar_csv_atomico(df_out, args.output, index=False, encoding='latin1')

    print(f"\nArquivo limpo salvo em: {args.output}")
    print(f"Coluna processada    : {args.text_column}")
    print(f"begin_point          : {args.begin_point}")
    print(f"Total de registros   : {len(df_out)}")


# ========= ENTRY POINT ========= #

if __name__ == "__main__":
    main()