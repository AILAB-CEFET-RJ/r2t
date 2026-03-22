import argparse
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os

# ========= FUNÇÕES DO PIPELINE ========= #

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

    args = parser.parse_args()

    # -------------------------------------------------
    # Setup NLTK
    # -------------------------------------------------
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # -------------------------------------------------
    # Leitura
    # -------------------------------------------------
    df = pd.read_csv(args.input)

    if args.text_column not in df.columns:
        raise ValueError(
            f"Coluna '{args.text_column}' não encontrada no CSV"
        )

    # -------------------------------------------------
    # Limpeza
    # -------------------------------------------------
    df_out = df.copy()
    df_out[args.text_column] = df_out[args.text_column].astype(str).apply(clean_text)

    # -------------------------------------------------
    # Salvar
    # -------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df_out.to_csv(args.output, index=False)

    print(f"\nArquivo limpo salvo em: {args.output}")
    print(f"Coluna processada: {args.text_column}")
    print(f"Total de registros: {len(df_out)}")

# ========= ENTRY POINT ========= #

if __name__ == "__main__":
    main()
