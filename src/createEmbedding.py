import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np
import pandas as pd
import time
import csv
import argparse
import sys
import string
import torch

verbose = False

def extract_relevant_text(doc, marker):
    doc = doc.lower()
    match = re.search(fr' {marker}[^\n]*', doc)
    if match:
        start = match.start()
        final_doc = doc[start:]
    else:
        final_doc = doc
    return final_doc

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
    
def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(["nº", "cep", "telefone", "rua", "avenida", "endereço", "fax", "fones"])
    stop_words.update(["egrégia", "egrégio", "eg", "e.g."])                   
    stop_words.update(["copy", "reg", "trade", "ldquo", "rdquo", "lsquo", "rsquo", "bull", "middot", "sdot", "ndash"])
    stop_words.update(["mdash", "cent", "pound", "euro", "ne", "frac12", "frac14", "frac34", "deg", "larr", "rarr", "uarr", "darr", "egrave", "eacute", "ccedil", "hellip"])
    
    tokens = word_tokenize(text, language='portuguese')
    tokens_cleaned = [token for token in tokens if token not in stop_words]
    detokenizer = TreebankWordDetokenizer()
    return detokenizer.detokenize(tokens_cleaned)

def clean_text(doc):
    match_patterns = [
        r',\s*,', r'\bpágina\s+(\d+)\s+(\d+)\b', r'\bpágina\s+(\d+)\s+de\s+(\d+)\b', r'\?', r'\b_+(?:\d+|[a-zA-Z]+)?\b',
        r'https?://\S+', r'www\.\S+', r'\S+@\S+', r'^\d{3}.\d{3}.\d{3}-\d{2}$', r'^\d{2}\.\d{3}\.\d{3}\/\d{4}\-\d{2}$',
        r'\d{2}/\d{2}/\d{4}[ ,]', r'\bprocuradoria regional (federal|da união) da \d+[ªa] região\b',
        r'\btribunal regional federal( da) \d+[ªa] região\b', r'\badvocacia( -)geral da união\b',
        r'\b(excelentíssimo|senhor|vice-presidente|desembargador|\(a\))\b', r'\bprocuradoria[ -]geral federal\b',
        r'\bescritório de advocacia\b', r'\b(superior) tribunal de justiça\b', r'\bsupremo tribunal federal\b',
        r'\bfones\b', r'\bfax\b'
    ]
    
    final_doc = doc
    for pattern in match_patterns:
        final_doc = re.sub(pattern, '', final_doc)
    
    return remove_stopwords(final_doc)

def process_corpus(file_path, clean, begin_point, column):
    try:
        data = pd.read_csv(file_path)
        size = len(data)
        docs, registered_ids, indices = [], [], []
        print("############### Reading corpus records ###############")
        
        for i, row in data.iterrows():
            sys.stdout.write(f'\r Progress: {i/size*100:.2f}%')
            sys.stdout.flush()
            text = row[column]
            try:
                if isinstance(text, str):
                    if begin_point:
                        text = extract_relevant_text(text, begin_point)
                    if clean:
                        text = clean_text(text)
                    
                    registered_ids.append(int(row['num_tema_cadastrado']))
                    docs.append(text)
                    indices.append(i)
            except Exception as error:
                print(f"Error processing record {i}: {error}")
                continue
                
        corpus_df = pd.DataFrame()
        corpus_df["index"] = indices
        corpus_df["num_tema_cadastrado"] = registered_ids
        corpus_df[column] = docs
        return corpus_df
        
    except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error: {e}")
        return None

def create_file_name(base_name, clean):
    return f'{base_name}_EMBEDDING_CLEAN.pkl' if clean else f'{base_name}_EMBEDDING.pkl'

def generate_embeddings(file, index, corpus, labels, model_name, verbose, clean, data_type):
    sentence_model = SentenceTransformer(model_name)
    embedding_file = create_file_name(file.name.split('.')[0], clean)
    
    if data_type == 'tema':
        print("############### Generating embeddings for topics ###############")
    
    if verbose:
        corpus_embeddings = sentence_model.encode(corpus, show_progress_bar=True)
    else:
        corpus_embeddings = sentence_model.encode(corpus, show_progress_bar=False)

    with open(embedding_file, "wb") as fOut:
        pickle.dump({'index': index, 'sentences': corpus, 'numTema': labels, 'embeddings': corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Embeddings saved to {embedding_file}")

def main(args):
    print("############### EMBEDDING GENERATION PROGRAM ###############")
    print(f"Sentence-BERT Model: {args.model}")
    print(f"Removing stopwords: {args.clean}")
    
    if args.begin_point:
        print(f"Processing text after the first occurrence of: {args.begin_point}")
    
    start_time = time.time()
    nltk.download('stopwords')
    nltk.download('punkt')
    
    corpus = process_corpus(args.corpus_csv_file, args.clean, args.begin_point, args.column)
    
    if corpus is not None:
        content = corpus[args.column].tolist()
        generate_embeddings(args.corpus_csv_file, corpus['index'], content, corpus['num_tema_cadastrado'], args.model, args.verbose, args.clean, args.data_type)
        
        total_time = time.time() - start_time
        minutes, seconds = divmod(int(total_time), 60)
        
        log_file = 'log_embedding_clean.txt' if args.clean else 'log_embedding.txt'
        with open(log_file, 'w') as log:
            log.write("############### Corpus Embedding Creation ###############\n")
            log.write(f"Sentence-BERT Model: {args.model}\n")
            log.write(f"Removing stopwords: {'Yes' if args.clean else 'No'}\n")
            log.write(f"Total execution time: {minutes} minutes and {seconds} seconds\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text embeddings using Sentence-BERT')
    parser.add_argument('corpus_csv_file', type=argparse.FileType('r'), help='File containing the corpus')
    parser.add_argument('data_type', choices=['recurso', 'tema'], help='Indicates whether the data type is "recurso" or "tema"')
    parser.add_argument('column', help='Column with text to be converted into embeddings')
    parser.add_argument('model', default='distiluse-base-multilingual-cased-v1', nargs='?', help='Sentence-BERT model used for embedding generation')
    parser.add_argument('--clean', action='store_true', help='Remove stopwords before creating embeddings')
    parser.add_argument('--begin_point', help='Keyword marking the start of the relevant text')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity')
    
    args = parser.parse_args()
    main(args)
