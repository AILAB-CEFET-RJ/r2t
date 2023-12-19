import nltk
from nltk.corpus import stopwords
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

verbose = False


def select_important(doc,mark):
    doc = doc.lower()
    match = re.search(fr' {mark}[^\n]*',doc)
    if match:
        start = match.start()
        final_doc = doc[start:]
    else:
        final_doc = doc
    return final_doc    
        #Palavras irrelevantes
    
def clean_text(text):
    stop_words = []
    match_stops = []
    match_patterns = []
    #print("###############################TEXTO ORIGINAL ######################3#")
    #print(text)
    text_cleaned = text.lower()
    stop_words_nltk = set(stopwords.words('portuguese'))
    stop_words.append(list(stop_words_nltk))
    stop_words.append(["nº","cep","telefone","rua","avenida","endereço","fax","fones"])
    stop_words.append(["egrégia","egrégio","eg","e.g."])
    stop_words.append(["copy","reg","trade","ldquo","rdquo","lsquo","rsquo","bull","middot","sdot","ndash","mdash","cent","pound"])
    stop_words.append(["euro","ne","frac12","frac14","frac34","deg","larr","rarr","uarr","darr","egrave","eacute","ccedil","hellip"])
    match_pattern = [r'\b_+(?:\d+|[a-zA-Z]+)?\b',r'https?://\S+',r'www\.\S+',r'\S+@\S+',r'^\d{3}.\d{3}.\d{3}-\d{2}$',r'^\d{2}\.\d{3}\.\d{3}\/\d{4}\-\d{2}$',r'\d{2}/\d{2}/\d{4}[ ,]',r'procuradoria regional (federal|da união) da \d+[ªa] região',r'tribunal regional federal( da)? \d+[ªa] região',r'advocacia( -)?geral da união',r'(excelentíssimo|senhor|vice-presidente|desembargador|\(a\))',r'procuradoria( -)?geral federal',r'escritório de advocacia',r'(superior )?tribunal de justiça',r'supremo tribunal federal',r'fones',r'fax',r'tribunal regional federal']

    stop_words_flat = [word for sublist in stop_words for word in sublist]
    match_pattern.extend([rf'\b{re.escape(palavra)}\b' for palavra in stop_words_flat])
    subs = [''] * len(match_stops)
    for match_pattern, subs in zip(match_pattern,subs):
        #print(match_pattern)
        text_cleaned = re.sub(match_pattern,subs,text_cleaned)
    #print("################################TEXTO LIMPO #################################")
    #print(text_cleaned)
    return text_cleaned
    
def process_corpus(path,clean,begin,column):
    try:
        resp = pd.read_csv(path)
        size = len(resp)
        docs = []
        num_cadastrado = []
        indice = []
        text = ""
        print("############### Lendo registros do corpus ###############")
        for i, linha in resp.iterrows():
            #Exibe percentual concluido
            sys.stdout.write("\r ")  # \r faz o cursor retroceder ao início da linha
            sys.stdout.write(f' Percentual concluído: {i/size*100:.2f}%')
            sys.stdout.flush()  # Força a impressão imediata
            tipo = type(linha[column])
            text = linha[column]
            try:
                #tratamento necessário, textos estavam sendo identificados como float
                if (tipo == str):
                    #Se houver um marco de trecho relevante do texto
                    if begin:
                        text = select_important(text,begin)
                        #print(text)
                    #Se tiver de limpar o texto
                    if clean:
                        text_cleaned = clean_text(text)
                        num_cadastrado.append(int(linha['num_tema_cadastrado']))
                        #print(text_cleaned)
                        docs.append(text_cleaned)
                        indice.append(i)     
                    else:
                        num_cadastrado.append(int(linha['num_tema_cadastrado']))
                        docs.append(text)
                        indice.append(i)

            except Exception as erro:
                print(f"Erro ao capturar numero de tema cadastrado {i} : {erro}")
                continue
                
        corpus = pd.DataFrame()
        corpus["indice"]=indice
        corpus["num_tema_cadastrado"]=num_cadastrado
        corpus[column]=docs 
        
    except FileNotFoundError:
        print(f"Error: File {path} not found")
    except PermissionError:
        print(f"Error: Permission denied for {path}")
    except pd.errors.EmptyDataError:
        print(f"Error: The path '{path}' is empty.")
    except pd.errors.ParserError as e:
        print(f"Error processing file: {e}")
        
    return corpus


def createFileName(name,clean):
    if clean:
        return f'{name}_EMBEDDING_CLEAN.pkl'
    else:
        return f'{name}_EMBEDDING.pkl'
    
def create_embedding(file,indice,corpus,num,model,v,c):
    sentence_model = SentenceTransformer(model)
    nameEmbedding = createFileName(file.name.split('.')[0],c)
    #verbose
    if v:
        corpus_embedding = sentence_model.encode(corpus,show_progress_bar=True)
    else:
        corpus_embedding = sentence_model.encode(corpus,show_progress_bar=False)
    
    with open(nameEmbedding, "wb") as fOut:
            pickle.dump({'indice':indice,'sentences': corpus,'numTema':num ,'embeddings': corpus_embedding}, fOut,protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Embedding salvo no arquivo {nameEmbedding}")   
    
def main(args):
    print("############### PROGRAMA DE GERAÇÃO DE EMBEDDINGS ###############")
    print("############### Configuração ###############")
    print(f"Sentence-BERT Model : {args.model}")
    print(f"Remoção de stopwords : {args.clean}")
    if args.begin_point:
        print(f"Considerar texto após primeira ocorrência da palavra: {args.begin_point}")
    tempo_inicio = time.time()
    print("\n\n############### Fazendo download de dependências ... ###############")
    nltk.download('stopwords')
    nltk.download('punkt')
    verbose = args.verbose
    corpus = process_corpus(args.corpus_csv_file,args.clean,args.begin_point,args.column)
    
    #Se não houver uma coluna indice, cria uma
    if 'indice' not in corpus.columns:
        corpus['indice'] = range(1,len(corpus)+1)
    create_embedding(args.corpus_csv_file,corpus['indice'],corpus[args.column],corpus['num_tema_cadastrado'],args.model,args.verbose,args.clean)
    
    print("Salvando log ...")
    tempo_fim = time.time()
    tempo_total_segundos = tempo_fim - tempo_inicio
    minutos, segundos = divmod(int(tempo_total_segundos), 60)
    if args.clean:
        log = 'log_corpus_embedding_clean.txt'
        with open(log, 'w') as arquivo:
            # Escrevendo os dados no arquivo
            arquivo.write("############### Criação de Embedding do Corpus ###############")
            arquivo.write(f"Sentence-BERT Model : {args.model}")
            arquivo.write("Remoção de stopwords : Sim")
            arquivo.write(f"Tempo total de execução: {minutos} minutos e {segundos} segundos")
    else:
        log = 'log_corpus_embedding.txt'
        with open(log, 'w') as arquivo:
            # Escrevendo os dados no arquivo
            arquivo.write("############### Criação de Embedding do Corpus ###############")
            arquivo.write(f"Sentence-BERT Model : {args.model}")
            arquivo.write("Remoção de stopwords : Não")
            arquivo.write(f"Tempo total de execução: {minutos} minutos e {segundos} segundos")
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Generate text embedding using Sentence-BERT model')
    parser.add_argument('corpus_csv_file', type=argparse.FileType('r'), help='File containing the corpus')
    parser.add_argument('column',help='Column that contains the text to be transformed into embedding')
    parser.add_argument('model',default='distiluse-base-multilingual-cased-v1',nargs='?', help='The Sentence-BERT model used to generate embedding : Default = distiluse-base-multilingual-cased-v1')
    parser.add_argument('--clean',action='store_true',help='Remove stopwords before creating embedding')
    parser.add_argument('--begin_point',help='Word that marks the beginning of essential part of the text')
    parser.add_argument('-v','--verbose',action='store_true',help='Increase the verbosity level')
    args = parser.parse_args()
    main(args)
    
    
    
