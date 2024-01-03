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
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation
    
def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(["nº","cep","telefone","rua","avenida","endereço","fax","fones"])
    stop_words.update(["egrégia","egrégio","eg","e.g."])                   
    
    stop_words.update(["copy","reg","trade","ldquo","rdquo","lsquo","rsquo","bull","middot","sdot","ndash"])  
    stop_words.update(["mdash","cent","pound","euro","ne","frac12","frac14","frac34","deg","larr","rarr","uarr","darr","egrave","eacute","ccedil","hellip"])
    # Tokenizando o texto
    tokens = word_tokenize(text, language='portuguese')
    tokens_cleaned = [token for token in tokens if token not in stop_words]
    detokenizer = TreebankWordDetokenizer()
    detokenized_text = detokenizer.detokenize(tokens_cleaned)
    text_cleaned = ' '.join(tokens_cleaned)
    return text_cleaned

    
def clean_text(doc):
    final_doc = doc
    #Padrões irrelevantes
    match_pattern = [r',\s*,',r'\bpágina\s+(\d+)\s+(\d+)\b',r'\bpágina\s+(\d+)\s+de\s+(\d+)\b',r'\?',r'\b_+(?:\d+|[a-zA-Z]+)?\b',r'https?://\S+',r'www\.\S+',r'\S+@\S+',r'^\d{3}.\d{3}.\d{3}-\d{2}$',r'^\d{2}\.\d{3}\.\d{3}\/\d{4}\-\d{2}$',r'\d{2}/\d{2}/\d{4}[ ,]',r'\bprocuradoria regional (federal|da união) da \d+[ªa] região\b',r'\btribunal regional federal( da) \d+[ªa] região\b',r'\badvocacia( -)geral da união\b',r'\b(excelentíssimo|senhor|vice-presidente|desembargador|\(a\))\b',r'\bprocuradoria[ -]geral federal\b',r'\bescritório de advocacia\b',r'\b(superior) tribunal de justiça\b',r'\bsupremo tribunal federal\b',r'\bfones\b',r'\bfax\b']
    subs = [''] * len(match_pattern)
    for match_pattern, subs in zip(match_pattern,subs):
        final_doc = re.sub(match_pattern,subs,final_doc)
    
    final_doc = remove_stopwords(final_doc)
    
    return final_doc
    
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

def create_embedding(file,indice,corpus,num,model,v,c,data_type):
    
    sentence_model = SentenceTransformer(model)
    nameEmbedding = createFileName(file.name.split('.')[0],c)
    if(data_type == 'tema'):
        print("############### Gerando embeddings dos temas ###############")
        i = 0
        for linha in corpus:
            i = i+1
             #Exibe percentual concluido
            sys.stdout.write("\r ")  # \r faz o cursor retroceder ao início da linha
            sys.stdout.write(f' Percentual concluído: {i/len(corpus)*100:.2f}%')
            sys.stdout.flush()  # Força a impressão imediata
            tema_clean = remove_punctuation(linha)          
            tokenized_temas = tema_clean.split(" ")
        #verbose
        if v:
            corpus_embedding = sentence_model.encode(corpus,show_progress_bar=True)
        else:
            corpus_embedding = sentence_model.encode(corpus,show_progress_bar=False)

        with open(nameEmbedding, "wb") as fOut:
                pickle.dump({'indice':indice,'sentences': corpus,'numTema':num ,'embeddings': corpus_embedding}, fOut,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Embedding salvo no arquivo {nameEmbedding}")
    else:
        #verbose
        if v:
            corpus_embedding = sentence_model.encode(corpus,show_progress_bar=True)
        else:
            corpus_embedding = sentence_model.encode(corpus,show_progress_bar=False)

        with open(nameEmbedding, "wb") as fOut:
                pickle.dump({'indice':indice,'sentences': corpus,'numTema':num ,'embeddings': corpus_embedding}, fOut,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Embedding salvo no arquivo {nameEmbedding}")
    
def main(args):
    conteudo = []
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
    #if 'indice' not in corpus.columns:
        #corpus['indice'] = range(1,len(corpus)+1)

    conteudo = corpus[args.column].tolist()
    create_embedding(args.corpus_csv_file,corpus['indice'],conteudo,corpus['num_tema_cadastrado'],args.model,args.verbose,args.clean, args.data_type)
    
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
    parser.add_argument('data_type',choices=['recurso','tema'],help='Indicates whether the data type is recurso or tema')
    parser.add_argument('column',help='Column that contains the text to be transformed into embedding')
    parser.add_argument('model',default='distiluse-base-multilingual-cased-v1',nargs='?', help='The Sentence-BERT model used to generate embedding : Default = distiluse-base-multilingual-cased-v1')
    parser.add_argument('--clean',action='store_true',help='Remove stopwords before creating embedding')
    parser.add_argument('--begin_point',help='Word that marks the beginning of essential part of the text')
    parser.add_argument('-v','--verbose',action='store_true',help='Increase the verbosity level')
    args = parser.parse_args()
    main(args)
    
    
    
