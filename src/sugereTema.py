from sentence_transformers import SentenceTransformer, util
from abc import ABC, abstractmethod
from LexRank import degree_centrality_scores
from rank_bm25 import BM25Okapi
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
import subprocess


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
    
def process_corpus(path,column):
    try:
        resp = pd.read_csv(path)
        size = len(resp)
        docs = []
        num_cadastrado = []
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
                    #if begin:
                        #text = select_important(text,begin)
                        #print(text)
                    #Se tiver de limpar o texto

                    text_cleaned = clean_text(text)
                    num_cadastrado.append(int(linha['num_tema_cadastrado']))
                    #print(text_cleaned)
                    docs.append(text_cleaned)

            except Exception as erro:
                print(f"Erro ao capturar numero de tema cadastrado {i} : {erro}")
                continue
                
        corpus = pd.DataFrame()
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


def createFileName(name):
    return f'{name}_EMBEDDING_CLEAN.pkl'

def create_embedding(file,corpus,num,model,data_type):
    
    sentence_model = SentenceTransformer(model)
    nameEmbedding = createFileName(file.name.split('.')[0])
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
            corpus_embedding = sentence_model.encode(corpus,show_progress_bar=True)
        with open(nameEmbedding, "wb") as fOut:
                pickle.dump({'sentences': corpus,'numTema':num ,'embeddings': corpus_embedding}, fOut,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Embedding salvo no arquivo {nameEmbedding}")
    else:
        #verbose
        corpus_embedding = sentence_model.encode(corpus,show_progress_bar=True)
        with open(nameEmbedding, "wb") as fOut:
                pickle.dump({'sentences': corpus,'numTema':num ,'embeddings': corpus_embedding}, fOut,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Embedding salvo no arquivo {nameEmbedding}")
        
    return nameEmbedding
    
def main(args):
    conteudo = []
    print("############### PROGRAMA DE SUGESTÃO DE TEMAS ###############")
    print("############### Configuração ###############")
    print(f"Sentence-BERT Model : {args.model}")
    tempo_inicio = time.time()
    print("\n\n############### Fazendo download de dependências ... ###############")
    nltk.download('stopwords')
    nltk.download('punkt')
    #Processa texto do recurso
    corpus_recurso = process_corpus(args.appeal_file,'recurso')
    
    #Processa lista de temas 
    corpus_tema = process_corpus(args.themes_file,'tema')
    
    #Cria embedding recurso
    conteudo_recurso = corpus_recurso['recurso'].tolist()
    arquivo_embedding_recurso = create_embedding(args.appeal_file,conteudo_recurso,corpus_recurso['num_tema_cadastrado'],args.model,'recurso')
    
    #Cria embedding temas
    conteudo_tema = corpus_tema['tema'].tolist()
    arquivo_embedding_tema = create_embedding(args.themes_file,conteudo_tema,corpus_tema['num_tema_cadastrado'],args.model,'tema')
    
    print("############### Criação do Resumo ###############")
    
    #Executa script de criação de resumo do recurso especial - estratégia lexrank guiada 15 sentenças
    comando = f"python createTopics.py {arquivo_embedding_recurso} 15 X -v --seed_list lista_temas.csv"
    subprocess.run(comando, shell=True)
    
    print("############### Calculo de similaridade ###############")
    
    #Executa script pra calcular similaridade entre resumo do recurso e temas usando BM25, 6 sugestoes
    comando_similaridade = "python calcSimilarity.py TOPICS_X15CLEAN.pkl lista_temas_EMBEDDING_CLEAN.pkl 6 B"
    subprocess.run(comando_similaridade, shell=True)
    
    print("Resultado salvo no arquivo CLASSFIED_TOPICS_X15CLEAN_BM25.csv")
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Generate text embedding using Sentence-BERT model')
    parser.add_argument('appeal_file', type=argparse.FileType('r'), help='File containing the appeal')
    parser.add_argument('themes_file', type=argparse.FileType('r'), help='File containing the themes')
    parser.add_argument('model',default='distiluse-base-multilingual-cased-v1',nargs='?', help='The Sentence-BERT model used to generate embedding : Default = distiluse-base-multilingual-cased-v1')

    args = parser.parse_args()
    main(args)
    
    
    
