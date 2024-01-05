from rank_bm25 import BM25Okapi
import csv
import pandas as pd
import string
import subprocess
from abc import ABC, abstractmethod
import argparse
import time
import pickle
import sys
from sentence_transformers import SentenceTransformer, util
import torch

# Instalação dos pacotes necessários
subprocess.run(["pip", "install", "rank_bm25","numexpr"])

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation


def read_text(dataframe, linha, coluna):
    texto = dataframe.at[linha, coluna]
    return texto

def sort_list(lista):
    return(sorted(lista, key = lambda x: x[1],reverse=True))


def create_list_similarity_bm25(temas ,k,tema_real):
  lista_similaridade = temas
  lista_tema_real = []

  sorted_list = sort_list(lista_similaridade)

  for i, linha in enumerate(sorted_list):
      if(linha[0]==tema_real):
        lista_tema_real.append(i+1)#identifica posição do tema real no ranking
        lista_tema_real.append(linha[1])#similaridade do tema real segundo bm25
        break
  ranking = sorted_list[:k]
  return ranking,lista_tema_real

def calc_similarity_cosine(topics, temas_embeddings,temas_numTema ,k,tema_real):
  lista_similaridade = []
  lista_tema_real = []
  
  for temas_embeddings,temas_numTema in zip(temas_embeddings,temas_numTema):
      query_embedding = topics
      #print(f"Tema {indice} : {tupla_num_tema[0]}")
      #tema_cleaned = clean_text(temas_embeddings)
      tensor_similaridade = util.cos_sim(query_embedding, temas_embeddings)
      valor_similaridade = tensor_similaridade.item()
      tupla = (temas_numTema,valor_similaridade)
      lista_similaridade.append(tupla)

  sorted_list = sort_list(lista_similaridade)
  for i, linha in enumerate(sorted_list):
      if(linha[0]==tema_real):
        lista_tema_real.append(i+1)#identifica posição do tema real no ranking
        lista_tema_real.append(linha[1]) 
        break
  ranking = sorted_list[:k]
  return ranking,lista_tema_real

def createColumns(k):
      #k refere-se ao numero de elementos no ranking
    colunas = []
    colunas.append("indice")
    colunas.append("num_tema_cadastrado")
    for i in range(1, k + 1):
        nome = f"sugerido_{i}"
        colunas.append(nome)
        nome = f"similaridade_{i}"
        colunas.append(nome)
    #Posicao que o tema cadastrado por analista ocupa no ranking
    colunas.append("posicao_tema_real")
    colunas.append("similaridade_tema_real")
    return colunas


def createFile(name,mode,rank):
    n = name.split('.')
    new=''
    if (mode=='B'):
        new =f'CLASSFIED_{n[0]}_BM25.csv'
    if (mode=='C'):
        new =f'CLASSFIED_{n[0]}_COSINE.csv'
    file_columns = createColumns(rank)
    df = pd.DataFrame(columns=file_columns)
    df.to_csv(new,index=False)
    return new
    
#Implementando padrao de projeto strategy

# Definindo interfaces (estratégias)
class Estrategia(ABC):
    @abstractmethod
    def executar(self, corpus, temas, typeSim,verbose=None):
        pass

class EstrategiaBM25(Estrategia):
    def __init__(self):
        self.bm25 = None
        
    def calc_scores_bm25(self,topico):
        if self.bm25 is None:
            raise ValueError("BM25 not initialized.")
        query = topico.replace("-","")
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        return doc_scores

    def executar(self, corpus,temas, typeSim,rank,verbose=None):
        temas_data = pd.DataFrame()
        corpus_data = pd.DataFrame()
        with open(corpus, "rb") as fIn:
            data = pickle.load(fIn)
            corpus_data['indice'] = data['indice']
            corpus_data['topics'] = data['topics']
            #corpus_data['topicsEmbeddings']
            corpus_data['numTema'] = data['numTema']
        with open(temas, "rb") as fIn:
            data = pickle.load(fIn)
            #Columns
            temas_data['indice'] = data['indice']
            temas_data['sentences'] = data['sentences']
            #temas_data['embeddings'] = data['embeddings']
            temas_data['numTema']= data['numTema']
        print("Calculando Similaridade com BM25")
        
        #Salva dados de similaridade em arquivo
        file_similarity = createFile(corpus,typeSim,rank)
        
        #Processamento do texto alvo
        temas = []
        for indice, linha in temas_data.iterrows():
            tema_clean = remove_punctuation(linha['sentences'])
            temas.append(tema_clean)           
        tokenized_corpus = [doc.split(" ") for doc in temas]
        self.bm25 = BM25Okapi(tokenized_corpus)

        
        
        #Processamento da consulta
        for indice, linha in corpus_data.iterrows():
            sys.stdout.write("\r ")  # \r faz o cursor retroceder ao início da linha
            sys.stdout.write(f' Percentual concluído: {indice/len(corpus_data)*100:.2f}%')
            sys.stdout.flush()  # Força a impressão imediata
            print(" ", end='\r')
            doc_scores = self.calc_scores_bm25(linha['topics'])
            temas_classificados = temas_data[['numTema']].copy()
            temas_classificados['similaridade']= doc_scores
            temas_classificados.columns = ['numTema','similaridade']
            list_temas_classificados = list(temas_classificados.itertuples(index=False, name=None))
            dados = []
            dados.append(indice)
              #numero do tema cadastrado por um analista
            try:
                  #numeracao de tema cadastrado pelo analista para o recurso
                dados.append(int(linha['numTema']))
            except Exception as erro:
                print(f"Erro ao capturar numero de tema cadastrado {indice}")
                continue
            try:  
                ranking , lista_tema_real = create_list_similarity_bm25( list_temas_classificados, rank, linha['numTema'])
            except Exception as erro:
                print(f"Erro calculo similaridade indice {indice}") 
                continue
                

            for i, tupla_num_tema in enumerate(ranking):
                #captura numero do tema sugerido e valor da similaridade
                dados.append(tupla_num_tema[0])
                dados.append(tupla_num_tema[1])

            if(lista_tema_real):
                dados.append(lista_tema_real[0])
                dados.append(lista_tema_real[1])
            else:
                dados.append("NA")
                dados.append("NA")
            with open(file_similarity, mode='a', newline='') as arquivo:
                writer = csv.writer(arquivo)
                writer.writerow(dados)

        print(f"Similaridade Computada -dados salvos no arquivo {file_similarity}")
class EstrategiaCosine(Estrategia):
    def executar(self, corpus,temas, typeSim,rank,verbose=None):
        temas_data = pd.DataFrame()
        corpus_data = pd.DataFrame()
        with open(corpus, "rb") as fIn:
            data = pickle.load(fIn)
            corpus_data['indice'] = data['indice']
            corpus_data['topics'] = data['topics']
            corpus_data['topicsEmbeddings'] = data['topicsEmbeddings'].tolist()
            corpus_data['numTema'] = data['numTema']
        with open(temas, "rb") as fIn:
            data = pickle.load(fIn)
            #Columns
            temas_data['indice'] = data['indice']
            temas_data['sentences'] = data['sentences']
            temas_data['embeddings'] = data['embeddings'].tolist()
            temas_data['numTema']= data['numTema']
        print("Calculando Similaridade por Cosseno")
        
        #Salva dados de similaridade em arquivo
        file_similarity = createFile(corpus,typeSim,rank)
        #Processamento do texto alvo
        temas = []

        for indice, linha in corpus_data.iterrows():
            sys.stdout.write("\r ")  # \r faz o cursor retroceder ao início da linha
            sys.stdout.write(f' Percentual concluído: {indice/len(corpus_data)*100:.2f}%')
            sys.stdout.flush()  # Força a impressão imediata
            print(" ", end='\r')
            dados = []
            dados.append(indice)
            #numero do tema cadastrado por um analista
            try:
                dados.append(int(linha['numTema']))
            except Exception as erro:
                print(f"Erro ao capturar numero de tema cadastrado {indice}")
                continue
            #try:
            ranking , lista_tema_real = calc_similarity_cosine(linha['topicsEmbeddings'], temas_data['embeddings'],temas_data['numTema'], rank, linha['numTema'])
            ##except Exception as erro:
                #print(f"Erro calculo similaridade indice {indice}") 
                #continue

            for i, tupla_num_tema in enumerate(ranking):
                #captura numero do tema sugerido e valor da similaridade
                dados.append(tupla_num_tema[0])
                dados.append(tupla_num_tema[1])

            if(lista_tema_real):
                dados.append(lista_tema_real[0])
                dados.append(lista_tema_real[1])
            else:
                dados.append("NA")
                dados.append("NA")
            with open(file_similarity, mode='a', newline='') as arquivo:
                writer = csv.writer(arquivo)
                writer.writerow(dados)

        print(f"Similaridade Computada -dados salvos no arquivo {file_similarity}")


# Classe que usa uma estratégia
class Contexto:
    def __init__(self, estrategia):
        self.estrategia = estrategia
        
    def executar_estrategia(self, corpus, temas, typeSim,rank, verbose=None):
        self.estrategia.executar(corpus, temas, typeSim,rank, verbose)



def main(args):
    print("############### PROGRAMA DE CÁLCULO DE SIMILARIDADE ###############")
    
    print("############### Configuração ###############")
    print(f"Type of similarity : {args.type}")

    tempo_inicio = time.time()
    verbose = args.verbose

    # Cria um contexto com a estratégia adequada
    if args.type == 'B':
        estrategia = EstrategiaBM25()
    elif args.type == 'C':
        estrategia = EstrategiaCosine()
    else:
        print(f"Tipo de geração de tópicos não reconhecido: {args.type}")
        return

    contexto = Contexto(estrategia)
    contexto.executar_estrategia(args.corpus,args.temas, args.type,int(args.rank), verbose)

    print("Salvando log ...")
    tempo_fim = time.time()
    tempo_total_segundos = tempo_fim - tempo_inicio
    minutos, segundos = divmod(int(tempo_total_segundos), 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculates  the similarity between texts')
    parser.add_argument('corpus', help='Path to corpus file')
    parser.add_argument('temas', help='Path to tema file')
    parser.add_argument('rank',help='Size of rank')
    parser.add_argument('type', choices=['B', 'C'], help='Type of similarity, by cosine or bm25')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase the verbosity level')



    args = parser.parse_args()
    main(args)
