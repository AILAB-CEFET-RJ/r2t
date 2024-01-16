import argparse
import time
from sentence_transformers import SentenceTransformer, util
import numpy as np
from bertopic import BERTopic
from abc import ABC, abstractmethod
import pickle
import pandas as pd
import csv
import sys
import nltk
from LexRank import degree_centrality_scores
import string


#Implementando padrao de projeto strategy

# Definindo interfaces (estratégias)
class Estrategia(ABC):
    @abstractmethod
    def executar(self, corpus_embedding, model, seed_list=None,verbose=None):
        pass

    @classmethod
    def createFileName(cls, stg, fen, sz):
        
        if 'clean' in fen.lower():
            return f'TOPICS_{stg}{sz}CLEAN.pkl'
        else:
            return f'TOPICS_{stg}{sz}.pkl'
        
    def remove_punctuation(self,text):
        translator = str.maketrans('', '', string.punctuation)
        text_without_punctuation = text.translate(translator)
        return text_without_punctuation
    
    def gerarResumo(self,strategy,size,fileEmbeddingName,indice, numTema,topics,topics_embeddings):
        name=self.createFileName(strategy,fileEmbeddingName,size)                  
        with open(name, "wb") as fOut:
            pickle.dump({'indice':indice,'topics': topics,'numTema':numTema,'topicsEmbeddings':topics_embeddings}, fOut,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Resumo de cada texto do corpus e respectivos embeddings salvos no arquivo {name}")
    
    def format_seed_list(self,seed):
        temas_repetitivos_eproc = pd.read_csv(seed, sep=',' )
        temas_seed_list = []
        temas_seed = temas_repetitivos_eproc[['tema']].copy()
        for indice,linha in temas_seed.iterrows():
            seed = self.remove_punctuation(linha['tema'])
            temas_seed_list.append(seed.split())
        return temas_seed_list

class EstrategiaBERTopic(Estrategia):
    def executar(self, corpus_embedding,size, model, seed_list=None,verbose=None):
        with open(corpus_embedding, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_indice = stored_data['indice']
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
            stored_number = stored_data['numTema']
        print("Executando Estratégia Bertopic")
        topic_model = BERTopic(embedding_model=model,top_n_words=size,verbose=verbose)
        topics, probs = topic_model.fit_transform(stored_sentences,stored_embeddings)
        representacao = topic_model.get_document_info(stored_sentences)
        
        representacao_topicos = [s.replace('-',' ') for s in representacao['Top_n_words']]
        
        sys.stdout.write("\r ")  # \r faz o cursor retroceder ao início da linha
        sys.stdout.write(f' Percentual concluído: {len(representacao)/len(stored_indice)*100:.2f}%')
        sys.stdout.flush()  # Força a impressão imediata
        print(" ", end='\r')
        #Cria embedding dos topicos
        sentence_model = SentenceTransformer(model)
        topics_embeddings = sentence_model.encode(representacao_topicos,show_progress_bar=True)       
        self.gerarResumo('B',size,corpus_embedding,stored_indice,stored_number,representacao_topicos,topics_embeddings)
        
class EstrategiaBERTopicGuiada(Estrategia):
    def executar(self, corpus_embedding,size, model, seed_list=None,verbose=None):
        with open(corpus_embedding, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_indice = stored_data['indice']
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
            stored_number = stored_data['numTema']
        print("Executando Estratégia Bertopic Guiada")
        seed = self.format_seed_list(seed_list)
        topic_model = BERTopic(embedding_model=model,top_n_words=size,verbose=verbose,seed_topic_list=seed)
        topics, probs = topic_model.fit_transform(stored_sentences,stored_embeddings)
        representacao = topic_model.get_document_info(stored_sentences)
        representacao_topicos = [s.replace('-',' ') for s in representacao['Top_n_words']]
        
        sys.stdout.write("\r ")  # \r faz o cursor retroceder ao início da linha
        sys.stdout.write(f' Percentual concluído: {len(representacao)/len(stored_indice)*100:.2f}%')
        sys.stdout.flush()  # Força a impressão imediata
        print(" ", end='\r')
        #Cria embedding dos topicos
        sentence_model = SentenceTransformer(model)
        topics_embeddings = sentence_model.encode(representacao_topicos,show_progress_bar=True)          
        self.gerarResumo('G',size,corpus_embedding,stored_indice,stored_number,representacao_topicos,topics_embeddings)
        
class EstrategiaLexrank(Estrategia):
    def executar(self, corpus_embedding,size, name_model, seed_list=None,verbose=None):
        stored_indice = []
        stored_sentences = []
        stored_embeddings = []
        stored_number = []
        representacao = []
        
        model = SentenceTransformer(name_model)
        
        with open(corpus_embedding, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_indice = stored_data['indice']
            stored_sentences = stored_data['sentences']
            stored_number = stored_data['numTema']
        print("Executando Estratégia Lexrank - Aguarde conclusão")

        for text in (stored_sentences):

            topics = []
            summary = ""

            #Split the document into sentences
            sentences = nltk.sent_tokenize(text,language='portuguese')

            #print("Num sentences:", len(sentences))
            #print(sentences)

            #Compute the sentence embeddings
            embeddings = model.encode(sentences, convert_to_tensor=True)

            #Compute the pair-wise cosine similarities
            cos_scores = util.cos_sim(embeddings, embeddings).numpy()

            #Compute the centrality for each sentence
            centrality_scores = degree_centrality_scores(cos_scores, threshold=0.3)

            #We argsort so that the first element is the sentence with the highest score
            most_central_sentence_indices = np.argsort(-centrality_scores)


            #print("\n\nSummary:")
            for idx in most_central_sentence_indices[0:size]:
                topics.append(sentences[idx].strip())
            summary = "".join(topics)
            representacao.append(summary)
            sys.stdout.write("\r ")  # \r faz o cursor retroceder ao início da linha
            sys.stdout.write(f' Percentual concluído: {len(representacao)/len(stored_indice)*100:.2f}%')
            sys.stdout.flush()  # Força a impressão imediata
            print(" ", end='\r')

        #print(representacao[1])
        #Cria embedding dos topicos
        sentence_model = SentenceTransformer(name_model)
        topics_embeddings = sentence_model.encode(representacao,show_progress_bar=True)
            
        self.gerarResumo('L',size,corpus_embedding,stored_indice,stored_number,representacao,topics_embeddings)

# Classe que usa uma estratégia
class Contexto:
    def __init__(self, estrategia):
        self.estrategia = estrategia
        
    def executarEstrategia(self, corpus_embedding, size, model, seed_list=None, verbose=None):
        self.estrategia.executar(corpus_embedding, size, model, seed_list, verbose)

def main(args):
    print("############### PROGRAMA DE GERAÇÃO DE TÓPICOS ###############")
    print("############### Configuração ###############")
    print(f"Topic generation type : {args.type}")

    tempo_inicio = time.time()
    verbose = args.verbose

    # Cria um contexto com a estratégia adequada
    if args.type == 'B':
        estrategia = EstrategiaBERTopic()
    elif args.type == 'G':
        estrategia = EstrategiaBERTopicGuiada()
    elif args.type == 'L':
        estrategia = EstrategiaLexrank()
    else:
        print(f"Tipo de geração de tópicos não reconhecido: {args.type}")
        return
    contexto = Contexto(estrategia)
    contexto.executarEstrategia(args.corpus_embedding,int(args.size), args.model, args.seed_list,verbose)

    print("Salvando log ...")
    tempo_fim = time.time()
    tempo_total_segundos = tempo_fim - tempo_inicio
    minutos, segundos = divmod(int(tempo_total_segundos), 60)

    # Verifica se o tipo é 'G' e se a lista foi fornecida
    if args.type == 'G' and not args.seed_list:
        parser.error("If type of generating topics is 'G', a seed list is required.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate topics from text')
    parser.add_argument('model', default='distiluse-base-multilingual-cased-v1', nargs='?', help='The Sentence-BERT model used to generate embedding : Default = distiluse-base-multilingual-cased-v1')
    parser.add_argument('corpus_embedding', help='Path to corpus_embedding file')
    parser.add_argument('size', help='Topic size')
    parser.add_argument('type', choices=['B', 'G', 'L'], help='Type of generating topics, Lexrank, Bertopic ou Guided Bertopic')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase the verbosity level')

    # Para modelagem guiada é necessario ter listas de topicos iniciais
    parser.add_argument('--seed_list', nargs='?', help='Seed list (required if type is G)')

    args = parser.parse_args()
    main(args)
