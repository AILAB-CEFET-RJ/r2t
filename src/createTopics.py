import argparse
import time
from sentence_transformers import SentenceTransformer, util
import numpy as np
from bertopic import BERTopic
from abc import ABC, abstractmethod
import pickle
import pandas as pd
import sys
import nltk
from LexRank import degree_centrality_scores
import string
from rank_bm25 import BM25Okapi

# Implementing the Strategy pattern

# Defining interfaces (strategies)
class Strategy(ABC):
    @abstractmethod
    def execute(self, corpus_embedding, model, seed_list=None, verbose=None):
        pass

    @classmethod
    def create_file_name(cls, strategy, embedding_name, size):
        if 'clean' in embedding_name.lower():
            return f'TOPICS_{strategy}{size}CLEAN.pkl'
        else:
            return f'TOPICS_{strategy}{size}.pkl'
        
    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def generate_summary(self, strategy, size, embedding_name, index, num_theme, topics, topic_embeddings):
        file_name = self.create_file_name(strategy, embedding_name, size)
        with open(file_name, "wb") as fOut:
            pickle.dump({'index': index, 'topics': topics, 'numTheme': num_theme, 'topicEmbeddings': topic_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Summaries and embeddings saved to {file_name}")
    
    def format_seed_list(self, seed):
        repetitive_themes = pd.read_csv(seed, sep=',')
        seed_list = []
        theme_seeds = repetitive_themes[['tema']].copy()
        for _, row in theme_seeds.iterrows():
            seed_text = self.remove_punctuation(row['tema'])
            seed_list.append(seed_text.split())
        return seed_list


class BertopicStrategy(Strategy):
    def execute(self, corpus_embedding, size, model, seed_list=None, verbose=None):
        with open(corpus_embedding, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_index = stored_data['index']
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
            stored_number = stored_data['numTheme']
        
        print("Executing Bertopic Strategy")
        topic_model = BERTopic(embedding_model=model, top_n_words=size, verbose=verbose)
        topics, probs = topic_model.fit_transform(stored_sentences, stored_embeddings)
        topic_representation = topic_model.get_document_info(stored_sentences)
        topic_words = [s.replace('-', ' ') for s in topic_representation['Top_n_words']]
        
        sys.stdout.write(f'\r Progress: {len(topic_representation) / len(stored_index) * 100:.2f}%')
        sys.stdout.flush()
        print(" ", end='\r')

        # Create topic embeddings
        sentence_model = SentenceTransformer(model)
        topic_embeddings = sentence_model.encode(topic_words, show_progress_bar=True)
        
        self.generate_summary('B', size, corpus_embedding, stored_index, stored_number, topic_words, topic_embeddings)


class GuidedBertopicStrategy(Strategy):
    def execute(self, corpus_embedding, size, model, seed_list=None, verbose=None):
        with open(corpus_embedding, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_index = stored_data['index']
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
            stored_number = stored_data['numTheme']

        print("Executing Guided Bertopic Strategy")
        seeds = self.format_seed_list(seed_list)
        topic_model = BERTopic(embedding_model=model, top_n_words=size, verbose=verbose, seed_topic_list=seeds)
        topics, probs = topic_model.fit_transform(stored_sentences, stored_embeddings)
        topic_representation = topic_model.get_document_info(stored_sentences)
        topic_words = [s.replace('-', ' ') for s in topic_representation['Top_n_words']]
        
        sys.stdout.write(f'\r Progress: {len(topic_representation) / len(stored_index) * 100:.2f}%')
        sys.stdout.flush()
        print(" ", end='\r')

        # Create topic embeddings
        sentence_model = SentenceTransformer(model)
        topic_embeddings = sentence_model.encode(topic_words, show_progress_bar=True)
        
        self.generate_summary('G', size, corpus_embedding, stored_index, stored_number, topic_words, topic_embeddings)


class LexrankStrategy(Strategy):
    def execute(self, corpus_embedding, size, model_name, seed_list=None, verbose=None):
        model = SentenceTransformer(model_name)
        
        with open(corpus_embedding, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_index = stored_data['index']
            stored_sentences = stored_data['sentences']
            stored_number = stored_data['numTheme']

        print("Executing Lexrank Strategy")
        summaries = []

        for text in stored_sentences:
            topics = []
            sentences = nltk.sent_tokenize(text, language='portuguese')

            # Compute sentence embeddings
            embeddings = model.encode(sentences, convert_to_tensor=True)
            cos_scores = util.cos_sim(embeddings, embeddings).numpy()

            # Compute centrality for each sentence
            centrality_scores = degree_centrality_scores(cos_scores, threshold=0.3)
            most_central_sentence_indices = np.argsort(-centrality_scores)

            for idx in most_central_sentence_indices[:size]:
                topics.append(sentences[idx].strip())
            summary = " ".join(topics)
            summaries.append(summary)
            
            sys.stdout.write(f'\r Progress: {len(summaries) / len(stored_index) * 100:.2f}%')
            sys.stdout.flush()
            print(" ", end='\r')

        # Create topic embeddings
        sentence_model = SentenceTransformer(model_name)
        topic_embeddings = sentence_model.encode(summaries, show_progress_bar=True)
        
        self.generate_summary('L', size, corpus_embedding, stored_index, stored_number, summaries, topic_embeddings)


class GuidedLexrankStrategy(Strategy):
    def execute(self, corpus_embedding, size, model_name, seed_list=None, verbose=None):
        alpha = 1
        beta = 3
        model = SentenceTransformer(model_name)
        
        with open(corpus_embedding, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_index = stored_data['index']
            stored_sentences = stored_data['sentences']
            stored_number = stored_data['numTheme']

        print("Executing Guided Lexrank Strategy")
        
        with open(seed_list, "r") as fIn:
            theme_data = pd.read_csv(fIn)
            
        themes = [self.remove_punctuation(row['tema']) for _, row in theme_data.iterrows()]
        theme_embeddings = model.encode(themes, convert_to_tensor=True)

        summaries = []
        
        for text in stored_sentences:
            topics = []
            sentences = nltk.sent_tokenize(text, language='portuguese')

            # Compute sentence embeddings
            embeddings = model.encode(sentences, convert_to_tensor=True)
            cos_scores = util.cos_sim(embeddings, embeddings).numpy()
            centrality_scores = degree_centrality_scores(cos_scores, threshold=0.05)

            # BM25 score calculation
            tokenized_themes = [doc.split(" ") for doc in themes]
            bm25 = BM25Okapi(tokenized_themes)

            bm25_scores = []
            for sentence in sentences:
                query = sentence.split()
                doc_scores = bm25.get_scores(query)
                bm25_scores.append(doc_scores.max())
            
            bm25_scores = np.array(bm25_scores)
            normalized_centrality_scores = (centrality_scores - centrality_scores.min()) / (centrality_scores.max() - centrality_scores.min())
            normalized_bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
            
            combined_scores = alpha * normalized_centrality_scores + beta * normalized_bm25_scores
            most_central_sentence_indices = np.argsort(-combined_scores)

            for idx in most_central_sentence_indices[:size]:
                topics.append(sentences[idx].strip())
            summary = " ".join(topics)
            summaries.append(summary)
            
            sys.stdout.write(f'\r Progress: {len(summaries) / len(stored_index) * 100:.2f}%')
            sys.stdout.flush()
            print(" ", end='\r')

        # Create topic embeddings
        sentence_model = SentenceTransformer(model_name)
        topic_embeddings = sentence_model.encode(summaries, show_progress_bar=True)

        self.generate_summary('X', size, corpus_embedding, stored_index, stored_number, summaries, topic_embeddings)


# Class that uses a strategy
class Context:
    def __init__(self, strategy):
        self.strategy = strategy
        
    def execute_strategy(self, corpus_embedding, size, model, seed_list=None, verbose=None):
        self.strategy.execute(corpus_embedding, size, model, seed_list, verbose)


def main(args):
    print("############### TOPIC GENERATION PROGRAM ###############")
    print("############### Configuration ###############")
    print(f"Topic generation type: {args.type}")

    start_time = time.time()
    verbose = args.verbose

    # Define the appropriate strategy
    if args.type == 'B':
        strategy = BertopicStrategy()
    elif args.type == 'G':
        strategy = GuidedBertopicStrategy()
    elif args.type == 'L':
        strategy = LexrankStrategy()
    elif args.type == 'X':
        strategy = GuidedLexrankStrategy()
    else:
        print(f"Unrecognized topic generation type: {args.type}")
        return

    context = Context(strategy)
    context.execute_strategy(args.corpus_embedding, int(args.size), args.model, args.seed_list, verbose)

    print("Saving log...")
    total_time = time.time() - start_time
    minutes, seconds = divmod(int(total_time), 60)
    print(f"Execution time: {minutes} minutes and {seconds} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate topics from text')
    parser.add_argument('model', default='distiluse-base-multilingual-cased-v1', nargs='?', help='The Sentence-BERT model used to generate embeddings (default: distiluse-base-multilingual-cased-v1)')
    parser.add_argument('corpus_embedding', help='Path to the corpus embedding file')
    parser.add_argument('size', help='Number of sentences or topics for summarization')
    parser.add_argument('type', choices=['B', 'G', 'L', 'X'], help='Type of topic generation: Bertopic (B), Guided Bertopic (G), Lexrank (L), or Guided Lexrank (X)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity')
    parser.add_argument('--seed_list', help='Seed list (required if type is G or X)')

    args = parser.parse_args()
    main(args)
