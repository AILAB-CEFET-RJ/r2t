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

# Install required packages
subprocess.run(["pip", "install", "rank_bm25", "numexpr"])

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def read_text(dataframe, row, column):
    return dataframe.at[row, column]

def sort_list(lst):
    return sorted(lst, key=lambda x: x[1], reverse=True)

def create_bm25_similarity_list(themes, k, real_theme):
    sorted_list = sort_list(themes)
    real_theme_info = []

    for i, item in enumerate(sorted_list):
        if item[0] == real_theme:
            real_theme_info.append(i + 1)  # Position of the real theme in the ranking
            real_theme_info.append(item[1])  # Similarity of the real theme according to BM25
            break

    ranking = sorted_list[:k]
    return ranking, real_theme_info

def calculate_cosine_similarity(topics, theme_embeddings, theme_numbers, k, real_theme):
    similarity_list = []
    real_theme_info = []

    for embedding, number in zip(theme_embeddings, theme_numbers):
        query_embedding = topics
        tensor_similarity = util.cos_sim(query_embedding, embedding)
        similarity_value = tensor_similarity.item()
        similarity_list.append((number, similarity_value))

    sorted_list = sort_list(similarity_list)
    for i, item in enumerate(sorted_list):
        if item[0] == real_theme:
            real_theme_info.append(i + 1)  # Position of the real theme in the ranking
            real_theme_info.append(item[1])
            break

    ranking = sorted_list[:k]
    return ranking, real_theme_info

def create_columns(k):
    columns = ["index", "registered_theme_number"]
    for i in range(1, k + 1):
        columns.append(f"suggested_{i}")
        columns.append(f"similarity_{i}")
    columns.append("real_theme_position")
    columns.append("real_theme_similarity")
    return columns

def create_file(name, mode, rank):
    name_parts = name.split('.')
    new_file_name = f'CLASSIFIED_{name_parts[0]}_BM25.csv' if mode == 'B' else f'CLASSIFIED_{name_parts[0]}_COSINE.csv'
    columns = create_columns(rank)
    pd.DataFrame(columns=columns).to_csv(new_file_name, index=False)
    return new_file_name

# Strategy pattern implementation

class SimilarityStrategy(ABC):
    @abstractmethod
    def execute(self, corpus, themes, similarity_type, rank, verbose=None):
        pass

class BM25Strategy(SimilarityStrategy):
    def __init__(self):
        self.bm25 = None
        
    def calculate_bm25_scores(self, topic):
        if self.bm25 is None:
            raise ValueError("BM25 not initialized.")
        query = topic.replace("-", "")
        tokenized_query = query.split()
        return self.bm25.get_scores(tokenized_query)

    def execute(self, corpus, themes, similarity_type, rank, verbose=None):
        themes_data = pd.DataFrame()
        corpus_data = pd.DataFrame()
        
        with open(corpus, "rb") as f:
            data = pickle.load(f)
            corpus_data['index'] = data['index']
            corpus_data['topics'] = data['topics']
            corpus_data['theme_number'] = data['theme_number']
        
        with open(themes, "rb") as f:
            data = pickle.load(f)
            themes_data['index'] = data['index']
            themes_data['sentences'] = data['sentences']
            themes_data['theme_number'] = data['theme_number']

        print("Calculating similarity using BM25")

        file_similarity = create_file(corpus, similarity_type, rank)
        
        cleaned_themes = [remove_punctuation(row['sentences']) for _, row in themes_data.iterrows()]
        tokenized_corpus = [doc.split(" ") for doc in cleaned_themes]
        self.bm25 = BM25Okapi(tokenized_corpus)

        for idx, row in corpus_data.iterrows():
            sys.stdout.write("\r ")
            sys.stdout.write(f' Progress: {idx/len(corpus_data)*100:.2f}%')
            sys.stdout.flush()
            print(" ", end='\r')
            doc_scores = self.calculate_bm25_scores(row['topics'])
            classified_themes = themes_data[['theme_number']].copy()
            classified_themes['similarity'] = doc_scores
            classified_themes.columns = ['theme_number', 'similarity']
            sorted_themes = list(classified_themes.itertuples(index=False, name=None))
            data_to_write = [idx, int(row['theme_number'])]

            try:
                ranking, real_theme_info = create_bm25_similarity_list(sorted_themes, rank, row['theme_number'])
            except Exception as e:
                print(f"Error calculating similarity index {idx}")
                continue

            for item in ranking:
                data_to_write.extend(item)

            if real_theme_info:
                data_to_write.extend(real_theme_info)
            else:
                data_to_write.extend(["NA", "NA"])

            with open(file_similarity, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_write)

        print(f"Similarity calculated - data saved to file {file_similarity}")

class CosineSimilarityStrategy(SimilarityStrategy):
    def execute(self, corpus, themes, similarity_type, rank, verbose=None):
        themes_data = pd.DataFrame()
        corpus_data = pd.DataFrame()

        with open(corpus, "rb") as f:
            data = pickle.load(f)
            corpus_data['index'] = data['index']
            corpus_data['topics'] = data['topics']
            corpus_data['topics_embeddings'] = data['topics_embeddings'].tolist()
            corpus_data['theme_number'] = data['theme_number']

        with open(themes, "rb") as f:
            data = pickle.load(f)
            themes_data['index'] = data['index']
            themes_data['sentences'] = data['sentences']
            themes_data['embeddings'] = data['embeddings'].tolist()
            themes_data['theme_number'] = data['theme_number']

        print("Calculating similarity using Cosine")

        file_similarity = create_file(corpus, similarity_type, rank)

        for idx, row in corpus_data.iterrows():
            sys.stdout.write("\r ")
            sys.stdout.write(f' Progress: {idx/len(corpus_data)*100:.2f}%')
            sys.stdout.flush()
            print(" ", end='\r')
            data_to_write = [idx, int(row['theme_number'])]

            ranking, real_theme_info = calculate_cosine_similarity(
                row['topics_embeddings'],
                themes_data['embeddings'],
                themes_data['theme_number'],
                rank,
                row['theme_number']
            )

            for item in ranking:
                data_to_write.extend(item)

            if real_theme_info:
                data_to_write.extend(real_theme_info)
            else:
                data_to_write.extend(["NA", "NA"])

            with open(file_similarity, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_write)

        print(f"Similarity calculated - data saved to file {file_similarity}")

class StrategyContext:
    def __init__(self, strategy):
        self.strategy = strategy
        
    def execute_strategy(self, corpus, themes, similarity_type, rank, verbose=None):
        self.strategy.execute(corpus, themes, similarity_type, rank, verbose)

def main(args):
    print("############### SIMILARITY CALCULATION PROGRAM ###############")
    print("############### Configuration ###############")
    print(f"Similarity type: {args.type}")

    start_time = time.time()
    verbose = args.verbose

    if args.type == 'B':
        strategy = BM25Strategy()
    elif args.type == 'C':
        strategy = CosineSimilarityStrategy()
    else:
        print(f"Unrecognized similarity type: {args.type}")
        return

    context = StrategyContext(strategy)
    context.execute_strategy(args.corpus, args.themes, args.type, int(args.rank), verbose)

    print("Saving log...")
    end_time = time.time()
    total_seconds = end_time - start_time
    minutes, seconds = divmod(int(total_seconds), 60)
    print(f"Execution time: {minutes} minutes and {seconds} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the similarity between texts')
    parser.add_argument('corpus', help='Path to corpus file')
    parser.add_argument('themes', help='Path to themes file')
    parser.add_argument('rank', help='Size of rank')
    parser.add_argument('type', choices=['B', 'C'], help='Type of similarity: BM25 (B) or Cosine (C)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity')

    args = parser.parse_args()
    main(args)
