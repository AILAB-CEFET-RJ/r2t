import pandas as pd
import csv
import os
import argparse
from rank_eval import Qrels, Run, evaluate

RESULTS_FILE = 'results.csv'

def save_results(file, data):
    """Saves the results to a CSV file."""
    try:
        file_exists = os.path.isfile(file)
        with open(file, 'a', newline='') as fl:
            csv_writer = csv.writer(fl)
            if not file_exists:
                csv_writer.writerow(['Source', 'Recall', 'F1-score', 'MAP', 'NDCG', 'MRR'])
            csv_writer.writerow(data)
    except Exception as e:
        print(f"Error saving results: {e}")

def recall_at_k(k, df):
    """Calculates recall@k."""
    relevant_retrieved = df[df['posicao_tema_real'].between(1, k)].shape[0]
    total_relevant = len(df)
    return relevant_retrieved / total_relevant if total_relevant > 0 else 0

def f1_score(map_score, recall):
    """Calculates the F1-score."""
    return (2 * map_score * recall) / (map_score + recall) if (map_score + recall) > 0 else 0

def process_corpus(data_file):
    """Processes the corpus and calculates the metrics."""
    results = [data_file]
    
    df = pd.read_csv(data_file)
    
    # Adjust DataFrame
    df[['indice', 'sugerido_1', 'sugerido_2', 'sugerido_3', 'sugerido_4', 'sugerido_5', 'sugerido_6', 'num_tema_cadastrado']] = df[
        ['indice', 'sugerido_1', 'sugerido_2', 'sugerido_3', 'sugerido_4', 'sugerido_5', 'sugerido_6', 'num_tema_cadastrado']
    ].astype(str)
    df['relevancia_tema_cadastrado'] = 10

    # Build ranx_dict
    ranx_dict = {
        "q_id": df['indice'].repeat(6).reset_index(drop=True),
        "doc_id": pd.concat([df[f'sugerido_{i}'] for i in range(1, 7)]).astype(int).reset_index(drop=True),
        "score": pd.concat([df[f'similaridade_{i}'] for i in range(1, 7)]).astype(float).reset_index(drop=True)
    }

    # Create Qrels and Run
    qrel = Qrels.from_df(df, q_id_col='indice', doc_id_col='num_tema_cadastrado', score_col='relevancia_tema_cadastrado')
    run_df = pd.DataFrame(ranx_dict)
    run = Run.from_df(df=run_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

    # Evaluate metrics
    metrics = evaluate(qrel, run, ["map@6", "ndcg@6", "mrr"])
    recall = recall_at_k(6, df)
    f1 = f1_score(metrics['map@6'], recall)

    results.extend([
        round(recall, 5),
        round(f1, 5),
        round(metrics['map@6'], 5),
        round(metrics['ndcg@6'], 5),
        round(metrics['mrr'], 5)
    ])

    save_results(RESULTS_FILE, results)
    print(f"Metrics data saved to file: {RESULTS_FILE}")

def main():
    parser = argparse.ArgumentParser(description='Calculates evaluation metrics for classified data.')
    parser.add_argument('corpus_csv_file', help='Path to the classified data CSV file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity level')
    args = parser.parse_args()

    if args.verbose:
        print("############### METRICS CALCULATION ###############")
    
    process_corpus(args.corpus_csv_file)

if __name__ == "__main__":
    main()
