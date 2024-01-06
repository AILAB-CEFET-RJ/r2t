#pip install rank_eval
#pip install ranx


from rank_eval import Qrels, Run, evaluate, compare
import pandas as pd
from pandas import DataFrame
import csv
import os
import argparse

def save_results(file, data):
    try:
        bolFile = os.path.isfile(file)

        with open(file, 'a', newline='') as fl:
            csv_writer = csv.writer(fl)

            if not bolFile:
                csv_writer.writerow(['Origem', 'Recall', 'F1-score', 'MAP', 'NDCG', 'MRR'])
            csv_writer.writerow(data)
    except Exception as e:
        print(f"Erro ao salvar resultados: {e}")



def recallAtK(k,df):
    Relevant_retrieved = 0
    Total_relevant = len(df)
    for indice,linha in df.iterrows():
        if(0 < linha['posicao_tema_real'] <= k):
            Relevant_retrieved +=1
    return Relevant_retrieved/Total_relevant

def f1_score(MAP, recall):
    f1 = (2*MAP*recall)/(MAP+recall)
    return f1

def process_corpus(data_file):
    results = []
    results.append(data_file)
    df = pd.read_csv(data_file)
    df['indice'] = df['indice'].astype(str)
    df['sugerido_1'] = df['sugerido_1'].astype(str)
    df['sugerido_2'] = df['sugerido_2'].astype(str)
    df['sugerido_3'] = df['sugerido_3'].astype(str)
    df['sugerido_4'] = df['sugerido_4'].astype(str)
    df['sugerido_5'] = df['sugerido_5'].astype(str)
    df['sugerido_6'] = df['sugerido_6'].astype(str)
    df['num_tema_cadastrado'] = df['num_tema_cadastrado'].astype(str)
    df['relevancia_tema_cadastrado']=10

    q_id = []
    doc_id = []
    score = []
    for indice, linha in df.iterrows():
        for i in range(1,7):
            q_id.append(linha[0])
            sug = f"sugerido_{i}"
            doc_id.append(int(linha[sug]))
            sim = f"similaridade_{i}"
            score.append(linha[sim])

    ranx_dict ={"q_id": q_id,
                "doc_id": doc_id,
                "score" : score
    }

    qrel = Qrels.from_df(df, q_id_col='indice', doc_id_col='num_tema_cadastrado', score_col='relevancia_tema_cadastrado')

    run_df = DataFrame.from_dict(ranx_dict)
    run_df['q_id'] = run_df['q_id'].astype(str)
    run_df['doc_id'] = run_df['doc_id'].astype(str)
    run_df['score'] = run_df['score'].astype(float)
    run = Run.from_df(
        df=run_df,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score"
    )
    run.name = "my_run"


    dicionario = evaluate(qrel, run, ["map@6","ndcg@6","mrr"])

    run.mean_scores
    
    recall = recallAtK(6,df)
    results.append(recall)

    f1 = f1_score(dicionario['map@6'],recall)

    f1 = round(f1,5)
    results.append(f1)
    
    results.append(dicionario['map@6'])
    results.append(dicionario['ndcg@6'])
    results.append(dicionario['mrr'])
    
    
    
    save_results('results.csv',results)
    print(f"Dados das métricas salvos em arquivo: results.csv")


def main(args):
    print("############### CÁLCULO DE MÉTRICAS ###############")

    verbose = args.verbose
    corpus = process_corpus(args.corpus_csv_file)
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Generates metrics')
    parser.add_argument('corpus_csv_file', help='File containing the classfied data')
    
    parser.add_argument('-v','--verbose',action='store_true',help='Increase the verbosity level')
    args = parser.parse_args()
    main(args)
