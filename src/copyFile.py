import pickle
import argparse
#Script criado para salvar uma cópia do arquivo original de recursos especiais e embeddings apenas alterando o nome das chaves do dict
#Objetivo calcular similaridade com temas a partir do texto inteiro, ou seja sem ser por resumo extrativo ou topicos, apenas com limpeza de termos irrelevantes


def createFileName(name):
    n = name.split('.')
    new=f'{n[0]}_V2.pkl'
    return new

def copy_file(file):
    name=''
    name=createFileName(file)
    # Carrega os dados do arquivo original
    with open(file, 'rb') as fIn:
        stored_data = pickle.load(fIn)

    # Renomeia as chaves
    renamed_data = {
        'indice': stored_data['indice'],
        'topics': stored_data['sentences'],
        'numTema': stored_data['numTema'],
        'topicsEmbeddings': stored_data['embeddings']
    }

    # Salva os dados renomeados em um novo arquivo
    with open(name, 'wb') as fOut:
        pickle.dump(renamed_data, fOut)
    print(f"Cópia do arquivo original em {name}")
    
def main(args):   
        copy_file(args.corpus_csv_file)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Copy file')
    parser.add_argument('corpus_csv_file', help='File containing the corpus')
    args = parser.parse_args()
    main(args)
