# r2t

## Exemplos  

criação embedding recursos :    python createEmbedding.py REsp_completo.csv recurso recurso --clean --begin_point cabimento -v

criação embedding temas : python createEmbedding.py temas_repetitivos.csv tema tema --clean -v

criação de tópicos lexrank 10 sentenças : python createTopics.py REsp_completo_EMBEDDING_CLEAN.pkl 10 L -v

criação de tópicos lexrank guiada 50 sentenças sem remoção de termos : python createTopics.py REsp_completo_EMBEDDING.pkl 50 X -v --seed_list temas_repetitivos.csv

criação de tópicos bertopic guiada 10 tópicos : python createTopics.py REsp_completo_EMBEDDING_CLEAN.pkl 10 G -v --seed_list temas_repetitivos.csv

criação de tópicos bertopic 10 tópicos: python createTopics.py REsp_completo_EMBEDDING_CLEAN.pkl 10 B -v

calculo de similaridade bm25 : python calcSimilarity.py TOPICS_L10CLEAN.pkl temas_repetitivos_EMBEDDING_CLEAN.pkl 6 B

calculo de similaridade cosseno : python calcSimilarity.py TOPICS_L10CLEAN.pkl temas_repetitivos_EMBEDDING_CLEAN.pkl 6 C

copia arquivo contendo recurso com embeddings alterando chaves do dicionário : python copyFile.py REsp_completo_EMBEDDING_CLEAN.pkl
cálculo de métricas da classificação feita pelo elasticsearch :python metrics.py CLASSFIED_REsp_Elasticsearch.csv

cálculo de métricas da classificação feita pelo BERTopic com 10 tópicos e similaridade por cosseno: CLASSFIED_TOPICS_B10_COSINE.csv

computação de métricas : python metrics.py CLASSFIED_TOPICS_B10CLEAN_BM25.csv


## Arquivos

Embeddings : temas_repetitivos_EMBEDDING_CLEAN.pkl | REsp_completo_EMBEDDING_CLEAN.pkl
