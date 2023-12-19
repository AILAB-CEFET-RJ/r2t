# r2t

## Exemplos  

criação embedding recursos :    python createEmbedding.py REsp_completo.csv recurso --clean --begin_point cabimento -v

criação embedding temas : python createEmbedding.py temas_repetitivos.csv tema --clean -v

criação de tópicos lexrank : python createTopics.py REsp_completo_EMBEDDING_CLEAN.pkl 10 L -v

criação de tópicos bertopic guiada : python createTopics.py REsp_completo_EMBEDDING_CLEAN.pkl 10 G -v --seed_list temas_repetitivos.csv

criação de tópicos bertopic : python createTopics.py REsp_completo_EMBEDDING_CLEAN.pkl 10 B -v

calculo de similaridade : python calcSimilarity.py TOPICS_L10CLEAN.pkl temas_repetitivos_EMBEDDING.pkl 6 B

computação de métricas : python metrics.py CLASSFIED_TOPICS_B10CLEAN_BM25.csv
