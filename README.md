# r2t

## Exemplos  

criação embedding  :    python createEmbedding.py REsp_completo.csv recurso --clean --begin_point cabimento -v

criação de tópicos : python createTopics.py corpus_embedding_clean.pkl 10 L -v

calculo de similaridade : python calcSimilarity.py TOPICS_L10CLEAN.pkl temas_repetitivos_EMBEDDING.pkl 6 B
