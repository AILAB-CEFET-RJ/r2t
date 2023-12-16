# r2t

## Exemplos  

criação embedding recursos :    python createEmbedding.py REsp_completo.csv recurso --clean --begin_point cabimento -v

criação embedding temas : python createEmbedding.py temas_repetitivos.csv tema --clean -v

criação de tópicos : python createTopics.py REsp_completo_EMBEDDING_CLEAN.pkl 10 L -v

calculo de similaridade : python calcSimilarity.py TOPICS_L10CLEAN.pkl temas_repetitivos_EMBEDDING.pkl 6 B
