# GLARE: Guided LexRank for Advanced Retrieval in Legal Analysis

This project implements the **GLARE** methodology, which is a system for classifying Brazilian legal documents, specifically **Recursos Especiais**, using unsupervised machine learning techniques. It combines text summarization and similarity evaluation to match legal documents to predefined themes from the Brazilian Superior Court of Justice (STJ).

## Project Overview

The goal of this project is to automate the classification of legal documents based on themes, without the need for labeled training data. The project includes:

- **Text Embedding Generation**: Creation of embeddings for legal documents (Recursos Especiais) and predefined themes.
- **Text Summarization**: Summarizing legal documents using one of four techniques: LexRank, Guided LexRank, BERTopic, or Guided BERTopic.
- **Similarity Calculation**: Evaluating the similarity between document summaries and themes using either BM25 (for text) or cosine similarity (for embeddings).
- **Performance Metrics**: Calculating various metrics to evaluate the accuracy and performance of the classification system.

## Project Structure

The project is composed of the following scripts:

1. **createEmbedding.py**: Generates text embeddings for legal documents and themes using Sentence-BERT.
2. **createTopics.py**: Summarizes legal documents using one of four summarization methods:
   - **LexRank**: A graph-based unsupervised summarization algorithm.
   - **Guided LexRank**: LexRank guided by predefined themes.
   - **BERTopic**: A topic modeling technique using sentence embeddings.
   - **Guided BERTopic**: BERTopic guided by predefined themes.
3. **calcSimilarity.py**: Calculates the similarity between the document summary and the themes.
   - Uses **BM25** for text-based similarity.
   - Uses **cosine similarity** for embedding-based similarity.
4. **metrics.py**: Computes relevant performance metrics (e.g., accuracy, recall, precision) for the classification results.

## Installation

1. Clone the repository:
   ```bash
   git clone ttps://github.com/AILAB-CEFET-RJ/r2t
   cd src
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Step 1: Generating Embeddings

Use createEmbedding.py to generate embeddings for both legal documents (Recursos Especiais) and themes.
- For legal documents:
  python createEmbedding.py REsp_completo.csv recurso recurso --clean --begin_point cabimento -v
- For themes:
  python createEmbedding.py temas_repetitivos.csv tema tema --clean -v

## Step 2: Summarizing Documents
Once embeddings are generated, you can summarize the documents using createTopics.py with one of the summarization methods.
python script.py <corpus_embedding> <size> <type> [--verbose] [--seed_list <seed_list>] [<model>]

### Parameters 
* corpus_embedding: Path to the corpus embeddings file (.pkl file).
* size: Number of sentences or topics to summarize.
* type: Type of topic generation:
   * B: Bertopic
   * G: Guided Bertopic
   * L: Lexrank
   * X: Guided Lexrank
* --verbose: Increase the verbosity of the process.
* --seed_list: Path to the seed list (required for type G or X).
* <model>: Sentence-BERT model used to generate embeddings (optional, default: distiluse-base-multilingual-cased-v1)

### Examples
* Topic generation with BERTopic:
  python script.py corpus.pkl 10 B

* Topic generation with Guided BERTopic:
  python script.py corpus.pkl 10 G --seed_list seeds.csv

* Summary generation with LexRank:
  python script.py corpus.pkl 5 L

* Summary generation with Guided LexRank:
  python script.py corpus.pkl 5 X --seed_list seeds.csv

## Step 3: Calculating Similarity
After summarizing the documents, use calcSimilarity.py to compute the similarity between the document summaries and the themes.
python calcSimilarity.py <corpus_file> <themes_file> <rank> <type>

### Parameters

* <corpus_file> is the path to the corpus file in pickle format.
* <themes_file> is the path to the themes file in pickle format.
* <rank> is the number of top results to retrieve.
* <type> type of similarity
   * B indicates that the BM25 method should be used for similarity calculation.
   * C indicates that the Cosine Similarity method should be used for similarity calculation.

### Usage
For text-based similarity (using BM25):
python calcSimilarity.py <corpus_file> <themes_file> <rank> B

### Output
The program will generate a CSV file with the similarity results. 
The file will be named CLASSIFIED_<corpus_name>_<METHOD>.csv, where <METHOD> is BM25 or COSINE, depending on the similarity method used.
Example Output
* For BM25:
  CLASSIFIED_TOPICS_L10CLEAN_BM25.csv
* For Cosine Similarity:
  CLASSIFIED_TOPICS_L10CLEAN_COSINE.csv

### Notes
* Ensure the input files are in pickle format and contain the expected structure.
* The rank parameter determines the number of items similar to the top to be retrieved and included in the output.


## Step 4: Evaluating Performance
Finally, use metrics.py to calculate metrics and evaluate the system’s performance.
It computes metrics such as Recall, F1-Score, MAP (Mean Average Precision), NDCG (Normalized Discounted Cumulative Gain), and MRR (Mean Reciprocal Rank) based on the provided classified data.

### Usage
python metrics.py CLASSFIED_TOPICS_B10CLEAN_BM25.csv





