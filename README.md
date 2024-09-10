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
   git clone https://github.com/yourusername/GLARE-Legal-Documents.git
   cd GLARE-Legal-Documents
