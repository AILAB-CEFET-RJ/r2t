import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

def recall_at_k(y_true, y_pred, k=6):
    recalls = []
    for true, pred in zip(y_true, y_pred):
        recalls.append(int(true in pred[:k]))
    return np.mean(recalls)

def average_precision_at_k(y_true, y_pred, k=6):
    aps = []
    for true, pred in zip(y_true, y_pred):
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(pred[:k]):
            if p == true:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        if num_hits > 0:
            aps.append(score / min(len(pred), k))
        else:
            aps.append(0.0)
    return np.mean(aps)

def dcg_at_k(r, k=6):
    r = np.asarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.0

def ndcg_at_k(y_true, y_pred, k=6):
    ndcgs = []
    for true, pred in zip(y_true, y_pred):
        r = [1 if p == true else 0 for p in pred[:k]]
        dcg = dcg_at_k(r, k)
        idcg = dcg_at_k(sorted(r, reverse=True), k)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs)

# Carregar o arquivo completo
df = pd.read_csv('REsp_completo.csv')

# Amostra dos dados
print('Amostra de dados do corpus completo')
print(df.head())

# Filtrar classes com mais de um registro
counts = df['num_tema_cadastrado'].value_counts()
df_filtered = df[df['num_tema_cadastrado'].isin(counts[counts > 1].index)]

# Extrair as colunas de interesse
texts = df_filtered['recurso']
labels = df_filtered['num_tema_cadastrado']

# Transformar texto em vetores TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Configurar validação cruzada estratificada com 10 folds
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
precision_scores = []
recall_scores = []
f1_scores = []
recall_at_6_scores = []
map_at_6_scores = []
ndcg_at_6_scores = []

results = []
detailed_results = []

# Iterar sobre cada fold
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Treinar o modelo com calibração
    base_classifier = OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    classifier = CalibratedClassifierCV(base_classifier, method='sigmoid', cv='prefit')
    base_classifier.fit(X_train, y_train)
    classifier.fit(X_train, y_train)

    # Fazer previsões
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)

    # Obter as 6 sugestões para cada texto com as probabilidades correspondentes
    top_suggestions = []
    for probs in y_pred_proba:
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:6]
        suggestions = [classifier.classes_[index] for index in sorted_indices]
        top_suggestions.append(suggestions)
        
    # Avaliar o modelo usando apenas a sugestão mais provável (primeira sugestão)
    top_1_suggestions = [suggestions[0] for suggestions in top_suggestions]
    report = classification_report(y_test, top_1_suggestions, zero_division=0, output_dict=True)

    accuracies.append(report['accuracy'])
    precision_scores.append(report['weighted avg']['precision'])
    recall_scores.append(report['weighted avg']['recall'])
    f1_scores.append(report['weighted avg']['f1-score'])
    recall_at_6_scores.append(recall_at_k(y_test, top_suggestions, k=6))
    map_at_6_scores.append(average_precision_at_k(y_test, top_suggestions, k=6))
    ndcg_at_6_scores.append(ndcg_at_k(y_test, top_suggestions, k=6))

    # Salvar y_true e recall para cada fold
    for true, pred in zip(y_test, top_suggestions):
        recall_value = int(true in pred[:6])
        results.append([true, recall_value])
        detailed_results.append([true, pred])
        


# Calcular a média das métricas ao longo dos folds
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_recall_at_6 = np.mean(recall_at_6_scores)
avg_map_at_6 = np.mean(map_at_6_scores)
avg_ndcg_at_6 = np.mean(ndcg_at_6_scores)

# Calcular o desvio padrão das métricas
std_accuracy = np.std(accuracies)
std_precision = np.std(precision_scores)
std_recall = np.std(recall_scores)
std_f1 = np.std(f1_scores)
std_recall_at_6 = np.std(recall_at_6_scores)
std_map_at_6 = np.std(map_at_6_scores)
std_ndcg_at_6 = np.std(ndcg_at_6_scores)

# Salvar as métricas em um arquivo de relatório
with open('xgboost_unbalanced.txt', 'w') as f:
    for fold, (accuracy, precision, recall, f1, recall_at_6, map_at_6, ndcg_at_6) in enumerate(zip(accuracies, precision_scores, recall_scores, f1_scores, recall_at_6_scores, map_at_6_scores, ndcg_at_6_scores)):
        f.write(f'Fold {fold + 1}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Recall@6: {recall_at_6}\n')
        f.write(f'MAP@6: {map_at_6}\n')
        f.write(f'NDCG@6: {ndcg_at_6}\n\n')

    f.write('Average Metrics\n')
    f.write(f'Average Accuracy: {avg_accuracy}\n')
    f.write(f'Average Precision: {avg_precision}\n')
    f.write(f'Average Recall: {avg_recall}\n')
    f.write(f'Average F1 Score: {avg_f1}\n')
    f.write(f'Average Recall@6: {avg_recall_at_6}\n')
    f.write(f'Average MAP@6: {avg_map_at_6}\n')
    f.write(f'Average NDCG@6: {avg_ndcg_at_6}\n')

    f.write('\nStandard Deviation of Metrics\n')
    f.write(f'Std Accuracy: {std_accuracy}\n')
    f.write(f'Std Precision: {std_precision}\n')
    f.write(f'Std Recall: {std_recall}\n')
    f.write(f'Std F1 Score: {std_f1}\n')
    f.write(f'Std Recall@6: {std_recall_at_6}\n')
    f.write(f'Std MAP@6: {std_map_at_6}\n')
    f.write(f'Std NDCG@6: {std_ndcg_at_6}\n')

print(f'Average Accuracy: {avg_accuracy}')
print(f'Average Precision: {avg_precision}')
print(f'Average Recall: {avg_recall}')
print(f'Average F1 Score: {avg_f1}')
print(f'Average Recall@6: {avg_recall_at_6}')
print(f'Average MAP@6: {avg_map_at_6}')
print(f'Average NDCG@6: {avg_ndcg_at_6}')
