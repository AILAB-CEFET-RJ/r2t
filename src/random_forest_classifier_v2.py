import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

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
    r = np.asfarray(r)[:k]
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

# Carregar os dados
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Extrair as colunas de interesse
texts_train = X_train['recurso']
labels_train = X_train['num_tema_cadastrado']

texts_test = X_test['recurso']
labels_test = X_test['num_tema_cadastrado']

# Transformar texto em vetores TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(texts_train)
X_test_tfidf = vectorizer.transform(texts_test)
y_train = labels_train
y_test = labels_test

# Configurar Stratified K-Fold Cross-Validation com shuffle
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
precision_scores = []
recall_scores = []
f1_scores = []
recall_at_6_scores = []
map_at_6_scores = []
ndcg_at_6_scores = []

# Obter todas as classes
all_classes = np.unique(y_train)

for fold, (train_index, val_index) in enumerate(kf.split(X_train_tfidf, y_train)):
    print(f"Starting fold {fold + 1}")
    
    X_fold_train, X_val = X_train_tfidf[train_index], X_train_tfidf[val_index]
    y_fold_train, y_val = y_train[train_index], y_train[val_index]
    
    # Aplicar reamostragem para aumentar as classes minoritárias
    ros = RandomOverSampler(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = ros.fit_resample(X_fold_train, y_fold_train)
    
    print("Training model")
    # Treinar o modelo
    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    classifier.fit(X_fold_train_resampled, y_fold_train_resampled)
    
    print("Predictions")
    # Fazer previsões
    y_val_pred_proba = classifier.predict_proba(X_val)
    y_val_pred = classifier.predict(X_val)

    # Obter as 6 sugestões para cada texto com as probabilidades correspondentes
    top_suggestions = []
    for probs in y_val_pred_proba:
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:6]
        suggestions = [classifier.classes_[index] for index in sorted_indices]
        top_suggestions.append(suggestions)

    # Avaliar o modelo usando apenas a sugestão mais provável (primeira sugestão)
    top_1_suggestions = [suggestions[0] for suggestions in top_suggestions]
    report = classification_report(y_val, top_1_suggestions, zero_division=0, output_dict=True)

    print("Calculating metrics")
    accuracies.append(report['accuracy'])
    precision_scores.append(report['weighted avg']['precision'])
    recall_scores.append(report['weighted avg']['recall'])
    f1_scores.append(report['weighted avg']['f1-score'])
    recall_at_6_scores.append(recall_at_k(y_val, top_suggestions, k=6))
    map_at_6_scores.append(average_precision_at_k(y_val, top_suggestions, k=6))
    ndcg_at_6_scores.append(ndcg_at_k(y_val, top_suggestions, k=6))

    # Contar a quantidade de registros para cada label distinto no conjunto de validação
    label_counts = y_val.value_counts().to_dict()

    print(f"Completed fold {fold + 1}")

# Avaliar o modelo no conjunto de teste final
y_test_pred_proba = classifier.predict_proba(X_test_tfidf)
y_test_pred = classifier.predict(X_test_tfidf)

# Obter as 6 sugestões para cada texto com as probabilidades correspondentes
top_suggestions_test = []
for probs in y_test_pred_proba:
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:6]
    suggestions = [classifier.classes_[index] for index in sorted_indices]
    top_suggestions_test.append(suggestions)

# Avaliar o modelo usando apenas a sugestão mais provável (primeira sugestão)
top_1_suggestions_test = [suggestions[0] for suggestions in top_suggestions_test]
report_test = classification_report(y_test, top_1_suggestions_test, zero_division=0, output_dict=True)

# Calcular métricas no conjunto de teste
test_accuracy = report_test['accuracy']
test_precision = report_test['weighted avg']['precision']
test_recall = report_test['weighted avg']['recall']
test_f1 = report_test['weighted avg']['f1-score']
test_recall_at_6 = recall_at_k(y_test, top_suggestions_test, k=6)
test_map_at_6 = average_precision_at_k(y_test, top_suggestions_test, k=6)
test_ndcg_at_6 = ndcg_at_k(y_test, top_suggestions_test, k=6)

# Calcular a média das métricas de validação
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_recall_at_6 = np.mean(recall_at_6_scores)
avg_map_at_6 = np.mean(map_at_6_scores)
avg_ndcg_at_6 = np.mean(ndcg_at_6_scores)

# Salvar as métricas em um arquivo de relatório
with open('random_forest_v2_classification_report.txt', 'w') as f:
    for fold, (accuracy, precision, recall, f1, recall_at_6, map_at_6, ndcg_at_6) in enumerate(zip(accuracies, precision_scores, recall_scores, f1_scores, recall_at_6_scores, map_at_6_scores, ndcg_at_6_scores)):
        f.write(f'Fold {fold + 1}\n')
        f.write(f'Accuracy: {accuracy:.6f}\n')
        f.write(f'Precision: {precision:.6f}\n')
        f.write(f'Recall: {recall:.6f}\n')
        f.write(f'F1 Score: {f1:.6f}\n')
        f.write(f'Recall@6: {recall_at_6:.6f}\n')
        f.write(f'MAP@6: {map_at_6:.6f}\n')
        f.write(f'NDCG@6: {ndcg_at_6:.6f}\n')
        
        # Adicionar a contagem de registros para cada label distinto
        f.write('Label Counts in Validation Set:\n')
        for label, count in label_counts.items():
            f.write(f'Label {label}: {count}\n')
        f.write('\n')

    f.write('Average Metrics on Validation Folds\n')
    f.write(f'Average Accuracy: {avg_accuracy:.6f}\n')
    f.write(f'Average Precision: {avg_precision:.6f}\n')
    f.write(f'Average Recall: {avg_recall:.6f}\n')
    f.write(f'Average F1 Score: {avg_f1:.6f}\n')

    f.write('\nMetrics on Test Set\n')
    f.write(f'Test Accuracy: {test_accuracy:.6f}\n')
    f.write(f'Test Precision: {test_precision:.6f}\n')
    f.write(f'Test Recall: {test_recall:.6f}\n')
    f.write(f'Test F1 Score: {test_f1:.6f}\n')
    f.write(f'Test Recall@6: {test_recall_at_6:.6f}\n')
    f.write(f'Test MAP@6: {test_map_at_6:.6f}\n')
    f.write(f'Test NDCG@6: {test_ndcg_at_6:.6f}\n')

print(f'Average Accuracy on Validation Folds: {avg_accuracy:.6f}')
print(f'Average Precision on Validation Folds: {avg_precision:.6f}')
print(f'Average Recall on Validation Folds: {avg_recall:.6f}')
print(f'Average F1 Score on Validation Folds: {avg_f1:.6f}')
print(f'Average Recall@6 on Validation Folds: {avg_recall_at_6:.6f}')
print(f'Average MAP@6 on Validation Folds: {avg_map_at_6:.6f}')
print(f'Average NDCG@6 on Validation Folds: {avg_ndcg_at_6:.6f}')

print(f'Test Accuracy: {test_accuracy:.6f}')
print(f'Test Precision: {test_precision:.6f}')
print(f'Test Recall: {test_recall:.6f}')
print(f'Test F1 Score: {test_f1:.6f}')
print(f'Test Recall@6: {test_recall_at_6:.6f}')
print(f'Test MAP@6: {test_map_at_6:.6f}')
print(f'Test NDCG@6: {test_ndcg_at_6:.6f}')