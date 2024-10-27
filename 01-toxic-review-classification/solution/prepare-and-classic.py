from pathlib import Path
import datasets
import pandas as pd
import re
from sklearn.model_selection import train_test_split

os.system("git clone https://github.com/WSU-SEAL/ToxiCR.git")

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    contractions = {
        "doesn't": "does not",
        "don't": "do not",
        "we're": "we are",
        "We're": "We are",
        "isn't": "is not",
        "haven't":"have not",
        "hasn't":"has not",
        "aren't":"are not",
        "can't":"can not",
        "couldn't":"could not",
        "wouldn't":"would not",
        "I'm":"I am",
        "i'm":"i am",
        "You're":"You are",
        "you're":"you are",
        "They're":"They are",
        "they're":"they are"
    }
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)

    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'[&^#*]', '', text)

    return text

def prepare(raw_data: Path):
    dataset = pd.read_excel(raw_data)
    dataset.dropna(inplace=True)

    dataset.insert(0, 'cleaned_text', dataset['message'].apply(clean_text))
    dataset.drop_duplicates(inplace=True)
    return dataset

data = prepare('ToxiCR/models/code-review-dataset-full.xlsx')

data.to_excel("./prepared.xlsx")


X, X_eval, y, y_eval = train_test_split(
    data['cleaned_text'], data['is_toxic'], test_size=0.2)

training = pd.DataFrame({'X': X, 'y': y})
training.to_excel("./training.xlsx")

eval = pd.DataFrame({'X': X_eval, 'y': y_eval})
eval.to_excel("./eval.xlsx")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
count_vectors = count_vectorizer.fit_transform(X)
tfidf_vectors = tfidf_vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(
    count_vectors, y, test_size=0.2)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    tfidf_vectors, y, test_size=0.2)


# Logistic Regression with CountVectorizer
simple_log_reg_count = LogisticRegression()
simple_log_reg_count.fit(X_train_count, y_train_count)
y_pred_count = simple_log_reg_count.predict(X_test_count)
f1_count = f1_score(y_test_count, y_pred_count)
print("F1-score для обычной LogisticRegression с CountVectorizer:", f1_count)

# Random Forest with CountVectorizer
simple_rf_count = RandomForestClassifier() # depth
simple_rf_count.fit(X_train_count, y_train_count)
y_pred_count = simple_rf_count.predict(X_test_count)
f1_count = f1_score(y_test_count, y_pred_count)
print("F1-score для обычного RandomForestClassifier с CountVectorizer:",
      f1_count)

# Logistic Regression with TfidfVectorizer
simple_log_reg_tfidf = LogisticRegression()
simple_log_reg_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_count = simple_log_reg_tfidf.predict(X_test_count)
f1_count = f1_score(y_test_count, y_pred_count)
print("F1-score для обычной LogisticRegression с TfidfVectorizer:", f1_count)

# Random Forest with TfidfVectorizer
simple_rf_tfidf = RandomForestClassifier()
simple_rf_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_count = simple_rf_tfidf.predict(X_test_count)
f1_count = f1_score(y_test_count, y_pred_count)
print("F1-score для обычного RandomForestClassifier с TfidfVectorizer:",
      f1_count)

from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=10, shuffle=True)

# ЭТО ДЛЯ CountVectorizer

cv_log_reg_count = simple_log_reg_count
lr_cv_scores = cross_val_score(cv_log_reg_count, count_vectors, y, cv=kf,
                               scoring='f1')
print("Logistic Regression with CountVectorizer - 10-Fold Cross-Validation Scores:")
print(lr_cv_scores)
print("Mean f1_score:", lr_cv_scores.mean())

cv_rf_count = simple_rf_count
rf_cv_scores = cross_val_score(cv_rf_count, count_vectors, y, cv=kf,
                               scoring='f1')
print("Random Forest with CountVectorizer - 10-Fold Cross-Validation Scores:")
print(rf_cv_scores)
print("Mean f1_score:", rf_cv_scores.mean())

# ЭТО ДЛЯ TfidfVectorizer

cv_log_reg_tfidf = simple_log_reg_tfidf
lr_cv_scores = cross_val_score(cv_log_reg_tfidf, tfidf_vectors, y, cv=kf,
                               scoring='f1')
print("Logistic Regression with TfidfVectorizer - 10-Fold Cross-Validation Scores:")
print(lr_cv_scores)
print("Mean f1_score:", lr_cv_scores.mean())

cv_rf_tfidf = simple_rf_tfidf
rf_cv_scores = cross_val_score(cv_rf_tfidf, tfidf_vectors, y, cv=kf,
                               scoring='f1')
print("Random Forest with TfidfVectorizer - 10-Fold Cross-Validation Scores:")
print(rf_cv_scores)
print("Mean f1_score:", rf_cv_scores.mean())

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict

log_reg_count_predictions = cross_val_predict(cv_log_reg_count, count_vectors, y, cv=kf)
log_reg_count_confusion_matrix = confusion_matrix(y, log_reg_count_predictions)
print("Logistic Regression with CountVectorizer Confusion Matrix:")
print(log_reg_count_confusion_matrix)

rf_count_predictions = cross_val_predict(cv_rf_count, count_vectors, y, cv=kf)
rf_count_confusion_matrix = confusion_matrix(y, rf_count_predictions)
print("Random Forest with CountVectorizer Confusion Matrix:")
print(rf_count_confusion_matrix)

log_reg_tfidf_predictions = cross_val_predict(cv_log_reg_tfidf, tfidf_vectors, y, cv=kf)
log_reg_tfidf_confusion_matrix = confusion_matrix(y, log_reg_tfidf_predictions)
print("Logistic Regression with TfidfVectorizer Confusion Matrix:")
print(log_reg_tfidf_confusion_matrix)

rf_tfidf_predictions = cross_val_predict(cv_rf_tfidf, tfidf_vectors, y, cv=kf)
rf_tfidf_confusion_matrix = confusion_matrix(y, rf_tfidf_predictions)
print("Random Forest with TfidfVectorizer Confusion Matrix:")
print(rf_tfidf_confusion_matrix)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

kf = KFold(n_splits=10, shuffle=True)

tr = pd.read_excel("./training.xlsx")

tr.dropna(inplace=True) 

X = tr['X']
y = tr['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lg_cv_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

lg_tf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

lg_param_grid = {
    'vectorizer__max_features': [1000, 5000],
    'vectorizer__min_df': [1, 5],
    #'vectorizer__max_df': [0.8, 0.9],
    #'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__stop_words': [None, 'english'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'newton-cholesky'], # 'sag', 'saga',
    'classifier__penalty': [None, 'elasticnet'], # 'l1', 'l2'
    'classifier__max_iter':[100, 1000]
}

rf_cv_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier())
])

rf_tf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', RandomForestClassifier())
])

rf_param_grid = {
    'vectorizer__max_features': [1000, 5000],
    'vectorizer__min_df': [1, 5],
    #'vectorizer__max_df': [0.8, 0.9],
    #'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__stop_words': [None, 'english'],
    'classifier__n_estimators': [10, 50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

lg_cv_grid_search = GridSearchCV(lg_cv_pipeline, lg_param_grid, cv=kf,
                                 scoring='f1')
lg_cv_grid_search.fit(X_train, y_train)
print("Best parameters for Logistic Regression with CountVectorizer:",
      lg_cv_grid_search.best_params_)
print("Best score for Logistic Regression with CountVectorizer:",
      lg_cv_grid_search.best_score_)

lg_tf_grid_search = GridSearchCV(lg_tf_pipeline, lg_param_grid, cv=kf,
                                 scoring='f1')
lg_tf_grid_search.fit(X_train, y_train)
print("Best parameters for Logistic Regression with TfidfVectorizer:",
      lg_tf_grid_search.best_params_)
print("Best score for Logistic Regression with TfidfVectorizer:",
      lg_tf_grid_search.best_score_)

rf_cv_grid_search = GridSearchCV(rf_cv_pipeline, rf_param_grid, cv=kf,
                                 scoring='f1')
rf_cv_grid_search.fit(X_train, y_train)
print("Best parameters for RandomForest with CountVectorizer:",
      rf_cv_grid_search.best_params_)
print("Best score for RandomForest with CountVectorizer:",
      rf_cv_grid_search.best_score_)

rf_tf_grid_search = GridSearchCV(rf_tf_pipeline, rf_param_grid, cv=kf,
                                 scoring='f1')
rf_tf_grid_search.fit(X_train, y_train)
print("Best parameters for RandomForest with TfidfVectorizer:",
      rf_tf_grid_search.best_params_)
print("Best score for RandomForest with TfidfVectorizer:",
      rf_tf_grid_search.best_score_)

ev = pd.read_excel("./eval.xlsx")

ev.dropna(inplace=True) 

X_eval = ev['X']
y_eval = ev['y']

y_pred = lg_cv_grid_search.predict(X_eval)
report = classification_report(y_eval, y_pred, output_dict=True)

print("lg_cv_grid_search")
print("Accuracy:", accuracy_score(y_eval, y_pred))
print("Precision:", report['weighted avg']['precision'])
print("Recall:", report['weighted avg']['recall'])
print("F1-Score:", report['weighted avg']['f1-score'])

y_pred = lg_tf_grid_search.predict(X_eval)
report = classification_report(y_eval, y_pred, output_dict=True)

print("lg_tf_grid_search")
print("Accuracy:", accuracy_score(y_eval, y_pred))
print("Precision:", report['weighted avg']['precision'])
print("Recall:", report['weighted avg']['recall'])
print("F1-Score:", report['weighted avg']['f1-score'])

y_pred = rf_cv_grid_search.predict(X_eval)
report = classification_report(y_eval, y_pred, output_dict=True)

print("rf_cv_grid_search")
print("Accuracy:", accuracy_score(y_eval, y_pred))
print("Precision:", report['weighted avg']['precision'])
print("Recall:", report['weighted avg']['recall'])
print("F1-Score:", report['weighted avg']['f1-score'])

y_pred = rf_tf_grid_search.predict(X_eval)
report = classification_report(y_eval, y_pred, output_dict=True)

print("rf_tf_grid_search")
print("Accuracy:", accuracy_score(y_eval, y_pred))
print("Precision:", report['weighted avg']['precision'])
print("Recall:", report['weighted avg']['recall'])
print("F1-Score:", report['weighted avg']['f1-score'])
