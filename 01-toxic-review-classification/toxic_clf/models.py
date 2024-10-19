from statistics import mean, stdev
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from tqdm import tqdm


def classifier(dataset, model):    
    if (model == 'rand_for'):
        real_model = RandomForestClassifier()
    elif (model == 'log_reg'):
        real_model = LogisticRegression()
    else:
        real_model = '' #codebert

    print(dataset)
    print(dataset['vectorized'][0])
    X, y = dataset['vectorized'], dataset['is_toxic']
    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scores = []
    for train_index, test_index in tqdm(skf.split(X, y), desc="Cross-Validating"):
        print(train_index, type(train_index))
        print(X)
        X_train, X_test = X[train_index], X[test_index]#[X[ind] for ind in train_index], [X[ind] for ind in test_index]
        y_train, y_test = y[train_index], y[test_index]#[y[ind] for ind in train_index], [y[ind]  for ind in test_index]
        real_model.fit(X_train, y_train)

        y_pred = real_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        scores.append((accuracy, f1))

    print('Some pretty description:')
    print(mean(scores), stdev(scores))
