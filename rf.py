
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics

PIPE_PCA = 'pca'
PIPE_RF = 'rf'

def load_and_split():
    path = "."
    data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
    X = data.drop(columns = "Class")
    y = data["Class"]
    return train_test_split(X, y, random_state=0)

def train(X_train, y_train, clf_params):
    # clf = RandomForestClassifier(random_state=0, verbose=1)
    pca = PCA(random_state=0)
    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    pipeline = Pipeline(steps=[(PIPE_PCA, pca), (PIPE_RF, rf)])
    search = RandomizedSearchCV(estimator=pipeline, param_distributions=clf_params, scoring='roc_auc', refit=True, random_state=0, n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    return search.best_estimator_

def evaluate(clf, X_test, y_test):
    print('roc_auc_score=%f' %(sklearn.metrics.roc_auc_score(y_test, clf.predict(X_test))))
    print('accuracy_score=%f' %(sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))))
    print('precision_score=%f' %(sklearn.metrics.precision_score(y_test, clf.predict(X_test))))
    print('recall_score=%f' %(sklearn.metrics.recall_score(y_test, clf.predict(X_test))))