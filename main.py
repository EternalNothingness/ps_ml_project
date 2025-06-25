import rf
import numpy as np
import joblib as jl

# load data and split it into train and test set
X_train, X_test, y_train, y_test = rf.load_and_split()

# find hyperparameters and train the random forest
n_features = X_train.shape[1]
step_size = int(n_features/5)
clf_params = {rf.PIPE_PCA + '__n_components': [i for i in range(1, n_features, step_size)], rf.PIPE_RF + '__n_estimators': np.random.randint(10, 100, 100), rf.PIPE_RF + '__max_features': np.random.rand(100), rf.PIPE_RF + '__criterion': ['gini', 'entropy', 'log_loss']}
clf = rf.train(X_train, y_train, clf_params)

# store the model
jl.dump(clf, "pca_rf.jl")

# evaluate the random forest and print the results
rf.evaluate(clf, X_test, y_test)