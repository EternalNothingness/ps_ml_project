# ps_ml_project
Closing project of the PS Machine Learning

Logistical Regression:

class_weight='balanced'

"The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))"

As the "transactional" dataset consists of 227 845 

It should compensate the class imbalance by giving more weight to the minority class (1, fraud).
Prevents the model from just predicting the majority class (0, non-fraud) all the time.

class_weight adjusts the training strategy, threshold adjusts the inference strategy.

ROC-AUC is the best between a test size of 0.2 and 0.4.

