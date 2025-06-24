import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pathlib

path = "/data/mlproject22" if os.path.exists("/data/mlproject22") else "."
train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
X_train = train_data.drop(columns = "Class")
y_train = train_data["Class"]
