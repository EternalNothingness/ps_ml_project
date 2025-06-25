import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
import joblib

import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns

# Define a random seed for reproducible results
random_seed = 100
np.random.seed(random_seed)

def plot_feature_distributions(data, features, target='Class', bins=60, grid_shape=(3, 3), figsize=(14, 10)):

    """
    Plot histograms with KDE overlays of given features grouped by the target class.
    
    Parameters:
    - data: pandas DataFrame containing the dataset
    - features: list of feature names (strings) to plot
    - target: name of the target column (default 'Class')
    - bins: number of bins for histogram (default 60)
    - grid_shape: tuple (rows, cols) defining subplot grid (default 3x3)
    - figsize: tuple defining figure size (default (14, 10))
    """
    num_features = len(features)
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.histplot(
            data=data,
            x=feature,
            hue=target,
            bins=bins,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=axes[i]
        )
        axes[i].set_title(f'Distribution of {feature} by {target}')
        axes[i].grid(True)
    
    # remove any unused subplots if features < grid size
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model, X_test, Y_test):
    """
    Plots the ROC curve for a given model and test data.
    
    Parameters:
    - model: trained classifier with decision_function or predict_proba method
    - X_test: features for test set (scaled if necessary)
    - Y_test: true target labels for test set
    """
    # Get model scores for the positive class
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(Y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_precision(precision, recall, thresholds):
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.show()

def load_split_data(data, correlations, split_method, top_n_features, test_size):
    """
    Loads dataset, selects top correlated features with the target, splits into train/test,
    and scales the features.
    
    Parameters:
    - filepath: str, path to CSV dataset
    - target_col: str, name of the target column
    - top_n_features: int, number of top correlated features to use
    - test_size: float, proportion of test data
    - random_seed: int, random seed for reproducibility
    
    Returns:
    - X_train_scaled, X_test_scaled: scaled feature arrays for train/test
    - Y_train, Y_test: target arrays for train/test
    - scaler: fitted StandardScaler object
    - features: list of selected feature names
    """
    
    features = correlations.head(top_n_features).index.tolist()
    #print("\nTop correlated features with 'Class':\n", correlations.head(27))

    X = data[features].values           # select features,  columns 'FeatureXX' from the dataset
    Y = data['Class'].values         # select target, column 'Class' from the dataset
    
    if (split_method == 1):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_seed, stratify=Y # preserves fraud/non-fraud ratio
        )
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=t_size, random_state=random_seed)
        for train_idx, test_idx in sss.split(X, Y):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # print the shape of the data
    #print(f'X_train.shape: {X_train.shape}')
    #print(f'Y_train.shape: {Y_train.shape}')
    #print(f'X_test.shape: {X_test.shape}')
    #print(f'Y_test.shape: {Y_test.shape}')
    
    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, features

def compute_best_threshold(precision, recall, thresholds):
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = f1_scores.argmax()
    #print(f"Best threshold by F1: {thresholds[best_idx]:.3f}")
    return thresholds[best_idx]

# load data
transactions_dataset = pd.read_csv('transactions.csv')
#print(transactions_dataset.head())
#print(transactions_dataset["Class"].value_counts())

# compute correlations with target
correlations = transactions_dataset.corr()['Class'].drop('Class').abs().sort_values(ascending=False)

feature_range = range(3, 27, 1)
test_sizes = np.arange(0.4, 0.5, 0.1)
best_config = {"f1": 0, "features": 0, "test_size": 0, "threshold": 0}
best_roc_auc = {"score": 0, "features": 0, "test_size": 0}

for t_size in test_sizes:
    for n_features in feature_range:
        print(f"\n=== Top {n_features} Features | Test Size: {t_size:.1f} ===")

        X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, features = load_split_data(
            transactions_dataset, 
            correlations,
            1,    
            n_features,
            t_size)

        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train_scaled, Y_train)
        y_pred = model.predict(X_test_scaled)
        y_scores = model.decision_function(X_test_scaled)
        roc_auc = roc_auc_score(Y_test, y_scores)

        print("ROC-AUC Score:", roc_auc)
        
        report = classification_report(Y_test, y_pred, output_dict=True)
        class_1_metrics = report['1']
        print(f"Class 1 - Precision: {class_1_metrics['precision']:.2f}, "
            f"Recall: {class_1_metrics['recall']:.2f}, "
            f"F1-Score: {class_1_metrics['f1-score']:.2f}")
        
        precision, recall, thresholds = precision_recall_curve(Y_test, y_scores)
        best_threshold = compute_best_threshold(precision, recall, thresholds)
        y_pred_threshold = (y_scores >= best_threshold).astype(int)
        
        reportThreshold = classification_report(Y_test, y_pred_threshold, output_dict=True)
        class_2_metrics = reportThreshold['1']
        print(f"Class 1 (Best Threshold) - Precision: {class_2_metrics['precision']:.2f}, "
            f"Recall: {class_2_metrics['recall']:.2f}, "
            f"F1-Score: {class_2_metrics['f1-score']:.2f}")

        if class_2_metrics['f1-score'] > best_config["f1"]:
            best_config.update({
                "f1": class_2_metrics['f1-score'],
                "features": n_features,
                "test_size": t_size,
                "threshold": best_threshold
            })

        if roc_auc > best_roc_auc["score"]:
            best_roc_auc.update({
                "score": roc_auc,
                "features": n_features,
                "test_size": t_size
            })
        
print("\n=== Best Configuration ===")
print(f"Top {best_config['features']} features | Test size: {best_config['test_size']}")
print(f"Best F1: {best_config['f1']:.2f} at threshold: {best_config['threshold']:.4f}")
print(f"Best ROC-ACU: {roc_auc_score(Y_test, y_scores):.3f}")

print("\n=== Best ROC-AUC Configuration ===")
print(f"Top {best_roc_auc['features']} features | Test size: {best_roc_auc['test_size']}")
print(f"Best ROC-AUC Score: {best_roc_auc['score']:.4f}")

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")