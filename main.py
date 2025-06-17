import argparse
import evaluate


import numpy as np
import pandas as pd


from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
from common import setup_seed, feature_importance, plot_confusion_matrix
from dataset import read_dataset
from model import get_model


def main(args):
    # Lists to store metrics for each seed
    accuracy = []
    recall = []
    precision = []
    f1 = []

    per_class_accuracy = {cls: [] for cls in range(4)}  # Assuming 4 classes (0, 1, 2, 3)

    # Loop over seeds
    for seed in [100, 200, 300]:
        setup_seed(seed)
        # Load dataset
        X_train, y_train, X_test, y_test = read_dataset(args.train_test_ratio, seed)

        # # # Apply SMOTE oversampling to training data
        if args.smote == True:
            smote = SMOTE(random_state=seed, k_neighbors=min(10, min(Counter(y_train).values())))
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Get and train model (assumes get_model handles grid search if args.grid == True)
        model = get_model(args.model_type, X_train, y_train, args.grid, seed)
        model.fit(X_train, y_train)

        # If grid search is enabled, assume get_model returns the best estimator
        if args.grid:
            # Assuming get_model returns a fitted GridSearchCV object when args.grid == True
            model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model

        # Predict on test set
        y_pred = model.predict(X_test)

        # feature_importance on test set
        if args.shap:
            feature_importance(model, X_test, seed)

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, args.model_type, args.smote)

        # Compute metrics
        acc_evaluate = evaluate.load('evaluate/metrics/accuracy')
        accuracy.append(acc_evaluate.compute(predictions=y_pred, references=y_test)['accuracy'])
        
        recall_evaluate = evaluate.load('evaluate/metrics/recall')
        recall.append(recall_evaluate.compute(predictions=y_pred, references=y_test, average='macro')['recall'])
        
        precision_evaluate = evaluate.load('evaluate/metrics/precision')
        precision.append(precision_evaluate.compute(predictions=y_pred, references=y_test, average='macro')['precision'])
    
        f1_evaluate = evaluate.load('evaluate/metrics/f1')
        f1.append(f1_evaluate.compute(predictions=y_pred, references=y_test, average='macro')['f1'])

    # Print mean and standard deviation for each metric
    print("\nSummary Across Seeds:")
    print(f"Ave Accuracy: {np.mean(accuracy):.4f}, Std Accuracy: {np.std(accuracy):.4f}")
    print(f"Ave Precision (macro): {np.mean(precision):.4f}, Std Precision (macro): {np.std(precision):.4f}")
    print(f"Ave Recall (macro): {np.mean(recall):.4f}, Std Recall (macro): {np.std(recall):.4f}")
    print(f"Ave F1-score (macro): {np.mean(f1):.4f}, Std F1-score (macro): {np.std(f1):.4f}")

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_ratio', type=float, default=0.3, help='train_test_ratio.')
    parser.add_argument('--model_type', choices=['random_forest', 'xgboost', 'balanced_random_forest', 'dnn', 'svm', 'gnn', 'olr'], type=str, default='olr', help='model_type.')
    parser.add_argument('--grid', default=False, help='Enable grid search.')
    parser.add_argument('--smote', default=False, help='Enable grid search.')
    parser.add_argument('--shap', default=False, help='feature importance.')
    args = parser.parse_args()
    main(args)

