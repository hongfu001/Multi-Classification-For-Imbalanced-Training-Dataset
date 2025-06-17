import os
import shap
import torch
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # 新增绘图库

from sklearn.metrics import confusion_matrix

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def feature_importance(model, X_test, seed):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
            # 定义特征名称列表
    feature_names = [
                'LConn', 'LLen', 'LFrac', 'LAC', 'LSin', 'LBear', 'MHDn', 'NQPDHn', 
                'BtHn', 'TPBtHn', 'Lnkn', 'Lenn', 'AngDn', 'MGLHn', 'MCFn', 'MI', 
                'MWPR', 'MPMR', 'MNOR', 'MBCR', 'MNAR', 'MCAR', 'MFAR'
            ]
    class_names = ['A', 'B', 'C', 'D']

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
    plt.figure()
    shap.summary_plot(shap_values, X_test, 
                                feature_names=feature_names,
                                plot_type="bar",
                                class_names=class_names,
                                show=False,
                                max_display=10)
    plt.title(f"Feature Importance of the variables for HOLC Value")
    plt.tight_layout()
    plt.savefig(f"shap_plots/feature_importance_seed_{seed}.png")
    plt.close()



def plot_confusion_matrix(y_true, y_pred, model_type, smote, output_dir="confusion_matrices"):
    """
    Plot and save confusion matrix for given true and predicted labels.
    """
    plt.rcParams.update({'font.size': 25})  # 改变所有字体大小
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot using seaborn
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['A', 'B', 'C', 'D'], yticklabels=['A', 'B', 'C', 'D'])  # Assuming 4 classes
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type}_{smote}.png'))
    plt.close()