from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def SVM(X_train, y_train, grid, seed):
    if grid == True:
        # Define parameter grid for SVC
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization parameter
            'kernel': ['linear', 'rbf'],  # Kernel type
            'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient for 'rbf'
        }

        # Initialize SVC
        model = SVC(random_state=seed)

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='f1_macro',  # Optimize for f1_macro (suitable for imbalanced data)
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
    else:
        # Default SVM model with fixed hyperparameters
        grid_search = SVC(
            C=1,
            kernel='rbf',
            gamma='scale',
            random_state=seed
        )

    return grid_search