from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def OrdinalLogisticRegression(X_train, y_train, grid, seed):
    if grid:
        # Define parameter grid for LogisticRegression
        param_grid = {
            'C': [0.1, 1, 10],  # Inverse of regularization strength
            'solver': ['lbfgs', 'liblinear'],  # Solvers suitable for ordinal classification
            'penalty': ['l2'],  # Regularization type (l2 is common for ordinal logistic)
            'max_iter': [1000]  # Ensure convergence
        }

        # Initialize LogisticRegression
        model = LogisticRegression(random_state=seed, multi_class='multinomial')

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='f1_macro',  # Optimize for f1_macro (suitable for imbalanced or ordinal data)
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
    else:
        # Default LogisticRegression model with fixed hyperparameters
        grid_search = LogisticRegression(
            C=1,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=seed,
            multi_class='multinomial'
        )

    return grid_search