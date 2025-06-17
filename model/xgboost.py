from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def XGBoost(X_train, y_train, grid, seed):
    if grid:
        # Define parameter grid for XGBClassifier
        param_grid = {
            'n_estimators': [100, 300, 500],  # Number of boosting rounds
            'max_depth': [3, 6, 9],           # Maximum depth of trees
            'learning_rate': [0.01, 0.1, 0.3], # Step size shrinkage
            'subsample': [0.6, 0.8, 1.0],     # Fraction of samples used per tree
            'colsample_bytree': [0.6, 0.8, 1.0] # Fraction of features used per tree
        }

        # Initialize XGBClassifier
        model = XGBClassifier(random_state=seed, n_jobs=-1, objective='multi:softmax', num_class=4)

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  
            scoring='f1_macro',  
            n_jobs=-1,  
            verbose=1,
        )

    else:
        # Default XGBClassifier with fixed parameters
        grid_search = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            objective='multi:softmax',
            num_class=4  
        )

    return grid_search