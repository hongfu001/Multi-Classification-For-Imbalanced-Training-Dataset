
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def RandomForest(X_train, y_train, grid, seed):
    if grid == True:
        # Define parameter grid for RandomForestClassifier
        param_grid = {
                'n_estimators': [100, 300, 500],  # Number of trees
                'max_depth': [10, 15, 20, None],  # Maximum depth of trees
                'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
                'min_samples_leaf': [1, 2, 4],    # Minimum samples at a leaf node
                'max_features': ['sqrt', 'log2']  # Number of features to consider at each split
            }

            # Initialize RandomForestClassifier
        model = RandomForestClassifier(random_state=seed, n_jobs=-1)

            # Set up GridSearchCV
        grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,  # 5-fold cross-validation
                scoring='f1_macro',  # Optimize for accuracy (can use 'f1_macro' for imbalanced data)
                n_jobs=-1,  # Use all available cores
                verbose=1
        )

    else:
        model = RandomForestClassifier(random_state=seed, n_jobs=-1)

        grid_search = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
        )
    return grid_search
