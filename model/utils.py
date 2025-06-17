from .randomforest import RandomForest
from .balancedrandomforest import BalancedRandomForest
from .dnn import DNN
from .xgboost import XGBoost
from .svm import SVM
from .gnn import GNN
from .olr import OrdinalLogisticRegression


def get_model(model_type, X_train, y_train, grid, seed):
    if model_type == 'random_forest':
        model = RandomForest(X_train, y_train, grid, seed)
        
    elif model_type == 'xgboost':
        model = XGBoost(X_train, y_train, grid, seed)
    
    elif model_type == 'balanced_random_forest':
        model = BalancedRandomForest(X_train, y_train, grid, seed)
    
    elif model_type == 'dnn':
        model = DNN(X_train, y_train, grid, seed)
    
    elif model_type == 'svm':
        model = SVM(X_train, y_train, grid, seed)

    elif model_type == 'gnn':
        model = GNN(X_train, y_train, grid, seed)

    elif model_type == 'olr':
        model = OrdinalLogisticRegression(X_train, y_train, grid, seed)


    else:
        print("Error Model")
    return model
