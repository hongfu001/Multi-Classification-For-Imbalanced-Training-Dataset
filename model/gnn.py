import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

def GNN(X_train, y_train, grid, seed):

    """Returns a GCN model with optional hyperparameter tuning."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale features
    scaler = StandardScaler()
    X_train, y_train = scaler.fit_transform(X_train), y_train
    
    
    # Construct k-NN graph
    def construct_graph(X, y, k=5):
        adj = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        x = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index).to(device)
        return data
    
    data = construct_graph(X_train, y_train, k=5)
    
    # Create train/validation split (90/10)
    n_samples = len(y_train)
    train_mask = np.random.choice(n_samples, int(0.9 * n_samples), replace=False)
    val_mask = np.setdiff1d(np.arange(n_samples), train_mask)
    data.train_mask = torch.tensor(train_mask, dtype=torch.long).to(device)
    data.val_mask = torch.tensor(val_mask, dtype=torch.long).to(device)
    
    # Define GCN model
    class GCN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
            self.dropout = torch.nn.Dropout(dropout)
            self.relu = torch.nn.ReLU()
        
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = self.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = self.fc(x)
            return x
    
    def train_gcn(model, data, optimizer, criterion, scheduler, epochs=3000):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            # Validation loss for scheduler
            model.eval()
            with torch.no_grad():
                val_out = model(data)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
            model.train()
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % 100 == 0:
                print(f"Seed {seed}, Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}")
    
    def evaluate_gcn(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1).cpu().numpy()
            true = data.y.cpu().numpy()
            return f1_score(true, pred, average='macro')
    
    if grid:
        # Hyperparameter tuning with Optuna
        def objective(trial):
            hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
            dropout = trial.suggest_float('dropout', 0.3, 0.5)
            lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 1e-2])
            weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3])
            scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.5)
            scheduler_patience = trial.suggest_int('scheduler_patience', 5, 20)
            
            model = GCN(
                input_dim=X_train.shape[1],
                hidden_dim=hidden_dim,
                output_dim=len(np.unique(y_train)),
                dropout=dropout
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
            
            train_gcn(model, data, optimizer, criterion, scheduler, epochs=3000)
            macro_f1 = evaluate_gcn(model, data)
            return macro_f1
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        print(f"Seed {seed} - Best parameters: {best_params}")
        
        # Train final model
        model = GCN(
            input_dim=X_train.shape[1],
            hidden_dim=best_params['hidden_dim'],
            output_dim=len(np.unique(y_train)),
            dropout=best_params['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=best_params['scheduler_factor'],
            patience=best_params['scheduler_patience'],
            verbose=True
        )
    else:
        # Default GCN
        model = GCN(
            input_dim=X_train.shape[1],
            hidden_dim=64,
            output_dim=len(np.unique(y_train)),
            dropout=0.5
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Train model
    train_gcn(model, data, optimizer, criterion, scheduler, epochs=3000)
    
    # Wrapper for scikit-learn compatibility
    class GNNWrapper:
        def __init__(self, model, data, scaler):
            self.model = model
            self.data = data
            self.scaler = scaler
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            test_data = construct_graph(X_scaled, np.zeros(len(X_scaled)), k=5).to(device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(test_data)
                pred = out.argmax(dim=1).cpu().numpy()
            return pred
        
        def fit(self, X, y):
            return self
    
    return GNNWrapper(model, data, scaler)
