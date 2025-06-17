import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

def DNN(X_train, y_train, grid, seed):
    """Returns an ANN model with optional hyperparameter tuning (renamed from GCN)."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale features and apply SMOTE
    scaler = StandardScaler()
    X_train, y_train = scaler.fit_transform(X_train), y_train
    
    # smote = SMOTE(random_state=seed, k_neighbors=min(10, min(Counter(y_train).values())))
    # X_train, y_train = smote.fit_resample(X_train_scaled, y_train)
    # print(f"Seed {seed} - Class distribution after SMOTE: {Counter(y_train)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
    # Create train/validation split (90/10)
    n_samples = len(y_train)
    train_mask = np.random.choice(n_samples, int(0.9 * n_samples), replace=False)
    val_mask = np.setdiff1d(np.arange(n_samples), train_mask)
    
    # Define ANN model
    class ANN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout):
            super(ANN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    def train_ann(model, X_train, y_train, train_mask, val_mask, optimizer, criterion, scheduler, epochs=3000):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out[train_mask], y_train[train_mask])
            loss.backward()
            optimizer.step()
            
            # Validation loss for scheduler
            model.eval()
            with torch.no_grad():
                val_out = model(X_train)
                val_loss = criterion(val_out[val_mask], y_train[val_mask])
            model.train()
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % 100 == 0:
                print(f"Seed {seed}, Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}")
    
    def evaluate_ann(model, X, y):
        model.eval()
        with torch.no_grad():
            out = model(X)
            pred = out.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()
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
            
            model = ANN(
                input_dim=X_train.shape[1],
                hidden_dim=hidden_dim,
                output_dim=len(np.unique(y_train)),
                dropout=dropout
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
            
            train_ann(model, X_train_tensor, y_train_tensor, train_mask, val_mask, optimizer, criterion, scheduler, epochs=3000)
            macro_f1 = evaluate_ann(model, X_train_tensor, y_train_tensor)
            return macro_f1
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        print(f"Seed {seed} - Best parameters: {best_params}")
        
        # Train final model
        model = ANN(
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
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=best_params['scheduler_factor'],
            patience=best_params['scheduler_patience'],
            verbose=True
        )
    else:
        # Default ANN
        model = ANN(
            input_dim=X_train.shape[1],
            hidden_dim=64,
            output_dim=len(np.unique(y_train)),
            dropout=0.5
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Train model
    train_ann(model, X_train_tensor, y_train_tensor, train_mask, val_mask, optimizer, criterion, scheduler, epochs=3000)
    
    # Wrapper for scikit-learn compatibility
    class DNNWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(X_tensor)
                pred = out.argmax(dim=1).cpu().numpy()
            return pred
        
        def fit(self, X, y):
            return self
    
    return DNNWrapper(model, scaler)