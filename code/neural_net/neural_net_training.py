import copy

import numpy as np
import torch
import tqdm
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score,
                             roc_curve)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class EarlyStopper:
    """
    Early stopping to prevent overfitting and save best model
    """
    def __init__(self, patience=5, min_delta=0, path='best_model.pth'):
        """
        Args:
            patience (int): Number of epochs to wait before early stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            path (str): Path to save the best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, model, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(model)
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model):
        """Save model when validation loss decreases"""
        torch.save(model.state_dict(), self.path)
        self.best_model = copy.deepcopy(model)
    
    def load_best_model(self):
        """Load the best model"""
        return self.best_model

def train_model_with_early_stopping(model, trainloader, valloader, criterion, optimizer, 
                                  num_epochs=100, patience=5, device=DEVICE):
    """
    Training loop with early stopping
    
    Args:
        model: ModNet instance
        trainloader: Training data loader
        valloader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on
    """
    # Initialize early stopper
    early_stopper = EarlyStopper(patience=patience, path='best_model.pth')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_roc_auc': [],
        'val_pr_auc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        loop = tqdm(enumerate(trainloader),
                    total=len(trainloader))

        for i, (signal_features, labels) in loop:
            signal_features = signal_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            read_level_probs = model(signal_features)
            site_level_probs = model.noisy_or_pooling(read_level_probs).squeeze()
            
            # Compute loss
            loss = criterion(site_level_probs, labels.float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{i+1}/{len(trainloader)}], '
                      f'Loss: {train_loss/10:.4f}')
                train_loss = 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for signal_features, labels in valloader:
                signal_features = signal_features.to(device)
                labels = labels.to(device)
                
                read_level_probs = model(signal_features)
                site_level_probs = model.noisy_or_pooling(read_level_probs).squeeze()
                
                loss = criterion(site_level_probs, labels.float())
                val_loss += loss.item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(site_level_probs.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(valloader)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        roc_auc = roc_auc_score(all_labels, all_predictions)
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        pr_auc = auc(recall, precision)
        
        # Update history
        history['train_loss'].append(train_loss / len(trainloader))
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(roc_auc)
        history['val_pr_auc'].append(pr_auc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'ROC-AUC: {roc_auc:.4f}')
        print(f'PR-AUC: {pr_auc:.4f}')
        
        # Early stopping check
        if early_stopper(model, val_loss):
            print("Early stopping triggered!")
            break
    
    # Load best model
    best_model = early_stopper.load_best_model()
    
    return best_model, history

def evaluate_model(model, testloader, criterion, device=DEVICE):
    model.eval()  
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in testloader:
            signal_features, labels = data
            
            # Move data to device
            signal_features = signal_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            read_level_probs = model(signal_features)
            site_level_probs = model.noisy_or_pooling(read_level_probs).squeeze() 

            # ensure single dimension tensor
            if site_level_probs.dim() > 1:
                site_level_probs = site_level_probs.squeeze()

            # Compute loss
            loss = criterion(site_level_probs, labels.float())
            total_loss += loss.item()

            # Collect predictions and labels for ROC and PR AUC
            all_labels.append(labels.cpu())
            all_predictions.append(site_level_probs.cpu())

    # Convert lists to tensors
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    # Compute ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    pr_auc = auc(recall, precision)

    # Average loss
    avg_loss = total_loss / len(testloader)
    
    print(f'Test Loss: {avg_loss:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}')
    
    return avg_loss, roc_auc, pr_auc
