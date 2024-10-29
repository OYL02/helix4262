import argparse

import numpy as np
import torch
import tqdm
from neural_net_model import ModNet
from neural_net_preproc import RNANanoporeDataset
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# TODO: implement model checkpoint-dict, modelstate-dict, evalresults-path? 
# TODO: might also need to implement labels_path and include preprocessing as part of the training workflow. depends.
def parse_arguments():
    """Parses the arguments that are supplied by the user
    
    Returns:
        Namespace: parsed arguments from the user
    """
    parser = argparse.ArgumentParser(description="Trains the Neural Network for predicting m6a modifications on RNA Transcriptomes")
    required = parser.add_argument_group("required arguments")
    required.add_argument("-dp", '--data-path', metavar='', type='str', required=True,
                          help="Full path to the .csv file including features and labels for training")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument('-ts', '--train-size', metavar='', default=0.8,
                          help='proportion of dataset used for training. Should be a float between 0 and 1. Default is 0.8.')
    optional.add_argument('-ne', '--num-epochs', metavar='', default=10,
                          help='number of epochs used for training. Default is 10.')
    optional.add_argument('-lr', '--learning-rate', metavar='', default=0.001,
                          help='Learning rate for training the neural network. Default is 0.001.')
    optional.add_argument('-bs', '--batch-size', metavar='', default=256,
                          help='Number of datapoints in each batch used to train the neural network. Default is 0.001.')
    return args

def evaluate_model(model, testloader, criterion, device=DEVICE):
    """Evaluates the performance of the Neural Network based on the Area under ROC and PRC

    Args:
        model (ModNet): neural network model that was used for training
        testloader (DataLoader): RNANanoporeDataset object created during the training process.
        criterion (BCEWithLogitsLoss): loss function optimized during training process.
        device (str): device used for training. Can take values 'cpu', 'mps', and 'cuda'. 

    Returns:
        tuple: tuple with average loss per row, area under ROC curve and area under PR curve
    """
    model.eval()  
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in tqdm(testloader, desc="Evaluating", leave=False):  # Wrap testloader with tqdm
            signal_features, labels = data
            
            # Move data to device
            signal_features = signal_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            read_level_probs = model(signal_features)
            site_level_probs = model.noisy_or_pooling(read_level_probs).squeeze() 

            # Ensure single dimension tensor
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

def train_model_with_checks(args):
    """
    Training loop with dimension checks.

    Args:
        args (Namespace): parsed arguments from the user 
    """
    device = DEVICE

    rna_data = RNANanoporeDataset(data_path=args.data_path)

    # perform train test split on data
    train_size = int(args.train_size * len(rna_data))
    test_size = len(rna_data) - train_size
    trainset, testset = random_split(rna_data, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # get number of signal features
    sample_features, _ = next(iter(trainloader))
    expected_input_dim = sample_features.shape[1]

    # instantiate model using number of signal features; set to training mode
    model = ModNet(signal_input_dim=79)
    model.to(device)
    model.train()

    # define number of epochs, optimizer and loss function to optimize
    # default num_epochs is 10, learning rate is 0.001
    num_epochs = args.num_epochs
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0

        loop = tqdm(enumerate(trainloader), total=len(trainloader))
        
        for i, data in loop:
            signal_features, labels = data
            
            # Check dimensions before forward pass
            current_input_dim = signal_features.shape[1]
            if current_input_dim != expected_input_dim:
                raise ValueError(
                    f"Input dimension mismatch! Model expects {expected_input_dim} "
                    f"features but got {current_input_dim} features. "
                    f"Full input shape: {signal_features.shape}"
                )
            
            signal_features = signal_features.to(device)
            labels = labels.to(device).float()
            
            try:
                # Forward pass
                read_level_probs = model(signal_features)
                site_level_probs = model.noisy_or_pooling(read_level_probs).squeeze()

                if site_level_probs.dim() > 1:
                    site_level_probs = site_level_probs.squeeze()
                
                # Compute loss
                loss = criterion(site_level_probs, labels)
                
                # Zero gradients, backward pass, and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Batch [{i+1}/{len(trainloader)}], '
                          f'Loss: {running_loss/10:.4f}')
                    running_loss = 0.0
                    
            except RuntimeError as e:
                print(f"\nError in batch {i}:")
                print(f"Input shape: {signal_features.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Error message: {str(e)}")
                raise e
    
    print('Model Training has ended. \nNow evaluating results...')

    full_data = DataLoader(rna_data, batch_size=256, shuffle=True)
    evaluate_model(model, full_data, criterion, device=DEVICE)

if __name__ == "__main__":
    torch.manual_seed(42)
    args = parse_arguments()
    train_model_with_checks(args)