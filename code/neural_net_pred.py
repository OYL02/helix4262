import argparse
import os

from joblib import dump, load
import torch
import pandas as pd
from pathlib import Path
from neural_net_preproc import NewRNANanoporeDataset
from neural_net_model import ModNet
from data_pre_process import json_to_csv, data_agg_mean
from torch.utils.data import DataLoader

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def parse_arguments():
    """
    Parses the arguments that are supplied by the user
    
    Returns:
        Namespace: parsed arguments from the user
    """
    parser = argparse.ArgumentParser(description="Predicting m6A modifications on RNA Transcriptomes using trained model")
    required = parser.add_argument_group("required arguments")
    required.add_argument('-dj', '--data-json-path', metavar='', type=str, required=True,
                          help="Path to direct RNA-Seq data, processed by m6Anet, in json format.") #full file path or dont need
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument('-msd', '--modelstate-dict', metavar='', type=str, default=os.path.join(".", "models/state", "model.pth"),
                        help="Filepath where model state (trained model) is stored. (Default: ./models/state/model.pth)") #need full filepath or dont need?
    optional.add_argument('-vp', '--vectorizer-path', metavar='', type=str, default=os.path.join(".", "data_preparators", "vectorizer.joblib"),
                          help="Full filepath to where we want to store the trained vectorizer. (Default: ./data_preparators/vectorizer.joblib)")
    optional.add_argument('-sp', '--standardizer-path', metavar='', type=str, default=os.path.join(".", "data_preparators", "standardizer.joblib"),
                          help="Full filepath to where we want to store the trained standardizer. (Default: ./data_preparators/standardizer.joblib)")
    optional.add_argument('-pp', '--prediction-path', metavar='', type=str,
                          help="Filepath to store predictions in csv format. (otherwise, would be saved in this format: ./prediction/predict_on_datafile.csv)")
    args = parser.parse_args()
    return args

def load_saved_model(model, model_path):
    """
    Load a saved model
    
    Args:
        model: Initialized model instance
        save_dir (str): directory where model is saved
        pth_file (str): name of the model file
    
    Returns:
        model: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")

    return model

def predict_on_new_dataset(model, dataset, batch_size=32, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Make predictions on new dataset/dataset without labels
    
    Args:
        model: trained PyTorch model
        dataset: any unlabelled dataset
        batch_size: batch size for DataLoader
        device: device to run predictions on
    
    Returns:
        tuple: (predictions, probabilities)
    """
    # Create DataLoader for full dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Lists to store predictions
    all_predictions = []
    all_probs = []
    
    print("Making predictions on dataset...")
    
    with torch.no_grad():
        for data in dataloader:
            signal_features = data
            
            # Move data to device
            signal_features = signal_features.to(device)
            
            # Forward pass
            read_level_probs = model(signal_features)
            site_level_probs = model.noisy_or_pooling(read_level_probs) # Removed .squeeze (31/10)

            # Added 31/10 
            if site_level_probs.dim() > 1:
                site_level_probs = site_level_probs.squeeze()
            
            # Store predictions and labels
            predictions = (site_level_probs > 0.5).float()
            
            all_predictions.append(predictions.cpu())
            all_probs.append(site_level_probs.cpu())
    
    # Concatenate all predictions and labels
    predictions = torch.cat(all_predictions)
    probabilities = torch.cat(all_probs)

    return predictions, probabilities

def predict_neural_network(args):
    #pass
    '''
    Predicting m6A modifications on RNA Transcriptomes using trained model

    Args:
        args (Namespace): parsed arguments from the user
        args.data_json_path
        args.modelstate_dict
        args.prediction_path
        args.vectorizer_path
        args.standardizer_path
    '''
    device = DEVICE

    path = Path(args.data_json_path)
    data_name = path.stem 
    data_dir = path.parent

    # data pre process: convert json to csv, read in csv as df and do aggregation
    # can we just make json to csv return a df since data_agg_mean takes in df too?
    data_file = data_name + ".csv"
    data_csv = os.path.join(data_dir, data_file)
    json_to_csv(path, data_csv)
    temp = pd.read_csv(data_csv)
    df = data_agg_mean(temp)

    # feature engineering: vectorizing, standardizing
    vectorizer = load(args.vectorizer_path)
    standardizer = load(args.standardizer_path)
    df_final = NewRNANanoporeDataset(df, vectorizer, standardizer)

    # instantiate model 
    model_instance = ModNet(signal_input_dim=79)

    # loading saved model
    model = load_saved_model(model_instance, args.modelstate_dict)

    # make predictions
    prediction, probabilities = predict_on_new_dataset(model, df_final, batch_size=32, device=device) #predicted class and predicted probability
    print("Prediction complete")
    prob = pd.DataFrame(probabilities.numpy().tolist(), columns=["score"])

    # merge predicted probabilities back to original dataframe that has transcript_id, transcript_position
    merged = pd.concat([df, prob], axis=1)
    final = merged[["transcript_name", "json_position", "score"]]
    final.rename(columns={'transcript_name':'transcript_id', 'json_position': 'transcript_position', 'score': 'score'}, inplace=True)

    # output final df to csv
    if args.prediction_path:
        prediction_path = args.prediction_path
    else:
        model_name = model.__class__.__name__
        file_name = model_name+"_predict_on_"+data_name+".csv"
        prediction_path = os.path.join(".", "prediction", file_name)

    os.makedirs(os.path.dirname(prediction_path),exist_ok=True)
    final.to_csv(prediction_path, index=False)

    print_statement = "Prediction saved in " + prediction_path
    print(print_statement)

if __name__ == "__main__":
    torch.manual_seed(42)
    args = parse_arguments()
    predict_neural_network(args)