import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from feature_eng_pipeline import pipeline_nn, process_new_data


class RNANanoporeDataset(Dataset):
    """Dataset used to train and test RNA Nanopore data"""

    def __init__(self, csv_file, v_outpath, s_outpath):
        """Initializes instance of class RNANanoporeDataset.

        Args:
            csv_file (str): Path to the csv file with the nanopore data
        """

        self.df = pd.read_csv(csv_file)
        v, s, X_df, y_df = pipeline_nn(self.df, v_outpath, s_outpath)
        #X_drop = X_df.drop(["transcript_name", "gene_id", "nucleotide_seq"], axis=1).reset_index(drop=True)  

        # keep all trigram columns in input dataset
        self.X = X_df
        self.y = y_df.reset_index(drop=True).squeeze()  
        self.v = v #need the vectorizer 
        self.s = s #need the standardizer

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.X)

    def __getitem__(self, idx):
    # Handle if idx is a tensor (converting to list if needed)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        signal_features = self.X.iloc[idx].values  
        label = self.y.iloc[idx]  

        # Convert to tensors
        signal_features = torch.tensor(signal_features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return signal_features, label
    
    def get_size(self):
        return len(self)

class NewRNANanoporeDataset(Dataset):
    """Dataset to predict new (non training) RNA Nanopore data"""

    def __init__(self, df, vectorizer, standardizer):
        """
        Initializes instance of class NewRNANanoporeDataset.

        Args:
            df (DataFrame): dataframe containing nanopore data
            vectorizer: trained on training data, to transform new data
            standardizer: trained on training data, to transform new data
        """
        
        self.df = df
        X_df = process_new_data(df, vectorizer, standardizer) 

        # keep all trigram columns in input dataset
        self.X = X_df

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.X)

    def __getitem__(self, idx):
    # Handle if idx is a tensor (converting to list if needed)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        signal_features = self.X.iloc[idx].values  

        # Convert to tensors
        signal_features = torch.tensor(signal_features, dtype=torch.float32)

        return signal_features
    
    def get_size(self):
        return len(self)

