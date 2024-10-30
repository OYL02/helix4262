import torch
import torch.nn as nn


class ModNet(nn.Module):
    def __init__(self, signal_input_dim):
        super(ModNet, self).__init__()

        # Read-level Encoder: MLP with two hidden layers
        self.encoder = nn.Sequential(
            nn.Linear(signal_input_dim, 150), 
            nn.ReLU(),
            nn.Linear(150, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Single output for binary classification
        )

    def forward(self, signal_features):
        read_level_probs = self.encoder(signal_features)
        return torch.sigmoid(read_level_probs)  

    def noisy_or_pooling(self, read_level_probs):
        """
        :param read_level_probs: Tensor of shape (batch_size, 1)
        :return: Site-level modification probability for each site (batch_size, 1)
        """
        site_level_probs = 1 - torch.prod(1 - read_level_probs, dim=1)
        return site_level_probs
