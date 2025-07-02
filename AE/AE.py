import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

# Define a Dataset class
class GeneExpressionDataset(data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, index):
        return self.data_tensor[index]

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_rate):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims: # Create layers for each hidden layer
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU()) # ReLU activation
            if dropout_rate > 0:
                 encoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Latent Space
        encoder_layers.append(nn.Linear(current_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers) # create encoder

        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        # Reverse the hidden layer dimensions for the decoder
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU()) # ReLU activation
            if dropout_rate > 0:
                 decoder_layers.append(nn.Dropout(dropout_rate))

            current_dim = h_dim

        # Output layer
        decoder_layers.append(nn.Linear(current_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z # Return both reconstruction and latent vector

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

