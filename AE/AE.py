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



class HiddenBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.0):
        super(HiddenBlock, self).__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),
            nn.ReLU()
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_rate):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims: # Create layers for each hidden layer
            encoder_layers.append(HiddenBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim
        # Latent Space
        encoder_layers.append(nn.Linear(current_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers) # create encoder

        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        # Reverse the hidden layer dimensions for the decoder
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(HiddenBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim

        # Output layer
        decoder_layers.append(nn.Linear(current_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z # Return both reconstruction and latent vector

    # def forward(self, x, noise_std=0.1):
    #     noisy_x = x + torch.randn_like(x) * noise_std  # add Gaussian noise
    #     z = self.encoder(noisy_x)
    #     recon_x = self.decoder(z)
    #     return recon_x, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

