#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import yaml
import scanpy as sc
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AE.AE import Autoencoder, GeneExpressionDataset


# load expression dataset

adata = sc.read_h5ad('Data/dataset.h5ad')

# load training data
with open('Data/training_data.pkl','rb') as f:
    data=pickle.load(f)


X_train_tensor = data['X_train_tensor']
X_full = data['full_dataset']  # Full dataset for embeddings
input_dim = X_train_tensor.shape[1]
loader_workers=4


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load parameters
with open('autoencoder_params.yaml', "r") as f:
    best_params=yaml.safe_load(f)



# Extract hidden dimensions based on the suggested params
best_hidden_dims = []
n_hidden_layers = best_params['n_hidden_layers']
for i in range(n_hidden_layers):
    # Append the hidden dimension for each layer
    if f'h_dim_{i}' in best_params:
        best_hidden_dims.append(best_params[f'h_dim_{i}'])


best_latent_dim = best_params['latent_dim']
best_dropout_rate = best_params['dropout_rate']
best_lr = best_params['lr']
best_batch_size = best_params['batch_size']

# Train the final model
final_model = Autoencoder(input_dim, best_latent_dim, best_hidden_dims, best_dropout_rate).to(device)
final_model.load_state_dict(torch.load('models/autoencoder_model.pth'))
final_model.eval() # Set to evaluation mode

# Get embeddings for the full dataset (assuming X_dense is your original scaled data array)
X_full_tensor = torch.FloatTensor(X_full.to_numpy()).to(device) # Use the full scaled data
with torch.no_grad(): # No need to calculate gradients for inference
    embeddings = final_model.encode(X_full_tensor).cpu().numpy() # Get embeddings and move back to CPU

# Store embeddings in the original anndata object (before subsetting genes)
# Or, map them back to the samples in the subsetted anndata object
adata.obsm['X_ae'] = embeddings # Store in .obsm attribute (observations/samples matrix)

print(f"Generated embeddings with shape: {adata.obsm['X_ae'].shape}")

adata.write('Data/adata_with_embeddings.h5ad')  # Save the anndata object with embeddings