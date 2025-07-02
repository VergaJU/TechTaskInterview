#!/usr/bin/python3
import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import yaml

import sys
import os


from AE.AE import Autoencoder, GeneExpressionDataset


# load training data
with open('Data/training_data.pkl','rb') as f:
    data=pickle.load(f)

X_train_tensor = data['X_train_tensor']
X_val_tensor = data['X_val_tensor']
input_dim = data['input_dim']
loader_workers=4


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load parameters
with open('optuna_best_params.yaml', "r") as f:
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
final_criterion = nn.MSELoss()
final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr)

# Data Loaders for final training WITH validation for early stopping
train_dataset_final = GeneExpressionDataset(X_train_tensor)
val_dataset_final = GeneExpressionDataset(X_val_tensor) # Use the validation set again
train_loader_final = DataLoader(train_dataset_final, batch_size=best_batch_size, shuffle=True, num_workers=loader_workers)
val_loader_final = DataLoader(val_dataset_final, batch_size=best_batch_size, shuffle=False, num_workers=loader_workers)

# --- Early Stopping Parameters ---
patience = 20 
min_delta = 1e-4 # Minimum change

# --- Training Parameters ---
MAX_N_EPOCHS = 500 


print("\nTraining final model with best hyperparameters and Early Stopping...")

best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None # To store the model state with the best validation loss

for epoch in range(MAX_N_EPOCHS):
    # --- Training Phase ---
    final_model.train()
    total_train_loss = 0
    for batch in train_loader_final:
        batch = batch.to(device)
        final_optimizer.zero_grad()
        recon_x, _ = final_model(batch)
        loss = final_criterion(recon_x, batch)
        loss.backward()
        final_optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader_final)

    # --- Validation Phase ---
    final_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader_final:
            batch = batch.to(device)
            recon_x, _ = final_model(batch)
            loss = final_criterion(recon_x, batch)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader_final)

    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        # Save the model state dictionary
        best_model_state = final_model.state_dict().copy() # Store a copy!
        print(f'Epoch {epoch+1}/{MAX_N_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Improved - Saving Model)')
    else:
        epochs_without_improvement += 1
        print(f'Epoch {epoch+1}/{MAX_N_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} ({epochs_without_improvement}/{patience} patience)')

    if epochs_without_improvement >= patience:
        print(f'\nEarly stopping triggered after {epoch+1} epochs (patience {patience} exceeded).')
        break

# --- Load the best model state ---
if best_model_state is not None:
    final_model.load_state_dict(best_model_state)
    print("Loaded model state from epoch with best validation loss.")
else:
    print("No model state was saved (did not improve). Using the model from the last epoch.") # Should not happen if training runs for > 0 epochs

print("Final model training complete.")

# ocreate models directory
if not os.path.exists('models'):
    os.makedirs('models')
# Save the trained model (the one with the best validation performance)
torch.save(final_model.state_dict(), 'models/autoencoder_model.pth')
print("Saved final model state_dict (with best validation loss) to 'models/autoencoder_model.pth'")