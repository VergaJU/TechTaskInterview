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

# Get absolute path to AE directory and add it to sys.path
# ae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AE'))
# sys.path.append(ae_path)
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
with open('optuna_parameter.yaml', "r") as f:
    optuna_parameters=yaml.safe_load(f)


# define objective function for Optuna
def objective(trial: optuna.Trial, optuna_parameters, input_dim, loader_workers):
    # --- Hyperparameter Suggestions ---
    latent_dim = trial.suggest_int("latent_dim", *optuna_parameters["latent_dim"]) # Range for latent space size
    dropout_rate = trial.suggest_float("dropout_rate", *optuna_parameters["dropout_rate"]) # Range for dropout
    lr = trial.suggest_float('lr',*optuna_parameters["lr"]) # Learning rate
    batch_size = trial.suggest_categorical('batch_size',optuna_parameters["batch_size"]) # Batch size options
    n_hidden_layers = trial.suggest_int('n_hidden_layers',*optuna_parameters["n_hidden_layers"]) # Number hidden layers


    hidden_dims=[]

    for i in range(n_hidden_layers):
         # Suggest dimensions between the current upper bound and latent_dim+buffer
         # relative to previous/next layer.
        if i == 0: # First hidden layer
             dim = trial.suggest_int(f'h_dim_{i}', int(latent_dim * 1.5), int(input_dim * 0.8))
        else: # Subsequent hidden layers
             # Ensure next layer is smaller than previous
             # Use the size of the previously suggested layer
             prev_dim = hidden_dims[-1]
             dim = trial.suggest_int(f'h_dim_{i}', int(latent_dim * 1.1), int(prev_dim * 0.9))
        hidden_dims.append(dim)

    # --- Model Training ---
    model = Autoencoder(input_dim, latent_dim, hidden_dims, dropout_rate).to(device)
    criterion = nn.MSELoss() # Mean Squared Error Loss for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam optimizer

    # Data Loaders
    train_dataset = GeneExpressionDataset(X_train_tensor)
    val_dataset = GeneExpressionDataset(X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers)

    # Training loop 
    N_EPOCHS = 50

    for epoch in range(N_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_x, _ = model(batch)
            loss = criterion(recon_x, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon_x, _ = model(batch)
                loss = criterion(recon_x, batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # --- Optuna Pruning ---
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the final validation loss
    return avg_val_loss


# Create a study object
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

print("Starting Optuna optimization...")
# Run the optimization for a number of trials
# Adjust n_trials based on available time and computational resources
study.optimize(lambda trial: objective(trial, optuna_parameters, input_dim, loader_workers),  n_trials=100, timeout=60*60*2) # 100 trials, max 2 hours
print("\nOptimization finished.")
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save the best hyperparameters to a YAML file
best_params = trial.params
with open('optuna_best_params.yaml', 'w') as f:
    yaml.dump(best_params, f)