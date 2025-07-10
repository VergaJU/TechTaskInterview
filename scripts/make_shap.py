import os
import sys
import gseapy
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import random
import matplotlib.pyplot as plt
from IPython.display import Markdown,display, Image, SVG
import io 
import shap
import shap.maskers as maskers 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AE.AE import Autoencoder
from AE.AEclassifier import AEClassifier, ClassificationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Load parameters
# Autoencoder params:
with open('autoencoder_params.yaml', "r") as f:
    best_params=yaml.safe_load(f)
# Classifier params
with open('classifier_params.yaml', "r") as f:
    classifier_params=yaml.safe_load(f)

# load lables
with open('Data/training_classifier_data.pkl','rb') as f:
    data=pickle.load(f)
    labels=data['labels_names']
    num_classes=data['num_classes']
    del data
    
# load gene names
with open('Data/training_data.pkl', 'rb') as f:
    data=pickle.load(f)
    genes=data['genes']
    full_data=data['full_dataset']
    del data

with open('models/standard_scaler.pkl','rb') as f:
    ss=pickle.load(f)




# recreate model

# Extract hidden dimensions based on the suggested params
best_hidden_dims = []
n_hidden_layers = best_params['n_hidden_layers']
for i in range(n_hidden_layers):
    # Append the hidden dimension for each layer
    if f'h_dim_{i}' in best_params:
        best_hidden_dims.append(best_params[f'h_dim_{i}'])


best_latent_dim = best_params['latent_dim']
best_dropout_rate=best_params['dropout_rate']
input_dim=len(genes)
# Extract hidden dimensions based on the suggested params
classifier_hidden_dims = []
n_hidden_layers = classifier_params['n_hidden_layers']
for i in range(n_hidden_layers):
    # Append the hidden dimension for each layer
    if f'h_dim_{i}' in classifier_params:
        classifier_hidden_dims.append(classifier_params[f'h_dim_{i}'])
classifier_dropout_rate=classifier_params['dropout_rate']

AE_arch = Autoencoder(input_dim,
                      best_latent_dim,
                      best_hidden_dims,
                     best_dropout_rate)


classifier_model = AEClassifier(AE_arch.encoder,num_classes=num_classes, 
                                        latent_dim=best_latent_dim,
                                        hidden_dims=classifier_hidden_dims, 
                                        dropout_rate=classifier_dropout_rate).to(device)
classifier_model.load_state_dict(torch.load('models/classifier_model.pth', map_location=device))
classifier_model = nn.Sequential(
    classifier_model,
    nn.Softmax(dim=1)  # apply softmax across classes
)

classifier_model.eval()
classifier_model.to(device)


background_features=torch.Tensor(full_data.to_numpy()).to(device)
explainer = shap.DeepExplainer(classifier_model, background_features)

with open('models/SHAP.pkl', 'wb') as f:
    pickle.dump(explainer, f)


