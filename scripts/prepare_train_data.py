#!/usr/bin/python

import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import pickle
import yaml
import os


data_dir = 'Data/'

adata = sc.read_h5ad(data_dir + 'dataset.h5ad')

# load parameters
with open('optuna_parameter.yaml', "r") as f:
    optuna_parameters=yaml.safe_load(f)


if optuna_parameters['hvg'] > 0:
    # find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=optuna_parameters['hvg']) 
    adata = adata[:, adata.var['highly_variable']].copy()
    hvg=adata.var
    hvg=hvg[hvg['highly_variable']].index.tolist()
    genes= hvg
else:
    # use all genes
    genes = adata.var_names.tolist()

# get expression df
df=sc.get.obs_df(adata, keys=adata.var_names.tolist())

# scale the data
ss=StandardScaler()
train_data = ss.fit_transform(df)


# ocreate models directory
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/standard_scaler.pkl', 'wb') as f:
    pickle.dump(ss, f)

train_df=pd.DataFrame(train_data, index=df.index, columns=df.columns)


# train test split
X_train, X_val = train_test_split(train_data, test_size=0.15, random_state=42) # 15% for validation
input_dim = X_train.shape[1]
n_samples_train = X_train.shape[0]
n_samples_val = X_val.shape[0]

print(f"Input dimension (number of genes): {input_dim}")
print(f"Training samples: {n_samples_train}, Validation samples: {n_samples_val}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)


# save files

files={
    'full_dataset':train_df,
    'X_train':X_train,
    'X_val':X_val,
    'input_dim':input_dim,
    'n_samples_train':n_samples_train,
    'n_samples_val':n_samples_val,
    'X_train_tensor':X_train_tensor,
    'X_val_tensor':X_val_tensor,
    'genes':genes,
    
}

with open(data_dir + "training_data.pkl", "wb") as f:
    pickle.dump(files, f)