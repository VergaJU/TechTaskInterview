#!/usr/bin/python

import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
import torch
import pickle
import yaml
import sys


# get label from command line argument

label = sys.argv[1]


data_dir = 'Data/'

adata = sc.read_h5ad(data_dir + 'dataset_annotated.h5ad')

labels = adata.obs[label]
labels = labels.astype('category')
label_ids = labels.cat.codes.values
num_classes = len(labels.cat.categories)
labels_names = labels.cat.categories.tolist()
label_ids_tensor = torch.LongTensor(label_ids.copy())

print(f"Number of classes ({label}): {num_classes}")


# load train data from autoencoder

with open('Data/training_data.pkl','rb') as f:
    data=pickle.load(f)

X_full = data['full_dataset'].to_numpy()  # Full dataset for embeddings

indices = np.arange(X_full.shape[0]) # Indices for the full dataset


train_indices, val_indices, y_train_split, y_val_split = train_test_split(
    indices, label_ids, test_size=0.15, random_state=42, stratify=label_ids # Stratify by labels
)

X_train_gene = torch.FloatTensor(X_full[train_indices])
X_val_gene = torch.FloatTensor(X_full[val_indices])


labels_train = label_ids_tensor[train_indices]
labels_val = label_ids_tensor[val_indices]

print(f"Train set size: {len(train_indices)}, Val set size: {len(val_indices)}")

# save files

files={
    'full_dataset':X_full,
    'X_train':X_train_gene,
    'X_val':X_val_gene,
    'labels_train':labels_train,
    'labels_val':labels_val,
    'num_classes':num_classes,
    'labels_names':labels_names
}

with open(data_dir + "training_classifier_data.pkl", "wb") as f:
    pickle.dump(files, f)