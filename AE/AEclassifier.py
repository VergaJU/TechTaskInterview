import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class ClassificationDataset(data.Dataset):
    def __init__(self, input_data, labels):
        self.input_data = input_data # Original scaled gene expression tensor
        self.labels = labels # Label tensor

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return (self.input_data[index], self.labels[index])


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.0):
        super(ClassifierHead, self).__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AEClassifier(nn.Module):
    def __init__(self, encoder, num_classes, latent_dim, hidden_dims, dropout_rate=0.2):
        super(AEClassifier, self).__init__()
        self.encoder = encoder # The pre-trained encoder module

        # Keep track of whether the encoder is frozen
        self._is_encoder_frozen = False

        # Assuming the last layer of the encoder is the latent layer
        latent_dim = self.encoder[-1].out_features
        current_dim = latent_dim
        classification_layers = []
        for h_dim in hidden_dims: # Create layers for each hidden layer
            classification_layers.append(ClassifierHead(current_dim, h_dim, dropout_rate))
            current_dim = h_dim
        
        # Add the final classification layer
        classification_layers.append(nn.Linear(current_dim, num_classes)) # Output layer: one score per
        self.classification_head = nn.Sequential(*classification_layers)


    def forward(self, gene_expression_data):
        # Pass gene expression data through the encoder
        # Ensure gradient tracking is off if encoder is frozen
        if self._is_encoder_frozen:
             with torch.no_grad():
                 latent_representation = self.encoder(gene_expression_data)
        else:
             latent_representation = self.encoder(gene_expression_data)
       

        # Pass the latent_representation features through the classification head
        logits = self.classification_head(latent_representation)

        return logits # Shape: (batch_size, num_classes)


    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._is_encoder_frozen = True
        print("Encoder frozen.")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._is_encoder_frozen = False
        print("Encoder unfrozen.")