#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import yaml
import numpy as np
import pandas as pd
import os


from AE.AE import Autoencoder, GeneExpressionDataset
from AE.AEclassifier import ClassificationDataset, AEClassifier

# load training data
with open('Data/training_classifier_data.pkl','rb') as f:
    data=pickle.load(f)

X_train_gene = data['X_train']
X_val_gene = data['X_val']
train_labels = data['labels_train']
val_labels = data['labels_val']
input_dim = X_train_gene.shape[1]
num_classes = data['num_classes']
labels_names = data['labels_names']  # List of class names for classification report
loader_workers=0


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# load parameters
with open('test_params.yaml', "r") as f:
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



# load classifier params
with open('classifier_params.yaml', "r") as f:
    classifier_params=yaml.safe_load(f)


# Extract hidden dimensions based on the suggested params
classifier_hidden_dims = []
n_hidden_layers = classifier_params['n_hidden_layers']
for i in range(n_hidden_layers):
    # Append the hidden dimension for each layer
    if f'h_dim_{i}' in classifier_params:
        classifier_hidden_dims.append(classifier_params[f'h_dim_{i}'])

classifier_batch_size = classifier_params['batch_size']
classifier_dropout_rate = classifier_params['dropout_rate']
classifier_lr = classifier_params['lr']


# Train the final model
final_model = Autoencoder(input_dim, best_latent_dim, best_hidden_dims, best_dropout_rate).to(device)
final_model.load_state_dict(torch.load('models/autoencoder_model.pth'))

classifier_model = AEClassifier(final_model.encoder,num_classes=num_classes, 
                                        latent_dim=best_latent_dim,
                                        hidden_dims=classifier_hidden_dims, 
                                        dropout_rate=classifier_dropout_rate).to(device)


classifier_model.freeze_encoder() # freeze weights of the encoder


classifier_criterion = nn.CrossEntropyLoss() # Includes Softmax
optimizer_classifier = optim.Adam(classifier_model.parameters(), lr=classifier_lr)



# Data Loaders for training WITH validation for early stopping


train_dataset_final = ClassificationDataset(X_train_gene, train_labels)
val_dataset_final = ClassificationDataset(X_val_gene, val_labels) # Use the validation set again
train_loader_final = DataLoader(train_dataset_final, batch_size=classifier_batch_size, shuffle=True, num_workers=loader_workers)
val_loader_final = DataLoader(val_dataset_final, batch_size=classifier_batch_size, shuffle=False, num_workers=loader_workers)


# --- Early Stopping Parameters ---
patience = 25
min_delta = 1e-5 # Minimum change


# --- Training Parameters ---
MAX_N_EPOCHS_CLASSIFIER = 1000 


print("\nTraining final model with best hyperparameters and Early Stopping...")

best_val_metric = -float('inf') # For accuracy (maximize)
epochs_without_improvement_classifier = 0
best_classifier_model_state = None

# Function to calculate validation accuracy
def calculate_accuracy(model, data_loader, device):
    model.eval() # Set model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for gene_batch, labels_batch in data_loader:
            gene_batch = gene_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(gene_batch) # Get logits
            _, predicted = torch.max(outputs.data, 1) # Get the predicted class index

            total_samples += labels_batch.size(0)
            correct_predictions += (predicted == labels_batch).sum().item()
    accuracy = correct_predictions / total_samples
    return accuracy



def evaluate_classifier_metrics(model, data_loader, device, target_names):
    """
    Evaluates classifier model and computes various metrics.

    Args:
        model: Trained PyTorch classifier model.
        data_loader: DataLoader for the evaluation dataset (validation or test).
        device: The device (cpu or gpu) to use.
        target_names: List of string names for the classes (matching label order).

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.eval() # Set model to evaluation mode
    all_labels = []
    all_preds = []
    all_logits = [] # Keep logits if needed later (e.g., for probability analysis)

    with torch.no_grad():
        for gene_batch, labels_batch in data_loader:
            gene_batch = gene_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(gene_batch) # Get logits
            _, predicted = torch.max(outputs.data, 1) # Get the predicted class index

            all_labels.append(labels_batch.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())


    # Concatenate lists of numpy arrays
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_logits = np.concatenate(all_logits)


    # Compute metrics using sklearn
    accuracy = accuracy_score(all_labels, all_preds)

    # For multi-class, specify average method
    # 'micro': calculates metrics globally, same as accuracy here
    # 'macro': calculates metrics for each class, then takes unweighted average
    # 'weighted': calculates metrics for each class, then takes average weighted by class size
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Classification Report (conveniently provides per-class metrics)
    # Use zero_division=0 to handle classes with no predicted samples gracefully
    class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)


    metrics = {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'true_labels': all_labels, # Return these for SHAP visualization later
        'predicted_labels': all_preds, # Return these for SHAP visualization later
        'logits': all_logits # Return logits/probabilities for SHAP or further analysis
    }

    return metrics


train_accuracies = []
train_losses = []
val_accuracies = []


for epoch in range(MAX_N_EPOCHS_CLASSIFIER):
    # --- Training Phase ---
    classifier_model.train() # Set model to training mode
    total_train_loss = 0
    for gene_batch, labels_batch in train_loader_final:
        gene_batch = gene_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer_classifier.zero_grad()
        logits = classifier_model(gene_batch) # Forward pass
        loss = classifier_criterion(logits, labels_batch) # Calculate loss

        loss.backward()
        optimizer_classifier.step()
        total_train_loss += loss.item()
    # compute train loss and accuracy
    avg_train_loss = total_train_loss / len(train_loader_final)
    # compute train accuracy
    train_accuracy = calculate_accuracy(classifier_model, train_loader_final, device)
    train_accuracies.append(train_accuracy)
    train_losses.append(avg_train_loss)
    # --- Validation Phase ---
    avg_val_accuracy = calculate_accuracy(classifier_model, val_loader_final, device)
    val_accuracies.append(avg_val_accuracy)
    # --- Early Stopping Check (using Accuracy) ---
    if avg_val_accuracy > best_val_metric + min_delta:
        best_val_metric = avg_val_accuracy
        epochs_without_improvement_classifier = 0
        best_classifier_model_state = classifier_model.state_dict().copy()
        print(f'Epoch {epoch+1}/{MAX_N_EPOCHS_CLASSIFIER}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {avg_val_accuracy:.4f} (Improved - Saving Model)')
    else:
        epochs_without_improvement_classifier += 1
        print(f'Epoch {epoch+1}/{MAX_N_EPOCHS_CLASSIFIER}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {avg_val_accuracy:.4f} ({epochs_without_improvement_classifier}/{patience} patience)')

    if epochs_without_improvement_classifier >= patience:
        print(f'\nEarly stopping triggered after {epoch+1} epochs (patience {patience} exceeded).')
        break



# --- Load the best model state ---
if best_classifier_model_state is not None:
    classifier_model.load_state_dict(best_classifier_model_state)
    print("Loaded classifier model state from epoch with best validation accuracy.")
else:
    print("No classifier model state was saved (did not improve). Using the model from the last epoch.")

print("Classifier model training complete.")

# Save the best classifier model
torch.save(classifier_model.state_dict(), 'models/classifier_model.pth')
print("Saved final classifier model state_dict to 'models/classifier_model.pth'")


if not os.path.exists('logs'):
    os.makedirs('logs')

metrics_df = pd.DataFrame({
    'epoch': list(range(1, len(train_accuracies)+1)),
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies,
    'train_loss': train_losses
})
metrics_df.to_csv("logs/classifier_training_logs.csv", index=False)

# Compute confusion matrix on validation set
metrics = evaluate_classifier_metrics(classifier_model, val_loader_final, device,labels_names)
print("Confusion Matrix on Validation Set:")
print(metrics['confusion_matrix'])



#save
with open('logs/classifier_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)