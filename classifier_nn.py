import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from prettytable import PrettyTable
from torchinfo import summary
from torchviz import make_dot
import random
import numpy as np

# ========== Preprocess ==========
# Load data
with open('data.json', 'r') as file:
    data = json.load(file)

# Normalize features
def normalize_features(data):
    all_features = np.array([[item[f'f_{i}'] for i in range(5)] +
                              [item[f'x_{i}'] for i in range(5)] +
                              [item[f'y_{i}'] for i in range(5)] for item in data])
    
    # Min-max scaling
    min_vals = all_features.min(axis=0)
    max_vals = all_features.max(axis=0)
    normalized_features = (all_features - min_vals) / (max_vals - min_vals)

    # Update the original data with normalized features
    for i, item in enumerate(data):
        for j in range(15):
            key = f'f_{j}' if j < 5 else f'x_{j - 5}' if j < 10 else f'y_{j - 10}'
            item[key] = normalized_features[i][j]

normalize_features(data)

# ========== Dataset ==========
class DopplerDataset(Dataset):
    def __init__(self, data):
        self.features = [[item[f'f_{i}'] for i in range(5)] +
                         [item[f'x_{i}'] for i in range(5)] +
                         [item[f'y_{i}'] for i in range(5)]
                         for item in data]
        self.labels = [item['label'] for item in data]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

dataset = DopplerDataset(data)
print(f"Dataset length: {dataset.__len__()}")

# Print number of class 1 and class 0 samples
num_class_1 = sum(dataset.labels)
num_class_0 = len(dataset.labels) - num_class_1
print(f"Number of class 1 samples: {num_class_1}")
print(f"Number of class 0 samples: {num_class_0}")

# Split dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========== Model ==========
class DopplerClassifier(nn.Module):
    def __init__(self):
        super(DopplerClassifier, self).__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.out(x))
        return x

# ========== Training ==========
criterion = nn.BCELoss()
epoch_results = []
thresholds = [0.5, 0.55, 0.6, 0.7, 0.75]

for num_epochs in [5, 10, 15]: 
    model = DopplerClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
    # Save the model
    model_filename = f"models/2d_static_classifier_{num_epochs}_epochs.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model trained for {num_epochs} epochs saved as '{model_filename}'.")

    # Evaluate the model at different thresholds
    model.eval()
    for threshold in thresholds:
        predictions, labels = [], []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features).squeeze()
                #print(f"Raw sigmoid outputs: {outputs[:10].cpu().numpy()}")
                predictions.extend((outputs >= threshold).float().cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())

        # Evaluate Metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        epoch_results.append((num_epochs, threshold, accuracy, 
                              round(precision, 4), round(recall, 4), 
                              round(f1, 3)))

# ========== Print Results ==========
table = PrettyTable()
table.field_names = ["Epochs", "Threshold", "Accuracy", "Precision", "Recall", "F1 Score"]
for row in epoch_results:
    table.add_row(row)

print(table)
