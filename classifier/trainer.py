# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model, train_dataset, test_dataset, batch_size: int = 16, lr: float = 0.01):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, epochs: int = 100):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")

    def evaluate(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        train_preds = []
        train_targets = []
        for X_batch, y_batch in train_loader:
            y_pred = self.model(X_batch)
            train_preds.extend(y_pred.argmax(axis=1).tolist())
            train_targets.extend(y_batch.tolist())

        test_preds = []
        test_targets = []
        for X_batch, y_batch in test_loader:
            y_pred = self.model(X_batch)
            test_preds.extend(y_pred.argmax(axis=1).tolist())
            test_targets.extend(y_batch.tolist())

        print("Train Set:")
        self._print_metrics(train_targets, train_preds)

        print("\nTest Set:")
        self._print_metrics(test_targets, test_preds)

    def plot_loss(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.show()

    def _print_metrics(self, targets, preds):
        print(f"Accuracy: {accuracy_score(targets, preds):.4f}")
        print(f"Precision: {precision_score(targets, preds):.4f}")
        print(f"Recall: {recall_score(targets, preds):.4f}")
        print(f"F1 Score: {f1_score(targets, preds):.4f}")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model is saved to {path}")