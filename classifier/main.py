import pandas as pd

from .data import preprocess_data, DopplerDataset
from .models import DopplerModel
from .trainer import ModelTrainer


HIDDEN_SIZE = 32
EPOCHS = 70

def main():
    # Read data as pd from file "data2.json"
    data = pd.read_json("data2.json")

    # Split data into features and labels
    X = data.drop(columns='label')
    y = data['label']

    (X_train_scaled, y_train), (X_test_scaled, y_test), scaler = preprocess_data(X, y)

    # Create datasets
    train_dataset = DopplerDataset(X_train_scaled, y_train)
    test_dataset = DopplerDataset(X_test_scaled, y_test)

    # Create model
    input_size = X_train_scaled.shape[1]
    model = DopplerModel(input_size, HIDDEN_SIZE)

    # Initialize trainer
    trainer = ModelTrainer(model, train_dataset, test_dataset)

    # Train and evaluate model
    trainer.train(epochs=EPOCHS)
    trainer.evaluate()
    trainer.plot_loss()

    # Save model
    trainer.save_model("model_distortion.pth")

if __name__ == "__main__":
    main()
