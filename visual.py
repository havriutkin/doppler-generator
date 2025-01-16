import torch
import torch.nn as nn

# Define the model class (same structure as trained model)
class DopplerClassifier(nn.Module):
    def __init__(self):
        super(DopplerClassifier, self).__init__()
        self.fc1 = nn.Linear(15, 15)
        self.fc2 = nn.Linear(15, 15)
        self.out = nn.Linear(15, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

# Load the trained model
model = DopplerClassifier()
model.load_state_dict(torch.load("models/2d_static_classifier_10_epochs.pth"))

# Pretty-print weights of each layer with neuron-level granularity
def print_model_weights(model):
    print(f"{'='*40}\nModel Weights (Per Neuron):\n{'='*40}")
    for name, param in model.named_parameters():
        if "weight" in name:  # Focus on weight tensors
            print(f"\nLayer: {name}")
            weights = param.data.numpy().round(3)  # Round weights to 3 decimal places
            for neuron_idx, neuron_weights in enumerate(weights):
                neuron_weights_str = ", ".join([f"{w:.3f}" for w in neuron_weights])
                print(f"  Neuron {neuron_idx + 1}:")
                print(f"    Weights: [{neuron_weights_str}]")
    print(f"{'='*40}")

# Print the weights
print_model_weights(model)
