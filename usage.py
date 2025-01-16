import torch
import torch.nn as nn
import sympy as sp
import numpy as np
from transmitter import TransmitterFactory
from receiver import ReceiverFactory
from propagation import SoundInAirPropagation
from noise import GaussianNoise

# ========== Define the Neural Network Model ==========
class DopplerClassifier(nn.Module):
    def __init__(self):
        super(DopplerClassifier, self).__init__()
        self.fc1 = nn.Linear(15, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

# Load the trained model
model_path = "models/2d_static_classifier_10_epochs.pth"  # Update the path to your saved model
model = DopplerClassifier()
model.load_state_dict(torch.load(model_path))
model.eval()

# ========== Set up a Doppler Problem ==========
# Define symbols
f, c = sp.symbols('f c')
x, y, v_x, v_y = sp.symbols('x y v_x v_y')
fi, xi, yi = sp.symbols('f_i x_i y_i')
f_i = sp.symbols('f_i:5')
x_i = sp.symbols('x_i:5')
y_i = sp.symbols('y_i:5')

NUM_OF_REC = 5
DIMENSION = 2
TOLERANCE = 0.1
NUM_OF_SIMS = 30  # Reduced for quick testing

# Define equation and its Jacobian
equation = (1 / f)**2 * (f - fi)**2 * ((xi - x)**2 + (yi - y)**2) - (1 / c)**2 * ((v_x * (xi - x) + v_y * (yi - y))**2)
jacobian = [sp.diff(equation, var) for var in (x, y, v_x, v_y)]

# Initialize components
transmitter_factory = TransmitterFactory()
receiver_factory = ReceiverFactory()
propagation = SoundInAirPropagation()
frequency = 1000

# Results list
results = []

for sim in range(NUM_OF_SIMS):
    print(f"Simulation {sim+1}/{NUM_OF_SIMS}...")

    # Create transmitter and receivers
    transmitter = transmitter_factory.create_random_transmitter_with_frequency(
        dimension=DIMENSION, min=-100, max=100, frequency=frequency)
    receivers = [receiver_factory.create_random_static_receiver(dimension=DIMENSION, min=-100, max=100)
                 for _ in range(NUM_OF_REC)]

    # Generate parameters
    p0 = [propagation.compute_observed_frequency(transmitter, receiver) for receiver in receivers]
    p1 = [GaussianNoise(noise_level=0.1).add_noise(frequency) for frequency in p0]

    # Compute true label using Newton's method
    Fp1 = [equation.subs({f: frequency, c: propagation.speed, xi: receivers[i]._position[0],
                          yi: receivers[i]._position[1], fi: p1[i]}) for i in range(NUM_OF_REC)]
    dFp1 = [[jacobian[j].subs({f: frequency, c: propagation.speed, xi: receivers[i]._position[0],
                               yi: receivers[i]._position[1], fi: p1[i]})
             for j in range(4)] for i in range(NUM_OF_REC)]

    s0 = [transmitter._position[0], transmitter._position[1], transmitter._velocity[0], transmitter._velocity[1]]
    Fp1_s0 = np.array([Fp1[i].subs({x: s0[0], y: s0[1], v_x: s0[2], v_y: s0[3]}) for i in range(NUM_OF_REC)],
                      dtype=np.float64)
    dFp1_s0 = np.array([[dFp1[i][j].subs({x: s0[0], y: s0[1], v_x: s0[2], v_y: s0[3]}) for j in range(4)]
                        for i in range(NUM_OF_REC)], dtype=np.float64)
    delta = np.linalg.lstsq(dFp1_s0, -Fp1_s0, rcond=None)[0]
    s1 = [s0[i] + delta[i] for i in range(4)]

    Fp1_s1 = sp.Matrix([Fp1[i].subs({x: s1[0], y: s1[1], v_x: s1[2], v_y: s1[3]}) for i in range(NUM_OF_REC)])
    norm = Fp1_s1.norm()
    true_label = 1 if norm < TOLERANCE else 0

    # Prepare features for the neural network
    features = torch.tensor(
        p1 + [receivers[i]._position[0] for i in range(NUM_OF_REC)] + 
            [receivers[i]._position[1] for i in range(NUM_OF_REC)], 
        dtype=torch.float32
    ).unsqueeze(0)  # Add batch dimension


    # Predict label using the model
    with torch.no_grad():
        predicted_label = model(features).round().item()

    # Compare results
    print(f"    True Label (Newton): {true_label}, Predicted Label (NN): {int(predicted_label)}")

    # Save result
    results.append({
        'true_label': true_label,
        'predicted_label': int(predicted_label),
        'features': features.tolist()
    })
