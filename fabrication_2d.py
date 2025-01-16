import sympy as sp
import numpy as np
from transmitter import TransmitterFactory
from receiver import ReceiverFactory
from propagation import SoundInAirPropagation
from noise import GaussianNoise
from termcolor import colored
import json

# ========== Set up a Doppler problem ==========
# Define the symbols
f = sp.Symbol('f')
c = sp.Symbol('c')
x, y = sp.symbols('x y')
v_x, v_y = sp.symbols('v_x v_y')
fi, xi, yi = sp.symbols('f_i x_i y_i')
f_i = sp.symbols('f_i:5')
x_i = sp.symbols('x_i:5')
y_i = sp.symbols('y_i:5')

NUM_OF_REC = 5
NUM_OF_VAR = 4
MIN = -100
MAX = 100
DIMENSION = 2
TOLERANCE = 0.1
TARGET_COUNT = 500  # Target for each label
frequency = 1000

# Define the equation and compute its Jacobian
equation = (1 / f)**2 * (f - fi)**2 * ((xi - x)**2 + (yi - y)**2) - (1 / c)**2 * ((v_x * (xi - x) + v_y * (yi - y))**2)
jacobian = [sp.diff(equation, var) for var in (x, y, v_x, v_y)]

# Create the transmitter, receiver, and propagation objects
transmitter_factory = TransmitterFactory()
receiver_factory = ReceiverFactory()
propagation = SoundInAirPropagation()

# Results and label counters
results = []
label_count = {0: 0, 1: 0}

print(colored("Starting generation process...", 'cyan'))

while label_count[0] < TARGET_COUNT or label_count[1] < TARGET_COUNT:
    # Generate a Doppler problem
    transmitter = transmitter_factory.create_random_transmitter_with_frequency(
        dimension=DIMENSION, min=MIN, max=MAX, frequency=frequency)
    receivers = [receiver_factory.create_random_static_receiver(dimension=DIMENSION, min=MIN, max=MAX)
                 for _ in range(NUM_OF_REC)]

    # Create a Family of Doppler problems
    F = [equation.subs({f: frequency, c: propagation.speed,
                        xi: receivers[i]._position[0],
                        yi: receivers[i]._position[1]}) for i in range(NUM_OF_REC)]

    # Compute the Jacobian of the Family
    J = []
    for i in range(NUM_OF_REC):
        row = [jacobian[j].subs({f: frequency, c: propagation.speed,
                                 xi: receivers[i]._position[0],
                                 yi: receivers[i]._position[1]})
               for j in range(NUM_OF_VAR)]
        J.append(row)

    # Generate parameters - observed frequencies
    p0 = [propagation.compute_observed_frequency(transmitter, receiver) for receiver in receivers]

    # Add noise to the observed frequencies
    p1 = [GaussianNoise(noise_level=0.1).add_noise(freq) for freq in p0]

    # Instance of the family F with parameters p0 and p1
    Fp1 = [F[i].subs({fi: p1[i]}) for i, _ in enumerate(p1)]

    # Compute the Jacobian at p1
    dFp1 = [[J[i][j].subs({fi: p1[i]}) for j in range(4)] for i, _ in enumerate(p1)]

    # Conduct one step of Newton's method
    s0 = [transmitter._position[0], transmitter._position[1], transmitter._velocity[0], transmitter._velocity[1]]
    Fp1_s0 = np.array([Fp1[i].subs({x: s0[0], y: s0[1], v_x: s0[2], v_y: s0[3]}) for i in range(NUM_OF_REC)],
                      dtype=np.float64)
    dFp1_s0 = np.array([[dFp1[i][j].subs({x: s0[0], y: s0[1], v_x: s0[2], v_y: s0[3]}) for j in range(4)]
                        for i in range(NUM_OF_REC)], dtype=np.float64)

    delta = np.linalg.lstsq(dFp1_s0, -Fp1_s0, rcond=None)[0]
    s1 = [s0[i] + delta[i] for i in range(4)]

    Fp1_s1 = sp.Matrix([Fp1[i].subs({x: s1[0], y: s1[1], v_x: s1[2], v_y: s1[3]}) for i in range(NUM_OF_REC)])
    norm = Fp1_s1.norm()
    label = 1 if norm < TOLERANCE else 0

    # Only save the result if it contributes to the balance
    if label_count[label] < TARGET_COUNT:
        label_count[label] += 1
        data = {
            'f': frequency,
            'c': propagation.speed,
            'x': transmitter._position[0],
            'y': transmitter._position[1],
            'v_x': transmitter._velocity[0],
            'v_y': transmitter._velocity[1],
        }
        data.update({f"f_{i}": p1[i] for i in range(NUM_OF_REC)})
        data.update({f"x_{i}": receivers[i]._position[0] for i in range(NUM_OF_REC)})
        data.update({f"y_{i}": receivers[i]._position[1] for i in range(NUM_OF_REC)})
        data.update({'label': label})
        results.append(data)

    print(f"Label counts: {label_count}", end='\r')

print(colored("\nData generation complete!", 'green'))
print(colored(f"Label 0: {label_count[0]}", 'cyan'))
print(colored(f"Label 1: {label_count[1]}", 'cyan'))

# Save the results to a JSON file
with open('data.json', 'w') as f:
    json.dump(results, f)

print(colored("Data saved successfully!", 'green'))
