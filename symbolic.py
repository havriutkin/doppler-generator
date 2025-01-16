import sympy as sp
import numpy as np
from transmitter import Transmitter, TransmitterFactory
from receiver import ReceiverFactory, Receiver
from propagation import PropagationBehavior, SoundInAirPropagation, UnitPropagation
from noise import GaussianNoise, NoNoise
from termcolor import colored

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
SCALE = 0.0001
TOLERANCE = 0.1
NUM_OF_SIMS = 100

# Define the equation and compute its Jacobian
equation = (1 / f)**2 * (f - fi)**2 * ((xi - x)**2 + (yi - y)**2) - (1 / c)**2 * ((v_x * (xi - x) + v_y * (yi - y))**2)
jacobian = [sp.diff(equation, x), sp.diff(equation, y), sp.diff(equation, v_x), sp.diff(equation, v_y)]

# Create the transmitter, receiver, and propagation objects
transmitter_factory = TransmitterFactory()
receiver_factory = ReceiverFactory()
propagation = SoundInAirPropagation() # Unit propagation speed for simplicity
frequency = 1000

results = []
label_count = {0: 0, 1: 0}
for i in range(NUM_OF_SIMS):
    # ========== Set up a Doppler problem ==========
    transmitter = transmitter_factory.create_random_transmitter_with_frequency(
        dimension=DIMENSION, 
        min=MIN, 
        max=MAX, 
        frequency=frequency)
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
                                xi: receivers[i]._position[0], yi: receivers[i]._position[1]}) 
                                for j in range(NUM_OF_VAR)]
        J.append(row)

    # Generate parameters - observed frequencies
    p0 = [propagation.compute_observed_frequency(transmitter, receiver) for receiver in receivers]

    # Add noise to the observed frequencies
    p1 = [GaussianNoise(noise_level=0.1).add_noise(frequency) for frequency in p0]

    # Instance of the family F with parameters p0 and p1
    Fp0 = [F[i].subs({fi: p0[i]}) for i, _ in enumerate(p0)]
    Fp1 = [F[i].subs({fi: p1[i]}) for i, _ in enumerate(p1)]

    # Compute the Jacobian at p1
    dFp1 = [[J[i][j].subs({fi: p1[i]}) for j in range(4)] for i, _ in enumerate(p1)]


    # ========== Conduct one step of Newton's method ==========
    # Initial guess
    s0 = [transmitter._position[0], transmitter._position[1], transmitter._velocity[0], transmitter._velocity[1]]

    # Compute the value of Fp0 at s0
    Fp0_s0 = [Fp0[i].subs({x: s0[0], y: s0[1], v_x: s0[2], v_y: s0[3]}) for i in range(NUM_OF_REC)]
    Fp0_s0 = np.array(Fp0_s0, dtype=np.float64)

    # Compute the value of Fp1 at s0
    Fp1_s0 = [Fp1[i].subs({x: s0[0], y: s0[1], v_x: s0[2], v_y: s0[3]}) for i in range(NUM_OF_REC)]
    Fp1_s0 = np.array(Fp1_s0, dtype=np.float64)

    # Compute the Jacobian of Fp1 at s0
    dFp1_s0 = [[dFp1[i][j].subs({x: s0[0], y: s0[1], v_x: s0[2], v_y: s0[3]}) 
                for j in range(NUM_OF_VAR)] for i in range(NUM_OF_REC)]
    dFp1_s0 = np.array(dFp1_s0, dtype=np.float64)

    # Find the delta by solving the linear system using least squares
    delta = np.linalg.lstsq(dFp1_s0, -Fp1_s0, rcond=None)[0]

    s1 = [s0[i] + delta[i] for i in range(4)]

    # Compute Fp1 at s1
    Fp1_s1 = sp.Matrix([Fp1[i].subs({x: s1[0], y: s1[1], v_x: s1[2], v_y: s1[3]}) for i in range(NUM_OF_REC)])
    norm = Fp1_s1.norm()
    label = 0
    if norm < TOLERANCE:
        label = 1
    
    label_count[label] += 1

    # ========== Save Results as JSON ==========
    data = {
        'f': frequency,
        'c': propagation.speed,
        'x': transmitter._position[0],
        'y': transmitter._position[1],
        'v_x': transmitter._velocity[0],
        'v_y': transmitter._velocity[1]
    }
    data.update({f"f_{i}": p1[i] for i in range(NUM_OF_REC)})
    data.update({f"x_{i}": receivers[i]._position[0] for i in range(NUM_OF_REC)})
    data.update({f"y_{i}": receivers[i]._position[1] for i in range(NUM_OF_REC)})
    data.update({'label': label})
    results.append(data)

print(colored("Data is generated successfully!", 'green'))
print(colored(f"Label 0: {label_count[0]}", 'cyan'))
print(colored(f"Label 1: {label_count[1]}", 'cyan'))

# Save to file
import json
with open('data.json', 'w') as f:
    json.dump(results, f)

print(colored("Data is saved successfully!", 'green'))


# ========== Print Results ==========
"""
def pretty_print_matrix(matrix, name):
    print(colored(f"\n{name}:", 'cyan'))
    for row in matrix:
        print(" ".join(f"{val:10.4f}" for val in row))

def pretty_print_vector(vector, name):
    print(colored(f"\n{name}:", 'cyan'))
    print(" ".join(f"{val:10.4f}" for val in vector))

print(colored("F:", 'cyan'))
for eq in F:
    print(eq)
print(colored("Fp0:", 'cyan'))
for eq in Fp0:
    print(eq)
print(colored("Fp1:", 'cyan'))
for eq in Fp1:
    print(eq)

pretty_print_vector(p0, "p0")
pretty_print_vector(p1, "p1")
pretty_print_vector(s0, "s0")
pretty_print_vector(Fp0_s0, "Fp0_s0")
pretty_print_vector(Fp1_s0, "Fp1_s0")
pretty_print_matrix(dFp1_s0, "dFp1_s0")
pretty_print_vector(delta, "delta")
pretty_print_vector(s1, "s1")
pretty_print_vector(Fp1_s1, "Fp1_s1")
print(colored(f"\nNorm ||Fp1(s1)||: {norm}", 'cyan'))
"""

