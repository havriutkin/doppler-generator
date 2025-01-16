import sympy as sp
import numpy as np
from transmitter import Transmitter, TransmitterFactory
from receiver import ReceiverFactory, Receiver
from propagation import PropagationBehavior, SoundInAirPropagation, UnitPropagation
from noise import GaussianNoise, NoNoise
from termcolor import colored

class DopplerProblem:
    def __init__(self, frequency, propagation, transmitter, receivers):
        self._frequency = frequency
        self._propagation = propagation
        self._transmitter = transmitter
        self._receivers = receivers
        self._volume = len(receivers)
        self._setup_symbols()
        self.equation = (1 / self.f)**2 * (self.f - self.fi)**2 * ((self.xi - self.x)**2 + (self.yi - self.y)**2) - (1 / self.c)**2 * ((self.v_x * (self.xi - self.x) + self.v_y * (self.yi - self.y))**2)
        self.jacobian = [sp.diff(self.equation, self.x), sp.diff(self.equation, self.y), sp.diff(self.equation, self.v_x), sp.diff(self.equation, self.v_y)]
        self._compute_label()
  
    def _setup_symbols(self):
        self.f = sp.Symbol('f')
        self.c = sp.Symbol('c')
        self.x, self.y = sp.symbols('x y')
        self.v_x, self.v_y = sp.symbols('v_x v_y')
        self.fi, self.xi, self.yi = sp.symbols('f_i x_i y_i')

    def _compute_label(self):
        # Create a Family of Doppler problems
        F = [self.equation.subs({self.f: self._frequency, self.c: self._propagation.speed, 
                            self.xi: self._receivers[i]._position[0], 
                            self.yi: self._receivers[i]._position[1]}) for i in range(self._volume)]

        # Compute the Jacobian of the Family
        J = []
        for i in range(self._volume):
            row = [self.jacobian[j].subs({self.f: self._frequency, self.c: self._propagation.speed,
                                    self.xi: self._receivers[i]._position[0], self.yi: self._receivers[i]._position[1]}) 
                                    for j in range(4)]
            J.append(row)

        # Generate parameters - observed frequencies
        p0 = [self._propagation.compute_observed_frequency(self._transmitter, receiver) for receiver in self._receivers]

        # Add noise to the observed frequencies
        p1 = [GaussianNoise(noise_level=0.1).add_noise(self._frequency) for frequency in p0]

        # Instance of the family F with parameters p0 and p1
        Fp0 = [F[i].subs({self.fi: p0[i]}) for i, _ in enumerate(p0)]
        Fp1 = [F[i].subs({self.fi: p1[i]}) for i, _ in enumerate(p1)]

        # Compute the Jacobian at p1
        dFp1 = [[J[i][j].subs({self.fi: p1[i]}) for j in range(4)] for i, _ in enumerate(p1)]

        # ========== Conduct one step of Newton's method ==========
        # Initial guess
        s0 = [self._transmitter._position[0], self._transmitter._position[1], self._transmitter._velocity[0], self._transmitter._velocity[1]]

        # Compute the value of Fp0 at s0
        Fp0_s0 = [Fp0[i].subs({self.x: s0[0], self.y: s0[1], self.v_x: s0[2], self.v_y: s0[3]}) for i in range(self._volume)]
        Fp0_s0 = np.array(Fp0_s0, dtype=np.float64)

        # Compute the value of Fp1 at s0
        Fp1_s0 = [Fp1[i].subs({self.x: s0[0], self.y: s0[1], self.v_x: s0[2], self.v_y: s0[3]}) for i in range(self._volume)]
        Fp1_s0 = np.array(Fp1_s0, dtype=np.float64)

        # Compute the Jacobian of Fp1 at s0
        dFp1_s0 = [[dFp1[i][j].subs({self.x: s0[0], self.y: s0[1], self.v_x: s0[2], self.v_y: s0[3]})
                    for j in range(4)] for i in range(self._volume)]
        dFp1_s0 = np.array(dFp1_s0, dtype=np.float64)

        # Find the delta by solving the linear system using least squares
        delta = np.linalg.lstsq(dFp1_s0, -Fp1_s0, rcond=None)[0]

        s1 = [s0[i] + delta[i] for i in range(4)]

        # Compute Fp1 at s1
        Fp1_s1 = sp.Matrix([Fp1[i].subs({self.x: s1[0], self.y: s1[1], self.v_x: s1[2], self.v_y: s1[3]}) for i in range(self._volume)])
        norm = Fp1_s1.norm()
        label = 0
        if norm < TOLERANCE:
            label = 1

        self._label = label

# ========== Set up a Doppler problem ==========
NUM_OF_REC = 5
NUM_OF_VAR = 4
MIN = -100
MAX = 100
DIMENSION = 2
TOLERANCE = 0.1
NUM_OF_SIMS = 100

# Define the equation and compute its Jacobian
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

    problem = DopplerProblem(frequency, propagation, transmitter, receivers)
    label = problem._label

    label_count[label] += 1

    # ========== Save Results as JSON ==========
    data = {
        'f': 
    }

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

