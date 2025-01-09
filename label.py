from receiver import Receiver, ReceiverFactory
from transmitter import Transmitter, TransmitterFactory
from propagation import SoundInAirPropagation, SoundInWaterPropagation, LightInAirPropagation
from doppler import DopplerBuilder, DopplerProblem, DopplerGenerator, DopplerProblemAggregator
from noise import GaussianNoise, NoNoise, NoiseBehavior

from abc import ABC, abstractmethod
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Label:
    """
        This class accepts Doppler Problem and generates a label for it using the following rules: 
        - Assume all problems are 2D and there are always 5 receivers.
        - Add noise to the observed frequencies
        - Run a one step of newton on the system of equations,
            with the initial guess being the transmitter's position and velocity in the pure case
        - If the system converges, the label is 1, otherwise it is 0
    """
    def __init__(self, doppler_problem: DopplerProblem, noise_behavior: NoiseBehavior, tolerance: float):
        self._doppler_problem = doppler_problem
        self._noise_behavior = noise_behavior
        self._tolerance = tolerance

    def get_equation_value(self, c, f, f_i, x, y, x_i, y_i, v_x, v_y):
        return c**2 * (f - f_i)**2 * ((x_i - x)**2 + (y_i - y)**2) - f**2 * ((v_x * (x_i - x) + v_y * (y_i - y))**2)
    
    def get_partials(self, c, f, f_i, x, y, x_i, y_i, v_x, v_y):
        return np.array([
            -2*x*v_x*f - 2*y*v_x*v_y*f + 2*x*c*f + 2*v_x*f*x_i - 2*c*f*x_i + 2*v_x*v_y*f*y_i - 4*x*c*f*f_i + 4*c*f*x_i*f_i + 2*x*c*f_i - 2*c*x_i*f_i,
            -2*x*v_x*v_y*f - 2*y*v_y**2*f + 2*y*c*f + 2*v_x*v_y*f*x_i + 2*v_y*f*y_i - 2*c*f*y_i - 4*y*c*f*f_i + 4*c*f*y_i*f_i + 2*y*c*f_i - 2*c*y_i*f_i,
            -2*x*v_x*f**2 - 2*x*y*v_y*f**2 + 4*x*v_x*f*x_i + 2*y*v_y*f*x_i - 2*v_x*f*x_i**2 + 2*x*v_y*f*y_i - 2*v_y*f*x_i*y_i,
            -2*x*y*v_x*f**2 - 2*y*v_y*f**2 + 2*y*v_x*f*x_i + 2*x*v_x*f*y_i + 4*y*v_y*f*y_i - 2*v_x*f*x_i*y_i - 2*v_y*f*y_i**2
        ])
    
    def get_jacobian(self, c, f, fi, x, y, v_x, v_y):
        jacobian = np.zeros((5, 4))
        for i in range(5):
            x_i = self._doppler_problem._receivers[i]._position[0]
            y_i = self._doppler_problem._receivers[i]._position[1]
            jacobian[i, :] = self.get_partials(c, f, fi[i], x, y, x_i, y_i, v_x, v_y)
        return jacobian
    
    def get_value(self, c, f, fi, x, y, v_x, v_y):
        value = np.zeros(5)
        for i in range(5):
            x_i = self._doppler_problem._receivers[i]._position[0]
            y_i = self._doppler_problem._receivers[i]._position[1]
            value[i] = self.get_equation_value(c, f, fi[i], x, y, x_i, y_i, v_x, v_y)
        return value

    def get_label(self):
        c = self._doppler_problem._propagation_speed
        f = self._doppler_problem._transmitter._frequency
        x = self._doppler_problem._transmitter._position[0]
        y = self._doppler_problem._transmitter._position[1]
        v_x = self._doppler_problem._transmitter._velocity[0]
        v_y = self._doppler_problem._transmitter._velocity[1]
        fi = self._doppler_problem._observed_frequencies
        fi_noisy = [self._noise_behavior.add_noise(fi_i)[0] for fi_i in fi]

        logging.debug(f"Pure observed frequencies: {fi}")
        logging.debug(f"Noisy observed frequencies: {fi_noisy}")

        logging.debug(f"Initial guess s_0: {x}, {y}, {v_x}, {v_y}")

        # Run one step of newton
        jacobian = self.get_jacobian(c, f, fi_noisy, x, y, v_x, v_y)
        value = self.get_value(c, f, fi_noisy, x, y, v_x, v_y)
        test_value = self.get_value(c, f, fi, x, y, v_x, v_y)

        logging.debug(f"Value of Fp0 at s_0: {test_value}")

        # Uncomment these lines if you want to log jacobian and value
        logging.debug(f"Jacobian of Fp1 at s_0: {jacobian}")
        logging.debug(f"Value of Fp1 at s_0: {value}")

        try:
            delta, _, _, _ = np.linalg.lstsq(jacobian, value)
            logging.debug(f"Delta: {delta}")
        except Exception as e:
            logging.error(f"Error: {e}")
            return 0

        x += delta[0]
        y += delta[1]
        v_x += delta[2]
        v_y += delta[3]

        logging.debug(f"New guess s_1: {x}, {y}, {v_x}, {v_y}")

        # Find new value
        new_value = self.get_value(c, f, fi_noisy, x, y, v_x, v_y)
        logging.debug(f"Value of Fp1 at s_1: {new_value}")

        # Check if the system converges
        norm = np.linalg.norm(new_value)
        logging.debug(f"Norm: {norm}")

        if norm < self._tolerance:
            return 1
        else:
            return 0


