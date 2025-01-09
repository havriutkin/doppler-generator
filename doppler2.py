from abc import ABC, abstractmethod
import numpy as np
import json

from receiver import Receiver
from transmitter import Transmitter
from noise import NoiseBehavior
from propagation import PropagationBehavior

"""
System of equations:
c^2(f - f_i)^2((x_i - x)^2 + (y_i - y)^2)) - f^2((v_x(x_i - x) + v_y(y_i - y))^2 = 0

Jacobian:
dx: - 2x v_x f - 2y v_x v_y f + 2x c f + 2 v_x f x_i - 2 c f x_i + 2 v_x v_y f y_i - 4x c f f_i + 4 c f x_i f_i + 2x c f_i - 2 c x_i f_i
dy: - 2x v_x v_y f - 2y v_y^2 f + 2y c f + 2 v_x v_y f x_i + 2 v_y f y_i - 2 c f y_i - 4y c f f_i + 4 c f y_i f_i + 2y c f_i - 2 c y_i f_i
dv_x: - 2x v_x f^2 - 2x y v_y f^2 + 4x v_x f x_i + 2y v_y f x_i - 2 v_x f x_i^2 + 2x v_y f y_i - 2 v_y f x_i y_i
dv_y: - 2x y v_x f^2 - 2y v_y f^2 + 2y v_x f x_i + 2x v_x f y_i + 4y v_y f y_i - 2 v_x f x_i y_i - 2 v_y f y_i^2

"""

def evaluate_system(c, f, f_i, x, y, x_i, y_i, v_x, v_y):
    return c**2 * (f - f_i)**2 * ((x_i - x)**2 + (y_i - y)**2) - f**2 * ((v_x * (x_i - x) + v_y * (y_i - y))**2)

def evaluate_jacobian(c, f, f_i, x, y, x_i, y_i, v_x, v_y):
    return np.array([
        -2*x*v_x*f - 2*y*v_x*v_y*f + 2*x*c*f + 2*v_x*f*x_i - 2*c*f*x_i + 2*v_x*v_y*f*y_i - 4*x*c*f*f_i + 4*c*f*x_i*f_i + 2*x*c*f_i - 2*c*x_i*f_i,
        -2*x*v_x*v_y*f - 2*y*v_y**2*f + 2*y*c*f + 2*v_x*v_y*f*x_i + 2*v_y*f*y_i - 2*c*f*y_i - 4*y*c*f*f_i + 4*c*f*y_i*f_i + 2*y*c*f_i - 2*c*y_i*f_i,
        -2*x*v_x*f**2 - 2*x*y*v_y*f**2 + 4*x*v_x*f*x_i + 2*y*v_y*f*x_i - 2*v_x*f*x_i**2 + 2*x*v_y*f*y_i - 2*v_y*f*x_i*y_i,
        -2*x*y*v_x*f**2 - 2*y*v_y*f**2 + 2*y*v_x*f*x_i + 2*x*v_x*f*y_i + 4*y*v_y*f*y_i - 2*v_x*f*x_i*y_i - 2*v_y*f*y_i**2
    ])

class DopplerDihedralProblem(ABC):
    def __init__(self, receivers: list[Receiver], transmitter: Transmitter, propagation_behavior: PropagationBehavior, noise_behavior: NoiseBehavior):
        self._receivers = receivers
        self._transmitter = transmitter
        self._propagation_behavior = propagation_behavior
        self._observed_frequencies = []
        for receiver in receivers:
            self._observed_frequencies.append(self._propagation_behavior.compute_observed_frequency(transmitter, receiver))

    def get_value_at(self, position: np.ndarray, velocity: np.ndarray):
        """ Returns a matrix - value of a system at a given position and velocity """
        c = self._propagation_behavior.speed
        f = self._transmitter._frequency
        x = self._transmitter._position[0]
        y = self._transmitter._position[1]
        v_x = self._transmitter._velocity[0]
        v_y = self._transmitter._velocity[1]

        result = []
        for i in range(len(self._receivers)):
            f_i = self._observed_frequencies[i]
            x_i = self._receivers[i]._position[0]
            y_i = self._receivers[i]._position[1]
            result.append(evaluate_system(c, f, f_i, x, y, x_i, y_i, v_x, v_y))

        return np.array(result)

    def get_jacobian_at(self, position: np.ndarray, velocity: np.ndarray):
        """ Returns a matrix - jacobian of a system at a given position and velocity """
        c = self._propagation_behavior.speed
        f = self._transmitter._frequency
        x = self._transmitter._position[0]
        y = self._transmitter._position[1]
        v_x = self._transmitter._velocity[0]
        v_y = self._transmitter._velocity[1]

        result = []
        for i in range(len(self._receivers)):
            f_i = self._observed_frequencies[i]
            x_i = self._receivers[i]._position[0]
            y_i = self._receivers[i]._position[1]
            result.append(evaluate_jacobian(c, f, f_i, x, y, x_i, y_i, v_x, v_y))

        return np.array(result)