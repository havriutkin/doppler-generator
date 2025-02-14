import numpy as np
import sympy as sp

from .. import Equations as eq
from .Transmitter import Transmitter
from .Noise import Noise

# Define dictionaries to map dimensions to the corresponding symbols and equations.
TRANS_POSITION_SYMBOLS = {
    2: [eq.x, eq.y],
    3: [eq.x, eq.y, eq.z]
}

TRANS_VELOCITY_SYMBOLS = {
    2: [eq.v_x, eq.v_y],
    3: [eq.v_x, eq.v_y, eq.v_z]
}

POSITION_SYMBOLS = {
    2: [eq.xi, eq.yi],
    3: [eq.xi, eq.yi, eq.zi]
}

VELOCITY_SYMBOLS = {
    2: [eq.vi_x, eq.vi_y],
    3: [eq.vi_x, eq.vi_y, eq.vi_z]
}

DOPPLER_EQUATION = {
    2: eq.doppler_eq_2d,
    3: eq.doppler_eq_3d
}

FREQUENCY_EQUATION = {
    2: eq.observer_freq_2d,
    3: eq.observer_freq_3d
}


class Receiver:
    """ Represents a receiver in the simulation """

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        if len(position) != len(velocity):
            raise ValueError("Position and velocity must have the same dimension.")

        self._position: np.ndarray = position
        self._velocity: np.ndarray = velocity
        self._dimension = len(position)

        if self._dimension not in DOPPLER_EQUATION:
            raise ValueError("Invalid dimension. Currently only 2D and 3D are supported.")

        subs_dict = {}
        for sym, value in zip(POSITION_SYMBOLS[self._dimension], position):
            subs_dict[sym] = value
        for sym, value in zip(VELOCITY_SYMBOLS[self._dimension], velocity):
            subs_dict[sym] = value

        self._equation = DOPPLER_EQUATION[self._dimension].subs(subs_dict)
        self._observed_freq = None

    def get_dimension(self) -> int:
        """ Returns the dimension of the receiver """
        return self._dimension

    def get_observed_frequency(self, prop_speed: float, transmitter: Transmitter, noise: Noise) -> float:
        """ Returns the frequency observed by the receiver """
        if self._observed_freq is not None:
            return self._observed_freq
        
        if (self._dimension != transmitter.get_dimension()):
            raise ValueError("The transmitter and receiver must have the same dimension.")

        subs_dict = {
            eq.f: transmitter._frequency,
            eq.c: prop_speed
        }

        for sym, value in zip(TRANS_POSITION_SYMBOLS[self._dimension], transmitter._position):
            subs_dict[sym] = value
        for sym, value in zip(TRANS_VELOCITY_SYMBOLS[self._dimension], transmitter._velocity):
            subs_dict[sym] = value
        for sym, value in zip(POSITION_SYMBOLS[self._dimension], self._position):
            subs_dict[sym] = value
        for sym, value in zip(VELOCITY_SYMBOLS[self._dimension], self._velocity):
            subs_dict[sym] = value

        freq = FREQUENCY_EQUATION[self._dimension].subs(subs_dict).evalf()
        freq = noise.add_noise(freq)
        self._observed_freq = freq

        return self._observed_freq

    def get_equation(self, prop_speed: float, transmitter: Transmitter, noise: Noise) -> sp.Expr:
        """ Returns the equation associated with the receiver """
        observed_freq = self.get_observer_frequency(prop_speed, transmitter, noise)
        substitute = self._equation.subs({
            eq.c: prop_speed,
            eq.f: transmitter._frequency,
            eq.fi: observed_freq
        })
        return substitute

    def to_json(self, index: int) -> dict:
        """ Returns a JSON representation of the receiver """
        if self._observed_freq is None:
            raise ValueError("Observed frequency must be calculated before converting to JSON.")
        
        if self._dimension == 2:
            return {
                f"v{index}_x": self._velocity[0],
                f"v{index}_y": self._velocity[1],
                f"x{index}": self._position[0],
                f"y{index}": self._position[1],
                f"f{index}": self._observed_freq
            }
        elif self._dimension == 3:
            return {
                f"v{index}_x": self._velocity[0],
                f"v{index}_y": self._velocity[1],
                f"v{index}_z": self._velocity[2],
                f"x{index}": self._position[0],
                f"y{index}": self._position[1],
                f"z{index}": self._position[2],
                f"f{index}": self._observed_freq
            }
        else:
            raise ValueError("Invalid dimension. Currently only 2D and 3D are supported.")
