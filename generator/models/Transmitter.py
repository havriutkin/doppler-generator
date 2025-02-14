import numpy as np

class Transmitter:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, frequency: float):
        if len(position) != len(velocity):
            raise ValueError("Position and velocity must have the same dimension")

        self._position: np.ndarray = position
        self._velocity: np.ndarray = velocity
        self._frequency: float = frequency

    def get_dimension(self) -> int:
        return len(self._position)