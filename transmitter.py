import numpy as np

class Transmitter:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, frequency: float):
        self._position: np.ndarray = position
        self._velocity: np.ndarray = velocity
        self._frequency: float = frequency

class TransmitterFactory:
    @staticmethod
    def create_random_transmitter(dimension: int, min: int, max: int) -> Transmitter:
        position = np.random.uniform(min, max, dimension)
        velocity = np.random.uniform(min, max, dimension)
        frequency = np.random.uniform(1, 100)
        return Transmitter(position, velocity, frequency)

    @staticmethod
    def create_random_transmitter_with_frequency(dimension: int, min: int, max: int, frequency: float) -> Transmitter:
        position = np.random.uniform(min, max, dimension)
        velocity = np.random.uniform(min, max, dimension)
        return Transmitter(position, velocity, frequency)
