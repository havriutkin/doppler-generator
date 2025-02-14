import numpy as np

from ..models.Transmitter import Transmitter

class TransmitterFactory:
    """ Factory class for creating Transmitter objects """

    @staticmethod
    def create_random_transmitter(dimension: int, min: int, max: int) -> Transmitter:
        if dimension < 2 or dimension > 3:
            raise ValueError("Dimension must be 2 or 3")

        position = np.random.uniform(min, max, dimension)
        velocity = np.random.uniform(min / 10, max / 10, dimension)
        frequency = np.random.uniform(1, 1000)
        return Transmitter(position, velocity, frequency)

    @staticmethod
    def create_random_transmitter_with_frequency(dimension: int, min: int, max: int, frequency: float) -> Transmitter:
        if dimension < 2 or dimension > 3:
            raise ValueError("Dimension must be 2 or 3")
        
        position = np.random.uniform(min, max, dimension)
        velocity = np.random.uniform(min / 2, max / 2, dimension)
        return Transmitter(position, velocity, frequency)
