import numpy as np

from ..models.Receiver import Receiver

class ReceiverFactory:
    """ Factory class for creating Receiver objects """

    @staticmethod
    def create_random_receiver(dimension: int, min: int, max: int) -> Receiver:
        if dimension < 2 or dimension > 3:
            raise ValueError("Dimension must be 2 or 3")

        position = np.random.uniform(min, max, dimension)
        velocity = np.random.uniform(min/2, max/2, dimension)
        return Receiver(position, velocity)
    
    @staticmethod
    def create_random_static_receiver(dimension: int, min: int, max: int) -> Receiver:
        if dimension < 2 or dimension > 3:
            raise ValueError("Dimension must be 2 or 3")
        
        position = np.random.uniform(min, max, dimension)
        velocity = np.zeros(dimension)
        return Receiver(position, velocity)
    