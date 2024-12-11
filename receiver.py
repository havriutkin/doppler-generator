import numpy as np

class Receiver:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self._position: np.ndarray = position
        self._velocity: np.ndarray = velocity

class ReceiverFactory:
    @staticmethod
    def create_random_receiver(dimension: int, min: int, max: int) -> Receiver:
        position = np.random.uniform(min, max, dimension)
        velocity = np.random.uniform(min, max, dimension)
        return Receiver(position, velocity)
    
    @staticmethod
    def create_random_static_receiver(dimension: int, min: int, max: int) -> Receiver:
        position = np.random.uniform(min, max, dimension)
        velocity = np.zeros(dimension)
        return Receiver(position, velocity)
    