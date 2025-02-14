from abc import ABC, abstractmethod
import numpy as np

class Noise(ABC):
    @abstractmethod
    def add_noise(self, value: float):
        pass

class GaussianNoise(Noise):
    def __init__(self, noise_level: float):
        self._noise_level = noise_level

    def add_noise(self, value: float) -> float:
        return value + np.random.normal(0, self._noise_level)
    
    def __str__(self):
        return "Gaussian Noise with noise level: " + str(self._noise_level)
    
class NoNoise(Noise):
    def add_noise(self, value: float) -> float:
        return value
    
    def __str__(self):
        return "No Noise"
    