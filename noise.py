from abc import ABC, abstractmethod
import numpy as np


class NoiseBehavior(ABC):
    @abstractmethod
    def add_noise(self, value: float):
        pass

class GaussianNoise(NoiseBehavior):
    def __init__(self, noise_level: float):
        self._noise_level = noise_level

    def add_noise(self, value: float):
        return value + np.random.normal(0, self._noise_level, 1)
    
class NoNoise(NoiseBehavior):
    def add_noise(self, value: float):
        return value
    