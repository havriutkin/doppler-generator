import numpy as np

from .models.Receiver import Receiver
from .models.Transmitter import Transmitter
from .models.Noise import Noise, NoNoise
from .DopplerProblem import DopplerProblem

class DopplerBuilder:
    """ Builder class for Doppler problems """
    def __init__(self):
        self._dimension = None
        self._receivers = []
        self._transmitter = None
        self._propagation_speed = None
        self._noise = None
    
    def set_dimension(self, dimension: int):
        self._dimension = dimension
        return self
    
    def add_receiver(self, receiver: Receiver):
        if self._dimension is None:
            self._dimension = receiver.get_dimension()

        if receiver.get_dimension() != self._dimension:
            raise ValueError("All receivers must have the same dimension.")
        
        self._receivers.append(receiver)
        return self
    
    def set_transmitter(self, transmitter: Transmitter):
        if self._dimension is None:
            self._dimension = transmitter.get_dimension()

        if transmitter.get_dimension() != self._dimension:
            raise ValueError("The transmitter and receivers must have the same dimension.")
        
        self._transmitter = transmitter
        return self
    
    def set_propagation_speed(self, speed: float):
        self._propagation_speed = speed
        return self
    
    def set_noise_behavior(self, noise: Noise):
        self._noise = noise
        return self
    
    def build(self):
        if self._dimension is None:
            raise ValueError("Dimension must be set.")
        
        if self._transmitter is None:
            raise ValueError("Transmitter must be set.")
        
        if self._propagation_speed is None:
            raise ValueError("Propagation speed must be set.")
        
        if self._noise is None:
            self._noise = NoNoise()
        
        if len(self._receivers) == 0:
            raise ValueError("At least one receiver must be added.")
        
        return DopplerProblem(self._receivers, self._transmitter, self._propagation_speed, self._noise)

    
