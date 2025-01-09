from abc import ABC, abstractmethod
import numpy as np

from receiver import Receiver
from transmitter import Transmitter

import logging

class PropagationBehavior(ABC):
    def __init__(self, speed: float):
        self.speed = speed

    @abstractmethod
    def compute_observed_frequency(self, transmitter: Transmitter, receiver: Receiver):
        pass

    def _doppler_effect(self, transmitter: Transmitter, receiver: Receiver):
        relative_position = receiver._position - transmitter._position
        relative_velocity = receiver._velocity - transmitter._velocity
        distance = np.sqrt(np.dot(relative_position, relative_position))
        
        frequency = transmitter._frequency * (1 - np.dot(relative_position, relative_velocity) / (distance * self.speed))

        return frequency

class SoundInAirPropagation(PropagationBehavior):
    def __init__(self):
        super().__init__(speed=343.2)  # m/s

    def compute_observed_frequency(self, transmitter: Transmitter, receiver: Receiver):
        return self._doppler_effect(transmitter, receiver)

class SoundInWaterPropagation(PropagationBehavior):
    def __init__(self):
        super().__init__(speed=1500)  # m/s

    def compute_observed_frequency(self, transmitter: Transmitter, receiver: Receiver):
        return self._doppler_effect(transmitter, receiver)

class LightInAirPropagation(PropagationBehavior):
    def __init__(self):
        super().__init__(speed=299792458)  # m/s

    def compute_observed_frequency(self, transmitter: Transmitter, receiver: Receiver):
        return self._doppler_effect(transmitter, receiver)
    
class UnitPropagation(PropagationBehavior):
    def __init__(self):
        super().__init__(speed=1)  # m/s

    def compute_observed_frequency(self, transmitter: Transmitter, receiver: Receiver):
        return self._doppler_effect(transmitter, receiver)