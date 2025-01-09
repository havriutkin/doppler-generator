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

    def _doppler_effect(self, transmitter: Transmitter, receiver: Receiver) -> np.float64:
        relative_position = np.array(receiver._position - transmitter._position, dtype=np.float64)
        relative_velocity = np.array(receiver._velocity - transmitter._velocity, dtype=np.float64)
        distance = np.sqrt(np.dot(relative_position, relative_position))
        inv_distance = 1 / distance
        inv_speed = 1 / self.speed

        frequency = transmitter._frequency * (1 - np.dot(relative_velocity, relative_position) * inv_distance * inv_speed)

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