from abc import ABC, abstractmethod
import numpy as np

from receiver import Receiver
from transmitter import Transmitter

class PropagationBehavior(ABC):
    def __init__(self, speed: float):
        self.speed = speed

    @abstractmethod
    def compute_observed_frequency(self, transmitter: Transmitter, receiver: Receiver):
        pass

    def _doppler_effect(self, transmitter: Transmitter, receiver: Receiver):
        # Shared Doppler effect formula logic
        distance = np.linalg.norm(receiver._position - transmitter._position)
        relative_position = receiver._position - transmitter._position
        relative_velocity = receiver._velocity - transmitter._velocity
        return transmitter._frequency - (transmitter._frequency / self.speed) * np.dot(relative_velocity, relative_position) / distance

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
    