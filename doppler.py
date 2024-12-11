from abc import ABC, abstractmethod
import numpy as np

from receiver import Receiver
from transmitter import Transmitter
from noise import NoiseBehavior
from propagation import PropagationBehavior


class DopplerData:
    def __init__(self, receivers: list[Receiver], transmitter: Transmitter, observed_frequencies: list[float], labels: list[int] | None):
        self._receivers = receivers
        self._transmitter = transmitter
        self._observed_frequencies = observed_frequencies
        self._labels = labels

    def export_to_csv(self, filename: str):
        pass

    def export_to_json(self, filename: str):
        pass

class DopplerGenerator:
    def __init__(self, receivers: list[Receiver], transmitter: Transmitter, propagation_behavior: PropagationBehavior, noise_behavior: NoiseBehavior):
        self._receivers = receivers
        self._transmitter = transmitter
        self._propagation_behavior = propagation_behavior
        self._noise_behavior = noise_behavior

    def generate_data(self):
        data = []
        for receiver in self._receivers:
            observed_frequency = self._propagation_behavior.compute_observed_frequency(self._transmitter, receiver)
            data.append(self._noise_behavior.add_noise(observed_frequency))
        return DopplerData(self._receivers, self._transmitter, data, None)
    
    def generate_data_with_labels(self, tolerance: float):
        # If noise is bigger than tolerance, label as 0, otherwise 1
        data = []
        labels = []
        for receiver in self._receivers:
            observed_frequency = self._propagation_behavior.compute_observed_frequency(self._transmitter, receiver)
            noisy_observed_frequency = self._noise_behavior.add_noise(observed_frequency)
            data.append(noisy_observed_frequency)
            labels.append(0 if np.abs(noisy_observed_frequency - observed_frequency) > tolerance else 1)
        return DopplerData(self._receivers, self._transmitter, data, labels)
    
class DopplerBuilder:
    def __init__(self):
        self._dimension = None
        self._receivers = []
        self._transmitter = None
        self._propagation_behavior = None
        self._noise_behavior = None

    def add_receiver(self, receiver: Receiver):
        # todo: check that dimension aligns

        self._receivers.append(receiver)
        return self
    
    def set_transmitter(self, transmitter: Transmitter):
        # todo: check that dimension aligns
        self._transmitter = transmitter
        return self
    
    def set_propagation_behavior(self, propagation_behavior: PropagationBehavior):
        self._propagation_behavior = propagation_behavior
        return self
    
    def set_noise_behavior(self, noise_behavior: NoiseBehavior):
        self._noise_behavior = noise_behavior
        return self
    
    def build(self):
        return DopplerGenerator(self, self._receivers, self._transmitter, 
                                self._propagation_behavior, self._noise_behavior)