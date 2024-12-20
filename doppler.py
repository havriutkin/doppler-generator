from abc import ABC, abstractmethod
import numpy as np
import json

from receiver import Receiver
from transmitter import Transmitter
from noise import NoiseBehavior
from propagation import PropagationBehavior


class DopplerData:
    def __init__(self, receivers: list[Receiver], transmitter: Transmitter, observed_frequencies: list[float], label: int | None):
        self._receivers = receivers
        self._transmitter = transmitter
        self._observed_frequencies = observed_frequencies
        self._label = label

    def get_label(self):
        return self._label

    def get_json(self):
        return {
            "transmitter": { "position": self._transmitter._position.tolist(),
                                "velocity": self._transmitter._velocity.tolist(),
                                "frequency": self._transmitter._frequency },
            "receivers": [{ "position": self._receivers[i]._position.tolist(),
                            "velocity": self._receivers[i]._velocity.tolist(),
                            "observed_frequency": self._observed_frequencies[i] } for i in range(len(self._receivers))],
            "label": self._label
        }

    def export_to_csv(self, filename: str):
        with open(filename, "w") as file:
            file.write("Transmitter Position, Transmitter Velocity, Transmitter Frequency, Receiver Position, Receiver Velocity, Observed Frequency, Label\n")
            for i in range(len(self._observed_frequencies)):
                file.write(f"{self._transmitter._position }, {self._transmitter._velocity}, {self._transmitter._frequency}, {self._receivers[i]._position}, {self._receivers[i]._velocity}, {self._observed_frequencies[i]}, {self._label}\n")

    def export_to_json(self, filename: str):
        data = self.get_json()

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

class DopplerDataAggregator:
    def __init__(self):
        self._data: list[DopplerData] = []

    def add_data(self, data: DopplerData):
        self._data.append(data)
        return self
    
    def export_to_json(self, filename: str):
        data_list = []
        for data in self._data:
            data_list.append(data.get_json())

        with open(filename, "w") as file:
            json.dump(data_list, file, indent=4)

    def get_data(self):
        return self._data

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
            data.append(self._noise_behavior.add_noise(observed_frequency)[0])
        return DopplerData(self._receivers, self._transmitter, data, None)
    
    def generate_data_with_label(self, tolerance: float):
        # If noise is bigger than tolerance, label as 0, otherwise 1
        data = []
        label = 1
        biggest_noise = 0
        for receiver in self._receivers:
            observed_frequency = self._propagation_behavior.compute_observed_frequency(self._transmitter, receiver)
            noisy_observed_frequency = self._noise_behavior.add_noise(observed_frequency)[0]
            data.append(noisy_observed_frequency)
            noise = abs(noisy_observed_frequency - observed_frequency)
            if noise > biggest_noise:
                biggest_noise = noise

        if biggest_noise > tolerance:
            label = 0

        return DopplerData(self._receivers, self._transmitter, data, label)
    
class DopplerBuilder:
    def __init__(self):
        self._dimension = None
        self._receivers = []
        self._transmitter = None
        self._propagation_behavior = None
        self._noise_behavior = None

    def add_receiver(self, receiver: Receiver):
        if self._dimension is None:
            self._dimension = receiver.get_dimension()
        else:
            assert self._dimension == receiver.get_dimension(), "Dimensions do not align"

        self._receivers.append(receiver)
        return self
    
    def set_transmitter(self, transmitter: Transmitter):
        if self._dimension is None:
            self._dimension = transmitter.get_dimension()
        else:
            assert self._dimension == transmitter.get_dimension(), "Dimensions do not align"

        self._transmitter = transmitter
        return self
    
    def set_propagation_behavior(self, propagation_behavior: PropagationBehavior):
        self._propagation_behavior = propagation_behavior
        return self
    
    def set_noise_behavior(self, noise_behavior: NoiseBehavior):
        self._noise_behavior = noise_behavior
        return self
    
    def build(self):
        return DopplerGenerator(self._receivers, self._transmitter, 
                                self._propagation_behavior, self._noise_behavior)