from abc import ABC, abstractmethod
import numpy as np
import json

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
        file = open(filename, "w")
        file.write("Transmitter Position, Transmitter Velocity, Transmitter Frequency, Receiver Position, Receiver Velocity, Observed Frequency, IsSolvable\n")
        for i in range(len(self._receivers)):
            receiver = self._receivers[i]
            file.write(f"{self._transmitter._position}, {self._transmitter._velocity}, {self._transmitter._frequency}, {receiver._position}, {receiver._velocity}, {self._observed_frequencies[i]}, {self._labels[i]}\n")
        file.close()


    def export_to_json(self, filename: str):
        data = {
            "transmitter": { "position": self._transmitter._position.tolist(), "velocity": self._transmitter._velocity.tolist(), "frequency": self._transmitter._frequency },
            "receivers": [],
            "observed_frequencies": self._observed_frequencies,
            "labels": self._labels
        }

        for receiver in self._receivers:
            data["receivers"].append({ "position": receiver._position.tolist(), "velocity": receiver._velocity.tolist() })

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)


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