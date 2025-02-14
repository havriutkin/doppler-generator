import sympy as sp
import numpy as np
from termcolor import colored
from prettytable import PrettyTable

from . import Equations as eq
from .models.Transmitter import Transmitter
from .models.Receiver import Receiver
from .models.Noise import Noise

class DopplerProblem:
    def __init__(self, receivers: list[Receiver], transmitter: Transmitter, 
                 propagation_speed: float, noise: Noise):
        self._receivers = receivers
        self._transmitter = transmitter
        self._propagation_speed = propagation_speed
        self._noise = noise
        self._dimension = transmitter.get_dimension()
        self._check_dimensions()
        self._system = []

        self._observed_frequencies = []
        self._observed_frequencies = self.get_observed_frequencies()

    def _check_dimensions(self):
        """ Check if all receivers and transmitter have the same dimension """
        for receiver in self._receivers:
            if receiver.get_dimension() != self._dimension:
                raise ValueError("All receivers and transmitter must have the same dimension.")

    def get_observed_frequencies(self) -> list[float]:
        """ Returns the observed frequencies by the receivers """
        if len(self._observed_frequencies) == 0:
            for receiver in self._receivers:
                observed_freq = receiver.get_observed_frequency(self._propagation_speed, self._transmitter, self._noise)
                self._observed_frequencies.append(observed_freq)
        
        return self._observed_frequencies

    def get_system(self) -> list[sp.Eq]:
        """ Returns a system of equations for the Doppler problem """
        system = []
        for receiver in self._receivers:
            equation = sp.Eq(receiver.get_equation(self._propagation_speed, self._transmitter, self._noise), 0)
            system.append(equation)

        self._system = system
        return system

    def update_observed_freq(self, index, new_freq):
        self._receivers[index]._observed_freq = new_freq
        print(f"Update freq: {self._receivers[index]._observed_freq}")

    def __str__(self):
        table = PrettyTable()
        table.title = colored("Doppler Problem Description", "blue")
        table.field_names = ["Dimension", 
                             "Number of Receivers", 
                             "Transmitter Frequency",
                             "Propagation Speed",
                             "Noise Behavior"]
        table.add_row([self._dimension,
                          len(self._receivers),
                          self._transmitter._frequency,
                          self._propagation_speed,
                          str(self._noise)])

        return str(table)
    
    def to_json(self):
        if len(self._observed_frequencies) == 0:
            self.get_observed_frequencies()

        receivers = []
        for i in range(len(self._receivers)):
            receivers.append(self._receivers[i].to_json(i))

        merged_dict = {}
        for dict in receivers:
            merged_dict.update(dict)
        merged_dict["c"] = self._propagation_speed
        merged_dict["f"] = self._transmitter._frequency

        return merged_dict