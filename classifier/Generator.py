# data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from generator.DopplerProblem import DopplerProblem
from generator.DopplerBuilder import DopplerBuilder
from generator.factories.ReceiverFactory import ReceiverFactory
from generator.factories.TransmitterFactory import TransmitterFactory
from generator.models.Noise import Noise, GaussianNoise
from ..generator.LabelProcedure import LabelByDistortion
from abc import ABC, abstractmethod

class DopplerGenerator(ABC):
    @abstractmethod
    def generate_problem(self) -> DopplerProblem:
        pass

class RandomStaticDopplerGenerator(DopplerGenerator):
    def __init__(self, num_receivers: int, dimension: int, propagation_speed: float, noise: Noise):
        self._num_receivers = num_receivers
        self._dimension = dimension
        self._propagation_speed = propagation_speed
        self._noise = noise
        self._receiver_factory = ReceiverFactory

    def generate_problem(self) -> DopplerProblem:
        transmitter_factory = TransmitterFactory()
        transmitter = transmitter_factory.create_random_transmitter(dimension=self._dimension, 
                                                                    min=-10000, max=10000)
        receivers = [self._receiver_factory.create_random_static_receiver(dimension=self._dimension, 
                                                                          min=-10000, max=10000)]
        builder = DopplerBuilder()
        builder.set_dimension(self._dimension)
        builder.set_transmitter(transmitter)
        builder.set_propagation_speed(self._propagation_speed)
        builder.set_noise_behavior(self._noise)
        for receiver in receivers:
            builder.add_receiver(receiver)
        return builder.build()
    