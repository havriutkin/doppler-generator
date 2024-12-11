from receiver import Receiver, ReceiverFactory
from transmitter import Transmitter, TransmitterFactory
from propagation import SoundInAirPropagation, SoundInWaterPropagation, LightInAirPropagation
from doppler import DopplerBuilder, DopplerData, DopplerGenerator
from noise import GaussianNoise, NoNoise

import numpy as np

if __name__ == "__main__":
    # Generate 100 overdetermined (6 receivers) Doppler data points with labels in 2D
    n = 100
    dimension = 2

    # Min and max positions and velocities in meters
    min = -500
    max = 500

    propagation = SoundInAirPropagation()
    noise = GaussianNoise()

    tolerance = 5

    for i in range(n):
        transmitter = TransmitterFactory.create_random_transmitter_with_frequency(dimension, min, max, 1000)
        receivers = [ReceiverFactory.create_random_receiver(dimension, min, max) for _ in range(6)]
        # TODO: Receivers can't be too close

        # Build 
        builder = DopplerBuilder()
        for receiver in receivers: builder = builder.add_receiver(receiver) 
        builder = builder.set_transmitter(transmitter)
        builder = builder.set_propagation_behavior(propagation)  
        builder = builder.set_noise_behavior(noise)
        doppler_generator = builder.build()
        doppler_data: DopplerData = doppler_generator.generate_data_with_labels(tolerance)

