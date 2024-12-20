from receiver import Receiver, ReceiverFactory
from transmitter import Transmitter, TransmitterFactory
from propagation import SoundInAirPropagation, SoundInWaterPropagation, LightInAirPropagation
from doppler import DopplerBuilder, DopplerData, DopplerGenerator, DopplerDataAggregator
from noise import GaussianNoise, NoNoise

import numpy as np
import os

if __name__ == "__main__":
    # Generate 100 overdetermined (6 receivers) Doppler data points with labels in 2D
    n = 1000
    dimension = 2

    # Min and max positions and velocities in meters
    min = -500
    max = 500

    propagation = SoundInAirPropagation()
    noise = GaussianNoise(noise_level=0.1)

    tolerance = 0.18

    data_aggregator: DopplerDataAggregator = DopplerDataAggregator()

    labels_count = {0: 0, 1: 0}

    print("Generating data...")
    for i in range(n):
        print(f"\tGenerating sample {i+1}/{n}...")

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
        doppler_data: DopplerData = doppler_generator.generate_data_with_label(tolerance)

        # Count labels
        labels_count[doppler_data.get_label()] += 1

        data_aggregator.add_data(doppler_data)

        # Save data in a file
        doppler_data.export_to_json(f"data/partial/doppler_data_{i}.json")
    
    # Unite all data in a single file
    data_aggregator.export_to_json("data/doppler_data.json")
    
    # Remove individual files
    """
    for i in range(n):
        os.remove(f"data/partial/doppler_data_{i}.csv")"""

    print("Data generation finished.")
    print(f"Labels count: {labels_count}")
