from receiver import Receiver, ReceiverFactory
from transmitter import Transmitter, TransmitterFactory
from propagation import SoundInAirPropagation, SoundInWaterPropagation, LightInAirPropagation
from doppler import DopplerBuilder, DopplerData, DopplerGenerator
from noise import GaussianNoise, NoNoise

import numpy as np
import os

if __name__ == "__main__":
    # Generate 100 overdetermined (6 receivers) Doppler data points with labels in 2D
    n = 100
    dimension = 2

    # Min and max positions and velocities in meters
    min = -500
    max = 500

    propagation = SoundInAirPropagation()
    noise = GaussianNoise(noise_level=0.1)

    tolerance = 0.1

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
        doppler_data: DopplerData = doppler_generator.generate_data_with_labels(tolerance)

        # Save data in a file
        doppler_data.export_to_csv(f"data/doppler_data_{i}.csv")
    
    # Unite all data in a single file
    print("Unifying data in a single file...")
    with open("data/doppler_data.csv", "w") as file:
        file.write("Transmitter Position, Transmitter Velocity, Transmitter Frequency, Receiver Position, Receiver Velocity, Observed Frequency, IsSolvable\n")
        for i in range(n):
            with open(f"data/doppler_data_{i}.csv", "r") as sample_file:
                lines = sample_file.readlines()
                for line in lines[1:]:
                    file.write(line)
    
    # Remove individual files
    for i in range(n):
        os.remove(f"data/doppler_data_{i}.csv")

    print("Data unified.")

    print("Data generation finished.")
