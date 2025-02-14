import json
import numpy as np
from termcolor import colored

from .factories.ReceiverFactory import ReceiverFactory
from .factories.TransmitterFactory import TransmitterFactory
from .models.Noise import Noise, GaussianNoise, NoNoise
from .DopplerProblem import DopplerProblem
from .DopplerBuilder import DopplerBuilder
from .LabelProcedure import LabelProcedure, LabelByDistortion, LabelByRelaxation

NUM_OF_PROBLEMS = 1000
DIMENSION = 2
NUM_OF_REC = 5
NOISE_LEVEL = 0.2

def np_encoder(obj):
    import sympy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, sympy.core.numbers.Float):
        return float(obj)
    raise TypeError(f"Unserializable object {obj} of type {type(obj)}")



if __name__ == "__main__":
    problems: list[DopplerProblem] = []
    receiver_factory = ReceiverFactory()
    transmitter_factory = TransmitterFactory()
    noise = GaussianNoise(noise_level=NOISE_LEVEL)
    label_procedure = LabelByDistortion(chance=0.5)

    print(colored("Generating data...", 'cyan'))

    for i in range(NUM_OF_PROBLEMS):
        print(colored(f"\tGenerating problem {i+1}/{NUM_OF_PROBLEMS}...", 'light_yellow'))
        builder = DopplerBuilder()
        transmitter = transmitter_factory.create_random_transmitter(dimension=DIMENSION, min=-1000, max=1000)
        receivers = [receiver_factory.create_random_static_receiver(dimension=DIMENSION, min=-1000, max=1000) 
                     for _ in range(NUM_OF_REC)]
        prop_speed = float(np.random.uniform(300, 500))

        builder.set_dimension(DIMENSION)
        builder.set_transmitter(transmitter)
        builder.set_noise_behavior(noise)
        builder.set_propagation_speed(prop_speed)
        for receiver in receivers:
            builder.add_receiver(receiver)
        problem = builder.build()
        problems.append(problem)
    print(colored("Data is generated successfully!\n", 'green'))

    print(colored("Labeling data...", 'cyan'))
    labels = []
    for i in range(NUM_OF_PROBLEMS):
        print(colored(f"\tLabeling problem {i+1}/{NUM_OF_PROBLEMS}...", 'light_yellow'))
        label = label_procedure.get_label(problems[i])
        labels.append(label)
    print(colored("Data is labeled successfully!", 'green'))
    print(colored(f"\tLabel 0: {labels.count(0)}", 'cyan'))
    print(colored(f"\tLabel 1: {labels.count(1)}\n", 'cyan'))

    print(colored("Merging data...", 'cyan'))
    data = [problem.to_json() for problem in problems]
    for i, features in enumerate(data):
        features['label'] = labels[i]

    # Save the data
    print(colored("Saving data...", 'cyan'))
    with open(f"data2.json", 'w') as f:
        json.dump(data, f, indent=4, default=np_encoder)
    print(colored("Data is saved successfully!\n", 'green'))

    

    


