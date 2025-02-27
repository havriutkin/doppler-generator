from abc import ABC, abstractmethod
import numpy as np
import sympy as sp
import logging

from generator.DopplerProblem import DopplerProblem
from generator import Equations as eq

UNKNOWNS_PER_DIMENSION = {
    2: 4,
    3: 6
}

class LabelProcedure(ABC):
    @abstractmethod
    def get_label(self, doppler_problem: DopplerProblem) -> int:
        pass

class LabelByRelaxation(LabelProcedure):
    """ Relax the system, solve it and check if it converges """
    def __init__(self, tolerance: float):
        super().__init__()
        self._tolerance = tolerance

    def get_label(self, doppler_problem: DopplerProblem) -> int:
        # Get the system
        system = doppler_problem.get_system()
        
        # Check if it is overdetermined
        dimension = doppler_problem._dimension
        unknowns = UNKNOWNS_PER_DIMENSION[dimension]
        if len(system) <= unknowns:
            raise ValueError("System is not overdetermined.")
        
        # Relax the system by getting rid of the last equation
        relaxed_system = system[:-1]

        # Solve the system
        solutions = sp.solve(relaxed_system, dict=True)
        logging.debug(f"Number of solutions: {len(solutions)}")

        if len(solutions) == 0:
            return 0
        
        # Substitute the solution in the last equation
        last_equation = system[-1]
        for solution in solutions:
            if last_equation.subs(solution) < self._tolerance:
                return 1
            
        return 0

class LabelByDistortion(LabelProcedure):
    """ 
        With the given chance replace one of the frequencies with a random one and return 0.
        Otherwise return 1. 
    """
    def __init__(self, chance: float):
        super().__init__()
        self._chance = chance

    def get_label(self, doppler_problem: DopplerProblem) -> int:
        doppler_problem.get_observed_frequencies()
        if np.random.rand() < self._chance:
            index = np.random.randint(len(doppler_problem._observed_frequencies))
            old_freq = doppler_problem._observed_frequencies[index]
            new_freq = old_freq
            while new_freq == old_freq:
                new_freq = np.random.randint(1000)
                
            doppler_problem.update_observed_freq(index, new_freq)

            print(f"Updated freq at index {index} from {old_freq} to {new_freq}")

            return 0
        
        return 1