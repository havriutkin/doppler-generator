import unittest
import numpy as np
from propagation import SoundInAirPropagation, SoundInWaterPropagation, LightInAirPropagation
from receiver import Receiver
from transmitter import Transmitter

class TestPropagation(unittest.TestCase):
    def setUp(self):
        self.transmitter1 = Transmitter(
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            frequency=1000
        )
        self.receiver1 = Receiver(
            position=np.array([1, 0, 0]),
            velocity=np.array([0, 0, 0])
        )
        
        self.transmitter2 = Transmitter(
            position=np.array([0, 0, 0]),
            velocity=np.array([10, 0, 0]),
            frequency=1500
        )
        self.receiver2 = Receiver(
            position=np.array([0, 10, 0]),
            velocity=np.array([0, -10, 0])
        )
        
        self.transmitter3 = Transmitter(
            position=np.array([0, 0, 0]),
            velocity=np.array([5, 5, 0]),
            frequency=2000
        )
        self.receiver3 = Receiver(
            position=np.array([10, 10, 0]),
            velocity=np.array([-5, -5, 0])
        )

    def test_sound_in_air_propagation(self):
        propagation = SoundInAirPropagation()
        observed_frequency1 = propagation.compute_observed_frequency(self.transmitter1, self.receiver1)
        self.assertAlmostEqual(observed_frequency1, 1000.0, delta=10)
        
        observed_frequency2 = propagation.compute_observed_frequency(self.transmitter2, self.receiver2)
        self.assertAlmostEqual(observed_frequency2, 1550.0, delta=10)
        
        observed_frequency3 = propagation.compute_observed_frequency(self.transmitter3, self.receiver3)
        self.assertAlmostEqual(observed_frequency3, 2080.0, delta=10)

    def test_sound_in_water_propagation(self):
        propagation = SoundInWaterPropagation()
        observed_frequency1 = propagation.compute_observed_frequency(self.transmitter1, self.receiver1)
        self.assertAlmostEqual(observed_frequency1, 1000.0, delta=10)
        
        observed_frequency2 = propagation.compute_observed_frequency(self.transmitter2, self.receiver2)
        self.assertAlmostEqual(observed_frequency2, 1500.0, delta=10)
        
        observed_frequency3 = propagation.compute_observed_frequency(self.transmitter3, self.receiver3)
        self.assertAlmostEqual(observed_frequency3, 2020.0, delta=10)

    def test_light_in_air_propagation(self):
        propagation = LightInAirPropagation()
        observed_frequency1 = propagation.compute_observed_frequency(self.transmitter1, self.receiver1)
        self.assertAlmostEqual(observed_frequency1, 1000.0, delta=10)
        
        observed_frequency2 = propagation.compute_observed_frequency(self.transmitter2, self.receiver2)
        self.assertAlmostEqual(observed_frequency2, 1500.0, delta=10)
        
        observed_frequency3 = propagation.compute_observed_frequency(self.transmitter3, self.receiver3)
        self.assertAlmostEqual(observed_frequency3, 2000.0, delta=10)

if __name__ == '__main__':
    unittest.main()