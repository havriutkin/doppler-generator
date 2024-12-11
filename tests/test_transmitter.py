import unittest
import numpy as np
from transmitter import Transmitter, TransmitterFactory

class TestTransmitter(unittest.TestCase):
    def test_transmitter_initialization(self):
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.3])
        frequency = 50.0
        transmitter = Transmitter(position, velocity, frequency)
        
        np.testing.assert_array_equal(transmitter._position, position)
        np.testing.assert_array_equal(transmitter._velocity, velocity)
        self.assertEqual(transmitter._frequency, frequency)

class TestTransmitterFactory(unittest.TestCase):
    def test_create_random_transmitter(self):
        dimension = 3
        min_val = 0
        max_val = 10
        transmitter = TransmitterFactory.create_random_transmitter(dimension, min_val, max_val)
        
        self.assertEqual(transmitter._position.shape[0], dimension)
        self.assertEqual(transmitter._velocity.shape[0], dimension)
        self.assertTrue(1 <= transmitter._frequency <= 100)

    def test_create_random_transmitter_with_frequency(self):
        dimension = 3
        min_val = 0
        max_val = 10
        frequency = 60.0
        transmitter = TransmitterFactory.create_random_transmitter_with_frequency(dimension, min_val, max_val, frequency)
        
        self.assertEqual(transmitter._position.shape[0], dimension)
        self.assertEqual(transmitter._velocity.shape[0], dimension)
        self.assertEqual(transmitter._frequency, frequency)

    def test_create_random_transmitter_various_dimensions(self):
        dimensions = [2, 3, 4, 5]
        min_val = 0
        max_val = 10
        for dimension in dimensions:
            with self.subTest(dimension=dimension):
                transmitter = TransmitterFactory.create_random_transmitter(dimension, min_val, max_val)
                self.assertEqual(transmitter._position.shape[0], dimension)
                self.assertEqual(transmitter._velocity.shape[0], dimension)
                self.assertTrue(1 <= transmitter._frequency <= 100)

    def test_create_random_transmitter_with_frequency_various_dimensions(self):
        dimensions = [2, 3, 4, 5]
        min_val = 0
        max_val = 10
        frequency = 60.0
        for dimension in dimensions:
            with self.subTest(dimension=dimension):
                transmitter = TransmitterFactory.create_random_transmitter_with_frequency(dimension, min_val, max_val, frequency)
                self.assertEqual(transmitter._position.shape[0], dimension)
                self.assertEqual(transmitter._velocity.shape[0], dimension)
                self.assertEqual(transmitter._frequency, frequency)

if __name__ == '__main__':
    unittest.main()