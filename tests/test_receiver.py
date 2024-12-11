import unittest
import numpy as np
from receiver import Receiver, ReceiverFactory

class TestReceiver(unittest.TestCase):
    def test_receiver_initialization(self):
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.3])
        receiver = Receiver(position, velocity)
        np.testing.assert_array_equal(receiver._position, position)
        np.testing.assert_array_equal(receiver._velocity, velocity)

class TestReceiverFactory(unittest.TestCase):
    def test_create_random_receiver(self):
        dimension = 3
        min_val = 0
        max_val = 10
        receiver = ReceiverFactory.create_random_receiver(dimension, min_val, max_val)
        self.assertEqual(receiver._position.shape[0], dimension)
        self.assertEqual(receiver._velocity.shape[0], dimension)
        self.assertTrue(np.all(receiver._position >= min_val) and np.all(receiver._position <= max_val))
        self.assertTrue(np.all(receiver._velocity >= min_val) and np.all(receiver._velocity <= max_val))

    def test_create_random_receiver_various_dimensions(self):
        dimensions = [2, 3, 4, 5]
        min_val = 0
        max_val = 10
        for dimension in dimensions:
            receiver = ReceiverFactory.create_random_receiver(dimension, min_val, max_val)
            self.assertEqual(receiver._position.shape[0], dimension)
            self.assertEqual(receiver._velocity.shape[0], dimension)
            self.assertTrue(np.all(receiver._position >= min_val) and np.all(receiver._position <= max_val))
            self.assertTrue(np.all(receiver._velocity >= min_val) and np.all(receiver._velocity <= max_val))

    def test_create_random_static_receiver(self):
        dimension = 3
        min_val = 0
        max_val = 10
        receiver = ReceiverFactory.create_random_static_receiver(dimension, min_val, max_val)
        self.assertEqual(receiver._position.shape[0], dimension)
        self.assertEqual(receiver._velocity.shape[0], dimension)
        self.assertTrue(np.all(receiver._position >= min_val) and np.all(receiver._position <= max_val))
        np.testing.assert_array_equal(receiver._velocity, np.zeros(dimension))

if __name__ == '__main__':
    unittest.main()