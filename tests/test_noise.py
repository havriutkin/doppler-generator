import unittest
import numpy as np
from noise import GaussianNoise, NoNoise

class TestNoiseBehavior(unittest.TestCase):

    def test_gaussian_noise(self):
        noise_level = 0.1
        gaussian_noise = GaussianNoise(noise_level)
        value = np.array([1.0, 2.0, 3.0])
        noisy_value = gaussian_noise.add_noise(value)
        
        self.assertEqual(len(value), len(noisy_value))
        self.assertFalse(np.array_equal(value, noisy_value))

    def test_no_noise(self):
        no_noise = NoNoise()
        value = np.array([1.0, 2.0, 3.0])
        noisy_value = no_noise.add_noise(value)
        
        self.assertTrue(np.array_equal(value, noisy_value))

if __name__ == '__main__':
    unittest.main()