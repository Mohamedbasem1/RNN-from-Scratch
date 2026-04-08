#!/usr/bin/env python
import sys
import os

# Add the src_to_implement directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src_to_implement'))

import unittest
from NeuralNetworkTests import TestConstraints

if __name__ == '__main__':
    # Create a test suite with just TestConstraints
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestConstraints)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
