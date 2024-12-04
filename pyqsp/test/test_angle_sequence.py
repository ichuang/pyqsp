import unittest

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev

from pyqsp.angle_sequence import (AngleFindingError,
                                  QuantumSignalProcessingPhases, poly2laurent)
from pyqsp.LPoly import LPoly, PolynomialToLaurentForm

# -----------------------------------------------------------------------------
# unit tests


class Test_angle_sequence(unittest.TestCase):
    def test_poly2laurent_1(self):
        pcoefs = np.array([-3 - 2j, 0., 26 + 10j, 0., -24 - 8j])
        expected = np.array([-3 - 1j, 1 + 1j, 2., 1 + 1j, -3 - 1j]) / 2.

        # Given chebyshev bypass, poly2laurent now no longer casts to Chebyshev basis, but expects it outright.
        pcoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)

        result = poly2laurent(pcoefs)
        self.assertAlmostEqual(np.max(np.abs(expected - result)), 0.)

    def test_poly2laurent_2(self):
        pcoefs = np.array([0., -5 + 5j, 0., 8 - 4j])
        expected = np.array([2 - 1j, 1 + 2j, 1 + 2j, 2 - 1j]) / 2.
        pcoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)
        result = poly2laurent(pcoefs)
        self.assertAlmostEqual(np.max(np.abs(expected - result)), 0.)

    def test_poly2laurent_3(self):
        pcoefs = np.array([1., -5 + 5j, 0., 8 - 4j])
        with self.assertRaises(AngleFindingError):
            poly2laurent(pcoefs)

    def test_response_1(self):
        pcoefs = [0, 1]

        poly = Chebyshev(pcoefs)

        QuantumSignalProcessingPhases(poly, signal_operator="Wx")
        QuantumSignalProcessingPhases(poly, signal_operator="Wz")

    def test_response_2(self):
        pcoefs = [0, 0, 1]
        poly = Chebyshev(pcoefs)

        QuantumSignalProcessingPhases(poly, signal_operator="Wx")
        QuantumSignalProcessingPhases(poly, signal_operator="Wz")

    def test_response_3(self):
        pcoefs = [0, -3, 0, 4]
        # Converting from monomial basis to expected Chebyshev one.
        pcoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)
        poly = Chebyshev(pcoefs)

        QuantumSignalProcessingPhases(poly, signal_operator="Wx")
        QuantumSignalProcessingPhases(poly, signal_operator="Wz")

    def test_response_4(self):
        pcoefs = [0., -2 + 1j, 0., 2.]
        pcoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)
        poly = Chebyshev(pcoefs)

        QuantumSignalProcessingPhases(
            poly, signal_operator="Wx", measurement="z")

    def test_response_5(self):
        pcoefs = [-1., 0., 50., 0., -400., 0., 1120., 0., -1280., 0., 512.]
        pcoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)
        poly = Chebyshev(pcoefs)

        QuantumSignalProcessingPhases(poly, signal_operator="Wx")
        QuantumSignalProcessingPhases(poly, signal_operator="Wz")

    def test_response_6(self):
        pcoefs = [-1., 0., (1 / 2) * (4 + 3j - (1 - 2j) * np.sqrt(3)),
                  0., (1 - 1j) * (-1j + np.sqrt(3))]
        pcoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)
        poly = Chebyshev(pcoefs)

        QuantumSignalProcessingPhases(
            poly, signal_operator="Wx", measurement="z")
