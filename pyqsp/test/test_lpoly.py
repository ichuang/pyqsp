import os
import numpy as np
from pyqsp import LPoly

# -----------------------------------------------------------------------------
# unit tests

import unittest


class Test_lpoly(unittest.TestCase):

    def test_simple_unitary_from_angles1(self):
        phiset = [0]  # identity
        ualg = LPoly.LAlg.unitary_from_angles(phiset)
        print(f"For phiset={phiset}, U={ualg}")
        print(f"diagonal poly = {ualg.IPoly}")
        assert ualg.IPoly == LPoly.LPoly([1])

    def test_simple_unitary_from_angles2(self):
        phiset = [0, 0]  # w
        ualg = LPoly.LAlg.unitary_from_angles(phiset)
        print(f"For phiset={phiset}, U={ualg}")
        print(f"diagonal poly = {ualg.IPoly}")
        assert ualg.IPoly == LPoly.LPoly([0, 1], -1)

    def test_simple_unitary_from_angles3(self):
        phiset = [0, 0, 0]  # w^2
        ualg = LPoly.LAlg.unitary_from_angles(phiset)
        print(f"For phiset={phiset}, U={ualg}")
        print(f"diagonal poly = {ualg.IPoly}")
        assert ualg.IPoly == LPoly.LPoly([0, 0, 1], -2)

    def test_simple_unitary_from_angles4(self):
        # fourier transform of sine
        phiset = [-np.pi / 4, 0, np.pi / 4]
        ualg = LPoly.LAlg.unitary_from_angles(phiset)
        print(f"For phiset={phiset}, U={ualg}")
        print(f"diagonal poly = {ualg.IPoly}")
        assert ualg.IPoly == LPoly.LPoly([0.5, 0, 0.5], -2)

    def test_simple_unitary_from_angles5(self):
        # fourier transform of sine
        phiset = [-np.pi / 4, 0, 0, 0, np.pi / 4]
        ualg = LPoly.LAlg.unitary_from_angles(phiset)
        print(f"For phiset={phiset}, U={ualg}")
        print(f"diagonal poly = {ualg.IPoly}")
        assert ualg.IPoly == LPoly.LPoly([0.5, 0, 0, 0, 0.5], -4)

    def test_lpoly1(self):
        # Exp[-i (pi/2) X] * w = w^(-1) * iX
        w = LPoly.w
        Q0 = LPoly.LAlg.rotation(np.pi / 2)
        Q0.IPoly.round_zeros()
        prod = Q0 * w
        print(f"w = {w}")
        print(f"~w = {~w}")
        print(f"Q0 = {Q0}")
        print(f"Q0 * w = {prod}")
        print(f"poly([1,0], -1) = {LPoly.LPoly([1], -1)}")
        print(f"prod.XPoly coefs={prod.XPoly.coefs}, dmin={prod.XPoly.dmin}")
        assert prod.XPoly == LPoly.LPoly([1], -1)

    def test_LPoly_mul1(self):
        '''
        Test product of two LPoly's
        '''
        lp1 = LPoly.LPoly([1 / 2, 1 / 2], -1)
        lp2 = lp1 * lp1
        print(f"lp1={lp1}")
        print(f"lp1 * lp1 = {lp2}")
        print(f"lp2 coefs={lp2.coefs}")
        assert lp2.dmin == -2
        assert abs(lp2.coefs - np.array([0.25, 0.5, 0.25])).sum() < 1.0e-4


class Test_PolynomialToLaurentForm(unittest.TestCase):

    def test_p2lf3(self):
        pcoefs = [-1, 0, 2]
        Plp = LPoly.PolynomialToLaurentForm(pcoefs)
        print(f"Polynomial {pcoefs} -> Laurent poly {Plp}")
        print(f"LPoly coefs = {Plp.coefs}")
        assert Plp.dmin == -2
        assert abs(Plp.coefs - np.array([0.5, 0, 0.5])).sum() < 1.0e-4

    def test_p2lf1(self):
        coefs = [0, 1]
        lp = LPoly.PolynomialToLaurentForm(coefs)
        print(f"Polynomial {coefs} -> Laurent poly {lp}")
        assert lp.dmin == -1
        assert abs(lp.coefs - np.array([1 / 2, 1 / 2])).sum() < 1.0e-8

    def test_p2lf2(self):
        '''
        Note how LPoly coefficient list elements correspond to powers of w which go up by 2 at a time, not one!
        '''
        coefs = [0, -3, 0, 4]
        lp = LPoly.PolynomialToLaurentForm(coefs)
        print(f"Polynomial {coefs} -> Laurent poly {lp}")
        print(f"LPoly coefs = {lp.coefs}")
        assert lp.dmin == -3
        assert abs(lp.coefs - np.array([0.5, 0, 0, 0.5])).sum() < 1.0e-8
