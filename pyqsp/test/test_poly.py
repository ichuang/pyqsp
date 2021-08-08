import unittest

import numpy as np

import pyqsp
from pyqsp import LPoly, angle_sequence, poly, response

# -----------------------------------------------------------------------------
# unit tests


class Test_poly(unittest.TestCase):

    def test_oneoverx0(self):
        pg = pyqsp.poly.PolyOneOverX()
        gp = pg.generate(8, 0.01, return_coef=False, ensure_bounded=False)
        # print(f"gp={gp}")
        print(f"gp(0.5) = {gp(0.5)}")
        assert abs(gp(0.5) - 2) < 0.01
        assert abs(gp(0.3) - 1 / (0.3)) < 0.01

    def test_oneoverx1(self):
        '''
        unit test to ensure that the polynomial approximation to 1/x really is close to 1/x
        '''
        kappa = 3
        epsilon = 0.01
        pg = pyqsp.poly.PolyOneOverX()
        gpoly = pg.generate(
            kappa,
            epsilon,
            return_coef=False,
            ensure_bounded=False)
        xpos = np.linspace(1 / kappa, 1)
        xval = np.concatenate([-xpos, xpos])
        expected = 1 / xval
        polyval = gpoly(xval)
        diff = abs(polyval - expected).mean()
        # print(f"diff={diff}")
        assert diff < 0.1

    def test_poly_one_over_x_response1(self):
        pg = pyqsp.poly.PolyOneOverX()
        pcoefs = pg.generate(3, 0.3, return_coef=True, ensure_bounded=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, signal_operator="Wx")
        # print(f"QSP angles = {phiset}")
        response.PlotQSPResponse(
            phiset, signal_operator="Wx", pcoefs=pcoefs, show=False)
        assert True

    def test_poly_sign1(self):
        pg = pyqsp.poly.PolySign()
        pfunc = pg.generate(17, 10)
        assert (pfunc(-0.9) < -0.2)
        assert (pfunc(0.9) > -0.2)

    def test_poly_thresh1(self):
        pg = pyqsp.poly.PolyThreshold()
        pcoefs = pg.generate(18, 10)
        pfunc = np.polynomial.Polynomial(pcoefs)
        # print(f"sign poly at -0.9 = {pfunc(-0.9)}")
        # print(f"sign poly at 0 = {pfunc(0)}")
        assert (pfunc(-0.9) < 0.1)
        assert (pfunc(0) > 0.3)
        assert (pfunc(0.9) < 0.1)

    def test_poly_linamp1(self):
        pg = pyqsp.poly.PolyLinearAmplification()
        pcoefs, scale = pg.generate(19, 0.25, return_scale=True)
        pfunc = np.polynomial.Polynomial(pcoefs)
        assert (pfunc(-0.9) < 0.05)
        assert (np.abs(pfunc(0.25) / scale - 0.5)) < 0.05
        assert (pfunc(0) < 0.05)
        assert (pfunc(0.9) < 0.05)

    def test_poly_phase1(self):
        pg = pyqsp.poly.PolyPhaseEstimation()
        pcoefs, scale = pg.generate(
            18, 10, ensure_bounded=True, return_scale=True)
        pfunc = np.polynomial.Polynomial(pcoefs)
        self.assertLess(pfunc(-0.9) / scale, -0.9)
        self.assertLess(pfunc(0.9) / scale, -0.9)
        self.assertGreater(pfunc(0) / scale, 0.9)

    def test_poly_gibbs1(self):
        pg = pyqsp.poly.PolyGibbs()
        pfunc = pg.generate(30, 4.5)
        # print(f"gibbs poly at 0.9 = {pfunc(0.9)}")
        # print(f"gibbs poly at 0 = {pfunc(0)}")
        assert (pfunc(0.9) < 0.3)
        assert (pfunc(0) > 0.9)

    def test_poly_cosine(self):
        tau = 10
        epsilon = 0.1
        pg = pyqsp.poly.PolyCosineTX()
        pcoefs, scale = pg.generate(
            tau,
            epsilon,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True)
        pfunc = np.polynomial.Polynomial(pcoefs)

        x = np.linspace(-1, 1)
        self.assertTrue(
            np.max(np.abs(pfunc(x) - scale * np.cos(tau * x))) < epsilon)

    def test_poly_sine(self):
        tau = 10
        epsilon = 0.1
        pg = pyqsp.poly.PolySineTX()
        pcoefs, scale = pg.generate(
            tau,
            epsilon,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True)
        pfunc = np.polynomial.Polynomial(pcoefs)

        x = np.linspace(-1, 1)
        self.assertTrue(
            np.max(np.abs(pfunc(x) - scale * np.sin(tau * x))) < epsilon)
