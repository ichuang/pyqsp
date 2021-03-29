import os
import numpy as np
import pyqsp
from pyqsp import poly
from pyqsp import LPoly
from pyqsp import response
from pyqsp import angle_sequence

#-----------------------------------------------------------------------------
# unit tests

import unittest

class Test_poly(unittest.TestCase):

    def test_oneoverx0(self):
        pg = pyqsp.poly.PolyOneOverX()
        gp = pg.generate(8, 0.01, return_coef=False, ensure_bounded=False)
        # print(f"gp={gp}")
        print(f"gp(0.5) = {gp(0.5)}")
        assert abs(gp(0.5) - 2) < 0.01
        assert abs(gp(0.3) - 1/(0.3)) < 0.01
    
    def test_oneoverx1(self):
        '''
        unit test to ensure that the polynomial approximation to 1/x really is close to 1/x
        '''
        kappa = 3
        epsilon = 0.01
        pg = pyqsp.poly.PolyOneOverX()
        gpoly = pg.generate(kappa, epsilon, return_coef=False, ensure_bounded=False)
        xpos = np.linspace(1/kappa, 1)
        xval = np.concatenate([-xpos, xpos])
        expected = 1/xval
        polyval = gpoly(xval)
        diff = abs(polyval - expected).mean()
        print(f"diff={diff}")
        assert diff < 0.1

    def test_poly_one_over_x_response1(self):
        pg = pyqsp.poly.PolyOneOverX()
        pcoefs = pg.generate(3, 0.3, return_coef=True, ensure_bounded=True)
        phiset = angle_sequence.QuantumSignalProcessingWxPhases(pcoefs)
        print(f"QSP angles = {phiset}")
        response.PlotQSPResponse(phiset, model="Wx", pcoefs=pcoefs, show=False)
        assert True

    def test_poly_sign1(self):
        pg = pyqsp.poly.PolySign()
        pcoefs = pg.generate(17, 10)
        poly = np.polynomial.Polynomial(pcoefs)
        print(f"sign poly at -0.9 = {poly(-0.9)}")
        assert (poly(-0.9) < -0.2)
        assert (poly(0.9) > -0.2)

    def test_poly_thresh1(self):
        pg = pyqsp.poly.PolyThreshold()
        pcoefs = pg.generate(18, 10)
        poly = np.polynomial.Polynomial(pcoefs)
        print(f"sign poly at -0.9 = {poly(-0.9)}")
        print(f"sign poly at 0 = {poly(0)}")
        assert (poly(-0.9) < 0.1)
        assert (poly(0) > 0.3)
        assert (poly(0.9) < 0.1)
        
    def test_poly_gibbs1(self):
        pg = pyqsp.poly.PolyGibbs()
        pcoefs = pg.generate(30, 4.5)
        poly = np.polynomial.Polynomial(pcoefs)
        print(f"gibbs poly at 0.9 = {poly(0.9)}")
        print(f"gibbs poly at 0 = {poly(0)}")
        assert (poly(0.9) < 0.3)
        assert (poly(0) > 0.9)
        
    def test_poly_efilter1(self):
        pg = pyqsp.poly.PolyEigenstateFiltering()
        pcoefs = pg.generate(20, 0.2, 0.9)
        poly = np.polynomial.Polynomial(pcoefs)
        print(f"ef poly at 0.9 = {poly(0.9)}")
        print(f"ef poly at 0 = {poly(0)}")
        assert (poly(0.9) < 0.1)
        assert (poly(0) > 0.7)
        
        
        
        
