import os
import numpy as np
from pyqsp import poly
from pyqsp import LPoly
from pyqsp import response
from pyqsp import angle_sequence

#-----------------------------------------------------------------------------
# unit tests

import unittest

class Test_poly(unittest.TestCase):

    def test_oneoverx0(self):
        gp = poly.PolyOneOverX(8, 0.01, return_coef=False, ensure_bounded=False)
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
        gpoly = poly.PolyOneOverX(kappa, epsilon, return_coef=False, ensure_bounded=False)
        xpos = np.linspace(1/kappa, 1)
        xval = np.concatenate([-xpos, xpos])
        expected = 1/xval
        polyval = gpoly(xval)
        diff = abs(polyval - expected).mean()
        print(f"diff={diff}")
        assert diff < 0.1

    def test_poly_one_over_x_response1(self):
        pcoefs = poly.PolyOneOverX(3, 0.3, return_coef=True, ensure_bounded=True)
        phiset = angle_sequence.QuantumSignalProcessingWxPhases(pcoefs)
        print(f"QSP angles = {phiset}")
        response.PlotQSPResponse(phiset, model="Wx", pcoefs=pcoefs, show=False)
        assert True
        
