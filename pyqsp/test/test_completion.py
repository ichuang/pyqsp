import os
import numpy as np
from pyqsp import LPoly
from pyqsp import completion

#-----------------------------------------------------------------------------
# unit tests

import unittest

class Test_completion(unittest.TestCase):
    
    def test_completion1(self):
        '''
        See if completion_from_root_finding can get the Q polynomial, given the P polynomial in Laurent form
        Test case: P = w should produce Q = 0
        '''
        lpcoefs = [0, 0, 1]
        Plp = LPoly.LPoly(lpcoefs, -len(lpcoefs) + 1)
    
        eps = 1.0e-4
        suc = 1 - 1.0e-4
        p = Plp
    
        # Capitalization: eps/2 amount of error budget is put to the highest power for sake of numerical stability.
        p_new = suc * (p + LPoly.LPoly([eps / 4], p.degree) + LPoly.LPoly([eps / 4], -p.degree))
    
        print(f"Laurent P poly {Plp}")
        Qalg = completion.completion_from_root_finding(p_new)
        Qlp = Qalg.XPoly
        print(f"Laurent Q poly {Qlp}")
        assert abs(Qlp.coefs - np.array([0] * 3)).sum() < 0.02
    
    def test_completion2(self):
        '''
        See if completion_from_root_finding can get the Q polynomial, given the P polynomial in Laurent form
        Test case: P = -1 + 2 a^2 should produce Q = i * 2 a sqrt(1-a^2), which has Laurent poly (w^2 - w^(-2))/2
        '''
        pcoefs = [-1, 0, 2]
        Plp = LPoly.PolynomialToLaurentForm(pcoefs)
    
        eps = 1.0e-4
        suc = 1 - 1.0e-4
        p = Plp
    
        # Capitalization: eps/2 amount of error budget is put to the highest power for sake of numerical stability.
        p_new = suc * (p + LPoly.LPoly([eps / 4], p.degree) + LPoly.LPoly([eps / 4], -p.degree))
    
        print(f"Laurent P poly {Plp}")
        Qalg = completion.completion_from_root_finding(p_new)
        Qlp = Qalg.XPoly
        print(f"Laurent Q poly {Qlp}")
        assert Qlp.dmin == -2
        assert abs(Qlp.coefs - np.array([-1,0, 1])/2).sum() < 0.02
    
    def test_completion3(self):
        '''
        See if completion_from_root_finding can get the Q polynomial, given the P polynomial in Laurent form
        Test case: P = a(-3+4a^2) = -3a + 4a^3 = [0, -3, 0, 4] should produce Q = (-1 + 4a^2) * sqrt(1-a^2) 
        Get:
            Laurent P poly 0.5 * w ^ (-3) + 0.0 * w ^ (-1) + 0.0 * w ^ (1) + 0.5 * w ^ (3)
            Laurent Q poly -0.4949998125160129 * w ^ (-3) + 8.24174368309573e-15 * w ^ (-1) + 7.204517437263955e-15 * w ^ (1) + 0.5050001874839872 * w ^ (3)
    
        cosine matching up with sine -- good
        '''
        pcoefs = [0, -3, 0, 4]
        Plp = LPoly.PolynomialToLaurentForm(pcoefs)
    
        eps = 1.0e-4
        suc = 1 - 1.0e-4
        p = Plp
    
        # Capitalization: eps/2 amount of error budget is put to the highest power for sake of numerical stability.
        p_new = suc * (p + LPoly.LPoly([eps / 4], p.degree) + LPoly.LPoly([eps / 4], -p.degree))
    
        print(f"Laurent P poly {Plp}")
        Qalg = completion.completion_from_root_finding(p_new)
        Qlp = Qalg.XPoly
        print(f"Laurent Q poly {Qlp}")
        assert Qlp.dmin == -3
        assert abs(Qlp.coefs - np.array([-1, 0, 0, 1])/2).sum() < 0.02
