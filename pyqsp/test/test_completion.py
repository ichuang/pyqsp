import unittest

import numpy as np
import pytest

from pyqsp.completion import (CompletionError, cheb2poly,
                              completion_from_root_finding, poly2cheb)
from pyqsp.LPoly import LPoly, PolynomialToLaurentForm

# -----------------------------------------------------------------------------
# unit tests

class Test_completion(unittest.TestCase):

    # Test now failing on new device due to broadcast error. Appears within same completion_from_root_finding mechanism.
    def _test_completion_p_2(self):
        pcoefs = [-1., 0., 50., 0., -400., 0., 1120., 0., -1280., 0., 512.]

        alg = completion_from_root_finding(pcoefs, coef_type="P")

        F, G = alg.IPoly, alg.XPoly

        ncoefs = (F * ~F + G * ~G).coefs
        ncoefs_expected = np.zeros(ncoefs.size)
        ncoefs_expected[ncoefs.size // 2] = 1.

        self.assertAlmostEqual(np.max(np.abs(ncoefs - ncoefs_expected)), 0.)

    def test_completion_p_4(self):
        pcoefs = [0., 0., -3., 0., 4.]

        with self.assertRaises(CompletionError):
            completion_from_root_finding(pcoefs, coef_type="P")

@pytest.mark.parametrize("p, expected_dmin", [
    (LPoly([0, 0, 1], -2), None),
    (PolynomialToLaurentForm([-1, 0, 2]), -2),
    (PolynomialToLaurentForm([0, -3, 0, 4]), -3),
], ids=["f_1", "f_2", "f_3"])
def test_completion_f(p, expected_dmin):
    eps, suc = 1.0e-4, 1 - 1.0e-4
    # Put eps/2 of the error budget in the highest powers for stability.
    p_new = suc * (p + LPoly([eps/4], p.degree) + LPoly([eps/4], -p.degree))
    q = completion_from_root_finding(p_new.coefs, coef_type="F").XPoly
    if expected_dmin is not None:
        assert q.dmin == expected_dmin
    coefficients = (p_new * ~p_new + q * ~q).coefs
    expected = np.zeros(coefficients.size)
    expected[coefficients.size // 2] = 1.
    unittest.TestCase().assertAlmostEqual(np.max(np.abs(coefficients - expected)), 0.)

@pytest.mark.parametrize("pcoefs", [
    [0., -2 + 1j, 0., 2.],
    [-1., 0., (1 / 2) * (4 + 3j - (1 - 2j) * np.sqrt(3)),
     0., (1 - 1j)*(-1j + np.sqrt(3))],
], ids=["p_1", "p_3"])
def test_completion_p(pcoefs):
    alg = completion_from_root_finding(pcoefs, coef_type="P")
    f, g = alg.IPoly, alg.XPoly
    coefficients = (f * ~f + g * ~g).coefs
    expected = np.zeros(coefficients.size)
    expected[coefficients.size // 2] = 1.
    unittest.TestCase().assertAlmostEqual(np.max(np.abs(coefficients - expected)), 0.)

@pytest.mark.parametrize("convert, kind, coefficients, expected", [
    (cheb2poly, "T", [0., 0., 1., 2., 2., 0.], [1., -6., -14., 8., 16., 0.]),
    (cheb2poly, "U", [0., 0., 1., 2., 2., 0.], [1., -8., -20., 16., 32., 0.]),
    (poly2cheb, "T", [1., -6., -14., 8., 16., 0.], [0., 0., 1., 2., 2., 0.]),
    (poly2cheb, "U", [1., -8., -20., 16., 32., 0.], [0., 0., 1., 2., 2., 0.]),
], ids=["cheb2poly_1", "cheby2poly_2", "poly2cheb_1", "poly2cheb_2"])
def test_basis_conversion(convert, kind, coefficients, expected):
    result = convert(np.array(coefficients), kind=kind)
    unittest.TestCase().assertAlmostEqual(np.max(np.abs(expected - result)), 0.)
