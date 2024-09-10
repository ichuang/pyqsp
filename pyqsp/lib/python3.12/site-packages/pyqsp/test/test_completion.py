import unittest

import numpy as np

from pyqsp.completion import (CompletionError, cheb2poly,
                              completion_from_root_finding, poly2cheb)
from pyqsp.LPoly import LPoly, PolynomialToLaurentForm

# -----------------------------------------------------------------------------
# unit tests


class Test_completion(unittest.TestCase):
    def test_completion_f_1(self):
        """
        See if completion_from_root_finding can get the G polynomial,
            given the F polynomial in Laurent form
        Test case: F = w should produce G = 0
        """
        lpcoefs = [0, 0, 1]
        Plp = LPoly(lpcoefs, -len(lpcoefs) + 1)

        eps = 1.0e-4
        suc = 1 - 1.0e-4
        p = Plp

        # Capitalization: eps/2 amount of error budget is put to the highest
        # power for sake of numerical stability.
        p_new = suc * \
            (p + LPoly([eps / 4], p.degree) + LPoly([eps / 4], -p.degree))

        # print(f"Laurent F poly {Plp}")
        Qalg = completion_from_root_finding(p_new.coefs, coef_type="F")
        Qlp = Qalg.XPoly
        # print(f"Laurent G poly {Qlp}")

        ncoefs = (p_new * ~p_new + Qlp * ~Qlp).coefs
        ncoefs_expected = np.zeros(ncoefs.size)
        ncoefs_expected[ncoefs.size // 2] = 1.

        self.assertAlmostEqual(np.max(np.abs(ncoefs - ncoefs_expected)), 0.)

    def test_completion_f_2(self):
        """
        See if completion_from_root_finding can get the G polynomial,
            given the F polynomial in Laurent form
        Test case: F = -1 + 2 a^2 should produce G = i * 2 a sqrt(1-a^2),
            which has Laurent poly (w^2 - w^(-2))/2
        """
        pcoefs = [-1, 0, 2]
        Plp = PolynomialToLaurentForm(pcoefs)

        eps = 1.0e-4
        suc = 1 - 1.0e-4
        p = Plp

        # Capitalization: eps/2 amount of error budget is put to the highest
        # power for sake of numerical stability.
        p_new = suc * \
            (p + LPoly([eps / 4], p.degree) + LPoly([eps / 4], -p.degree))

        # print(f"Laurent F poly {Plp}")
        Qalg = completion_from_root_finding(p_new.coefs, coef_type="F")
        Qlp = Qalg.XPoly
        # print(f"Laurent G poly {Qlp}")

        self.assertEqual(Qlp.dmin, -2)

        ncoefs = (p_new * ~p_new + Qlp * ~Qlp).coefs
        ncoefs_expected = np.zeros(ncoefs.size)
        ncoefs_expected[ncoefs.size // 2] = 1.

        self.assertAlmostEqual(np.max(np.abs(ncoefs - ncoefs_expected)), 0.)

    def test_completion_f_3(self):
        """
        See if completion_from_root_finding can get the G polynomial, given the F polynomial in Laurent form
        Test case: F = a(-3+4a^2) = -3a + 4a^3 = [0, -3, 0, 4] should produce G = (-1 + 4a^2) * sqrt(1-a^2)
        Get:
            Laurent F poly 0.5 * w ^ (-3) + 0.0 * w ^ (-1) + 0.0 * w ^ (1) + 0.5 * w ^ (3)
            Laurent G poly -0.4949998125160129 * w ^ (-3) + 8.24174368309573e-15 * w ^ (-1) + 7.204517437263955e-15 * w ^ (1) + 0.5050001874839872 * w ^ (3)

        cosine matching up with sine -- good
        """
        pcoefs = [0, -3, 0, 4]
        Plp = PolynomialToLaurentForm(pcoefs)

        eps = 1.0e-4
        suc = 1 - 1.0e-4
        p = Plp

        # Capitalization: eps/2 amount of error budget is put to the highest
        # power for sake of numerical stability.
        p_new = suc * \
            (p + LPoly([eps / 4], p.degree) + LPoly([eps / 4], -p.degree))

        # print(f"Laurent F poly {Plp}")
        Qalg = completion_from_root_finding(p_new.coefs, coef_type="F")
        Qlp = Qalg.XPoly
        # print(f"Laurent G poly {Qlp}")

        self.assertEqual(Qlp.dmin, -3)

        ncoefs = (p_new * ~p_new + Qlp * ~Qlp).coefs
        ncoefs_expected = np.zeros(ncoefs.size)
        ncoefs_expected[ncoefs.size // 2] = 1.

        self.assertAlmostEqual(np.max(np.abs(ncoefs - ncoefs_expected)), 0.)

    def test_completion_p_1(self):
        pcoefs = [0., -2 + 1j, 0., 2.]

        alg = completion_from_root_finding(pcoefs, coef_type="P")

        F, G = alg.IPoly, alg.XPoly

        ncoefs = (F * ~F + G * ~G).coefs
        ncoefs_expected = np.zeros(ncoefs.size)
        ncoefs_expected[ncoefs.size // 2] = 1.

        self.assertAlmostEqual(np.max(np.abs(ncoefs - ncoefs_expected)), 0.)

    def test_completion_p_2(self):
        pcoefs = [-1., 0., 50., 0., -400., 0., 1120., 0., -1280., 0., 512.]

        alg = completion_from_root_finding(pcoefs, coef_type="P")

        F, G = alg.IPoly, alg.XPoly

        ncoefs = (F * ~F + G * ~G).coefs
        ncoefs_expected = np.zeros(ncoefs.size)
        ncoefs_expected[ncoefs.size // 2] = 1.

        self.assertAlmostEqual(np.max(np.abs(ncoefs - ncoefs_expected)), 0.)

    def test_completion_p_3(self):
        pcoefs = [-1., 0., (1 / 2) * (4 + 3j - (1 - 2j) * np.sqrt(3)),
                  0., (1 - 1j) * (-1j + np.sqrt(3))]

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

    def test_cheb2poly_1(self):
        pcoefs = np.array([0., 0., 1., 2., 2., 0.])
        expected = np.array([1., -6., -14., 8., 16., 0.])
        result = cheb2poly(pcoefs, kind='T')
        self.assertAlmostEqual(np.max(np.abs(expected - result)), 0.)

    def test_cheby2poly_2(self):
        pcoefs = np.array([0., 0., 1., 2., 2., 0.])
        expected = np.array([1., -8., -20., 16., 32., 0.])
        result = cheb2poly(pcoefs, kind='U')
        self.assertAlmostEqual(np.max(np.abs(expected - result)), 0.)

    def test_poly2cheb_1(self):
        pcoefs = np.array([1., -6., -14., 8., 16., 0.])
        expected = np.array([0., 0., 1., 2., 2., 0.])
        result = poly2cheb(pcoefs, kind='T')
        self.assertAlmostEqual(np.max(np.abs(expected - result)), 0.)

    def test_poly2cheb_2(self):
        pcoefs = np.array([1., -8., -20., 16., 32., 0.])
        expected = np.array([0., 0., 1., 2., 2., 0.])
        result = poly2cheb(pcoefs, kind='U')
        self.assertAlmostEqual(np.max(np.abs(expected - result)), 0.)
