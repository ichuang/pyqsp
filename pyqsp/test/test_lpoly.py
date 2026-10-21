import numpy as np
import pytest

from pyqsp import LPoly


@pytest.mark.parametrize("phases, coefficients, dmin", [
    ([0], [1], 0),
    ([0, 0], [0, 1], -1),
    ([0, 0, 0], [0, 0, 1], -2),
    ([-np.pi/4, 0, np.pi/4], [.5, 0, .5], -2),
    ([-np.pi/4, 0, 0, 0, np.pi/4], [.5, 0, 0, 0, .5], -4),
], ids=["angles1", "angles2", "angles3", "angles4", "angles5"])
def test_simple_unitary_from_angles(phases, coefficients, dmin):
    unitary = LPoly.LAlg.unitary_from_angles(phases)
    assert unitary.IPoly == LPoly.LPoly(coefficients, dmin)


def test_lpoly1():
    # Exp[-i (pi/2) X] * w = w^(-1) * iX.
    rotation = LPoly.LAlg.rotation(np.pi/2)
    rotation.IPoly.round_zeros()
    product = rotation * LPoly.w
    assert product.XPoly == LPoly.LPoly([1], -1)


def test_LPoly_mul1():
    polynomial = LPoly.LPoly([1/2, 1/2], -1)
    squared = polynomial * polynomial
    assert squared.dmin == -2
    assert abs(squared.coefs - np.array([.25, .5, .25])).sum() < 1.0e-4


@pytest.mark.parametrize("coefficients, expected, dmin, tolerance", [
    ([-1, 0, 2], [.5, 0, .5], -2, 1.0e-4),
    ([0, 1], [1/2, 1/2], -1, 1.0e-8),
    ([0, -3, 0, 4], [.5, 0, 0, .5], -3, 1.0e-8),
], ids=["p2lf3", "p2lf1", "p2lf2"])
def test_polynomial_to_laurent_form(coefficients, expected, dmin, tolerance):
    # Adjacent LPoly coefficients represent exponents two apart.
    polynomial = LPoly.PolynomialToLaurentForm(coefficients)
    assert polynomial.dmin == dmin
    assert abs(polynomial.coefs - np.array(expected)).sum() < tolerance
