"""Traditional coefficient, positional and signed-amplitude contracts."""

from types import SimpleNamespace

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import pytest

from pyqsp import angle_sequence, response
from pyqsp.LPoly import LAlg, LPoly
from pyqsp.test import qsp_amplitude


POINTS = np.array([-1., -.83, -.4, 0., .27, .61, .94, 1.])
COMPLEX_QUADRATIC = (4 + 3j - (1 - 2j)*np.sqrt(3))/2
COMPLEX_QUARTIC = (1 - 1j)*(-1j + np.sqrt(3))


@pytest.mark.parametrize("signal, measurement", [("Wx", "x"), ("Wz", "z")])
@pytest.mark.parametrize("polynomial", [[.1, 0, .2], np.array([.1, 0, .2]),
                                        Chebyshev([.1, 0, .2])])
def test_legacy_basis_and_positional_arguments(polynomial, signal, measurement):
    phases = angle_sequence.QuantumSignalProcessingPhases(
        polynomial, .02, .8, signal, measurement, 1e-8, "laurent", False)
    assert type(phases) is list and len(phases) == 3
    assert all(np.isscalar(phase) and np.isreal(phase) for phase in phases)
    # False gates the method; the input still represents .4*x**2 - .1.
    expected = .8*(.42*POINTS**2 - .11)
    np.testing.assert_allclose(qsp_amplitude(phases, POINTS, signal), expected,
                               atol=1e-10, rtol=0)


@pytest.mark.parametrize("coefficients, expected", [
    ([0, -.5 + 1j, 0, .5], lambda x: 2*x**3 + (-2 + 1j)*x),
    ([-1 + COMPLEX_QUADRATIC/2 + 3*COMPLEX_QUARTIC/8, 0,
      (COMPLEX_QUADRATIC + COMPLEX_QUARTIC)/2, 0, COMPLEX_QUARTIC/8],
     lambda x: -1 + COMPLEX_QUADRATIC*x*x + COMPLEX_QUARTIC*x**4),
])
def test_specialized_complex_wx_z_amplitude(coefficients, expected):
    phases = angle_sequence.QuantumSignalProcessingPhases(
        coefficients, eps=.02, suc=.8, signal_operator="Wx", measurement="z",
        method="laurent")
    # Specialized P completion does not perturb or scale these admissible targets.
    np.testing.assert_allclose(qsp_amplitude(phases, POINTS, measurement="z"),
                               expected(POINTS), atol=1e-8, rtol=0)


def test_legacy_stride_two_nonreciprocal_response():
    coefficients, eps, suc = [.1, .05, -.2], .01, .7
    phases = angle_sequence.angle_sequence(coefficients, eps, suc)
    assert type(phases) is list and len(phases) == 3
    w = POINTS + 1j*np.sqrt(1 - POINTS**2)
    target = .1/w**2 + .05 - .2*w**2
    expected = suc*(target + eps/4*(w**2 + w**-2))
    actual = qsp_amplitude(phases, POINTS, "Wz")
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    assert np.max(np.abs(actual/suc - target)) < eps


@pytest.mark.parametrize("residual, succeeds", [(.0625, True), (.125, False), (.25, False)])
def test_lower_acceptance_retains_strict_unscaled_error(monkeypatch, residual, succeeds):
    # Binary-exact values distinguish strict inequality at the boundary.
    target, suc, eps = LPoly([.25, 0, .125], -2), .5, .125
    reconstructed = suc*(target + LPoly([residual]))
    monkeypatch.setattr(LAlg, "unitary_from_angles",
                        lambda phases: SimpleNamespace(IPoly=reconstructed))
    if succeeds:
        assert type(angle_sequence.angle_sequence(target.coefs, eps, suc)) is list
    else:
        with pytest.raises(ValueError, match="angle finding program failed"):
            angle_sequence.angle_sequence(target.coefs, eps, suc)


def test_legacy_laurent_reversal_does_not_conjugate():
    polynomial = LPoly([1 + 2j, 3 - 4j], -1)
    assert polynomial[-1] == 1 + 2j
    assert polynomial[1] == 3 - 4j
    np.testing.assert_array_equal((~polynomial).coefs, [3 - 4j, 1 + 2j])


@pytest.mark.parametrize("signal, measurement", [("Wx", "x"), ("Wz", "z"), ("Wx", "z")])
def test_response_dictionary_and_default_measurement(signal, measurement):
    phases = [.17, -.31, .29]
    data = response.ComputeQSPResponse(POINTS, phases, signal)
    assert set(data) == {"adat", "pdat", "model", "phiset"}
    assert data["adat"] is POINTS and data["phiset"] is phases
    assert data["model"] == (signal, {"Wx": "x", "Wz": "z"}[signal])
    np.testing.assert_allclose(data["pdat"], qsp_amplitude(phases, POINTS, signal),
                               atol=1e-14, rtol=0)
    explicit = response.ComputeQSPResponse(POINTS, phases, signal, measurement)
    np.testing.assert_allclose(explicit["pdat"],
                               qsp_amplitude(phases, POINTS, signal, measurement),
                               atol=1e-14, rtol=0)
