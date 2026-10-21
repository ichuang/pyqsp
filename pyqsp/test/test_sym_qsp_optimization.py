import os
import unittest

import numpy as np
import pytest
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.special

import pyqsp
from pyqsp import angle_sequence, sym_qsp_opt, poly

# from .. import sym_qsp_opt
# from .. import poly

# -----------------------------------------------------------------------------
"""
    A series of unit tests for optimization subroutines for symmetric QSP phases, based on the algorithms presented in "Efficient phase-factor evaluation in quantum signal processing" of Dong, Meng, Whaley, and Lin (https://arxiv.org/abs/2002.11649), and "Robust iterative method for symmetric quantum signal processing in all parameter regimes" by Dong, Lin, Ni, Wang (https://arxiv.org/abs/2307.12468).

    Some of the companion modules invoked in these tests reimplement basic functionality already provided by the pyqsp package (e.g., in response.py). However, given the important parity requirements required by [DMWL20], as well as slightly different numerical requirements, we try to limit dependencies between these modules as much as is reasonable.

    Run with: pytest pyqsp/test/test_sym_qsp_optimization.py
    Or alt: python -m unittest pyqsp/test/test_sym_qsp_optimization.py
"""

class Test_sym_qsp_optimization(unittest.TestCase):

    def _test_unitary_form(self):
        pass

    def test_gen_sym_qsp_obj(self):
        # Generate a symmetric QSP object with default arguments.
        sym_qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol()
        assert len(sym_qsp_protocol.reduced_phases) == 0
        assert sym_qsp_protocol.parity == None
        assert sym_qsp_protocol.full_phases == None
        assert sym_qsp_protocol.poly_deg == None

    def test_trivial_protocol(self):
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[0,0],parity=0)
        samples = [0, 0.5, 1]
        qsp_unitary = qsp_protocol.gen_unitary(samples)

        # Check that the second Chebyshev polynomial is evaluated.
        assert len(qsp_protocol.full_phases) == 3
        (s0, s1, s2) = (qsp_unitary[0], qsp_unitary[1], qsp_unitary[2])
        assert (s0[0,0] - (2*samples[0]**2 - 1)) <= 10**(-3)
        assert (s1[0,0] - (2*samples[1]**2 - 1)) <= 10**(-3)
        assert (s2[0,0] - (2*samples[2]**2 - 1)) <= 10**(-3)

        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[0,0],parity=1)
        samples = [0, 0.5, 1]
        qsp_unitary = qsp_protocol.gen_unitary(samples)

        # Check that the third Chebyshev polynomial is evaluated.
        assert len(qsp_protocol.full_phases) == 4
        (s0, s1, s2) = (qsp_unitary[0], qsp_unitary[1], qsp_unitary[2])
        assert (s0[0,0] - (4*samples[0]**3 - 3*samples[0])) <= 10**(-3)
        assert (s1[0,0] - (4*samples[1]**3 - 3*samples[1])) <= 10**(-3)
        assert (s2[0,0] - (4*samples[2]**3 - 3*samples[2])) <= 10**(-3)

    def test_protocol_response(self):
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[0,0],parity=0)
        samples = [0, 0.5, 1]
        qsp_response = qsp_protocol.gen_response_re(samples)
        # Check that the second Chebyshev polynomial is evaluated.
        (s0, s1, s2) = (qsp_response[0], qsp_response[1], qsp_response[2])
        assert (s0 - (2*samples[0]**2 - 1)) <= 10**(-3)
        assert (s1 - (2*samples[1]**2 - 1)) <= 10**(-3)
        assert (s2 - (2*samples[2]**2 - 1)) <= 10**(-3)

        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[0,np.pi/8],parity=0)
        samples = [0, 0.5, 1]
        qsp_response_re = qsp_protocol.gen_response_re(samples)
        qsp_response_im = qsp_protocol.gen_response_im(samples)
        # Check that that proper real part is evaluated for modified phases.
        (s0, s1, s2) = (qsp_response_re[0], qsp_response_re[1], qsp_response_re[2])
        (t0, t1, t2) = (qsp_response_im[0], qsp_response_im[1], qsp_response_im[2])
        assert (s0 - (2*samples[0]**2 - 1)/np.sqrt(2)) <= 10**(-3)
        assert (s1 - (2*samples[1]**2 - 1)/np.sqrt(2)) <= 10**(-3)
        assert (s2 - (2*samples[2]**2 - 1)/np.sqrt(2)) <= 10**(-3)
        assert (t0 - 1/np.sqrt(2)) <= 10**(-3)
        assert (t1 - 1/np.sqrt(2)) <= 10**(-3)
        assert (t2 - 1/np.sqrt(2)) <= 10**(-3)

    def test_poly_jacobian_components_trivial(self):
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[0,0,0], parity=0)

        argument = 0
        result = qsp_protocol.gen_poly_jacobian_components(argument)

        assert result.shape == (1, 4)
        assert result[0,0] == 2
        assert result[0,1] == -2
        assert result[0,2] == 2
        assert result[0,3] == 0

        # Same protocol at different evaluation point; note parity is being treated as even for these cases.
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[0,0,0], parity=0)

        argument = np.pi/4
        result = qsp_protocol.gen_poly_jacobian_components(argument)

        assert result.shape == (1, 4)
        assert np.abs((result[0,0] - 2)) < 1e-3
        assert np.abs((result[0,1] - 0.4674)) < 1e-3
        assert np.abs((result[0,2] - -1.7815)) < 1e-3
        assert np.abs((result[0,3] - 0)) < 1e-3

        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[np.pi/4,np.pi/4,np.pi/4], parity=0)

        argument = np.pi/4
        result = qsp_protocol.gen_poly_jacobian_components(argument)

        assert result.shape == (1, 4)
        assert np.abs((result[0,0] - 1.8908)) < 1e-3
        assert np.abs((result[0,1] - 0)) < 1e-3
        assert np.abs((result[0,2] - 0)) < 1e-3
        assert np.abs((result[0,3] - -0.2337)) < 1e-3

    def test_jacobian_full_trivial(self):
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(reduced_phases=[0,0,0], parity=0)

        f, df = qsp_protocol.gen_jacobian()

        assert f.shape == (3,)
        assert df.shape == (3, 3)

        assert np.abs((f[0] - 0)) < 1e-3
        assert np.abs((f[1] - 0)) < 1e-3
        assert np.abs((f[2] - 0)) < 1e-3

        assert np.abs((df[0,0] - 2)) < 1e-3
        assert np.abs((df[1,1] - 2)) < 1e-3
        assert np.abs((df[2,2] - 2)) < 1e-3

    def test_qsp_newton_solver(self):
        coef = np.array([0.2,0.1,0.3])
        parity = 0

        (phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_solver(coef, parity)

        print("phases: %s\nerror: %s\niter: %s\n"%(str(phases), str(err), str(total_iter)))

        assert (np.abs(err) < 1e-3)

    def test_cosine_newton_solver(self):
        freq = 16
        eps = 1e-12
        # Generate approximation to cosine.
        pg = poly.PolyCosineTX()
        pcoefs = pg.generate(tau=freq, epsilon=eps, chebyshev_basis=True)
        # Specify parity and remove trivial coefficients.
        parity = 0
        coef = pcoefs[parity::2]
        # Anonymous function (approx to cosine) using pcoefs.
        approx_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)
        # Anonymous ideal function (true cosine).
        true_fun = lambda x: 0.5*np.cos(freq*x)

        # Optimize for the desired function using Newton solver.
        crit=1e-12
        (phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_solver(coef, parity, crit=crit)

        # Generate samples over which to analyze approximation.
        num_samples = 200
        samples = np.linspace(-1,1,num=num_samples)

        # Map the desired function and achieved function over samples.
        qsp_im_vals = np.array(qsp_seq_opt.gen_response_im(samples))
        approx_vals = np.array(list(map(approx_fun, samples)))
        true_vals = np.array(list(map(true_fun, samples)))

        # Compute 1-norm of diff between approx and true, and qsp and approx.
        approx_to_true_err = np.linalg.norm(np.abs(true_vals - approx_vals),ord=1)
        qsp_to_approx_err = np.linalg.norm(np.abs(approx_vals - qsp_im_vals),ord=1)

        assert (approx_to_true_err < 1e-10)
        assert (qsp_to_approx_err < 1e-10)

    def test_cheb_poly_approximation(self):
        # Generate second Chebyshev polynomial and fit.
        degree = 4
        test_fun = lambda x: 2*(x**2) - 1

        # Use taylor series with cheb basis, over 20 equispaced samples, and with no subnormalization.
        poly_obj = poly.PolyTaylorSeries()
        cheb_approx = poly_obj.taylor_series(func=test_fun, degree=degree, npts=20, chebyshev_basis=True, cheb_samples=20, max_scale=1.0)

        smp_pts = 100
        samples = np.linspace(-1,1,smp_pts)

        # Map chebyshev function over samples
        evaluation = cheb_approx(samples)
        # Expecting second chebyshev polynomial
        exp_value = np.array(list(map(test_fun, samples)))

        # Take the one norm of the difference and assure small.
        diff = np.linalg.norm(np.abs(exp_value - evaluation), ord=1)
        assert (diff < 10-6)

    def test_cheb_poly_sine_approximation(self):
        # Generate second Chebyshev polynomial and fit.
        degree = 20
        test_fun = lambda x: np.cos(3*x)

        # Use taylor series with cheb basis, over 100 equispaced samples, and with no subnormalization.
        poly_obj = poly.PolyTaylorSeries()
        cheb_approx = poly_obj.taylor_series(func=test_fun, degree=degree, npts=20, chebyshev_basis=True, cheb_samples=100, max_scale=1.0)

        smp_pts = 100
        samples = np.linspace(-1,1,smp_pts)

        # Map chebyshev function over samples
        evaluation = cheb_approx(samples)
        # Expecting second chebyshev polynomial
        exp_value = np.array(list(map(test_fun, samples)))

        # Take the one norm of the difference and assure small.
        diff = np.linalg.norm(np.abs(exp_value - evaluation), ord=1)
        assert (diff < 10-6)

    def test_sym_qsp_sign_norm_violation(self):
        # Test inserted to investigate problems when finding the extrema of polynomial approximations to the sign function over the interval [-1,1] as determined by the following command-line call
        # pyqsp --plot --func "np.sign(x)" --polydeg 151 sym_qsp_func
        #
        # Running this test individually and verbosely can be done with:
        # pytest pyqsp/test/test_sym_qsp_optimization.py -s -k 'test_sym_qsp_sign_norm_violation'
        #
        # As it stands, this optimization reveals that the naïve, scipy-based methods for extrema calculation over the interval fail for even modestly high-degree approximations to the sign function. This causes a failure of both the `sym_qsp` iterative and `laurent` completion methods if fed to them, and (incorrectly) does not throw errors.
        degree = 151
        test_fun = lambda x: np.sign(x)

        # Instantiate Taylor series object and optimize toward test function.
        poly_obj = poly.PolyTaylorSeries()
        cheb_poly, scale = poly_obj.taylor_series(func=test_fun, degree=degree, chebyshev_basis=True, return_scale=True)

        # Compute extremal properties of rescaled test function; find minimum of function and its negation to extremize absolute value over interval.
        res_1 = scipy.optimize.minimize(-1*cheb_poly, (0.1,), bounds=[(-1, 1)])
        res_2 = scipy.optimize.minimize(cheb_poly, (0.1,), bounds=[(-1, 1)])
        pmax_1 = res_1.x[0]
        pmax_2 = res_2.x[0]
        max_abs_1 = np.abs(cheb_poly(pmax_1))
        max_abs_2 = np.abs(cheb_poly(pmax_2))

        # Note that the code below seems to suggest that the polynomial approximation `cheb_poly` has had its maximum rescaled to 0.9, as the value of `max_abs_1` suggests. Plainly, however, we see that plotting the resulting function shows that the rescaled polynomial exceeds norm 1 near the origin, indicating failure of the original optimization.
        #
        # In fact, we see that the initial location for optimization (0.1) is directly next to the local minimum at 0.103 which has been rescaled to the default `scale=0.9` parameter supplied to the Taylor series function.

        # Print the locations and evaluations of the rescaled function's extrema.
        print("Optimization results:")
        print(f"Max arg 1: {pmax_1}")
        print(f"Max arg 2: {pmax_2}")
        print(f"Max val 1: {max_abs_1}")
        print(f"Max val 2: {max_abs_2}")

        # Plot the approximating polynomial, rescaled target function, and computed extrema to show clear improper optimization.
        npts = 500
        adat = np.linspace(-1, 1, npts)
        pdat_0 = cheb_poly(adat)
        pdat_1 = np.array(list(map(lambda x: max_abs_1, adat)))
        pdat_2 = np.array(list(map(lambda x: max_abs_2, adat)))
        pdat_3 = np.array(list(map(lambda x: 0.9*test_fun(x), adat)))

        plt.plot(adat, pdat_3, 'k--', alpha=0.4)
        plt.plot(adat, pdat_0)
        plt.plot(adat, pdat_1)
        plt.plot(adat, pdat_2)
        plt.plot(adat, -1*pdat_1)
        plt.plot(adat, -1*pdat_2)

        plt.ylim([-1.1, 1.1])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.show() # Comment out if desire to run all tests quietly.

        # Changing the initial point (sensitive) shows norm violation.
        ext_0 = scipy.optimize.minimize(-1*cheb_poly, (0.02,), bounds=[(0, 1)])
        ext_1 = scipy.optimize.minimize(cheb_poly, (0.02,), bounds=[(0, 1)])
        arg_0 = ext_0.x[0]
        arg_1 = ext_1.x[0]
        max_val_0 = np.abs(cheb_poly(arg_0))
        max_val_1 = np.abs(cheb_poly(arg_1))

        print(f"Alt max arg 0: {arg_0}")
        print(f"Alt max arg 1: {arg_1}")
        print(f"Alt max val 0: {max_val_0}")
        print(f"Alt max val 1: {max_val_1}")


@pytest.mark.parametrize("parity, budget", [(0, 1), (1, 1), (1, .5)])
def test_one_newton_update_reports_returned_iterate_residual(parity, budget):
    # Positive fractional budgets retain their historical rounding to one step.
    result = sym_qsp_opt.newton_solver(np.array([.2]), parity, maxiter=budget)
    assert isinstance(result, tuple) and len(result) == 4
    phases, residual, iterations, protocol = result
    # For one reduced phase, the only coefficient is sin(2*phi),
    # both for the constant target and for the coefficient of T_1.
    expected_phase = .1 - (np.sin(.2) - .2)/(2*np.cos(.2))
    expected_residual = abs(np.sin(2*expected_phase) - .2)
    np.testing.assert_allclose(phases, [expected_phase], atol=1e-16, rtol=0)
    assert residual == pytest.approx(expected_residual, abs=1e-16)
    assert iterations == 1
    assert protocol.solver_status == "maxiter"
    np.testing.assert_array_equal(phases, protocol.reduced_phases)
    # Previously this returned the pre-update residual, about 1.33e-3.
    assert 1e-7 < residual < 2e-7


@pytest.mark.parametrize("budget", [0, 0.0])
def test_zero_iteration_budget_preserves_initial_iterate(budget):
    phases, residual, iterations, protocol = sym_qsp_opt.newton_solver(
        np.array([.2]), 1, maxiter=budget)
    np.testing.assert_array_equal(phases, [.1])
    assert residual == pytest.approx(abs(np.sin(.2) - .2), abs=1e-16)
    assert iterations == 0
    assert protocol.solver_status == "maxiter"


def test_initial_convergence_does_not_solve_or_update(monkeypatch):
    def unexpected_solve(*args):
        pytest.fail("An iterate already within crit must not take a Newton step")

    monkeypatch.setattr(np.linalg, "solve", unexpected_solve)
    phases, residual, iterations, protocol = sym_qsp_opt.newton_solver(
        np.array([.2]), 1, crit=.01)
    np.testing.assert_array_equal(phases, [.1])
    assert residual == pytest.approx(abs(np.sin(.2) - .2), abs=1e-16)
    assert iterations == 0
    assert protocol.solver_status == "converged"


def test_zero_criterion_runs_to_budget_without_claiming_convergence():
    phases, residual, iterations, protocol = sym_qsp_opt.newton_solver(
        np.array([0.]), 1, crit=0, maxiter=1)
    np.testing.assert_array_equal(phases, [0.])
    assert residual == 0
    assert iterations == 1
    assert protocol.solver_status == "maxiter"


def test_low_level_exhaustion_retains_unsuccessful_iterate():
    phases, residual, iterations, protocol = sym_qsp_opt.newton_solver(
        np.array([1.1]), 1, maxiter=3)
    assert iterations == 3
    assert protocol.solver_status == "maxiter"
    assert np.isfinite(phases).all()
    assert residual == pytest.approx(abs(np.sin(2*phases[0]) - 1.1), abs=1e-14)
    assert residual >= .1


@pytest.mark.parametrize("coefficients", [[], [[.2]], [np.nan], [np.inf], [.2j]])
def test_low_level_rejects_invalid_coefficients(coefficients):
    with pytest.raises(ValueError):
        sym_qsp_opt.newton_solver(np.array(coefficients), 1)


@pytest.mark.parametrize("kwargs", [
    {"crit": -1}, {"crit": np.nan}, {"crit": np.inf},
    {"maxiter": -1}, {"maxiter": np.inf},
])
def test_low_level_rejects_invalid_stopping_parameters(kwargs):
    with pytest.raises(ValueError):
        sym_qsp_opt.newton_solver(np.array([.2]), 1, **kwargs)


@pytest.mark.parametrize("parity", [-1, 2, None])
def test_low_level_rejects_invalid_parity(parity):
    with pytest.raises(ValueError):
        sym_qsp_opt.newton_solver(np.array([.2]), parity)


@pytest.mark.parametrize("failure, expected_status", [
    ("singular", "singular_jacobian"), ("nonfinite", "nonfinite"),
])
def test_failed_newton_step_retains_last_evaluated_iterate(monkeypatch, failure, expected_status):
    def failed_solve(matrix, residual):
        if failure == "singular":
            raise np.linalg.LinAlgError("singular test system")
        return np.full_like(residual, np.inf)

    monkeypatch.setattr(np.linalg, "solve", failed_solve)
    phases, residual, iterations, protocol = sym_qsp_opt.newton_solver(
        np.array([.2]), 1)
    np.testing.assert_array_equal(phases, [.1])
    assert residual == pytest.approx(abs(np.sin(.2) - .2), abs=1e-16)
    assert iterations == 0
    assert protocol.solver_status == expected_status


@pytest.mark.parametrize("bad_residual", [False, True])
def test_nonfinite_evaluation_returns_diagnostic_iterate(monkeypatch, bad_residual):
    def evaluation(protocol):
        return (np.array([np.nan if bad_residual else np.sin(.2)]),
                np.array([[np.nan]]))

    monkeypatch.setattr(sym_qsp_opt.SymmetricQSPProtocol, "gen_jacobian", evaluation)
    phases, residual, iterations, protocol = sym_qsp_opt.newton_solver(
        np.array([.2]), 1)
    np.testing.assert_array_equal(phases, [.1])
    assert np.isnan(residual) if bad_residual else residual == pytest.approx(abs(np.sin(.2)-.2))
    assert iterations == 0
    assert protocol.solver_status == "nonfinite"


def imaginary_response(phases, points):
    """Independent scalar Wx/Z-rotation products for the promised readout."""
    values = []
    for x in points:
        signal = np.array([[x, 1j*np.sqrt(1-x*x)],
                           [1j*np.sqrt(1-x*x), x]])
        unitary = np.diag(np.exp(1j*np.array([phases[0], -phases[0]])))
        for phase in phases[1:]:
            rotation = np.diag(np.exp(1j*np.array([phase, -phase])))
            unitary = unitary @ signal @ rotation
        values.append(unitary[0, 0].imag)
    return np.array(values)


@pytest.mark.parametrize("coefficients, parity, target", [
    ([-.4], 0, lambda x: -.4*np.ones_like(x)),
    ([0, -.6], 1, lambda x: -.6*x),
    ([.2, 0, -.1, 0, .15], 0,
     lambda x: .2 - .1*(2*x*x - 1) + .15*(8*x**4 - 8*x*x + 1)),
    ([0, -.2, 0, .15, 0, -.1], 1,
     lambda x: -.2*x + .15*(4*x**3 - 3*x)
     - .1*(16*x**5 - 20*x**3 + 5*x)),
])
def test_successful_public_symmetric_response(coefficients, parity, target):
    result = angle_sequence.QuantumSignalProcessingPhases(
        coefficients, method="sym_qsp", chebyshev_basis=True,
        signal_operator="Wz", measurement="x", eps=.3, suc=.2,
        tolerance=.01, crit=1., maxiter=0)
    # These legacy options remain ignored by symmetric dispatch. Its fixed
    # accuracy criterion and Wx imaginary readout still apply.
    assert isinstance(result, tuple) and len(result) == 3
    full, reduced, actual_parity = result
    assert isinstance(full, np.ndarray) and isinstance(reduced, np.ndarray)
    assert actual_parity == parity
    assert len(full) == len(coefficients)
    assert np.isfinite(full).all() and np.isfinite(reduced).all()
    np.testing.assert_array_equal(full, full[::-1])
    points = np.array([-1., -.91, -.37, 0., .23, .77, 1.])
    np.testing.assert_allclose(imaginary_response(full, points), target(points),
                               atol=1e-12, rtol=0)


def test_public_symmetric_rejects_unattainable_target():
    with pytest.raises(angle_sequence.AngleFindingError) as error:
        angle_sequence.QuantumSignalProcessingPhases(
            [0, 1.1], method="sym_qsp", chebyshev_basis=True)
    message = str(error.value).lower()
    assert "residual" in message and "iteration" in message
    assert "100" in message


@pytest.mark.parametrize("residual, status", [
    (0., "singular_jacobian"),
    (1e-6, "converged"),
    (np.nan, "converged"),
])
def test_public_checks_status_and_residual(monkeypatch, residual, status):
    protocol = sym_qsp_opt.SymmetricQSPProtocol([np.arcsin(.2)/2], parity=1)
    protocol.solver_status = status
    monkeypatch.setattr(
        angle_sequence, "newton_solver",
        lambda *args, **kwargs: (protocol.reduced_phases, residual, 7, protocol))
    with pytest.raises(angle_sequence.AngleFindingError) as error:
        angle_sequence.QuantumSignalProcessingPhases(
            [0, .2], method="sym_qsp", chebyshev_basis=True, tolerance=.1)
    message = str(error.value).lower()
    assert "residual" in message and "iteration" in message and "7" in message


@pytest.mark.parametrize("phase_attribute", ["reduced_phases", "full_phases"])
def test_public_checks_phase_finiteness(monkeypatch, phase_attribute):
    protocol = sym_qsp_opt.SymmetricQSPProtocol([np.arcsin(.2)/2], parity=1)
    protocol.solver_status = "converged"
    getattr(protocol, phase_attribute)[0] = np.nan
    monkeypatch.setattr(
        angle_sequence, "newton_solver",
        lambda *args, **kwargs: (protocol.reduced_phases, 0., 7, protocol))
    with pytest.raises(angle_sequence.AngleFindingError):
        angle_sequence.QuantumSignalProcessingPhases(
            [0, .2], method="sym_qsp", chebyshev_basis=True)


@pytest.mark.parametrize("coefficients", [[0, np.nan], [0, np.inf], [0, .2j]])
def test_public_symmetric_rejects_invalid_coefficients(coefficients):
    with pytest.raises(ValueError):
        angle_sequence.QuantumSignalProcessingPhases(
            coefficients, method="sym_qsp", chebyshev_basis=True)


def test_real_coefficients_in_complex_array_are_accepted():
    full, reduced, parity = angle_sequence.QuantumSignalProcessingPhases(
        np.array([0, -.2], dtype=complex), method="sym_qsp",
        chebyshev_basis=True)
    assert parity == 1
    assert np.isrealobj(full) and np.isrealobj(reduced)
    points = np.array([-1., -.3, 0., .8, 1.])
    np.testing.assert_allclose(imaginary_response(full, points), -.2*points,
                               atol=1e-12, rtol=0)


@pytest.mark.parametrize("failure", ["singular", "nonfinite"])
def test_public_symmetric_rejects_failed_newton_step(monkeypatch, failure):
    def failed_solve(matrix, residual):
        if failure == "singular":
            raise np.linalg.LinAlgError("singular test system")
        return np.full_like(residual, np.inf)

    monkeypatch.setattr(np.linalg, "solve", failed_solve)
    with pytest.raises(angle_sequence.AngleFindingError) as caught:
        angle_sequence.QuantumSignalProcessingPhases(
            [0, .2], method="sym_qsp", chebyshev_basis=True)
    message = str(caught.value)
    assert failure in message
    assert "residual=" in message and "iterations=0" in message
