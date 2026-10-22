import unittest

import numpy as np
import pytest

from pyqsp import angle_sequence, sym_qsp_opt, poly
from pyqsp.test import qsp_matrix

# -----------------------------------------------------------------------------
"""
    A series of unit tests for optimization subroutines for symmetric QSP phases, based on the algorithms presented in "Efficient phase-factor evaluation in quantum signal processing" of Dong, Meng, Whaley, and Lin (https://arxiv.org/abs/2002.11649), and "Robust iterative method for symmetric quantum signal processing in all parameter regimes" by Dong, Lin, Ni, Wang (https://arxiv.org/abs/2307.12468).

    Some of the companion modules invoked in these tests reimplement basic functionality already provided by the pyqsp package (e.g., in response.py). However, given the important parity requirements required by [DMWL20], as well as slightly different numerical requirements, we try to limit dependencies between these modules as much as is reasonable.

    Run with: pytest pyqsp/test/test_sym_qsp_optimization.py
    Or alt: python -m unittest pyqsp/test/test_sym_qsp_optimization.py
"""

class Test_sym_qsp_optimization(unittest.TestCase):

    def test_gen_sym_qsp_obj(self):
        # Generate a symmetric QSP object with default arguments.
        sym_qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol()
        assert len(sym_qsp_protocol.reduced_phases) == 0
        assert sym_qsp_protocol.parity == None
        assert sym_qsp_protocol.full_phases == None
        assert sym_qsp_protocol.poly_deg == None

    def test_trivial_protocol(self):
        samples = np.array([-1, -0.5, 0, 0.5, 1])
        # W(x)**2 and W(x)**3 have closed-form diagonal/off-diagonal entries.
        root = np.sqrt(1 - samples**2)
        for parity, diagonal, off_diagonal in (
                (0, 2*samples**2 - 1, 2j*samples*root),
                (1, 4*samples**3 - 3*samples, 1j*(4*samples**2 - 1)*root)):
            qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(
                reduced_phases=[0, 0], parity=parity)
            qsp_unitary = qsp_protocol.gen_unitary(samples)
            assert len(qsp_protocol.full_phases) == 3 + parity
            expected = np.empty((len(samples), 2, 2), dtype=complex)
            expected[:, 0, 0] = expected[:, 1, 1] = diagonal
            expected[:, 0, 1] = expected[:, 1, 0] = off_diagonal
            np.testing.assert_allclose(qsp_unitary, expected, rtol=0, atol=1e-12)

    def test_protocol_response(self):
        samples = np.array([-1, -0.5, 0, 0.5, 1])
        second_chebyshev = 2*samples**2 - 1
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(
            reduced_phases=[0, 0], parity=0)
        np.testing.assert_allclose(
            qsp_protocol.gen_response_re(samples), second_chebyshev,
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            qsp_protocol.gen_response_im(samples), 0, rtol=0, atol=1e-12)

        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(
            reduced_phases=[0, np.pi/8], parity=0)
        # Z(pi/8) W(x)**2 Z(pi/8) has U00 = exp(i*pi/4)*T_2(x).
        expected = second_chebyshev / np.sqrt(2)
        np.testing.assert_allclose(
            qsp_protocol.gen_response_re(samples), expected,
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            qsp_protocol.gen_response_im(samples), expected,
            rtol=0, atol=1e-12)

    def test_poly_jacobian_components_trivial(self):
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(
            reduced_phases=[0, 0, 0], parity=0)
        # At zero phases, differentiating each mirrored phase pair gives
        # 2*T_{2k}(x); the central phase is itself doubled.
        for argument in (0, np.pi/4):
            result = qsp_protocol.gen_poly_jacobian_components(argument)
            expected = [[2, 2*(2*argument**2 - 1),
                         2*(8*argument**4 - 8*argument**2 + 1), 0]]
            assert result.shape == (1, 4)
            if argument == 0:
                np.testing.assert_array_equal(result, expected)
            else:
                np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(
            reduced_phases=[np.pi/4, np.pi/4, np.pi/4], parity=0)
        argument = np.pi/4
        result = qsp_protocol.gen_poly_jacobian_components(argument)
        # Directly multiplying Z(pi/4) W Z(pi/4) W Z(pi/2) W
        # Z(pi/4) W Z(pi/4) gives Im(U00) = 1 - 2*x**2.
        # Differentiating its reduced phases gives the first three entries.
        expected = [[8*argument**2*(1 - argument**2), 0, 0,
                     1 - 2*argument**2]]
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_jacobian_full_trivial(self):
        qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol(
            reduced_phases=[0, 0, 0], parity=0)
        f, df = qsp_protocol.gen_jacobian()
        assert f.shape == (3,)
        assert df.shape == (3, 3)
        # The derivatives above are exactly the Chebyshev basis functions,
        # so the coefficient Jacobian is twice the identity, including zeros.
        np.testing.assert_allclose(f, 0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(df, 2*np.eye(3), rtol=0, atol=1e-12)

    def test_qsp_newton_solver(self):
        coef = np.array([0.2,0.1,0.3])
        parity = 0

        (phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_solver(coef, parity)

        print("phases: %s\nerror: %s\niter: %s\n"%(str(phases), str(err), str(total_iter)))

        assert np.isfinite(err) and 0 <= err < 1e-12
        assert np.all(np.isfinite(phases))
        samples = np.linspace(-1, 1, 101)
        expected = (0.2 + 0.1*(2*samples**2 - 1)
                    + 0.3*(8*samples**4 - 8*samples**2 + 1))
        np.testing.assert_allclose(
            qsp_seq_opt.gen_response_im(samples), expected,
            rtol=0, atol=1e-12)

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


    def test_sym_qsp_sign_norm_violation(self):
        degree = 151
        fitted, scale = poly.PolyTaylorSeries().taylor_series(
            np.sign, degree, chebyshev_basis=True, return_scale=True)
        theta = (np.arange(2*degree) + .5)*np.pi/(2*degree)
        modes = np.arange(degree + 1)
        # First-kind-node cosine orthogonality gives the least-squares fit.
        coefficients = 2*np.mean(
            np.sign(np.cos(theta))[:, None]
            * np.cos(theta[:, None]*modes), axis=0)
        coefficients[0] /= 2
        assert np.isfinite(scale) and scale > 0
        # This checks the returned scale, not a certified global bound.
        np.testing.assert_allclose(
            fitted.coef, scale*coefficients, atol=1e-12, rtol=0)


@pytest.mark.parametrize("degree, cheb_samples, target", [
    (4, 20, lambda x: 2*(x**2) - 1),
    (20, 100, lambda x: np.cos(3*x)),
], ids=["chebyshev_T2", "cosine"])
def test_cheb_poly_approximation(degree, cheb_samples, target):
    fitted = poly.PolyTaylorSeries().taylor_series(
        target, degree, npts=20, chebyshev_basis=True,
        cheb_samples=cheb_samples, max_scale=1.0)
    samples = np.linspace(-1, 1, 100)
    expected = np.array(list(map(target, samples)))
    assert np.linalg.norm(np.abs(expected - fitted(samples)), ord=1) < 1e-6


# Newton diagnostics and public symmetric synthesis accuracy contracts.

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
    return np.array([qsp_matrix(phases, x)[0, 0].imag for x in points])


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
