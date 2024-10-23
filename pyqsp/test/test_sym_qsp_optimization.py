import os
import unittest

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.special

import pyqsp
from pyqsp import sym_qsp_opt, poly

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
