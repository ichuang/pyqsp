import os
import unittest

import numpy as np
from .. import sym_qsp_opt

# import pyqsp
# from pyqsp import sym_qsp_opt
# import pyqsp

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

		(phases, err, total_iter) = sym_qsp_opt.newton_Solver(coef, parity)
		
		print("phases: %s\nerror: %s\niter: %s\n"%(str(phases), str(err), str(total_iter)))

		assert (np.abs(err) < 1e-3)











