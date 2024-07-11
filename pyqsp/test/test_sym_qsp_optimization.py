import os
import unittest

import numpy as np
from .. import sym_qsp_opt

# import pyqsp
# from pyqsp import sym_qsp_opt
# import pyqsp

# -----------------------------------------------------------------------------
"""
	A series of unit tests for optimization subroutines for symmetric QSP phases, based on the algorithms presented in "Efficient phase-factor evaluation in quantum signal processing" of Dong, Meng, Whaley, and Lin (https://arxiv.org/abs/2002.11649).

	Some of the companion modules invoked in these tests reimplement basic functionality already provided by the pyqsp package (e.g., in response.py). However, given the important parity requirements required by [DMWL20], as well as slightly different numerical requirements, we try to limit dependencies between these modules as much as is reasonable. 
"""

class Test_sym_qsp_optimization(unittest.TestCase):

	def test_unitary_form(self):
		pass

	def test_gen_sym_qsp_obj(self):
		# Generate a symmetric QSP object with default arguments.
		sym_qsp_protocol = sym_qsp_opt.SymmetricQSPProtocol()
		assert sym_qsp_protocol.poly_deg == 0
		assert len(sym_qsp_protocol.reduced_phases) == 0
		assert len(sym_qsp_protocol.full_phases) == 0