import numpy as np
import scipy.linalg

class SymmetricQSPProtocol:

	def __init__(self, poly_deg=0, reduced_phases=[], full_phases=[]):
		
		"""
		TODO: assert that the degree matches the length of the phases, and that the reduced and full phases agree, and that if one or the other is not instantiated, then full phases are recovered. Also check parity.
		"""

		self.poly_deg = poly_deg
		self.reduced_phases = reduced_phases
		self.full_phases = full_phases
		self.parity = False # Include this to match with DMWL convention.

	def help(self):
		return("Basic class for storing classical description (e.g., length, phases) of symmetric QSP protocol, with auxiliary methods for generating derived quantities (e.g., response function, various gradients).")

	def get_reduced_phases(self):
		pass

	def get_full_phases(self):
		pass

	def signal(self, a):
		return np.array(
			[[a, 1j * np.sqrt(1 - a**2)],
             [1j * np.sqrt(1 - a**2), a]])

	def phase(self, phi):
		return np.array(
            [[np.exp(1j * phi), 0.],
             [0., np.exp(-1j * phi)]])

	def gen_unitary(self, samples):
		pass

	def gen_response(self, samples):
		"""
		Check for name collisions. This method returns the real part of the (0,0) matrix element, which is what sym qsp optimization expects.
		"""
		pass

	def gen_loss(self, samples, target_poly):
		"""
		Compute L(phi) function over samples with respect to target_poly.
		"""
		pass

	def gen_grad(self, samples, index):
		"""
		Return gradient with respect to index phase over samples. The intended use case for this is that the phases of a given protocol can be internally updated based on the computed gradient, resulting in the ultimately desired object.

		TODO: determine if this whole thing is too slow; in principle a new object doesn't need to be instantiated a bunch of times, just one, and every other method just performs a derived calculation. If we can guarantee that only internal methods can update phases, then we can have a set-method that (called rarely) checks whether the required conditions are preserved.
		"""
		pass


"""
	In what remains we include methods for actually performing L-BFGS with respect to some target, maxiter, accuracy, and other parameters.

	If there are methods in original package for computing a bounded polynomial approximation, these can be used to generate a target function, which can then be passed to the gradient computation of the symmetric QSP class.

"""

