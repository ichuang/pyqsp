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
        # Parity is (d mod 2), where len(full_phases) = d + 1.
        self.parity = (self.poly_deg + 1)%2

        # Eventually, we should be able to call with just a poly, and series of parameters, and automatically drive function toward desired behavior.
        self.target_poly = None

        """
        We assert the following:
        len(full_phases) == poly_deg + 1
        If d is even, start with half the middle phase onward. (dt - 1) onward with dt = ceil([d + 1]/2).
        If d is odd, start with the second half of the phases. dt onward.
        Conventions found in (2.5) of (https://arxiv.org/pdf/2307.12468)
        """

    def help(self):
        return("Basic class for storing classical description (e.g., length, phases) of symmetric QSP protocol, with auxiliary methods for generating derived quantities (e.g., response function, gradients and Jacobians).")

    def get_reduced_phases(self):
        return self.reduced_phases

    def get_full_phases(self):
        return self.full_phases

    def signal(self, a):
        return np.array(
            [[a, 1j * np.sqrt(1 - a**2)],
             [1j * np.sqrt(1 - a**2), a]])

    def phase(self, phi):
        return np.array(
            [[np.exp(1j * phi), 0.],
             [0., np.exp(-1j * phi)]])

    def gen_unitary(self, samples):
        phi_mats = []
        u_mats = []
        for phi in self.full_phases:
            phi_mats.append(self.phase(phi))

        for s in samples:
            W = self.signal(s)
            U = phi_mats[0]
            for mat in phi_mats[1:]:
                U = U @ W @ mat
            u_mats.append(U)

        u_mats = np.array(u_mats)
        return u_mats

    def gen_response_re(self, samples):
        """
        Returns real part of (0,0) matrix element over samples.
        """
        u_mats = self.gen_unitary(samples)
        return np.array(list(map(lambda x: np.real(x[0,0]), u_mats)))

    def gen_response_im(self, samples):
        """
        Returns real part of (0,0) matrix element over samples.
        """
        u_mats = self.gen_unitary(samples)
        return np.array(list(map(lambda x: np.imag(x[0,0]), u_mats)))

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

    def gen_jacobian(self):
        pass


    def gen_poly_jacobian_components(self, a):
        pass


"""
    In what remains we include methods for actually performing L-BFGS with respect to some target, maxiter, accuracy, and other parameters.

    If there are methods in original package for computing a bounded polynomial approximation, these can be used to generate a target function, which can then be passed to the gradient computation of the symmetric QSP class.

"""

