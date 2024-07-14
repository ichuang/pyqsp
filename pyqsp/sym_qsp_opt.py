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
        
        # Reference to an external function; see if we can recover this sort of syntax in python.
        # f = @(x) QSPGetPimDeri_sym_real(phi, x, parity);
        f = lambda x: self.gen_poly_jacobian_components(x)
        
        d = len(self.reduced_phases)
        dd = 2*d

        # Check transpose syntax and range syntax
        # CHECK WHETHER MATLAB EXPECTS FINAL INDEX OR NOT.
        theta = np.transpose(np.arange(0,d+1)) * (np.pi/dd)

        M = np.zeros((2*dd, d+1))

        for n in range(1,d+2):
            M[n-1,:] = f(np.cos(theta[n-1]))

        # TODO: Note convention for start, end, step ordering, 'end' keyword. We may also need to adjust indices after slice. IMPORTANT
        M[d+1:dd+1,:] = (-1)**self.parity * M[d-1::-1,:]
        M[dd+1:,:] = M[dd-1:0:-1,:]

        M = np.fft.fft(M) # FFT over columns.
        M = np.real(M[0:dd+1,:])
        M[1:-1,:] = M[1:-1,:]*2
        M = M/(2*dd)

        f = M[self.parity:2*d:2,-1]
        df = M[self.parity:2*d:2,0:-1]

        return (f, df)

        """
        M(d+2:dd+1,:)=(-1)^parity * M(d:-1:1,:);
        M(dd+2:end,:)= M(dd:-1:2,:);

        M = fft(M); %FFT w.r.t. columns.
        M = real(M(1:(dd+1),:));
        M(2:end-1,:) = M(2:end-1,:)*2;
        M = M/(2*dd);

        f = M(parity+1:2:2*d,end);
        df = M(parity+1:2:2*d,1:end-1);
        """

        """
        We have to replace parentheses with square brackets and subtract 1 from all indices (except after :)!!! Note that the second criterion seems to not occur in the code below.
        """

    def gen_poly_jacobian_components(self, a):
        """
        Following the structure and conventions of `QSPGetPimDeri_sym_real.m' in QSPPACK, which in turn follows the conventions of Alg 3.2 in (https://arxiv.org/pdf/2307.12468).
        """

        n = len(self.reduced_phases)
        t = np.arccos(a)
        
        B = np.array([
            [np.cos(2*t), 0, -1*np.sin(2*t)],
            [0, 1, 0],
            [np.sin(2*t), 0, np.cos(2*t)]])
        L = np.zeros((n, 3))

        # TODO: new indexing schemes begin here! Also change phi to reduced phases, and make sure indecing above and below starts at zero versus one. Also make an auto conversion from non-reduced to reduced, or vice versa, and check in all of these cases on instantiation.

        L[n-1,:] = np.array([0,1,0]) # Set final column of L. Note zero indexing
        
        
        # Note that original iterator is xrange(n-1:-1:1):
        # The Matlab convention is start:skip:end, with inclusive handling.
        for k in range(n-1,1,-1): # We have lowered all called indices by one.
            L[k-1,:] = L[k,:] @ np.array([
                [np.cos(2*self.reduced_phases[k]), -1*np.sin(2*self.reduced_phases[k]), 0], 
                [np.sin(2*self.reduced_phases[k]), np.cos(2*self.reduced_phases[k]), 0], 
                [0, 0, 1]]) @ B
        
        R = np.zeros((3, n))

        if self.parity == 0:
            R[:,0] = np.array([1,0,0]) # Note flat indexing.
        else:
            R[:,0] = np.array([np.cos(t),0,np.sin(t)]) # Note flat indexing.

        # Lowering all indices by 1.
        for k in range(1,n-1):
            R[:,k] = B @ (np.array([[np.cos(2*self.reduced_phases[k-1]), -1*np.sin(2*self.reduced_phases[k-1]), 0],[np.sin(2*self.reduced_phases[k-1]), np.cos(2*self.reduced_phases[k-1]), 0],[0, 0, 1]]) @ R[:,k-1])

        y = np.zeros((1,n+1)) # Note extending this by one; is this automatically handled in matlab? Apparently this is a thing, lol.

        for k in range(0,n-1):
            # TODO: Change to setting an entire row; note this is different than matlab code; why do they allow casting like this?
            y[:,k] = 2*L[k,:] @ np.array([
                [-1*np.sin(2*self.reduced_phases[k]), -1*np.cos(2*self.reduced_phases[k]), 0],
                [np.cos(2*self.reduced_phases[k]), -1*np.sin(2*self.reduced_phases[k]), 0],
                [0, 0, 0]]) @ R[:,k]

        y[:,n] = L[n-1,:] @ np.array([
            [np.cos(2*self.reduced_phases[n-1]), -1*np.sin(2*self.reduced_phases[n-1]), 0],
            [np.sin(2*self.reduced_phases[n-1]), np.cos(2*self.reduced_phases[n-1]), 0],
            [0, 0, 1]]) @ R[:,n-1]

        # Finally, return Jacobian matrix evaluations
        return y


"""
    In what remains we include methods for actually performing L-BFGS with respect to some target, maxiter, accuracy, and other parameters.

    If there are methods in original package for computing a bounded polynomial approximation, these can be used to generate a target function, which can then be passed to the gradient computation of the symmetric QSP class.

"""

