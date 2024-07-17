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
        self.parity = (self.poly_deg + 1)%2 # Check for degree here.

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
        """
        Following the structure and conventions of `F_Jacobian.m' in QSPPACK, which in turn follows the conventions of Alg 3.2 in (https://arxiv.org/pdf/2307.12468).

        This method has been checked numerous times for index-convention agreement with the original source.
        """
        
        # Anonymous function to component method for columns.
        f = lambda x: self.gen_poly_jacobian_components(x)
        
        d = len(self.reduced_phases)
        dd = 2*d

        # Generate equispaced sample angles.
        theta = np.transpose(np.arange(0,d+1)) * (np.pi/dd)
        M = np.zeros((2*dd, d+1))

        # Evaluate columns over each angle.
        for n in range(0,d+1):
            M[n,:] = f(np.cos(theta[n]))


        # Flip sign of second half of rows, and flip order of remaining elements.
        M[d+1:dd+1,:] = ((-1)**self.parity) * np.copy(M[d-1::-1,:])
        M[dd+1:,:] = np.copy(M[dd-1:0:-1,:])

        # Take FFT over columns.
        M = np.fft.fft(M)
        M = np.copy(np.real(M[0:dd+1,:]))
        M[1:-1,:] = np.copy(M[1:-1,:]*2)
        M = M/(2*dd)

        f = np.copy(M[self.parity:2*d:2,-1])
        df = np.copy(M[self.parity:2*d:2,0:-1])

        return (f, df)

    def gen_poly_jacobian_components(self, a):
        """
        Following the structure and conventions of `QSPGetPimDeri_sym_real.m' in QSPPACK, which in turn follows the conventions of Alg 3.3 in (https://arxiv.org/pdf/2307.12468).

        This method has been checked numerous times for index-convention agreement with the original source.
        """

        n = len(self.reduced_phases)
        t = np.arccos(a)
        
        B = np.array([
            [np.cos(2*t), 0, -1*np.sin(2*t)],
            [0, 1, 0],
            [np.sin(2*t), 0, np.cos(2*t)]])
        L = np.zeros((n, 3))


        # Fix the final column of L.
        L[n-1,:] = np.array([0,1,0])
        
        # Modify range parameters to update elements n-2 to 0.
        for k in range(n-2,-1,-1):
            L[k,:] = np.copy(L[k+1,:]) @ np.array([
                [np.cos(2*self.reduced_phases[k+1]), -1*np.sin(2*self.reduced_phases[k+1]), 0], 
                [np.sin(2*self.reduced_phases[k+1]), np.cos(2*self.reduced_phases[k+1]), 0], 
                [0, 0, 1]]) @ B
        
        R = np.zeros((3, n))

        if self.parity == 0:
            R[:,0] = np.array([1,0,0]) # Updating column with flat list.
        else:
            R[:,0] = np.array([np.cos(t),0,np.sin(t)])

        # Updating R matrix with a second pass; ranging over 1 to n-1.
        for k in range(1,n):
            R[:,k] = B @ (np.array([
                [np.cos(2*self.reduced_phases[k-1]), -1*np.sin(2*self.reduced_phases[k-1]), 0],
                [np.sin(2*self.reduced_phases[k-1]), np.cos(2*self.reduced_phases[k-1]), 0],
                [0, 0, 1]]) @ np.copy(R[:,k-1]))

        y = np.zeros((1,n+1)) 

        # Here we have to convert from Matlab's 'single indexing' scheme.
        for k in range(n):
            y[0,k] = np.copy(2*L[k,:]) @ np.array([
                [-1*np.sin(2*self.reduced_phases[k]), -1*np.cos(2*self.reduced_phases[k]), 0],
                [np.cos(2*self.reduced_phases[k]), -1*np.sin(2*self.reduced_phases[k]), 0],
                [0, 0, 0]]) @ np.copy(R[:,k])

        y[0,n] = np.copy(L[n-1,:]) @ np.array([
            [np.cos(2*self.reduced_phases[n-1]), -1*np.sin(2*self.reduced_phases[n-1]), 0],
            [np.sin(2*self.reduced_phases[n-1]), np.cos(2*self.reduced_phases[n-1]), 0],
            [0, 0, 1]]) @ np.copy(R[:,n-1])

        return y

def newton_Solver(coef, **kwargs):
    """
        In what remains we include methods for actually performing Newton with respect to some target, maxiter, accuracy, and other parameters.

        If there are methods in original package for computing a bounded polynomial approximation, these can be used to generate a target function, which can then be passed to the gradient computation of the symmetric QSP class.

        In one of the examples online (https://qsppack.gitbook.io/qsppack/examples/quantum-linear-system-problems#solving-the-phase-factors-for-qlsp), they call the following two methods, which serve as a good example
        -----------------------------------------
        coef_full=cvx_poly_coef(targ, deg, opts);
        coef = coef_full(1+parity:2:end);
    """

    maxiter = 1e3 # In original is 1e5
    crit = 1e-4 # In original is 1e-12
    targetPre = True
    # self.useReal = True

    # Determines if targeting real or imaginary component.
    if targetPre:
        coef = -1*coef
    
    reduced_phases = coef/2 # Determine if the order of setting is proper here
    poly_deg = len(reduced_phases)

    qsp_seq_opt = SymmetricQSPProtocol(reduced_phases=reduced_phases, poly_deg=poly_deg)

    curr_iter = 0

    # Start the main loop
    while True:
        (Fval,DFval) = qsp_seq_opt.gen_jacobian()
        res = Fval - coef
        err = np.linalg.norm(res, ord=1) # Take the one norm.
        curr_iter = curr_iter + 1

        # Break conditions
        if curr_iter >= maxiter:
            print("Max iteration reached.\n")
            break

        if err < crit:
            print("Stop criteria satisfied.\n")
            break

        # Use Matlab's strange backslash division: DFval\res
        print("CURRENT ITER:%s"%(curr_iter))
        print(DFval)
        print(Fval)
        print(coef)
        print(err)
        print(qsp_seq_opt.reduced_phases)
        print("\n")

        lin_sol = np.linalg.solve(DFval, res)
        qsp_seq_opt.reduced_phases = qsp_seq_opt.reduced_phases - lin_sol

    return (qsp_seq_opt.reduced_phases, err, curr_iter)



