import numpy as np
import scipy.linalg

class SymmetricQSPProtocol:

    def __init__(self, reduced_phases=[], parity=None):
        
        """
        TODO: assert that the degree matches the length of the phases, and that the reduced and full phases agree, and that if one or the other is not instantiated, then full phases are recovered. Also check parity.
        """

        self.reduced_phases = np.array(reduced_phases)
        self.parity = parity

        if (len(reduced_phases) != 0) and (parity != None):
            if parity == 1:
                # Start with reversed phases and append phases.
                phi_front = np.flip(np.copy(self.reduced_phases),0)
                phi_back = np.copy(self.reduced_phases)
                self.full_phases = np.concatenate((phi_front, phi_back), axis=0)
            else:
                # Combine final element of reversed and standard phases.
                if len(reduced_phases) == 1:
                    # Trivial case of length 'zero' QSP protocol consisting of only a phase.
                    self.full_phases = 2*self.reduced_phases
                else:
                    phi_front = np.flip(np.copy(self.reduced_phases)[1:])
                    phi_back = np.copy(self.reduced_phases)[1:]
                    middle_phase = 2*np.array([self.reduced_phases[0]])
                    self.full_phases = np.concatenate((phi_front,middle_phase,phi_back), axis=0)
            self.poly_deg = len(self.full_phases) - 1
        else:
            self.full_phases = None
            self.poly_deg = None
        
        # Currently vestigial variable.
        self.target_poly = None

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

        if len(self.full_phases) == 1:
            for s in samples:
                U = phi_mats[0]
                u_mats.append(U)
        else:
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


        ## print("INITIAL")
        ## print(M)
        # CORRECT AFTER THIS STEP FOR THE TRIVIAL CASE.

        # Flip sign of second half of rows, and flip order of remaining elements.
        M[d+1:dd+1,:] = ((-1)**self.parity) * np.copy(M[d-1::-1,:])
        M[dd+1:,:] = np.copy(M[dd-1:0:-1,:])

        ## print("BEFORE FFT")
        ## print(M)
        # CORRECT AFTER THIS STEP FOR THE TRIVIAL CASE.

        # Take FFT over columns.
        M = np.fft.fft(M,axis=0) # SEEMS FFT TAKEN OVER THIS AXIS
        M = np.copy(np.real(M[0:dd+1,:]))

        ## print("AFTER FFT")
        ## print(M)

        M[1:-1,:] = np.copy(M[1:-1,:]*2)
        M = M/(2*dd)

        f = np.copy(M[self.parity:2*d:2,-1])
        df = np.copy(M[self.parity:2*d:2,0:-1])

        ## print("AFTER ALL")
        ## print(f)
        ## print(df)

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
            R[:,0] = np.array([1,0,0]) # Updating column with flat list. Why is this seemingly different from the convention in Alg A.1.?
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

def newton_Solver(coef, parity, **kwargs):
    """
        In what remains we include methods for actually performing Newton with respect to some target, maxiter, accuracy, and other parameters.

        If there are methods in original package for computing a bounded polynomial approximation, these can be used to generate a target function, which can then be passed to the gradient computation of the symmetric QSP class.

        In one of the examples online (https://qsppack.gitbook.io/qsppack/examples/quantum-linear-system-problems#solving-the-phase-factors-for-qlsp), they call the following two methods, which serve as a good example
        -----------------------------------------
        coef_full=cvx_poly_coef(targ, deg, opts);
        coef = coef_full(1+parity:2:end);
    """

    maxiter = 1e3 # In original is 1e5.
    crit = 1e-4 # In original is 1e-12.
    targetPre = True

    # Determines if targeting real or imaginary component.
    if targetPre:
        coef = -1*coef
    
    reduced_phases = coef/2 # Determine if the order of setting is proper here

    # This is necessary for giving a parity argument for internal methods, but is slightly backwards. We should convert everything to reduced phases and parity, which is sufficient for the rest.
    qsp_seq_opt = SymmetricQSPProtocol(reduced_phases=reduced_phases, parity=parity)

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

        # qsp_seq_opt.reduced_phases = qsp_seq_opt.reduced_phases - res/2 # Standard fixed point method.

    return (qsp_seq_opt.reduced_phases, err, curr_iter)



