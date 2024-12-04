import numpy as np

class SymmetricQSPProtocol:

    def __init__(self, reduced_phases=[], parity=None, target_poly=None):

        """
        Initialize a symmetric QSP protocol in terms of its reduced phases
        and parity, following the convention of Dong, Lin, Ni, & Wang in
        (https://arxiv.org/pdf/2307.12468). For even (0) parity, the central
        phase is split in half across the initial position of the reduced
        phases, while for odd (1) parity, the reduced phase list is mirrored
        and concatenated with itself.
        """

        self.reduced_phases = np.array(reduced_phases)
        self.parity = parity

        if (len(self.reduced_phases) != 0) and (self.parity != None):
            if self.parity == 1:
                # Append reduced phases to reversed reduced phases.
                phi_front = np.flip(np.copy(self.reduced_phases),0)
                phi_back = np.copy(self.reduced_phases)
                self.full_phases = np.concatenate((phi_front, phi_back), axis=0)
            else:
                if len(self.reduced_phases) == 1:
                    # For trivial case of length 'zero' QSP protocol, double the only phase.
                    self.full_phases = 2*self.reduced_phases
                else:
                    # Otherwise create new middle phase which is twice the initial phase.
                    phi_front = np.flip(np.copy(self.reduced_phases)[1:])
                    phi_back = np.copy(self.reduced_phases)[1:]
                    middle_phase = 2*np.array([self.reduced_phases[0]])
                    self.full_phases = np.concatenate((phi_front,middle_phase,phi_back), axis=0)
            # Set the overall degree of P.
            self.poly_deg = len(self.full_phases) - 1
        else:
            # Otherwise, instantiate all to null. We could also throw an error here.
            self.full_phases = None
            self.poly_deg = None

        # Currently vestigial, but can eventually allow for immediate self-optimization upon instantiation with target poly of proper dimension.
        if target_poly:
            self.target_poly = target_poly
        else:
            self.target_poly = None

    def help(self):
        return("Class for storing classical description (i.e., reduced phases, parity) of symmetric QSP protocol, with auxiliary methods for generating derived quantities (e.g., response function, gradients and Jacobians).")

    def signal(self, a):
        # QSP signal unitary at scalar arguement a.
        return np.array(
            [[a, 1j * np.sqrt(1 - a**2)],
             [1j * np.sqrt(1 - a**2), a]])

    def phase(self, phi):
        # QSP phase unitary at scalar phase phi.
        return np.array(
            [[np.exp(1j * phi), 0.],
             [0., np.exp(-1j * phi)]])

    def update_reduced_phases(self, new_reduced_phases):
        # Update reduced phases and full phases simultaneously with a single call
        self.reduced_phases = np.array(new_reduced_phases)

        if (len(self.reduced_phases) != 0) and (self.parity != None):
            if self.parity == 1:
                # Append reduced phases to reversed reduced phases.
                phi_front = np.flip(np.copy(self.reduced_phases),0)
                phi_back = np.copy(self.reduced_phases)
                self.full_phases = np.concatenate((phi_front, phi_back), axis=0)
            else:
                if len(self.reduced_phases) == 1:
                    # For trivial case of length 'zero' QSP protocol, double the only phase.
                    self.full_phases = 2*self.reduced_phases
                else:
                    # Otherwise create new middle phase which is twice the initial phase.
                    phi_front = np.flip(np.copy(self.reduced_phases)[1:])
                    phi_back = np.copy(self.reduced_phases)[1:]
                    middle_phase = 2*np.array([self.reduced_phases[0]])
                    self.full_phases = np.concatenate((phi_front,middle_phase,phi_back), axis=0)
            # Set the overall degree of P.
            self.poly_deg = len(self.full_phases) - 1
        else:
            # TODO: Otherwise, instantiate all to null. We could also throw an error here.
            self.full_phases = None
            self.poly_deg = None

    def gen_unitary(self, samples):
        # From full phases, generate overall QSP unitary mapped over sample signal values (i.e., the argument a in the signal method).
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
        # Return real part of (0,0) matrix element of QSP protocol over samples.
        u_mats = self.gen_unitary(samples)
        return np.array(list(map(lambda x: np.real(x[0,0]), u_mats)))

    def gen_response_im(self, samples):
        # Return real part of (0,0) matrix element of QSP protocol over samples.
        u_mats = self.gen_unitary(samples)
        return np.array(list(map(lambda x: np.imag(x[0,0]), u_mats)))

    def gen_loss(self, samples, target_poly):
        # Currently vestigial; compute L(phi) function over samples with respect to target_poly.
        pass

    def gen_jacobian(self):
        """
        Following the structure and conventions of `F_Jacobian.m' in QSPPACK,
        which in turn follows the conventions of Alg 3.2 in
        (https://arxiv.org/pdf/2307.12468). Compute the Jacobian matrix of the
        overall loss function (difference between desired matrix element
        implicit in gen_poly_jacobian_components and the achieved matrix
        element at the Chebyshev nodes of order len(reduced_phases)) against
        the reduced QSP phases.
        """

        d = len(self.reduced_phases)
        dd = 2*d

        # Anonymous function to generate columns.
        f = lambda x: self.gen_poly_jacobian_components(x)

        # Generate equispaced sample angles.
        theta = np.transpose(np.arange(0,d+1)) * (np.pi/dd)
        M = np.zeros((2*dd, d+1))

        # Evaluate columns over each angle.
        for n in range(0,d+1):
            M[n,:] = f(np.cos(theta[n]))

        # Flip sign of second half of rows, and flip order of remaining elements.
        M[d+1:dd+1,:] = ((-1)**self.parity) * np.copy(M[d-1::-1,:])
        M[dd+1:,:] = np.copy(M[dd-1:0:-1,:])

        # Take FFT over rows. Note difference from MATLAB convention.
        M = np.fft.fft(M,axis=0)
        M = np.copy(np.real(M[0:dd+1,:]))

        # Double initial element and rescale matrix.
        M[1:-1,:] = np.copy(M[1:-1,:]*2)
        M = M/(2*dd)

        # Slice out phases and the full Jacobian.
        f = np.copy(M[self.parity:2*d:2,-1])
        df = np.copy(M[self.parity:2*d:2,0:-1])

        return (f, df)

    def gen_poly_jacobian_components(self, a):
        """
        Following the structure and conventions of `QSPGetPimDeri_sym_real.m'
        in QSPPACK, which in turn follows the conventions of Alg 3.3 in
        (https://arxiv.org/pdf/2307.12468). Compute individual columns of the
        overall jacobian at a given scalar signal a by direct computation of
        the product of QSP signal and phase unitaries composing the derivative
        of the unitary with respect to each reduced phase index.
        """

        n = len(self.reduced_phases)
        t = np.arccos(a)

        B = np.array([
            [np.cos(2*t), 0, -1*np.sin(2*t)],
            [0, 1, 0],
            [np.sin(2*t), 0, np.cos(2*t)]])
        L = np.zeros((n, 3))

        # Fix the final row of L.
        L[n-1,:] = np.array([0,1,0])

        # Modify range parameters to update rows from n-2 to 0.
        for k in range(n-2,-1,-1):
            L[k,:] = np.copy(L[k+1,:]) @ np.array([
                [np.cos(2*self.reduced_phases[k+1]), -1*np.sin(2*self.reduced_phases[k+1]), 0],
                [np.sin(2*self.reduced_phases[k+1]), np.cos(2*self.reduced_phases[k+1]), 0],
                [0, 0, 1]]) @ B

        R = np.zeros((3, n))

        # Fix the initial column of R depending on parity.
        if self.parity == 0:
            R[:,0] = np.array([1,0,0])
        else:
            R[:,0] = np.array([np.cos(t),0,np.sin(t)])

        # Updating R with a second pass ranging from 1 to n-1.
        for k in range(1,n):
            R[:,k] = B @ (np.array([
                [np.cos(2*self.reduced_phases[k-1]), -1*np.sin(2*self.reduced_phases[k-1]), 0],
                [np.sin(2*self.reduced_phases[k-1]), np.cos(2*self.reduced_phases[k-1]), 0],
                [0, 0, 1]]) @ np.copy(R[:,k-1]))

        y = np.zeros((1,n+1))

        # Finally, update y according to L, R.
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

def newton_solver(coef, parity, **kwargs):
    """
        External method for performing Newton iteration with respect to
        some target polynomial, maxiter, and accuracy.

        If there are methods in original package for computing a
        bounded polynomial approximation, these can be used to generate
        a target function (n.b., in the Chebyshev basis, with zero-components
        due to definite parity removed, from low to high order!), which can
        then be passed to the Jacobian computation of the symmetric QSP class.
    """

    if 'crit' in kwargs:
        crit = kwargs['crit']
    else:
        crit = 1e-12
    if 'maxiter' in kwargs:
        maxiter = kwargs['maxiter']
    else:
        maxiter = 1e2

    # Currently deprecated, but real and imaginary parts can be alternately targeted by flipping overall sign of coef.
    # # targetPre = True
    # # Determines if targeting real or imaginary component.
    # if targetPre:
    #     coef = -1*coef

    # Set an initial guess for the reduced phases which is approximately correct locally around the origin.
    reduced_phases = coef/2

    # Generate symmetric QSP phases with these parameters, which will be iteratively updated.
    qsp_seq_opt = SymmetricQSPProtocol(reduced_phases=reduced_phases, parity=parity)
    curr_iter = 0

    print(f"[sym_qsp] Iterative optimization to err {crit:.3e} or max_iter {int(maxiter)}.")

    # Start the main loop
    while True:
        # Recover evaluated differences and Jacobian.
        (Fval,DFval) = qsp_seq_opt.gen_jacobian()
        res = Fval - coef
        err = np.linalg.norm(res, ord=1) # Take the one norm error.
        curr_iter = curr_iter + 1

        # Format string to show running error computation
        print(f"iter: {curr_iter:03} --- err: {err:.3e}")

        # Invert Jacobian at evaluated point to determine direction of next step.
        lin_sol = np.linalg.solve(DFval, res)
        # Note: update reduced and full phases together through specialized method, to prevent schism.
        qsp_seq_opt.update_reduced_phases(qsp_seq_opt.reduced_phases - lin_sol)

        # Break conditions (beyond maxiter or within err.)
        if curr_iter >= maxiter:
            print("[sym_qsp] Max iteration reached.")
            break
        if err < crit:
            print("[sym_qsp] Stop criteria satisfied.")
            break

    return (qsp_seq_opt.reduced_phases, err, curr_iter, qsp_seq_opt)
