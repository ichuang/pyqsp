import numpy as np
import scipy.optimize
import scipy.special
from scipy.interpolate import approximate_taylor_polynomial

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

class StringPolynomial:
    '''
    Representation of a polynomial using a python string which specifies a numpy function,
    and an integer giving the desired polynomial's degree.
    '''

    def __init__(self, funcstr, poly_deg):
        '''
        funcstr: (str) specification of function using "x" as the argument, e.g. "np.where(x<0, -1 ,np.where(x>0,1,0))"
                 The function should accept a numpy array as "x"
        poly_deg: (int) degree of the polynoimal to be used to approximate the specified function
        '''
        self.funcstr = funcstr
        self.poly_deg = int(poly_deg)
        try:
            self.__call__(0.5)
        except Exception as err:
            raise ValueError(
                f"Invalid function specifciation, failed to evaluate at x=0.5, err={err}")

    def degree(self):
        return self.poly_deg

    def __call__(self, arg):
        ret = eval(self.funcstr, globals(), {'x': arg})
        return ret

    def target(self, arg):
        return self.__call__(arg)

# -----------------------------------------------------------------------------

# Note: use of this class is deprecated, but retained here for legacy applications.
class TargetPolynomial(np.polynomial.Polynomial):
    '''
    Polynomial with ideal target
    '''

    def __init__(self, *args, target=None, scale=None, **kwargs):
        '''
        target = function which accepts argument and gives ideal response, e.g. lambda x: x**2
        scale = metadata about scale of polynomial
        '''
        self.target = target
        self.scale = scale
        super().__init__(*args, **kwargs)

# -----------------------------------------------------------------------------

class PolyGenerator:
    '''
    Abstract base class for polynomial generators
    '''

    def __init__(self, verbose=True):
        self.verbose = verbose
        return

    def help(self):
        '''
        return help text
        '''
        return "help text about the expected polynomial arguments"

    def generate(self):
        '''
        return list of floats specifying the [const, a, a^2, ...] coefficients of the polynomial
        '''
        return [0, 0]

# -----------------------------------------------------------------------------

class PolyCosineTX(PolyGenerator):

    def help(self):
        return "Used for Hamiltonian simultion for time tau. Error is epsilon"

    def generate(
            self,
            tau=10.,
            epsilon=0.1,
            return_coef=True,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False):
        '''
        Approximation to cos(tx) polynomial, using sums of Chebyshev
        polynomials, from Optimal Hamiltonian Simulation by Quantum Signal
        Processing by Low and Chuang,
        https://arxiv.org/abs/1606.02685
        ensure_bounded: True if polynomial should be normalized to be between
        +/- 1
        '''
        r = scipy.optimize.fsolve(lambda r: (
            np.e * np.abs(tau) / (2 * r))**r - (5 / 4) * epsilon, tau)[0]
        print(r)
        R = np.floor(r / 2).astype(int)
        R = max(R, 1)

        print(f"R={R}")

        g = scipy.special.jv(0, tau) * np.polynomial.chebyshev.Chebyshev([1])
        for k in range(1, R + 1):
            gcoef = 2 * scipy.special.jv(2 * k, tau)
            deg = 2 * k
            g += (-1)**k * gcoef * \
                np.polynomial.chebyshev.Chebyshev([0] * deg + [1])

        if ensure_bounded:
            scale = 0.5
            g = scale * g
            print(f"[PolyCosineTX] rescaling by {scale}.")

        if return_coef:
            pcoefs = g.coef
            # Note that now all approximating polynomials are in the Chebyshev basis, bypassing the branching conditional below.
            # if chebyshev_basis:
            #     pcoefs = g.coef
            # else:
            #     pcoefs = np.polynomial.chebyshev.cheb2poly(g.coef)
            if ensure_bounded and return_scale:
                return pcoefs, scale
            else:
                return pcoefs
        return g

# -----------------------------------------------------------------------------

class PolySineTX(PolyGenerator):

    def help(self):
        return "Used for Hamiltonian simultion for time tau. Error is epsilon"

    def generate(
            self,
            tau=10.,
            epsilon=0.1,
            return_coef=True,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False):
        '''
        Approximation to cos(tx) polynomial, using sums of Chebyshev
        polynomials, from Optimal Hamiltonian Simulation by Quantum Signal
        Processing by Low and Chuang,
        https://arxiv.org/abs/1606.02685
        ensure_bounded: True if polynomial should be normalized to be between
        +/- 1
        '''
        r = scipy.optimize.fsolve(lambda r: (
            np.e * np.abs(tau) / (2 * r))**r - (5 / 4) * epsilon, tau)[0]
        print(r)
        R = np.floor(r / 2).astype(int)
        R = max(R, 1)

        print(f"R={R}")

        g = np.polynomial.chebyshev.Chebyshev([0])
        for k in range(0, R + 1):
            gcoef = 2 * scipy.special.jv(2 * k + 1, tau)
            deg = 2 * k + 1
            g += (-1)**k * gcoef * \
                np.polynomial.chebyshev.Chebyshev([0] * deg + [1])

        if ensure_bounded:
            scale = 0.5
            g = scale * g
            print(f"[PolySineTX] rescaling by {scale}.")

        if return_coef:
            pcoefs = g.coef
            # Note: all polynomials are expected in the Chebyshev basis, bypassing the conditional below.
            # if chebyshev_basis:
            #     pcoefs = g.coef
            # else:
            #     pcoefs = np.polynomial.chebyshev.cheb2poly(g.coef)
            if ensure_bounded and return_scale:
                return pcoefs, scale
            else:
                return pcoefs
        return g

# -----------------------------------------------------------------------------

class PolyOneOverX(PolyGenerator):

    def help(self):
        return "Region of validity is from 1/kappa to 1, and from -1/kappa to -1.  Error is epsilon"

    def generate(
            self,
            kappa=3,
            epsilon=0.1,
            return_coef=True,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False):
        '''
        Approximation to 1/x polynomial, using sums of Chebyshev polynomials,
        from Quantum algorithm for systems of linear equations with exponentially
        improved dependence on precision, by Childs, Kothari, and Somma,
        https://arxiv.org/abs/1511.02306v2.

        Note in the above paper, Lemma 14 (page 16), that the given function is
        2*epsilon-close to the desired function over the region specified below.

        Define region D_kappa to be from 1/kappa to 1, and from -1/kappa to -1.  A good
        approximation is desired only in this region.

        ensure_bounded: True if polynomial should be normalized to be between +/- 1
        '''
        b = int(kappa**2 * np.log(kappa / epsilon))
        j0 = int(np.sqrt(b * np.log(4 * b / epsilon)))
        print(f"b={b}, j0={j0}")

        """
        # Analytic form for inverse function; note large integer divisions.
        g = np.polynomial.chebyshev.Chebyshev([0])
        for j in range(j0 + 1):
            gcoef = 0
            for i in range(j + 1, b + 1):
                gcoef += scipy.special.binom(2 * b, b + i) / 2**(2 * b)
            deg = 2 * j + 1
            g += (-1)**j * gcoef * \
                np.polynomial.chebyshev.Chebyshev([0] * deg + [1])
        g = 4 * g
        """

        # Iterative subroutine replacing above block to avoid integer overflow.
        # Following analytic form of Lemma 18 from https://arxiv.org/abs/1511.02306v2.
        # I.e., (1 - (1 - x**2)**b)/x, as per (77) on page (19) in cited work.
        g = np.polynomial.chebyshev.Chebyshev([1])
        for j in range(b):
            g *= np.polynomial.chebyshev.Chebyshev([0.5, 0, -0.5])
        g = -1*g
        g += np.polynomial.chebyshev.Chebyshev([1])

        # Perform polynomial division; remainder is trivial and ignored.
        g_coef = g.coef
        div_result = np.polynomial.chebyshev.chebdiv(g_coef, [0, 1])[0]

        # Replace g with its divided value.
        g = np.polynomial.chebyshev.Chebyshev(div_result)

        if ensure_bounded:
            # Given explicit form, the one-sided optimization is enough; compare to Taylor series setting.
            res = scipy.optimize.minimize(g, (-0.1,), bounds=[(-0.8, 0.8)])
            pmin = res.x
            print(
                f"[PolyOneOverX] minimum {g(pmin)} is at {pmin}: normalizing")
            scale = 1 / abs(g(pmin))
            scale = scale * 0.9
            print("[PolyOneOverX] bounding to 0.9")
            g = scale * g

        if return_coef:
            pcoefs = g.coef
            # Note: as all polynomials are now treated in the Chebyshev basis, the conditional below has been bypassed.
            # if chebyshev_basis:
            #     pcoefs = g.coef
            # else:
            #     pcoefs = np.polynomial.chebyshev.cheb2poly(g.coef)
            # print(f"[pyqsp.PolyOneOverX] pcoefs={pcoefs}")
            if ensure_bounded and return_scale:
                return pcoefs, scale
            else:
                return pcoefs

        return g

# -----------------------------------------------------------------------------

class PolyOneOverXRect(PolyGenerator):

    def help(self):
        return "Region of validity is from 1/kappa to 1, and from -1/kappa to -1. Error is epsilon."

    def generate(
            self,
            degree=6,
            delta=2,
            kappa=3,
            epsilon=0.1,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False):

        coefs_invert, scale1 = PolyOneOverX().generate(2 * kappa,
                                                       epsilon,
                                                       ensure_bounded,
                                                       return_scale=True,
                                                       chebyshev_basis=chebyshev_basis)

        coefs_rect, scale2 = PolyRect().generate(degree,
                                                 delta,
                                                 kappa,
                                                 epsilon,
                                                 ensure_bounded,
                                                 return_scale=True,
                                                 chebyshev_basis=chebyshev_basis)


        poly_invert = np.polynomial.chebyshev.Chebyshev(coefs_invert)
        poly_rect = np.polynomial.chebyshev.Chebyshev(coefs_rect)

        mult_result = np.polynomial.chebyshev.chebmul(poly_invert.coef, poly_rect.coef)
        pcoefs = mult_result

        # Note: all polynomials are now considered in Chebyshev basis, bypassing the original conditional below.
        # if not chebyshev_basis:
        #     poly_invert = np.polynomial.Polynomial(coefs_invert)
        #     poly_rect = np.polynomial.Polynomial(coefs_rect)

        #     pcoefs = (poly_invert * poly_rect).coef
        # else:
        #     poly_invert = np.polynomial.chebyshev.Chebyshev(coefs_invert)
        #     poly_rect = np.polynomial.chebyshev.Chebyshev(coefs_rect)

        #     mult_result = np.polynomial.chebyshev.chebmul(poly_invert.coef, poly_rect.coef)
        #     pcoefs = mult_result

        if return_scale:
            return pcoefs, scale1 * scale2
        else:
            return pcoefs

# -----------------------------------------------------------------------------

"""
    As an addendum, we can add a boolean parameter in the taylor series method to compute, instead of the monomial expansion, the chebyshev expansion, which can be passed down through the inheriting classes. This requires understanding the reversal of coefficients before instantiating a numpy polynomial. It's also worth determining whether numpy polynomial objects can themselves be implemented with chebyshev coefficients, in which case the normalization that occurs can be handled in situ.

    # Another question; we only attempt to optimize for the minimum that the negative of the polynomial acquires (which is the same as maximizing on the interval); we have definite parity, so this would always seem to give the maximum magnitude regardless (unless the polynomial is of even parity and is highly negative?). This seems to be a bug.

    It looks like we can directly instantiate a Chebyshev expansion object. This can be paired with the chebfit method to directly determine coefficients.
"""

class PolyTaylorSeries(PolyGenerator):
    '''
    Base class for PolySign and PolyThreshold
    '''

    def taylor_series(
            self,
            func,
            degree,
            ensure_bounded=True,
            return_scale=False,
            npts=100,
            max_scale=0.9,
            chebyshev_basis=False,
            cheb_samples=20):
        '''
        If chebyshev_basis is True:
            Return numpy Chebyshev approximation for func, using numpy methods for Chebyshev approximation of specified degree.
            We also evaluate the mean absolute difference on equispaced points over the interval [-1,1].

        If chebyshev_basis is False:
            Return numpy Polynomial approximation for func, constructed using
            taylor series, of specified degree.
            We also evaluate the mean absolute difference on equispaced points over the interval [-1,1].
        '''

        # Note: PolyTaylorSeries now no longer generates approximating Taylor polynomials, but only Chebyshev interpolations as contained in the assured branch indicated below. This exhibits better stability and convergence.

        cheb_samples = 2*degree # Set to prevent aliasing; note that all methods calling TaylorSeries implicitly have their cheb_samples specifications overruled here.
        # Generate x and y values for fit according to func; note use of chebyshev nodes of the first kind.
        samples = np.polynomial.chebyshev.chebpts1(cheb_samples)
        scale = 1.0 # Binding variable.

        vals = np.array(list(map(func, samples)))
        # Generate cheb fit for function.
        cheb_coefs = np.polynomial.chebyshev.chebfit(samples, vals, degree)
        # Generate chebyshev polynomial object from coefs.
        cheb_poly = np.polynomial.chebyshev.Chebyshev(cheb_coefs)

        # Determine maximum over interval and rescale.
        if ensure_bounded:
            # Minimize polynomial and negative of polynomial to find overall bound on absolute value.
            res_1 = scipy.optimize.minimize(-1*cheb_poly, (0.1,), bounds=[(-1, 1)])
            res_2 = scipy.optimize.minimize(cheb_poly, (0.1,), bounds=[(-1, 1)])
            pmax_1 = res_1.x[0]
            pmax_2 = res_2.x[0]

            # Compute the smaller of the two rescaling values.
            arg_array = np.array([pmax_1, pmax_2])
            max_index = np.argmax([abs(cheb_poly(pmax_1)), abs(cheb_poly(pmax_2))])
            scale = 1.0/np.max([abs(cheb_poly(pmax_1)), abs(cheb_poly(pmax_2))])

            # Compute overal rescaling factor and apply to poly approx.
            scale = scale * max_scale
            print(f"[PolyTaylorSeries] (Cheb) max {scale} is at {arg_array[max_index]}: normalizing")
            cheb_poly = scale * cheb_poly

        # Determine average error on interval and print.
        adat = np.linspace(-1, 1, npts)
        pdat = cheb_poly(adat)
        edat = scale * func(adat) # Compare to rescaled function.
        avg_err = abs(edat - pdat).mean()
        print(
            f"[PolyTaylorSeries] (Cheb) average error = {avg_err} in the domain [-1, 1] using degree {degree}")

        if ensure_bounded and return_scale:
            return cheb_poly, scale
        else:
            return cheb_poly

# -----------------------------------------------------------------------------

class PolySign(PolyTaylorSeries):

    def help(self):
        return "approximation to the sign function using erf(delta*a) ; given delta"

    def generate(
            self,
            degree=7,
            delta=2,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False,
            cheb_samples=20,
            max_scale=0.9):
        '''
        Approximation to sign function, using erf(delta * x)
        '''
        degree = int(degree)
        print(f"[pyqsp.poly.PolySign] degree={degree}, delta={delta}")
        if not (degree % 2):
            raise Exception("[PolyErf] degree must be odd")

        def erf_delta(x):
            return scipy.special.erf(x * delta)

        if ensure_bounded and return_scale:
            the_poly, scale = self.taylor_series(
                erf_delta,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)
        else:
            the_poly = self.taylor_series(
                erf_delta,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)

        pcoefs = the_poly.coef
        # force even coefficients to be zero, since the polynomial must be odd
        pcoefs[0::2] = 0
        if (ensure_bounded and return_scale) or chebyshev_basis:
            return pcoefs, scale
        else:
            return TargetPolynomial(pcoefs, target=lambda x: np.sign(x))

# -----------------------------------------------------------------------------

class PolyThreshold(PolyTaylorSeries):

    def help(self):
        return "approximation to a thresholding function at threshold 1/2, using linear combination of erf(delta * a); give degree and delta"

    def generate(self,
                degree=6,
                delta=2,
                ensure_bounded=True,
                return_scale=False,
                chebyshev_basis=False,
                cheb_samples=20,
                max_scale=0.9):
        '''
        Approximation to threshold function at a=1/2; use a bandpass built from two erf's
        '''
        degree = int(degree)
        print(f"[pyqsp.poly.PolyThreshold] degree={degree}, delta={delta}")
        if (degree % 2):
            raise Exception("[PolyThreshold] degree must be even")

        def erf_delta(x):
            return scipy.special.erf(x * delta)

        def threshold(x):
            return (erf_delta(x + 0.5) - erf_delta(x - 0.5)) / 2

        if ensure_bounded and return_scale:
            the_poly, scale = self.taylor_series(
                threshold,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)
        else:
            the_poly = self.taylor_series(
                threshold,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)

        pcoefs = the_poly.coef
        # force odd coefficients to be zero, since the polynomial must be even
        pcoefs[1::2] = 0
        if ensure_bounded and return_scale:
            return pcoefs, scale
        else:
            return pcoefs

# -----------------------------------------------------------------------------

class PolyPhaseEstimation(PolyTaylorSeries):

    def help(self):
        return "phase estimation polynomial given "

    def generate(self,
                degree=6,
                delta=2,
                ensure_bounded=True,
                return_scale=False,
                chebyshev_basis=False,
                cheb_samples=20,
                max_scale=0.9):
        '''
        Approximation to threshold function at a=1/2; use a bandpass built from two erf's
        '''
        degree = int(degree)
        print(f"[pyqsp.poly.PolyThreshold] degree={degree}, delta={delta}")
        if (degree % 2):
            raise Exception("[PolyThreshold] degree must be even")

        def erf_delta(x):
            return scipy.special.erf(x * delta)

        def threshold(x):
            return (-1 + erf_delta(1/np.sqrt(2) - x) + erf_delta(1/np.sqrt(2) + x))

        if ensure_bounded and return_scale:
            the_poly, scale = self.taylor_series(
                threshold,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)
        else:
            the_poly = self.taylor_series(
                threshold,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)

        pcoefs = the_poly.coef
        # force odd coefficients to be zero, since the polynomial must be even
        pcoefs[1::2] = 0
        if ensure_bounded and return_scale:
            return pcoefs, scale
        else:
            return pcoefs

# -----------------------------------------------------------------------------

class PolyRect(PolyTaylorSeries):

    def help(self):
        return "approximation to a thresholding function at threshold 1/2, using linear combination of erf(delta * a); give degree and delta"

    def generate(self,
                degree=6,
                delta=2,
                kappa=3,
                epsilon=0.1,
                ensure_bounded=True,
                return_scale=False,
                chebyshev_basis=False,
                cheb_samples=20,
                max_scale=0.9):
        '''
        Approximation to threshold function at a=1/2; use a bandpass built from two erf's
        '''
        degree = int(degree)
        print(f"[pyqsp.poly.PolyRect] degree={degree}, delta={delta}")
        if (degree % 2):
            raise Exception("[PolyRect] degree must be even")

        if np.log(2 / (np.pi * epsilon**2)) <= 0:
            raise ValueError("epsilon must be specified smaller to ensure valid specification for erf approximation to rectangle function.")
        else:
            pass

        k = (np.sqrt(2) / delta) * np.sqrt(np.log(2 / (np.pi * epsilon**2)))

        def erf_delta(x):
            return scipy.special.erf(x * k)

        def rect(x):
            return 1 + (erf_delta(x - 3 / (4 * kappa)) +
                        erf_delta(-x - 3 / (4 * kappa))) / 2

        # target=lambda x: scale *
        # (1 - (np.sign(x + 1 / args.seqargs[2])
        # - np.sign(x - 1 / args.seqargs[2])) / 2)

        if ensure_bounded and return_scale:
            the_poly, scale = self.taylor_series(
                rect,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)
        else:
            the_poly = self.taylor_series(
                rect,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)

        pcoefs = the_poly.coef
        # force odd coefficients to be zero, since the polynomial must be even
        pcoefs[1::2] = 0
        if ensure_bounded and return_scale:
            return pcoefs, scale
        else:
            return pcoefs

# -----------------------------------------------------------------------------

class PolyLinearAmplification(PolyTaylorSeries):

    def help(self):
        return "approximates x/(2*gamma) in region (-2*gamma, 2*gamma) capped to +/- 1 outside for some constant gamma"

    def generate(self,
                degree=7,
                gamma=0.25,
                kappa=10,
                ensure_bounded=True,
                return_scale=False,
                chebyshev_basis=False,
                cheb_samples=20,
                max_scale=0.9):
        '''
        Approximation to the truncated linear function described in Low's thesis (2017)
        '''
        degree = int(degree)
        print(
            f"[pyqsp.poly.PolyLinearAmplification] degree={degree}, gamma={gamma}")
        if (degree % 2) != 1:
            raise Exception("[PolyLinearAmplification] degree must be odd")

        def erf_delta(x):
            return scipy.special.erf(x * kappa)

        def rect(x):
            return (erf_delta(x + 2 * gamma) - erf_delta(x - 2 * gamma)) / 2

        def linear_amplification(x):
            return x * rect(x) / (2 * gamma)

        result = self.taylor_series(
            linear_amplification,
            degree,
            ensure_bounded=ensure_bounded,
            return_scale=return_scale,
            max_scale=max_scale,
            chebyshev_basis=chebyshev_basis,
            cheb_samples=cheb_samples)

        if ensure_bounded and return_scale:
            the_poly, scale = result
        else:
            the_poly = result

        pcoefs = the_poly.coef
        # force even coefficients to be zero, since the polynomial must be odd
        pcoefs[0::2] = 0
        if ensure_bounded and return_scale:
            return pcoefs, scale
        else:
            return pcoefs

# -----------------------------------------------------------------------------

class PolyGibbs(PolyTaylorSeries):
    '''
    exponential decay polynomial
    '''

    def help(self):
        return "approximation to exp(-beta*a) ; specify degree and beta"

    def generate(self,
                degree=6,
                beta=2,
                ensure_bounded=True,
                return_scale=False,
                chebyshev_basis=False,
                cheb_samples=20,
                max_scale=0.9):
        degree = int(degree)
        print(f"[pyqsp.poly.PolyGibbs] degree={degree}, beta={beta}")
        if (degree % 2):
            raise Exception("[PolyGibbs] degree must be even")

        def gibbs(x):
            return np.exp(-beta * abs(x))

        if ensure_bounded and return_scale:
            the_poly, scale = self.taylor_series(
                gibbs,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)
        else:
            the_poly = self.taylor_series(
                gibbs,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)

        pcoefs = the_poly.coef
        # force odd coefficients to be zero, since the polynomial must be even
        pcoefs[1::2] = 0
        if (ensure_bounded and return_scale) or chebyshev_basis:
            return pcoefs, scale
        else:
            return TargetPolynomial(pcoefs, target=lambda x: gibbs(x))

# -----------------------------------------------------------------------------

class PolyEigenstateFiltering(PolyTaylorSeries):
    '''
    Lin and Tong's eigenstate filtering polynomial
    '''

    def help(self):
        return "Lin and Tong's eigenstate filtering polynomial ; specify degree, delta, max_scale"

    def generate(
            self,
            degree=6,
            delta=0.2,
            max_scale=0.9,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False,
            cheb_samples=20):
        degree = int(degree)
        print(f"[pyqsp.poly.PolyEfilter] degree={degree}, delta={delta}")
        if (degree % 2):
            raise Exception("[PolyEfilter] degree must be even")

        def cheb(x):
            Tk = np.polynomial.chebyshev.Chebyshev([0] * degree + [1])
            return Tk(-1 + 2 * (x**2 - delta**2) / (1 - delta**2))
        scale = 1 / cheb(0)

        def efpoly(x):
            return scale * cheb(x)

        if ensure_bounded and return_scale:
            the_poly, scale = self.taylor_series(
                efpoly,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)
        else:
            the_poly = self.taylor_series(
                efpoly,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)

        pcoefs = the_poly.coef
        # force odd coefficients to be zero, since the polynomial must be even
        pcoefs[1::2] = 0
        if ensure_bounded and return_scale:
            return pcoefs, scale
        else:
            return pcoefs

# -----------------------------------------------------------------------------

class PolyRelu(PolyTaylorSeries):
    '''
    Relu function
    '''

    def help(self):
        return "symmetric Relu function sigma(|a-delta|) = 0 if |a| < delta, else |a|-delta ; specify degree, delta"

    def generate(
            self,
            degree=6,
            delta=0.2,
            max_scale=0.9,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False,
            cheb_samples=20):
        degree = int(degree)
        print(f"[pyqsp.poly.PolyRelu] degree={degree}, delta={delta}")
        if (degree % 2):
            raise Exception("[PolyRelu] degree must be even")

        def cdf(x):
            return (1 + scipy.special.erf(x / np.sqrt(2))) / 2

        def gelu(x):
            return abs(x) * cdf(abs(x) - delta)
        the_poly = self.taylor_series(
            gelu,
            degree,
            ensure_bounded=ensure_bounded,
            max_scale=max_scale,
            chebyshev_basis=chebyshev_basis,
            cheb_samples=cheb_samples)
        pcoefs = the_poly.coef
        # Force odd coefficients to be zero, since the polynomial must be even.
        pcoefs[1::2] = 0
        if ensure_bounded and return_scale:
            scale = max_scale
            return pcoefs, scale
        else:
            return pcoefs

# -----------------------------------------------------------------------------

class PolySoftPlus(PolyTaylorSeries):
    '''
    SoftPlus function
    '''

    def help(self):
        return "symmetric softplus function sigma(|a-delta|) = 0 if |a| < delta, else |a| ; specify degree, delta"

    def generate(
            self,
            degree=6,
            delta=0.2,
            kappa=1,
            max_scale=0.9,
            ensure_bounded=True,
            return_scale=False,
            chebyshev_basis=False,
            cheb_samples=20):
        degree = int(degree)
        print(
            f"[pyqsp.poly.PolySoftPlus] degree={degree}, delta={delta}, kappa={kappa}")
        if (degree % 2):
            raise Exception("[PolySoftPlus] degree must be even")

        def func(x):
            return np.log(1 + np.exp(kappa * (abs(x) - delta))) / kappa
        if ensure_bounded and return_scale:
            the_poly, scale = self.taylor_series(
                func,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)
        else:
            the_poly = self.taylor_series(
                func,
                degree,
                ensure_bounded=ensure_bounded,
                return_scale=return_scale,
                max_scale=max_scale,
                chebyshev_basis=chebyshev_basis,
                cheb_samples=cheb_samples)

        pcoefs = the_poly.coef
        # Force odd coefficients to be zero, since the polynomial must be even.
        pcoefs[1::2] = 0
        if ensure_bounded and return_scale:
            return pcoefs, scale
        else:
            return pcoefs

# -----------------------------------------------------------------------------

polynomial_generators = {'invert': PolyOneOverX,
                         'poly_sign': PolySign,
                         'poly_thresh': PolyThreshold,
                         'gibbs': PolyGibbs,
                         'efilter': PolyEigenstateFiltering,
                         'relu': PolyRelu,
                         'softplus': PolySoftPlus,
                         }
