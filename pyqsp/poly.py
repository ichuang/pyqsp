import numpy as np
import scipy.special
import scipy.optimize
from scipy.interpolate import approximate_taylor_polynomial

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


#-----------------------------------------------------------------------------

class PolyOneOverX(PolyGenerator):

    def help(self):
        return "Region of validity is from 1/kappa to 1, and from -1/kappa to -1.  Error is epsilon"

    def generate(self, kappa=3, epsilon=0.1, return_coef=True, ensure_bounded=True):
        '''
        Approximation to 1/x polynomial, using sums of Chebyshev polynomials,
        from Quantum algorithm for systems of linear equations with exponentially
        improved dependence on precision, by Childs, Kothari, and Somma, 
        https://arxiv.org/abs/1511.02306v2 
    
        Define region D_kappa to be from 1/kappa to 1, and from -1/kappa to -1.  A good
        approximation is desired only in this region.
    
        ensure_bounded: True if polynomial should be normalized to be between +/- 1 
        '''
        b = int(kappa**2  * np.log(kappa/epsilon))
        j0 = int(np.sqrt(b * np.log(4 * b/ epsilon)))
        print(f"b={b}, j0={j0}")
    
        g = np.polynomial.chebyshev.Chebyshev([0])
        for j in range(j0+1):
            gcoef = 0
            for i in range(j+1, b+1):
                gcoef += scipy.special.binom(2*b, b+i) / 2**(2*b)
            deg = 2*j + 1
            g += (-1)**j * gcoef * np.polynomial.chebyshev.Chebyshev([0] * deg + [1])
        g = 4 * g
    
        if ensure_bounded:
            res = scipy.optimize.minimize(g, (-0.1,), bounds=[(-0.8, 0.8)])
            pmin = res.x
            print(f"[PolyOneOverX] minimum {g(pmin)} is at {pmin}: normalizing")
            scale = 1/abs(g(pmin))
            if 0:
                scale = scale * 0.9
            else:
                scale = scale * 0.5
                print(f"[PolyOneOverX] bounding to 0.5")
            g = scale * g
    
        if return_coef:
            if 1:
                pcoefs = np.polynomial.chebyshev.cheb2poly(g.coef)
            else:
                pcoefs = g.coef
            print(f"[pyqsp.PolyOneOverX] pcoefs={pcoefs}")
            return pcoefs
    
        return g

#-----------------------------------------------------------------------------

class PolyTaylorSeries(PolyGenerator):
    '''
    Base clas for PolySign and PolyThreshold
    '''

    def taylor_series(self, func, degree, ensure_bounded=True, npts=100, max_scale=0.5):
        '''
        Return numpy Polynomial approximation for func, constructed using taylor series, of specified degree.
        Evaluate approximation using mean absolut difference on npts points in the domain from -1 to 1.
        '''
        the_poly = approximate_taylor_polynomial(func, 0, degree, 1)
        the_poly = np.polynomial.Polynomial(the_poly.coef[::-1])
        if ensure_bounded:
            res = scipy.optimize.minimize(-the_poly, (0.1,), bounds=[(-1, 1)])
            pmax = res.x
            scale = 1/abs(the_poly(pmax))
            scale = scale * max_scale	# use this for the new QuantumSignalProcessingWxPhases code, which employs np.polynomial.chebyshev.poly2cheb(pcoefs)
            print(f"[PolyTaylorSeries] max {scale} is at {pmax}: normalizing")
            the_poly = scale * the_poly
        adat = np.linspace(-1, 1, npts)
        pdat = the_poly(adat)
        edat = func(adat)
        avg_err = abs(edat - pdat).mean()
        print(f"[PolyTaylorSeries] average error = {avg_err} in the domain [-1, 1] using degree {degree}")
        return the_poly
        
#-----------------------------------------------------------------------------

class PolySign(PolyTaylorSeries):

    def help(self):
        return "approximation to the sign function using erf(kappa*a) ; give degree and kappa"
    
    def generate(self, degree=7, kappa=2, ensure_bounded=True):
        '''
        Approximation to sign function, using erf(kappa * x)
        '''
        degree = int(degree)
        print(f"[pyqsp.poly.PolySign] degree={degree}, kappa={kappa}")
        if not (degree % 2):
            raise Exception("[PolyErf] degree must be odd")
        def erf_kappa(x):
            return scipy.special.erf(x*kappa)
        the_poly = self.taylor_series(erf_kappa, degree, ensure_bounded=ensure_bounded, max_scale=0.5)
        if kappa > 4:
             the_poly = 0.7 * the_poly	# smaller, to handle imperfect approximation
        pcoefs = the_poly.coef
        # force even coefficients to be zero, since the polynomial must be odd
        pcoefs[0::2] = 0
        return pcoefs

class PolyThreshold(PolyTaylorSeries):

    def help(self):
        return "approximation to a thresholding function at threshold 1/2, using linear combination of erf(kappa * a); give degree and kappa"

    def generate(self, degree=6, kappa=2, ensure_bounded=True):
        '''
        Approximation to threshold function at a=1/2; use a bandpass built from two erf's
        '''
        degree = int(degree)
        print(f"[pyqsp.poly.PolyThreshold] degree={degree}, kappa={kappa}")
        if (degree % 2):
            raise Exception("[PolyThreshold] degree must be even")
        def erf_kappa(x):
            return scipy.special.erf(x*kappa)
        def threshold(x):
            return (erf_kappa(x+0.5) - erf_kappa(x-0.5))/2
        the_poly = self.taylor_series(threshold, degree, ensure_bounded=ensure_bounded, max_scale=0.5)
        if kappa > 4:
             the_poly = 0.7 * the_poly	# smaller, to handle imperfect approximation
        pcoefs = the_poly.coef
        # force odd coefficients to be zero, since the polynomial must be even
        pcoefs[1::2] = 0
        return pcoefs

#-----------------------------------------------------------------------------

class PolyGibbs(PolyTaylorSeries):
    '''
    exponential decay polynomial
    '''

    def help(self):
        return "approximation to exp(-beta*a) ; specify degree and beta"

    def generate(self, degree=6, beta=2, ensure_bounded=True):
        degree = int(degree)
        print(f"[pyqsp.poly.PolyGibbs] degree={degree}, beta={beta}")
        if (degree % 2):
            raise Exception("[PolyGibbs] degree must be even")
        def gibbs(x):
            return np.exp(-beta * abs(x))
        the_poly = self.taylor_series(gibbs, degree, ensure_bounded=ensure_bounded, max_scale=1)
        if 0:
             the_poly = 0.8 * the_poly	# smaller, to handle imperfect approximation
        pcoefs = the_poly.coef
        # force odd coefficients to be zero, since the polynomial must be even
        pcoefs[1::2] = 0
        return pcoefs

#-----------------------------------------------------------------------------

polynomial_generators = {'invert': PolyOneOverX,
                         'poly_sign': PolySign,
                         'poly_thresh': PolyThreshold,
                         'gibbs': PolyGibbs,
}
