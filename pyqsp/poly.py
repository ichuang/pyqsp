import numpy as np
import scipy.special
import scipy.optimize

def PolyOneOverX(kappa=3, epsilon=0.1, return_coef=True, ensure_bounded=True):
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
        scale = scale * 0.9
        g = scale * g

    if return_coef:
        if 1:
            pcoefs = np.polynomial.chebyshev.cheb2poly(g.coef)
        else:
            pcoefs = g.coef
        return pcoefs

    return g
