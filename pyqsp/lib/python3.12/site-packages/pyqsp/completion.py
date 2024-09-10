import numpy as np
from numpy.polynomial.polynomial import Polynomial, polyfromroots
from scipy.special import chebyt, chebyu

from pyqsp.LPoly import Id, LAlg, LPoly


class CompletionError(Exception):
    """Raised when completion step fails."""
    pass


def cheb2poly(ccoefs, kind="T"):
    pcoefs = np.zeros(len(ccoefs), dtype=ccoefs.dtype)
    cfunc = None
    if kind == "T":
        cfunc = chebyt
    elif kind == "U":
        cfunc = chebyu
    else:
        raise Exception("Invalid kind specifier: {}".format(kind))

    for deg, ccoef in enumerate(ccoefs):
        basis = cfunc(deg).coef[::-1]
        pcoefs[:deg + 1] += ccoef * basis

    return pcoefs


def poly2cheb(pcoefs, kind="T"):
    ccoefs = np.zeros(len(pcoefs), dtype=pcoefs.dtype)
    cfunc = None
    if kind == "T":
        cfunc = chebyt
    elif kind == "U":
        cfunc = chebyu
    else:
        raise Exception("Invalid kind specifier: {}".format(kind))

    for i, pcoef in enumerate(pcoefs[::-1]):
        deg = pcoefs.size - i - 1
        basis = cfunc(deg).coef[::-1]
        ccoefs[deg] = pcoef / basis[-1]
        pcoefs[:deg + 1] -= ccoefs[deg] * basis

    return ccoefs


def _pq_completion(P):
    """
    Find polynomial Q given P such that the following matrix is unitary.
    [[P(a), i Q(a) sqrt(1-a^2)],
      i Q*(a) sqrt(1-a^2), P*(a)]]

    Args:
        P: Polynomial object P.

    Returns:
        Polynomial object Q giving described unitary matrix.
    """
    pcoefs = P.coef

    P = Polynomial(pcoefs)
    Pc = Polynomial(pcoefs.conj())
    roots = (1. - P * Pc).roots()

    # catagorize roots
    real_roots = np.array([], dtype=np.float64)
    imag_roots = np.array([], dtype=np.float64)
    cplx_roots = np.array([], dtype=np.complex128)

    tol = 1e-6
    for root in roots:
        if np.abs(np.imag(root)) < tol:
            # real roots
            real_roots = np.append(real_roots, np.real(root))
        elif np.real(root) > -tol and np.imag(root) > -tol:
            if np.real(root) < tol:
                imag_roots = np.append(imag_roots, np.imag(root))
            else:
                cplx_roots = np.append(cplx_roots, root)

    # remove root r_i = +/- 1
    ridx = np.argmin(np.abs(1. - real_roots))
    real_roots = np.delete(real_roots, ridx)
    ridx = np.argmin(np.abs(-1. - real_roots))
    real_roots = np.delete(real_roots, ridx)

    # choose real roots in +/- pairs
    real_roots = np.sort(real_roots)
    real_roots = real_roots[::2]

    # include negative conjugate of complex roots
    cplx_roots = np.r_[cplx_roots, -cplx_roots]

    # construct Q
    Q = Polynomial(
        polyfromroots(
            np.r_[
                real_roots,
                1j *
                imag_roots,
                cplx_roots]))
    Qc = Polynomial(Q.coef.conj())

    # normalize
    norm = np.sqrt((1 - P * Pc).coef[-1] /
                   (Q * Qc * Polynomial([1, 0, -1])).coef[-1])

    return Q * norm


def _fg_completion(F, seed):
    """
    Find polynomial G given Laurent polynomial F such that the following matrix
    is unitary.
    [[F(w), i G(w)],
      i G(1/w), F(1/w)]]

    Args:
        F: Laurent Polynomial object P.
        seed: Random seed.

    Returns:
        Laurent polynomial object G corresponding to a unitary matrix.
    """
    # Find all the roots of Id - (p * ~p) that lies within the upper unit
    # circle.
    poly = (Id - (F * ~F)).coefs
    roots = np.roots(poly)
    # poly is a real, self-inverse Laurent polynomial with no root on the unit circle. All real roots come in reciprocal pairs,
    # and all complex roots come in quadruples (r, r*, 1/r, 1/r*).
    # For each pair of real roots, select the one within the unit circle.
    # For each quadruple of complex roots, select the pair within the unit
    # circle.
    imag_roots = []
    real_roots = []
    for i in roots:
        if (np.abs(i) < 1) and (np.imag(i) > -1e-8):
            if np.imag(i) == 0.:
                real_roots.append(np.real(i))
            else:
                imag_roots.append(i)
    norm = poly[-1]

    # Randomly choose whether to pick the real root (the pair of complex roots)
    #   inside or outside the unit circle.
    # This is to reduce the range of the coefficients appearing in the final
    #   product.
    degree = len(real_roots) + 2 * len(imag_roots)
    lst = []
    if seed is None:
        seed = np.random.randint(2, size=len(imag_roots) + len(real_roots))
    for i, root in enumerate(imag_roots):
        if seed[i]:
            root = 1 / root
        lst.append(LPoly([np.abs(root) ** 2, -2 * np.real(root), 1]))
    for i, root in enumerate(real_roots):
        if seed[i + len(imag_roots)]:
            root = 1 / root
        lst.append(LPoly([-root, 1]))

    # Multiply all the polynomial factors via fft for numerical stability.
    pp = int(np.floor(np.log2(degree))) + 1
    lst_fft = np.pi * np.linspace(0, 1 - 1 / 2**pp, 2**pp)
    coef_mat = np.log(np.array([i.eval(lst_fft) for i in lst]))
    coef_fft = np.exp(np.sum(coef_mat, axis=0))
    gcoefs = np.real(np.fft.fft(coef_fft, 1 << pp))[
        :degree + 1] / (1 << pp)

    # Normalization
    G = LPoly(gcoefs * np.sqrt(norm / gcoefs[0]), -len(gcoefs) + 1)

    return G


def completion_from_root_finding(coefs, coef_type="F", seed=None, tol=1e-6):
    """
    Find a Low Algebra element g such that the identity components are given by
    the input p.

    Args:
        coefs: Array corresponding to the (Laurent) polynomial coefficients.
        coef_type: Quantum signal processing convention in ['F', 'P'].
        seed: Random seed.
        tol: Error tolerance for completion.

    Returns:
        LAlg corresponding to unitary element of the F(w) + G(w) * iX algebra.

    Raises:
        CompletionError: Raised when polynomial cannot be completed.
    """
    if coef_type == "F" or coef_type == "f":
        F = LPoly(coefs, -len(coefs) + 1)
        G = _fg_completion(F, seed)

        ipoly = F
        xpoly = G
    elif coef_type == "P" or coef_type == "p":
        pcoefs = np.array(coefs, dtype=np.complex128)
        P = Polynomial(pcoefs)
        Q = _pq_completion(P)
        deg = pcoefs.size - 1

        pcheb = poly2cheb(pcoefs, kind='T')
        qcheb = np.r_[0., poly2cheb(Q.coef, kind='U')]

        pcheb = pcheb[deg % 2::2]
        qcheb = qcheb[deg % 2::2]

        fcoefs = np.zeros(deg + 1)
        gcoefs = np.zeros(deg + 1)

        if deg % 2 == 0:
            fcoefs[:(deg + 1) // 2] = np.real(pcheb[:0:-1] - qcheb[:0:-1]) / 2
            fcoefs[(deg + 1) // 2 + 1:] = np.real(pcheb[1:] + qcheb[1:]) / 2

            gcoefs[:(deg + 1) // 2] = np.imag(pcheb[:0:-1] + qcheb[:0:-1]) / 2
            gcoefs[(deg + 1) // 2 + 1:] = np.imag(pcheb[1:] - qcheb[1:]) / 2

            fcoefs[(deg + 1) // 2] = np.real(pcheb[0])
            gcoefs[(deg + 1) // 2] = np.imag(pcheb[0])
        else:
            fcoefs[:(deg + 1) // 2] = np.real(pcheb[::-1] - qcheb[::-1]) / 2
            fcoefs[(deg + 1) // 2:] = np.real(pcheb + qcheb) / 2

            gcoefs[:(deg + 1) // 2] = np.imag(pcheb[::-1] + qcheb[::-1]) / 2
            gcoefs[(deg + 1) // 2:] = np.imag(pcheb - qcheb) / 2

        ipoly = LPoly(fcoefs, -len(fcoefs) + 1)
        xpoly = LPoly(gcoefs, -len(gcoefs) + 1)
    else:
        raise CompletionError(
            "Invalid QSP coef_type specifier: {}".format(coef_type))

    # check completion
    ncoefs = (ipoly * ~ipoly + xpoly * ~xpoly).coefs
    ncoefs_expected = np.zeros(ncoefs.size)
    ncoefs_expected[ncoefs.size // 2] = 1.
    success = np.max(np.abs(ncoefs - ncoefs_expected)) < tol

    if not success:
        raise CompletionError(
            "Completion Failed. Input {} = {} could not be completed".format(
                coef_type, coefs))

    return LAlg(ipoly, xpoly)
