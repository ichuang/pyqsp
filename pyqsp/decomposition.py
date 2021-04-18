import numpy
from scipy.linalg import toeplitz
from pyqsp.LPoly import LPoly, LAlg, w


def linear_system(g, ldeg):
    """
    Let vec(l) = l.IPoly.coefs + l.XPoly.coefs, i.e. to regard a Low Algebra element l of degree ldeg as a real vector in R^{2n+2}.
    M is the matrix representation of g in the sense that M * vec(l) = vec(l * g) for all l with degree ldeg.
    The linear system m, s consist of two parts; the first part ensures that the degree of g * l does not exceed deg - ldeg;
    the second part ensures that l(Id) = Id, fixing the SU(2) freedom left from the previous set of constraints.
    """
    deg = g.degree
    aligned_icoefs = g.IPoly.aligned(-deg, deg)
    aligned_xcoefs = g.XPoly.aligned(-deg, deg)

    def vec_to_mat(vec):
        return toeplitz(numpy.hstack((vec, [0] * ldeg)), [0] * (ldeg + 1))
    M = numpy.vstack((numpy.hstack((vec_to_mat(aligned_icoefs),
                                    vec_to_mat(-aligned_xcoefs[::-1]))),
                      numpy.hstack((vec_to_mat(aligned_xcoefs),
                                    vec_to_mat(aligned_icoefs[::-1])))))

    a = numpy.ones(ldeg + 1, dtype=float)
    b = numpy.zeros(ldeg + 1, dtype=float)
    m = numpy.vstack((numpy.hstack((a, b)), numpy.hstack(
        (b, a)), M[:ldeg, :], M[deg + 1: deg + 2 * ldeg + 1, :], M[-ldeg:, :]))
    s = numpy.zeros(m.shape[0], dtype=float)
    s[0] = 1
    return m, s


def decompose(g, ldeg):
    """
    Let
    g = exp(iθ_0 * X) * w * exp(iθ_1 * X) * ... w * exp(iθ_deg * X)
    One wants to solve for
    l = exp(iθ_0 * X) * w * exp(iθ_1 * X) * ... w * exp(-i(Σ_{i=1}^{deg-ldeg-1}θ_j) * X),
    and
    r = ~l * g = (exp(iΣ_{j=1}^{deg-ldeg}θ_j * X) * w * exp(iθ_{deg-ldeg+1} * X) * ... w * exp(iθ_deg * X).

    The linear system (m, s) is such that

    deg(l * g) <= deg - ldeg
    l(Id) = Id

    One can show that such a system has a unique solution.
    """
    deg = g.degree
    m, s = linear_system(g, ldeg)
    lstsq = numpy.linalg.lstsq(m, s, rcond=-1)
    v = lstsq[0]

    li = v[:ldeg + 1]
    lx = v[-ldeg - 1:]
    l = LAlg(LPoly(li, -ldeg), LPoly(lx, -ldeg))
    r = LAlg.truncate(l * g, -g.degree + ldeg, g.degree - ldeg)

    return ~l, r


def angseq(g):
    '''
    Divide-and-conquer approach to solve for the whole angle sequence.
    '''
    deg = g.degree
    if deg == 1:
        return g.left_and_right_angles
    else:
        l, r = decompose(g, deg // 2)
        a = angseq(l)
        b = angseq(r)
        return a[:-1] + [a[-1] + b[0]] + b[1:]
