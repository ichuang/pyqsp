import numpy
from pyqsp.LPoly import LPoly, LAlg, Id

def root_classes(p):
    '''
    Find all the roots of Id - (p * ~p) that lies within the upper unit circle.
    '''
    poly = (Id - (p * ~p)).coefs
    roots = numpy.roots(poly)
    # poly is a real, self-inverse Laurent polynomial with no root on the unit circle. All real roots come in reciprocal pairs,
    # and all complex roots come in quadruples (r, r*, 1/r, 1/r*).
    # For each pair of real roots, select the one within the unit circle.
    # For each quadruple of complex roots, select the pair within the unit circle.
    imag_roots = []
    real_roots = []
    for i in roots:
        if (numpy.abs(i) < 1) and (numpy.imag(i) > -1e-8):
            if numpy.imag(i) == 0.:
                real_roots.append(numpy.real(i))
            else:
                imag_roots.append(i)
    norm = poly[-1]
    return real_roots, imag_roots, norm

def poly_from_roots(real_roots, imag_roots, norm, seed=None):
    '''
    Construct the counter part with the roots and the norm.
    '''
    # Randomly choose whether to pick the real root (the pair of complex roots) inside or outside the unit circle.
    # This is to reduce the range of the coefficients appearing in the final product.
    degree = len(real_roots) + 2 * len(imag_roots)
    lst = []
    if seed is None:
        seed = numpy.random.randint(2, size=len(imag_roots) + len(real_roots))
    for i, root in enumerate(imag_roots):
        if seed[i]:
            root = 1 / root
        lst.append(LPoly([numpy.abs(root) ** 2, -2 * numpy.real(root), 1]))
    for i, root in enumerate(real_roots):
        if seed[i + len(imag_roots)]:
            root = 1 / root
        lst.append(LPoly([-root, 1]))

    # Multiply all the polynomial factors via fft for numerical stability.
    pp = int(numpy.floor(numpy.log2(degree)))+1
    lst_fft = numpy.pi * numpy.linspace(0, 1 - 1/2**pp, 2**pp)
    coef_mat = numpy.log(numpy.array([i.eval(lst_fft) for i in lst]))
    coef_fft = numpy.exp(numpy.sum(coef_mat, axis=0))
    coefs = numpy.real(numpy.fft.fft(coef_fft, 1 << pp))[:degree+1] / (1 << pp)


    # Normalization
    xpoly = LPoly(coefs * numpy.sqrt(norm / coefs[0]), -len(coefs) + 1)
    return xpoly

def completion_from_root_finding(p, seed=None):
    """
    Find a Low Algebra element g such that the identity components are given by the input p.
    """
    real_roots, imag_roots, norm = root_classes(p)
    xpoly = poly_from_roots(real_roots, imag_roots, norm)
    return LAlg(p, xpoly)
