import numpy
PRES = 5000


class LPoly():
    '''
    Laurent polynomials with parity constraint.
    '''

    def __init__(self, coefs, dmin=0):
        '''
        coefs = one-dimensional list or array of coefficients for the Laurent polynomial
        dmin = minimum power for the polynomial, e.g. 0 or -1 or -4
        '''
        self.coefs = numpy.array(coefs)
        if len(self.coefs) == 0:
            self.dmin = dmin
            self.iszero = True
            self.coefs = [0]
        else:
            assert(len(self.coefs.shape) == 1), self.coefs
            self.dmin = dmin
            self.iszero = False

    @property
    def dmax(self):
        return 2 * len(self.coefs) + self.dmin - 2

    @property
    def degree(self):
        """
        The degree of a Laurent polynomial is defined to be the maximum absolute value of the powers over all nonzero terms.
        """
        return max(-self.dmin, self.dmax)

    @property
    def norm(self):
        """
        The norm of a Laurent polynomial is defined to be the l2 norm of the coefficients regarded as a vector.
        """
        return numpy.linalg.norm(self.coefs)

    @property
    def inf_norm(self):
        """
        The infinity norm of a Laurent polynomial is defined to be the maximum modulus over the unit circle.
        """
        i, x = self.curve
        return numpy.amax(
            numpy.absolute(i + 1j * x))

    @property
    def parity(self):
        '''
        Parity of the polynomial. 0 for even parity Laurent polynomials and 1 for odd parity Laurent polynomials.
        '''
        return self.dmin % 2

    @property
    def curve(self):
        values = numpy.exp(
            numpy.outer(
                numpy.linspace(
                    -self.parity * 1j * numpy.pi,
                    1j * numpy.pi,
                    PRES),
                range(
                    self.dmin,
                    self.dmax + 1,
                    2))).dot(
            self.coefs)
        return numpy.real(values), numpy.imag(values)

    def __getitem__(self, key):
        if (key - self.dmin) % 2:
            return 0
        pos = (key - self.dmin) // 2
        if (pos < len(self.coefs)) and (pos >= 0):
            return self.coefs[pos]
        else:
            return 0

    def __mul__(self, other):
        if isinstance(other, LAlg):
            return LAlg(self * other.IPoly, self * other.XPoly)
        if not isinstance(other, LPoly):
            return LPoly(other * self.coefs, self.dmin)
        if self.iszero or other.iszero:
            return LPoly([])
        dmin = self.dmin + other.dmin
        coefs = numpy.convolve(self.coefs, other.coefs)
        return LPoly(coefs, dmin)

    def __rmul__(self, other):
        if isinstance(other, LAlg):
            return LAlg(self * other.IPoly, ~self * other.XPoly)
        elif not isinstance(other, LPoly):
            return LPoly(other * self.coefs, self.dmin)

    def __add__(self, other):
        '''
        Only Laurent polynomials of the same parity can be added together in order to preserve parity.
        '''
        if self.iszero:
            return LPoly(other.coefs, other.dmin)
        if other.iszero:
            return LPoly(self.coefs, self.dmin)
        assert (self.parity == other.parity), "not of the same parity"
        dmin = min(self.dmin, other.dmin)
        dmax = max(self.dmax, other.dmax)
        coefs = self.aligned(dmin, dmax) + other.aligned(dmin, dmax)
        return LPoly(coefs, dmin)

    def __neg__(self):
        return LPoly(-1 * self.coefs, self.dmin)

    def __invert__(self):
        '''
        Conjugation of a Laurent polynomial maps f(w) to f(w^-1).
        '''
        dmin = -self.dmax
        coefs = self.coefs[::-1]
        return LPoly(coefs, dmin)

    def __sub__(self, other):
        return self + (-other)

    def __str__(self):
        return " + ".join(["{} * w ^ ({})".format(self.coefs[i],
                                                  self.dmin + 2 * i)
                           for i in range(len(self.coefs))])

    def aligned(self, dmin, dmax):
        if(self.iszero):
            return numpy.zeros((dmax - dmin) // 2 + 1)
        else:
            assert (
                dmin <= self.dmin) and (
                dmax >= self.dmax), "interval not valid"
            return numpy.hstack((numpy.zeros((self.dmin - dmin) // 2),
                                 numpy.array(self.coefs),
                                 numpy.zeros((dmax - self.dmax) // 2)))

    def eval(self, angles):
        '''
        Evalute the Laurent polynomial f(w) at w = exp(i * angle) for angle iterating over angles. Returns a complex array .
        '''
        if self.iszero:
            return 1
        res = self.coefs.dot(
            numpy.exp(
                1j *
                numpy.outer(
                    numpy.arange(
                        self.dmin,
                        self.dmax +
                        1,
                        2),
                    angles)))
        return res

    @classmethod
    def truncate(cls, p, dmin, dmax):
        lb = min(dmin, p.dmin)
        ub = max(dmax, p.dmax)
        return LPoly(p.aligned(lb, ub + 2)
                     [(dmin - lb) // 2:(dmax - ub) // 2 - 1], dmin)

    @classmethod
    def isconsistent(cls, a, b):
        if a.iszero:
            return True
        if b.iszero:
            return True
        return a.parity == b.parity

    def __eq__(self, other):
        '''
        Equality test between two LPoly instances.
        Note that this doesn't check for when two LPoly's are equal but have different dmin
        and different coefficients, e.g. because of zeros in the coefficients
        '''
        iseq = abs(self.coefs - other.coefs).sum() < 0.001
        iseq = iseq and (self.dmin == other.dmin)
        return iseq

    def round_zeros(self, thresh=1.0e-5):
        '''
        round small coefficients down to zero
        '''
        self.coefs[self.coefs < thresh] = 0

    def pos_half(self):
        '''
        Return Laurent polynomial with coefficients of only positive exponents
        The negative (and zero) exponent coefficients are set to zero.
        '''
        new_coefs = numpy.copy(self.coefs)
        nhalf = int(numpy.ceil(len(new_coefs) / 2))
        new_coefs[:nhalf] = 0
        return LPoly(new_coefs, self.dmin)

    def neg_half(self):
        '''
        Return Laurent polynomial with coefficients of only negative (or zero) exponents
        The positive exponent coefficients are set to zero.
        '''
        new_coefs = numpy.copy(self.coefs)
        nhalf = int(numpy.ceil(len(new_coefs) / 2))
        new_coefs[nhalf:] = 0
        return LPoly(new_coefs, self.dmin)


class LAlg():
    '''
    Low algebra elements with parity constraints.
    A Low algebra element g is a matrix-valued Laurent polynomial with the given form
    g(w) = IPoly(w) + XPoly(w) * iX,
    where IPoly and XPoly are real Laurent polynomials sharing the same parity,
    and the elements w and X satisfy the relations (iX)^2 = Id and XwXw = Id.
    '''

    def __init__(self, IPoly=LPoly([], 0), XPoly=LPoly([], 0)):
        self.IPoly = IPoly
        self.XPoly = XPoly
        assert LPoly.isconsistent(self.IPoly, self.XPoly),\
            "The algebra element does not have a consistent parity"

    @property
    def degree(self):
        return max(self.IPoly.degree, self.XPoly.degree)

    @property
    def norm(self):
        return numpy.sqrt(self.IPoly.norm**2 + self.XPoly.norm**2)

    @property
    def parity(self):
        return self.IPoly.parity

    def __str__(self):
        res = []
        if not self.IPoly.iszero:
            res.append(str(self.IPoly))
        if not self.XPoly.iszero:
            res.append("( " + str(self.XPoly) + " ) * iX")
        return " + ".join(res)

    def __add__(self, other):
        if isinstance(other, LPoly):
            return LAlg(self.IPoly + other, self.XPoly)
        return LAlg(self.IPoly + other.IPoly, self.XPoly + other.XPoly)

    def __neg__(self):
        return LAlg(-self.IPoly, -self.XPoly)

    def __sub__(self, other):
        return self + (-other)

    def __invert__(self):
        '''
        Conjugation of g(w) = A(w) + B(w) * iX is ~g(w) = A(w^-1) - B(w) * iX.
        '''
        return LAlg(~self.IPoly, -self.XPoly)

    def __mul__(self, other):
        if isinstance(other, LPoly):
            return LAlg(self.IPoly * other, self.XPoly * ~other)
        if not isinstance(other, LAlg):
            return LAlg(self.IPoly * other, self.XPoly * other)
        return LAlg(self.IPoly * other.IPoly - self.XPoly * (~other.XPoly),
                    self.IPoly * other.XPoly + self.XPoly * (~other.IPoly))

    @property
    def pnorm(self):
        return (self * (~self)).IPoly

    @property
    def unitarity(self):
        '''
        Unitarity of an element A is defined to be ||Id-a*~a||_2^2.
        '''
        return (LPoly([1]) - self.pnorm).norm

    @property
    def angle(self):
        '''
        For degree 0 element M, the corresponding angle is defined such that M \propto exp(angle * iX).
        '''
        assert (self.degree == 0), "deg = {}".format(self.degree)
        return numpy.angle(self.IPoly[0] + self.XPoly[0] * 1j)

    @property
    def left_and_right_angles(self):
        '''
        For a degree 1 element g(w) = exp(a * iX) * w * exp(b * iX), return the left and right rotation angles [a, b].
        Note that g(Id) = exp((a + b) * iX) and g(iZ) = exp((a-b) * iX) * iZ.
        '''
        assert self.degree == 1
        summation = numpy.angle(
            self.IPoly.eval(0)[0] +
            1j *
            self.XPoly.eval(0)[0])
        difference = numpy.angle(self.IPoly.eval(
            numpy.pi / 2)[0] - 1j * self.XPoly.eval(numpy.pi / 2)[0]) - numpy.pi / 2
        res = [(summation + difference) / 2, (summation - difference) / 2]
        return res

    @classmethod
    def rotation(cls, ang):
        '''
        Degree 0 rotation with a certain angle. Inverse of `LAlg.angle`.
        '''
        return LAlg(LPoly([numpy.cos(ang)]), LPoly([numpy.sin(ang)]))

    @property
    def curve(self):
        i, z = self.IPoly.curve
        y, x = self.XPoly.curve
        return numpy.angle(1j * i - y) * numpy.sqrt(1 - x**2 - z**2), z, x

    @classmethod
    def truncate(cls, g, dmin, dmax):
        return LAlg(LPoly.truncate(g.IPoly, dmin, dmax),
                    LPoly.truncate(g.XPoly, dmin, dmax))

    @classmethod
    def generator(cls, ang):
        '''
        Generator elements of the unitary group; each one is w conjugated by some rotation.
        '''
        return cls.rotation(ang) * w * cls.rotation(-ang)

    @classmethod
    def unitary_from_conjugations(cls, ang):
        '''
        Generating a special unitary element from the conjugation angles.
        '''
        res = Id
        for i in ang:
            res *= cls.generator(i)
        return res

    @classmethod
    def unitary_from_angles(cls, ang):
        '''
        Generating a unitary element from the rotation angles sandwiching w.
        '''
        res = cls.rotation(ang[0])
        for i in ang[1:]:
            res = res * w * cls.rotation(i)
        return res

# -----------------------------------------------------------------------------


def PolynomialToLaurentForm(coefs):
    '''
    Given coefficients of a polynomial in a = cos(theta), return the Laurent form of the polynomial, as an instance of LPoly
    coefs[0] = constant term, coefs[1] ~= a, coefs[2] ~= a^2, ...
    We substitute a = (w + 1/w)/2, based on w = exp[i * theta], where theta = arccos[a]
    Note i sqrt(1-a^2) = (w - 1/w)/2
    '''
    lp = LPoly([])
    for k, c in enumerate(coefs):
        if c == 0:
            continue
        if k == 0:
            lpcoefs = [1]
            dmin = 0
        else:
            lpcoefs = [1 / 2, 1 / 2]
            dmin = -1
        nlp = LPoly(lpcoefs, dmin)
        for j in range(k - 1):
            nlp = nlp * LPoly(lpcoefs, dmin)
        lp = lp + c * nlp
    return lp

# -----------------------------------------------------------------------------
# Definition of elements


Id = LPoly([1])
w = LPoly([1], 1)
iX = LAlg(XPoly=LPoly([1]))
