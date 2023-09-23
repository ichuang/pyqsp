"""
This file contains notes on the steps needed to retrieve the QSP phases for the polynomial used in the correction protocol.
"""

#################################################
#################################################

"""
Both of these are currently contained in poly.py in the main repo.
"""
class PolyExtraction(PolyTaylorSeries):
    def help(self):
        return "approximation to the extraction function"

    def generate(
            self,
            degree=7):
        '''
        Approximation to the extraction function
        '''
        degree = int(degree)

        def extraction(x):
            return 1 / np.sqrt(1 - x ** 2)

        # Generates the coefficients
        pcoefs = np.array([0 if k % 2 == 1 else scipy.special.binom(-0.5, (k/2)) * (-1) ** (k/2) for k in range(degree + 1)])
        # force odd coefficients to be zero, since the polynomial must be odd
        pcoefs[1::2] = 0
        return TargetPolynomial(pcoefs, target=lambda x: extraction(x))

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

#################################################
#################################################

"""
The above is called by this in turn, located in phases.py; this is enough to solve our problem, as it gives the phases. In turn it depends only on PolyExtraction, which is a core QSP method.

This method also uses QuantumSignalProcessingPhases(), located in angle_sequence.py, which is also a core QSP method. With the right imports, this phi should be simply generable and insertable into gadget_assemblage.py.
"""
class ExtractionSequence(PhaseGenerator):
    """
    Extraction M-QSP protocol
    """
    def generate(self, n):
        """Phase angles"""
        Q = PolyExtraction().generate(degree=n)
        phi = QuantumSignalProcessingPhases(Q, poly_type="Q", measurement="z")
        return phi

#################################################
#################################################
"""
Remaining items to consider include the square root sequence, which is also in phases.py, and its related versions for inverse chebyshev.
"""

"""
The path to getting these phases is relatively simple; in the body of gadget_assemblage.py (or whichever series of files we end up breaking things into), we need to import ExtractionSequence from phases.py as well as qsp_models from pyqsp; these are already included, commented out.

Then calling phi = ExtractionSequence().generate(deg) within get_correction_phases() immediately gives us a list of angles, to be turned into the proper protocol.
"""