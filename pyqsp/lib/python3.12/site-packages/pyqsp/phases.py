'''
pyqsp/phases.py

Known QSP phases for specific responses, in the Wx convention
'''

import numpy as np


class PhaseGenerator:
    '''
    Abstract base class for phase generators
    '''

    def __init__(self, verbose=True):
        self.verbose = verbose
        return

    def help(self):
        '''
        return help text
        '''
        return "help text about the expected sequence arguments"

    def generate(self):
        '''
        return list of floats specifying the QSP phase angles
        '''
        return [0, 0]


class FPSearch(PhaseGenerator):
    '''
    Return phases for fixed point quantum search, following https://arxiv.org/abs/1409.3305

    d = length of search sequence (length of phase vector returned = 2*d)
    delta:  error probability allowed = 1-delta^2, at the fixed point
    gamma: can be specified instead of delta (it is calculated if delta given)
    return_alpha: if True, then return alpha_k instead of phivec
    '''

    def help(self):
        txt = """Return phases for fixed point quantum search, following https://arxiv.org/abs/1409.3305
    Arguments are:
        d = length of search sequence (length of phase vector returned = 2*d)
        delta:  error probability allowed = 1-delta^2, at the fixed point
        gamma: can be specified instead of delta (it is calculated if delta given)"""

        return txt

    def generate(self, d, delta=None, gamma=None, return_alpha=False):
        d = int(d)
        L = 2 * d + 1
        print(
            f"[pyqsp.fixed_point_search] generating length {2*d} sequence in the Wx convention")
        kvec = np.arange(1, d + 1)
        if gamma is None:
            if delta is None:
                delta = 0.1
            # T_{1/L}(1/delta)
            gamma = 1 / np.cosh((1 / L) * np.arccosh(1 / delta))
        sg = np.sqrt(1 - gamma**2)
        if self.verbose:
            print("[phi_fp]: gamma=%s" % gamma)
        avec = 2 * np.arctan2(1, (np.tan(2 * np.pi * kvec / L) * sg))
        if return_alpha:
            return avec
        bvec = - avec[::-1]
        self.avec = avec
        self.bvec = bvec
        phivec = np.zeros(2 * d)
        for k in range(d):
            # reverse order & scale to match QSVT convention
            phivec[2 * k] = -avec[d - k - 1] / 2
            phivec[2 * k + 1] = bvec[d - k - 1] / 2

        return phivec


class erf_step(PhaseGenerator):
    def help(self):
        return """Step function polynomial using erf(), but only for specific pre-computed values.  Argument is n, where n may be 7 or 23"""

    def generate(self, n):
        # n=7 poly approximation to erf(1.5 x), defined using W(x) convention
        # made for re(P)
        phi_n7_erf = [1.58019, 0.00172821, 0.251897, -
                      0.834542, -0.834542, 0.251897, 0.00172821, 0.00939863]

        # n=23 poly approximation to erf(2 x)
        # made for re(P)
        phi_n23_erf = [1.5708, 2.87883E-8, 5.83909E-7, 1.84144E-6, 0.0000209995,
                       -0.0000120126, 0.000564903, -0.0022922, 0.0150024,
                       -0.064666, 0.263754, -0.926685, -0.926912, 0.263756,
                       -0.0645576, 0.014932, -0.00225885, 0.000553565,
                       -8.29284E-6, 0.0000200459, 2.09476E-6, 5.3026E-7,
                       4.38578E-8, 3.66401E-9]
        if n == 7:
            return phi_n7_erf
        elif n == 23:
            return phi_n23_erf
        raise Exception("[pyqsp.phases.erf_step] n must be 7 or 23")

# -----------------------------------------------------------------------------


phase_generators = {'fpsearch': FPSearch,
                    'erf_step': erf_step,
                    }
