# Import relevant modules and methods.
import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import (polynomial_generators, PolyTaylorSeries)

# Specify definite-parity target function for QSP.
func = lambda x: np.cos(3*x)
polydeg = 12 # Desired QSP protocol length.
max_scale = 0.9 # Maximum norm (<1) for rescaling.
true_func = lambda x: max_scale * func(x) # For error, include scale.

"""
Within PolyTaylorSeries class, compute /Chebyshev interpolant/ up to degree
'polydeg' (using twice as many Chebyshev nodes to prevent aliasing).
"""
poly = PolyTaylorSeries().taylor_series(
    func=func,
    degree=polydeg,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)

# Compute full phases (and reduced phases, parity) using symmetric QSP.
(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True)

"""
Plot response according to full phases.
Note that `pcoefs` are coefficients of the approximating polynomial,
while `target` is the true function (rescaled) being approximated.
"""
response.PlotQSPResponse(
    phiset,
    pcoefs=poly,
    target=true_func,
    sym_qsp=True,
    simul_error_plot=True)
