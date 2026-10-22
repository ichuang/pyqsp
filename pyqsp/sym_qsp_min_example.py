# Import relevant modules and methods.
import numpy as np
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries

# Specify definite-parity target function for QSP.
func = lambda x: np.cos(3*x)
polydeg = 12 # Desired QSP protocol length.
max_scale = 0.9 # Desired magnitude at the estimated extremum.

"""
Fit Chebyshev coefficients through degree 'polydeg' at 2*polydeg nodes.
This sampling rule does not rule out aliasing for an arbitrary callable.
"""
poly, scale = PolyTaylorSeries().taylor_series(
    func=func,
    degree=polydeg,
    max_scale=max_scale,
    return_scale=True,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)
# max_scale is an estimated magnitude; scale is the multiplier actually used.
true_func = lambda x: scale * func(x)

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
