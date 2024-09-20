# Import relevant modules and methods.
import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import (polynomial_generators, PolyTaylorSeries)

# Specify definite-parity target function for QSP.
func = lambda x: np.cos(3*x)
polydeg = 6 # Desired QSP protocol length.
max_scale = 0.9 # Maximum norm (<1) for rescaling.

# Within PolyTaylorSeries class, compute /Chebyshev interpolant/ up to a specified degree (using twice as many Chebyshev nodes).
poly = PolyTaylorSeries().taylor_series(
    func=func,
    degree=polydeg,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)

# Compute full phases using symmetric QSP method.
(phiset, reduced_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True)

# Plot response according to full phases.
response.PlotQSPResponse(
    phiset,
    target=poly,
    sym_qsp=True)
