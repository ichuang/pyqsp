import time

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev

from pyqsp.completion import completion_from_root_finding
from pyqsp.decomposition import angseq
from pyqsp.LPoly import LAlg, LPoly
from pyqsp.poly import StringPolynomial, TargetPolynomial
from pyqsp.response import ComputeQSPResponse
from pyqsp.sym_qsp_opt import newton_solver

class AngleFindingError(Exception):
    """Raised when angle finding step fails."""
    pass


def angle_sequence(p, eps=1e-4, suc=1 - 1e-4):
    """
    Solve for the angle sequence corresponding to the array p, with eps error budget and suc success probability.
    The bigger the error budget and the smaller the success probability, the better the numerical stability of the process.
    The array p specifies the coefficients of a Laurent polynomial, as powers of w: [ w^(-n), ..., w^(-2), w^(-1), const, w^1, w^2, ..., w^n ]
    This polynomial specifies the 0,0 component of the unitary.
    The QSP phases returned are for the Wz QSP convention (signal unitaries are Z-rotations and QSP phases are X-rotations).
    """
    p = LPoly(p, -len(p) + 1)

    # Capitalization: eps/2 amount of error budget is put to the highest power
    # for sake of numerical stability.
    p_new = suc * \
        (p + LPoly([eps / 4], p.degree) +
         LPoly([eps / 4], -p.degree))

    # Completion phase
    t = time.time()
    g = completion_from_root_finding(p_new.coefs)
    t_comp = time.time()
    print("Completion part finished within time ", t_comp - t)

    # Decomposition phase
    seq = angseq(g)
    t_dec = time.time()
    print("Decomposition part finished within time ", t_dec - t_comp)
    # print(seq)

    # Make sure that the reconstructed element lies in the desired error
    # tolerance regime
    g_recon = LAlg.unitary_from_angles(seq)
    final_error = (1 / suc * g_recon.IPoly - p).inf_norm
    print(f"Final error = {final_error}")
    if final_error < eps:
        return seq
    else:
        raise ValueError(
            "The angle finding program failed on given instance, with an error of {}. Please relax the error budget and/ or the success probability.".format(final_error))


def poly2laurent(pcoefs):
    """
    Convert polynomial coefficients to Laurent coefficients for polynomials
    with definite parity.

    Args:
        pcoefs: array of polynomial coefficients

    Returns:
        lcoefs: corresponding parity restricted Laurent coefficients

    Raises:
        AngleFindingError: if pcoefs does not have definite parity.
    """
    # convert polynomial coefficients to laurent; as we are already in the Chebyshev basis, this transformation is trivial
    ccoefs = pcoefs
    # ccoefs = np.polynomial.chebyshev.poly2cheb(pcoefs) # Conversion in old monomial convention.

    # determine parity of polynomial
    is_even = np.max(np.abs(ccoefs[0::2])) > 1e-8
    is_odd = np.max(np.abs(ccoefs[1::2])) > 1e-8

    if is_even and is_odd:
        raise AngleFindingError(
            "Polynomial must have definite parity: {}".format(str(pcoefs)))

    if is_odd:
        lcoefs = ccoefs[1::2] / 2
        lcoefs = np.r_[lcoefs[::-1], lcoefs]
    else:
        lcoefs = ccoefs[0::2] / 2
        lcoefs = np.r_[lcoefs[-1:0:-1], 2 * lcoefs[0], lcoefs[1:]]

    return lcoefs

def QuantumSignalProcessingPhases(
        poly,
        eps=1e-4,
        suc=1 - 1e-4,
        signal_operator="Wx",
        measurement=None,
        tolerance=1e-6,
        method="laurent",
        chebyshev_basis=False,
        **kwargs):
    """
    Generate QSP phase angles for a given polynomial.

    Args:
        poly: polynomial object
        eps: capitilization parameter for numerical stability
        suc: scaling factor for numerical stability
        signal_operator: QSP signal-dependent operation ['Wx', 'Wz']
        measurement: measurement basis (defaults to signal operator basis)
        tolerance: error tolerance in final reconstruction
        method: method to use for computing phase angles ['laurent', 'tf']

    Returns:
        Array of QSP angles.

    Raises:
        CompletionError: Raised when polynomial cannot be completed in given
            model.
        AngleFindingError: Raised if angle finding algorithm cannot find
        sequence to specified tolerance.
        ValueError: Raised if invalid model (or method) is specified.
    """

    # As all inputs are assured to be in the Chebyshev basis, we check whether we have been given a Chebyshev object or an ndarray of coefficients, and cast to Chebyshev object.
    if isinstance(poly, np.ndarray) or isinstance(poly, list):
        poly = Chebyshev(poly)
    elif isinstance(poly, Chebyshev):
        pass
    else:
        raise ValueError(
            f"Must provide Chebyshev coefficient list/array, or Chebyshev polynomial object.")

    # Note: as all polynomials are now guaranteed to be in the Chebyshev basis, we bypass the conditional below.
    # if not chebyshev_basis:
    #     if isinstance(poly, np.ndarray) or isinstance(poly, list):
    #         poly = Polynomial(poly)
    #     # For TargetPolynomial class, cast back to parent object.
    #     elif isinstance(poly, Polynomial):
    #         poly = Polynomial(poly.coef)
    #     else:
    #         raise ValueError(
    #             f"Must provide coefficient list/array, or Polynomial object.")
    # else:
    #     if isinstance(poly, np.ndarray) or isinstance(poly, list):
    #         poly = Chebyshev(poly)
    #     elif isinstance(poly, Chebyshev):
    #         # Currently, there should be no sub-class that we have to cast from in this branch.
    #         pass
    #     else:
    #         raise ValueError(
    #             f"Must provide Chebyshev coefficient list/array, or Chebyshev polynomial object.")

    if measurement is None:
        if signal_operator == "Wx":
            measurement = "x"
        elif signal_operator == "Wz":
            measurement = "z"

    if method == "tf":
        if chebyshev_basis:
            raise ValueError("Working with the `tf` method requires setting chebyshev_basis to False, or not specifying it.")
        else:
            pass
        if not signal_operator == "Wx":
            raise ValueError(
                f"Must use Wx signal operator model with tf method")
        else:
            pass
        return QuantumSignalProcessingPhasesWithTensorflow(poly,
                                                           measurement=measurement,
                                                           **kwargs)
    elif method == "laurent":
        model = (signal_operator, measurement)

        if chebyshev_basis:
            raise ValueError("Working with the `laurent` method requires setting chebyshev_basis to False, or not specifying it.")
        else:
            pass

        # Perform completion
        if model in {("Wx", "x"), ("Wz", "z")}:
            # Capitalization: eps/2 amount of error budget is put to the highest power for sake of numerical stability. We are assured to be in the Chebyshev basis.
            poly = suc * \
                 (poly + Chebyshev([0, ] * poly.degree() + [eps / 2, ]))

            lcoefs = poly2laurent(poly.coef)
            lalg = completion_from_root_finding(lcoefs, coef_type="F")
        elif model == ("Wx", "z"):
            # Unlike in the Laurent picture, the 'P' type coefficients are expected in the monomial basis. Note this is now the only setting in which we need to convert back from the Chebyshev basis to the expected monomial basis, and should be used rarely.
            monomial_coef = np.polynomial.chebyshev.cheb2poly(poly.coef)
            lalg = completion_from_root_finding(monomial_coef, coef_type="P")
        else:
            raise ValueError(
                "Invalid model: {}".format(str(model))
            )

        # Decomposition phase
        phiset = angseq(lalg)

        # Verify by reconstruction
        adat = np.linspace(-1., 1., 100)
        response = ComputeQSPResponse(
            adat,
            phiset,
            signal_operator=signal_operator,
            measurement=measurement)["pdat"]
        expected = poly(adat)

        max_err = np.max(np.abs(response - expected))
        if max_err > tolerance:
            raise AngleFindingError(
                "The angle finding program failed on given instance, with an error of {}. Please relax the error budget and / or the success probability.".format(max_err))

        return phiset

    # New branch for third, symmetric QSP based phase-finding method.
    elif method == "sym_qsp":
        # This method requires working in the Chebyshev basis.
        if not chebyshev_basis:
            raise ValueError("Working with the `sym_qsp` method requires setting chebyshev_basis to True.")
        else:
            pass

        # Gather coefficients from pre-converted Chebyshev object above.
        coefs = poly.coef

        # Check that the (non-reduced) phases have definite parity.
        is_even = np.max(np.abs(coefs[0::2])) > 1e-8
        is_odd = np.max(np.abs(coefs[1::2])) > 1e-8

        # We could allow for the zero polynomial, but choose not to.
        if (is_even and is_odd) or (not is_even and not is_odd):
            raise AngleFindingError(
                "Polynomial must have definite parity and be non-zero: {}".format(str(coefs)))

        # Set parity, and reduce Chebyshev coefs accordingly.
        parity = 0 if is_even else 1
        reduced_coefs = coefs[parity::2]

        # Call main method, currently with `crit` hardcoded.
        (phases, err, total_iter, qsp_seq_opt) = newton_solver(reduced_coefs, parity, crit=1e-12)

        """
        Note that this method currently ignores choice of signal,
        measurement, and eps, suc, tolerance. These can eventually be matched with the known internal parameters for the sym_qsp method, but many of them are instroduced for numerical stability in root finding methods, which are deprecated here.

        TODO: we can also provide an internal check for good
        agreement with the requested polynomial.

        adat = np.linspace(-1., 1., 100)
        response = ComputeQSPResponse(
            adat,
            phiset,
            signal_operator=signal_operator,
            measurement=measurement)["pdat"]
        expected = poly(adat)

        max_err = np.max(np.abs(response - expected))
        if max_err > tolerance:
            raise AngleFindingError(
                "The angle finding program failed on given instance, with an error of {}. Please relax the error budget and / or the success probability.".format(max_err))
        """

        # For now, return reduced, full phases, and parity together.
        return (qsp_seq_opt.full_phases, phases, parity)

    else:
        raise ValueError(f"Invalid method {method}")

def QuantumSignalProcessingPhasesWithTensorflow(
        poly,
        npts_theta=30,
        nepochs=5000,
        verbose=0,
        measurement="z",
        return_all=False):
    '''
    Compute QSP phase angles using optimization, with tensorflow, via the qsp_model submodule
    Running this imports pyqsp.qsp_models
    This import is done in the procedure because qsp_models requires tensorflow, and is heavyweight.
    We want to avoid requiring tensorflow to run pyqsp, if this is not needed.

    Args:
        poly: polynomial object, or StringPolynomial instance
        npts_theta: number of points to discretize theta axis into
        nepochs: number of epochs of training to do
        verbose: flag to give verbose debugging output during training
        return_all: if True, return dict with model, training history, and more

    Returns:
        Array of QSP angles (or if return_all: dict with model, training history, ...)
    '''
    import pyqsp.qsp_models as qsp_models
    import tensorflow as tf

    if not (
        isinstance(
            poly,
            Polynomial) or isinstance(
            poly,
            StringPolynomial)):
        raise ValueError(
            f"poly={poly} should be a Polynomial or StringPolynomial")

    poly_deg = poly.degree()

    # The intput theta training values
    th_in = np.arange(0, np.pi, np.pi / npts_theta)
    th_in = tf.reshape(th_in, (th_in.shape[0], 1))

    # The desired real part of p(x) which is the upper left value in the unitary of the qsp sequence
    # and the desired real part of q(x) = 0
    expected_outputs = [poly(np.cos(th_in)), np.zeros(th_in.shape[0])]

    # the tensorflow keras model
    model = qsp_models.construct_qsp_model(poly_deg, measurement=measurement)
    history = model.fit(
        x=th_in,
        y=expected_outputs,
        epochs=nepochs,
        verbose=verbose)
    phis = model.trainable_weights[0].numpy()
    if return_all:
        data = {'model': model,
                'history': history,
                'phis': phis,
                'th_in': th_in,
                'expected_outputs': expected_outputs,
                'poly': poly,
                }
        return data

    return phis
