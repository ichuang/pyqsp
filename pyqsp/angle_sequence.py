import time
import numpy as np
import scipy.optimize
from pyqsp import LPoly
from pyqsp import completion
from pyqsp import decomposition
from pyqsp import response

def angle_sequence(p, eps=1e-4, suc=1-1e-4):
    """
    Solve for the angle sequence corresponding to the array p, with eps error budget and suc success probability.
    The bigger the error budget and the smaller the success probability, the better the numerical stability of the process.

    The array p specifies the coefficients of a Laurent polynomial, as powers of w: [ w^(-n), ..., w^(-2), w^(-1), const, w^1, w^2, ..., w^n ]
    This polynomial specifies the 0,0 component of the unitary.
    The QSP phases returned are for the Wz QSP convention (signal unitaries are Z-rotations and QSP phases are X-rotations).
    """
    p = LPoly.LPoly(p, -len(p) + 1)
    # Capitalization: eps/2 amount of error budget is put to the highest power for sake of numerical stability.
    p_new = suc * (p + LPoly.LPoly([eps / 4], p.degree) + LPoly.LPoly([eps / 4], -p.degree))

    # Completion phase
    t = time.time()
    g = completion.completion_from_root_finding(p_new)
    t_comp = time.time()
    print("Completion part finished within time ", t_comp - t)

    # Decomposition phase
    seq = decomposition.angseq(g)
    t_dec = time.time()
    print("Decomposition part finished within time ", t_dec - t_comp)
    print(seq)

    # Make sure that the reconstructed element lies in the desired error tolerance regime
    g_recon = LPoly.LAlg.unitary_from_angles(seq)
    final_error = (1/suc * g_recon.IPoly - p).inf_norm
    print(f"Final error = {final_error}")
    if  final_error < eps:
        return seq
    else:
        raise ValueError("The angle finding program failed on given instance, with an error of {}. Please relax the error budget and/ or the success probability.".format(final_error))

def QuantumSignalProcessingPhases(pcoefs=None, max_nretries=1, tolerance=0.1, verbose=True, model="Wx"):
    '''
    Compute QSP phase angles for the specified polynomial, in the specified model.

    Model can be:
        Wx - phases do Z-rotations and signal does X-rotations
        Wz - phases do X-rotations and signal does Z-rotations

    return a list of floats, giving the QSP phases
    '''
    if max_nretries > 1 and (pcoefs is not None):
        return QuantumSignalProcessingPhasesOptimizer(pcoefs,
                                                      max_nretries=max_nretries,
                                                      tolerance=tolerance,
                                                      verbose=verbose,
                                                      model=model)
    
    if model=="Wx":
        return QuantumSignalProcessingWxPhases(pcoefs, max_nretries=max_nretries, tolerance=tolerance, verbose=verbose)        
    elif model=="Wz":
        return QuantumSignalProcessingWzPhases(pcoefs, max_nretries=max_nretries, tolerance=tolerance, verbose=verbose)
    else:
        raise Exception(f"[QuantumSignalProcessingPhases] Unknown model {model}: must be Wx or Wz")

def QuantumSignalProcessingPhasesOptimizer(pcoefs=None, max_nretries=1, tolerance=0.1, verbose=True, model="Wx"):
    '''
    Run QuantumSignalProcessingPhases until error tolerance reached or max number of retries exceeded
    '''
    best_phiset = None
    best_error = None
    for k in range(max_nretries):
        phiset = QuantumSignalProcessingPhases(pcoefs=pcoefs, max_nretries=0, verbose=False, model=model)
        qspr = response.ComputeQSPResponse(phiset, model="Wx", npts=100)
        adat = qspr['adat']
        pdat = qspr['pdat']
        poly = np.polynomial.Polynomial(pcoefs)
        expected = poly(adat)
        if 0:
            def error_func(b):
                return abs(expected - np.real(pdat * np.exp((0+1j)*b))).mean()
            res = scipy.optimize.minimize(error_func, (0.2,), bounds=[(0, 2*np.pi)])
            bmin = res.x
            phiset[0] = phiset[0] + bmin
            qspr = response.ComputeQSPResponse(phiset, model=model, npts=100)
            pdat = qspr['pdat']
            avg_error = abs(expected - np.real(pdat)).mean()
        else:
            avg_error = abs(expected - np.real(pdat)).mean()
        print(f"[QuantumSignalProcessingPhases]                    avg_error={avg_error}")
        if avg_error < tolerance:
            print(f"[QuantumSignalProcessingPhases]     QSP angles = {phiset}")
            return phiset

        if (not best_error) or avg_error < best_error:
            best_error = avg_error
            best_phiset = phiset
    phiset = best_phiset
    print(f"[QuantumSignalProcessingPhases] failed to obtain phases with avg_error less than tolerance {tolerance}, aborting with best set, err={best_error}")
    print(f"[QuantumSignalProcessingPhases]    QSP angles = {phiset}")
    return phiset
    

def QuantumSignalProcessingWzPhases(pcoefs=None, max_nretries=1, tolerance=0.1, verbose=True):
    '''
    Generate QSP phase angles for the Laurent polynomial specified by pcoefs.
    '''
    return angle_sequence(pcoefs)

def QuantumSignalProcessingWxPhases(pcoefs=None, laurent_poly=None, max_nretries=1, tolerance=0.1, verbose=True):
    '''
    Take a polynomial P(a) as specified by pcoefs, convert to Laurent Poly, 
    complete to find Q Laurent Poly, perform Hadamard transform to get P' = P + Q.  
    Generate QSP phases for P'
    These phases should be the QSP phases for the W(x) = Rx convention of QSP.

    pcoefs       - a list, with coefficients for [constant, a, a^2, a^3, ...]
    laurent_poly - if provided, use this instead of pcoefs

    The QSP phases returned are for the Wx QSP convention (signal unitaries are X-rotations and QSP phases are Z-rotations)
    '''
    if max_nretries > 1 and (pcoefs is not None):
        return QuantumSignalProcessingPhasesOptimizer(pcoefs,
                                                      max_nretries=max_nretries,
                                                      tolerance=tolerance,
                                                      verbose=verbose,
                                                      model=model)

    if laurent_poly is not None:
        Plp = laurent_poly
    else:
        if pcoefs is None:
            pcoefs = [0, -3, 0, 4]		# default, as an example
        Plp = LPoly.PolynomialToLaurentForm(pcoefs)

    eps = 1.0e-4
    suc = 1 - 1.0e-4
    p = Plp

    # Capitalization: eps/2 amount of error budget is put to the highest power for sake of numerical stability.
    p_new = suc * (p + LPoly.LPoly([eps / 4], p.degree) + LPoly.LPoly([eps / 4], -p.degree))

    if verbose:
        print(f"Laurent P poly {Plp}")
    Qalg = completion.completion_from_root_finding(p_new)
    Qlp = Qalg.XPoly
    if verbose:
        print(f"Laurent Q poly {Qlp}")
    
    Pprime_lp = Plp + Qlp
    Qprime_lp = Plp - Qlp
    if verbose:
        print(f"Laurent Pprime poly {Pprime_lp}")

    g = LPoly.LAlg(Pprime_lp, Qprime_lp)
    # g = completion.completion_from_root_finding(Pprime_lp)
    seq = decomposition.angseq(g)

    g_recon = LPoly.LAlg.unitary_from_angles(seq)
    final_error = (1/suc * g_recon.IPoly - Pprime_lp).inf_norm
    print(f"[QuantumSignalProcessingWxPhases] Error in reconstruction from QSP angles = {final_error}")
    if verbose:
        print(f"QSP angles = {seq}")
    seq = np.array(seq)
    # seq[0] = seq[0] - 0.25
    return(seq)
