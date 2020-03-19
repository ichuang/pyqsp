import completion
import decomposition
import LPoly
import time

def angle_sequence(p, eps=1e-4, suc=1-1e-4):
    """
    Solve for the angle sequence corresponding to the array p, with eps error budget and suc success probability.
    The bigger the error budget and the smaller the success probability, the better the numerical stability of the process.
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
    if  final_error < eps:
        return seq
    else:
        raise ValueError("The angle finding program failed on given instance, with an error of {}. Please relax the error budget and/ or the success probability.".format(final_error))