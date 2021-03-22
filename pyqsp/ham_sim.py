import time
import numpy
from scipy.special import jn
from pyqsp.LPoly import LPoly
from pyqsp.angle_sequence import angle_sequence

def hamiltonian_coefficients(tau, eps):
    n = 2*int(numpy.e / 4 * tau - numpy.log(eps) / 2)
    return jn(numpy.arange(-n, n + 1, 1), tau)

def ham_sim(tau, eps, suc):
    t = time.time()
    a = hamiltonian_coefficients(tau, eps / 10)
    return angle_sequence(a, .9 * eps, suc), time.time()-t

if __name__=="__main__":
    ham_sim(100, 1e-4, 1-1e-4)
