import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def ComputeQSPResponse(phiset, model="Wx", npts=100, align_first_point_phase=True, positive_only=False):
    '''
    Compute QSP response.

    Model can be:
        Wx - phases do z-rotations and signal does x-rotations
        Wz - phases do x-rotations and signal does z-rotations
        WxH - phases do z-rotations and signal does x-rotations, but conjugate by Hadamard at the end

    Return dict with 
    { adat: 1-d array of a-values,
      pdat: array of complex-valued U[0,0] values
      model: model
    }

    positive_only: if True, then only use positive a (polynomial argument) values
    '''
    if positive_only:
        adat = np.linspace(0, 1, npts)
    else:
        adat = np.linspace(-1, 1, npts)
    pdat = []
    sz = np.matrix([[1,0],[0,-1]])
    sx = np.matrix([[0,1],[1,0]])
    H = (sx + sz)/np.sqrt(2)
    if model in ["Wx", "WxH"]:
        s_phase = sz
    elif model=="Wz":
        s_phase = sx
    else:
        raise Exception(f"[PlotQSPRsponse] model={model} unknown - must be Wx (signal is x-rot) or Wz (signal is z-rot)")
    i = (0+1j)
    pmats = [np.exp(i * phiset[0]) * np.eye(2)]
    for phi in phiset[1:]:
        pmats.append( scipy.linalg.expm(i * phi * s_phase))
    # print(f"pm[-1] = {pmats[-1]}")
    for a in adat:
        ao = i * np.sqrt(1-a**2)
        if model in ["Wx", "WxH"]:
            W = np.matrix([[a, ao], [ao, a]])
        elif model=="Wz":
            W = np.matrix([[a, ao], [ao, a]])
            W = H @ W @ H
            # W = np.matrix([[a, 0], [0, -i * ao]])
        U = pmats[0]
        for pm in pmats[1:]:
            U = U @ W @ pm
        if model=="WxH":
            U = H @ U @ H
        pdat.append( U[0,0] )

    pdat = np.array(pdat)
    if align_first_point_phase:
        pdat = pdat * np.exp(i * np.arctan2(np.imag(pdat[0]), np.real(pdat[0])))

    ret = {'adat': adat,
           'pdat': pdat,
           'model': model,
           'phiset': phiset,
    }
    return ret

def PlotQSPResponse(phiset, model="Wx", npts=100, pcoefs=None, show=True, align_first_point_phase=True,
                    plot_magnitude=False, plot_positive_only=False, plot_real_only=False):
    '''
    Generate plot of QSP response function polynomial, i.e. Re( <0| U |0> )
    For values of model, see ComputeQSPResponse.

    pcoefs - coefficients for expected polynomial response; will be plotted, if provided
    align_first_point_phase - if True, change the complex phase of phase such that the first point has phase angle zero
    plot_magnitude - if True, show magnitude instead of real and imaginary parts
    plot_positive_only - if True, then only show positive ordinate values
    '''
    qspr = ComputeQSPResponse(phiset, model, npts, align_first_point_phase=align_first_point_phase, positive_only=plot_positive_only)
    adat = qspr['adat']
    pdat = qspr['pdat']

    if plot_magnitude:
        plt.plot(adat, abs(pdat), 'k')
        ytext = "magnitude (black)"
    elif plot_real_only:
        plt.plot(adat, np.real(pdat), 'r')
        ytext = "real (red)"
    else:
        plt.plot(adat, np.real(pdat), 'r')
        plt.plot(adat, np.imag(pdat), 'g')
        ytext = "real (red), imag (green)"

    # plt.plot(adat, abs(pdat), 'k')

    if pcoefs is not None:
        poly = np.polynomial.Polynomial(pcoefs)
        expected = poly(adat)
        plt.plot(adat, expected, 'b')
        ytext += ', blue=goal'

    plt.ylabel(ytext)
    plt.xlabel("a")
    plt.legend(loc="upper right")

    if show:
        plt.show()    
