import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def ComputeQSPResponse(
        phiset,
        model="Wx",
        npts=100,
        align_first_point_phase=True,
        positive_only=False):
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
    sz = np.matrix([[1, 0], [0, -1]])
    sx = np.matrix([[0, 1], [1, 0]])
    H = (sx + sz) / np.sqrt(2)
    if model in ["Wx", "WxH"]:
        s_phase = sz
        p_state = np.matrix([[1], [1]]) / np.sqrt(2)
    elif model == "Wz":
        s_phase = sx
        p_state = np.matrix([[1], [0]])
    else:
        raise Exception(
            f"[PlotQSPRsponse] model={model} unknown - must be Wx (signal is x-rot) or Wz (signal is z-rot)")
    i = (0 + 1j)
    pmats = []
    for phi in phiset:
        pmats.append(scipy.linalg.expm(i * phi * s_phase))
    # print(f"pm[-1] = {pmats[-1]}")
    for a in adat:
        ao = i * np.sqrt(1 - a**2)
        if model in ["Wx", "WxH"]:
            W = np.matrix([[a, ao], [ao, a]])
        elif model == "Wz":
            W = np.matrix([[a, ao], [ao, a]])
            W = H @ W @ H
            # W = np.matrix([[a, 0], [0, -i * ao]])
        U = pmats[0]
        for pm in pmats[1:]:
            U = U @ W @ pm
        if model == "WxH":
            U = H @ U @ H
        pdat.append((p_state.T @ U @ p_state)[0, 0])

    pdat = np.array(pdat, dtype=np.complex128)
    if align_first_point_phase:
        pdat = pdat * \
            np.exp(i * np.arctan2(np.imag(pdat[0]), np.real(pdat[0])))

    ret = {'adat': adat,
           'pdat': pdat,
           'model': model,
           'phiset': phiset,
           }
    return ret


def PlotQSPResponse(
        phiset,
        model="Wx",
        npts=100,
        pcoefs=None,
        target=None,
        show=True,
        align_first_point_phase=False,
        plot_magnitude=False,
        plot_positive_only=False,
        plot_real_only=False,
        plot_tight_y=False):
    '''
    Generate plot of QSP response function polynomial, i.e. Re( <0| U |0> )
    For values of model, see ComputeQSPResponse.

    pcoefs - coefficients for expected polynomial response; will be plotted, if provided
    target - reference function, if provided
    align_first_point_phase - if True, change the complex phase of phase such that the first point has phase angle zero
    plot_magnitude - if True, show magnitude instead of real and imaginary parts
    plot_positive_only - if True, then only show positive ordinate values
    plot_tight_y - if True, set y-axis scale to be from min to max of real part; else go from +1.5 max to -1.5 max
    '''
    qspr = ComputeQSPResponse(
        phiset,
        model,
        npts,
        align_first_point_phase=align_first_point_phase,
        positive_only=plot_positive_only)
    adat = qspr['adat']
    pdat = qspr['pdat']

    plt.figure(figsize=[8, 5])

    if pcoefs is not None:
        poly = np.polynomial.Polynomial(pcoefs)
        expected = poly(adat)
        plt.plot(adat, expected, 'b', label="target polynomial")

    if target is not None:
        L = np.max(np.abs(adat))
        xref = np.linspace(-L, L, 101)
        plt.plot(xref, target(xref), 'k', label="target function")

    if plot_magnitude:
        plt.plot(adat, abs(pdat), 'b', label="abs(P)")
    else:
        plt.plot(adat, np.real(pdat), 'r', label="Re(P)")
        if not plot_real_only:
            plt.plot(adat, np.imag(pdat), 'g', label="Im(P)")
    #plt.plot(adat, abs(pdat), 'k')

    # format plot
    plt.ylabel("response")
    plt.xlabel("a")
    plt.legend(loc="upper right")

    ymax = np.max(np.abs(np.real(pdat)))
    ymin = np.min(np.abs(np.real(pdat)))
    plt.xlim([np.min(adat), np.max(adat)])
    if plot_tight_y:
        plt.ylim([1.05 * ymin, 1.05 * ymax])
    else:
        plt.ylim([-1.5 * ymax, 1.5 * ymax])

    if show:
        plt.show()


def PlotQSPPhases(phiset, show=True):
    '''
    Generate plot of QSP response function polynomial, i.e. Re( <0| U |0> )
    For values of model, see ComputeQSPResponse.

    pcoefs - coefficients for expected polynomial response; will be plotted, if provided
    target - reference function, if provided
    '''
    plt.figure(figsize=[8, 5])

    plt.stem(phiset, markerfmt='bo', basefmt='k-')
    plt.xlabel("k")
    plt.ylabel("phi_k")
    plt.ylim([-np.pi, np.pi])

    if show:
        plt.show()
