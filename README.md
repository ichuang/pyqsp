# Quantum Signal Processing

## Introduction

[Quantum signal processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) is a framework for quantum algorithms including Hamiltonian simulation, quantum linear system solving, amplitude amplification, etc. 

Quantum signal processing performs spectral transformation of any unitary $U$, given access to an ancilla qubit, a controlled version of $U$ and single-qubit rotations on the ancilla qubit. It first truncates an arbitrary spectral transformation function into a Laurent polynomial, then finds a way to decompose the Laurent polynomial into a sequence of products of controlled-$U$ and single qubit rotations (by certain "QSP phase angles") on the ancilla. Such routines achieve optimal gate complexity for many of the quantum algorithmic tasks mentioned above.  The task achieved is essentially entirely defined by the QSP phase angles employed in the QSP operation sequence, and as such a central part is finding these QSP phase angles, given the desired Laurent polynomial.

This python package genrates QSP phase angles using the code based on [Finding Angles for Quantum Signal Processing with Machine Precision](https://arxiv.org/abs/2003.02831), and extending the original code for QSP phase angle calculation, at https://github.com/alibaba-edu/angle-sequence.  Two QSP model conventions are used in the literature: Wx, where the signal W(a) is an X-rotation and QSP phase shifts are Z-rotations, and Wz, where the signal W(a) is a Z-rotation and QSP phase shifts are X-rotations.

Specifically, in the Wx convention, the QSP operation sequence is:

<!-- U_x = e^{i\phi_0 Z} \sum_{k=1}^L W(a) e^{i\phi_k Z} ~~~~{\rm where} ~~~ W(x)=\left[ \begin{array}{cc} a & i\sqrt{1-a^2} \\ i\sqrt{1-a^2} & a  \end{array} \right ] -->

<center>
<img src="https://latex.codecogs.com/svg.latex?%5Clarge%20U_x%20%3D%20e%5E%7Bi%5Cphi_0%20Z%7D%20%5Csum_%7Bk%3D1%7D%5EL%20W%28a%29%20e%5E%7Bi%5Cphi_k%20Z%7D%20%7E%7E%7E%7E%7B%5Crm%20where%7D%20%7E%7E%7E%20W%28a%29%3D%5Cleft%5B%20%5Cbegin%7Barray%7D%7Bcc%7D%20a%20%26%20i%5Csqrt%7B1-a%5E2%7D%20%5C%5C%20i%5Csqrt%7B1-a%5E2%7D%20%26%20a%20%5Cend%7Barray%7D%20%5Cright%20%5D" />
</center>

And in the Wz convention, the QSP operation sequence is:

<!-- U_z = e^{i\phi_0 X} \sum_{k=1}^L W(a) e^{i\phi_k X} ~~~~{\rm where} ~~~ W(a)=\left[ \begin{array}{cc} a + i\sqrt{1-a^2} & 0 \\ 0 & a - i\sqrt{1-a^2}  \end{array} \right ] -->

<center>
<img src="https://latex.codecogs.com/svg.latex?%5Clarge%20U_z%20%3D%20e%5E%7Bi%5Cphi_0%20X%7D%20%5Csum_%7Bk%3D1%7D%5EL%20W%28a%29%20e%5E%7Bi%5Cphi_k%20X%7D%20%7E%7E%7E%7E%7B%5Crm%20where%7D%20%7E%7E%7E%20W%28a%29%3D%5Cleft%5B%20%5Cbegin%7Barray%7D%7Bcc%7D%20a%20&plus;%20i%5Csqrt%7B1-a%5E2%7D%20%26%200%20%5C%5C%200%20%26%20a%20-%20i%5Csqrt%7B1-a%5E2%7D%20%5Cend%7Barray%7D%20%5Cright%20%5D" />
</center>

They are related by a Hadamard transform:

<!-- U_x = H U_z H -->

<center>
<img src="https://latex.codecogs.com/svg.latex?%5Clarge%20U_x%20%3D%20H%20U_z%20H" />
</center>

The Wz convention is convenient for and employed in [Laurent polynomial formulations of QSP](https://arxiv.org/abs/2003.02831), whereas the Wx convention is more traditional, e.g. as employed in [quantum singular value transform](https://arxiv.org/abs/1806.01838) applications of QSP.

This package can generate QSP phase angles for both conventions (whereas the original code only handles the Wz convention).  The challenge is that if one wants a certain polynomial  <!-- P_x(a) = \langle 0|U_x|0\rangle --> <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20P_x%28a%29%20%3D%20%5Clangle%200%7CU_x%7C0%5Crangle"/> in the Wx convention, one cannot just use the phases generated for this polynomial in the Wz convention.  Instead, first the Q_x(a) corresponding to P_x(a) is needed to complete the full U_x.  This then gives <!-- P_z(a) = \langle 0|U_z|0\rangle = P_x(a) + Q_x(a) --> <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20P_z%28a%29%20%3D%20%5Clangle%200%7CU_z%7C0%5Crangle%20%3D%20P_x%28a%29%20&plus;%20Q_x%28a%29" />.  Computing the QSP phases for P_z(a) in the Wz convention then gives the desired QSP phases for P_x(a) in the Wx convention.  Note that this is not entirely satisfactory, since there is a phase between P_x and Q_x which is left indeterminate, and Q_x may contain errors due to the sensitivity of the completion process to roots of the polynomials involved, but these issues can likely be fixed.

## Examples

This package also plots the QSP response function, and can be run from the command line.  In the example below, the blue line shows the target ideal polynomial QSP response function P_x(a); the red line shows the real part of the response achieved by the QSP phases, and the green line shows the imaginary part of the QSP response, with L=20.

![Example QSP response function for 1/a](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-one-over-x-kappa3.png?raw=true)

This was generated by running `pyqsp --plot invert`, which generated the following text output:

```
b=20, j0=10
[PolyOneOverX] minimum [-2.90002589] is at [-0.25174082]: normalizing
Laurent P poly 0.0002108924030879319 * w ^ (-21) + -0.0006894043260278334 * w ^ (-19) + 0.001994436843136674 * w ^ (-17) + -0.005148265426149545 * w ^ (-15) + 0.011941126989562179 * w ^ (-13) + -0.02504164571900347 * w ^ (-11) + 0.04774921151669176 * w ^ (-9) + -0.08322978307559126 * w ^ (-7) + 0.1333200017468883 * w ^ (-5) + -0.1973241700493915 * w ^ (-3) + 0.2714342596621151 * w ^ (-1) + 0.2714342596621151 * w ^ (1) + -0.1973241700493915 * w ^ (3) + 0.1333200017468883 * w ^ (5) + -0.08322978307559126 * w ^ (7) + 0.04774921151669176 * w ^ (9) + -0.02504164571900347 * w ^ (11) + 0.011941126989562179 * w ^ (13) + -0.005148265426149545 * w ^ (15) + 0.001994436843136674 * w ^ (17) + -0.0006894043260278334 * w ^ (19) + 0.0002108924030879319 * w ^ (21)
Laurent Q poly -2.784691352688532e-05 * w ^ (-21) + -3.467046922689296e-06 * w ^ (-19) + -0.00023447352272530086 * w ^ (-17) + 0.00018731079814912242 * w ^ (-15) + -0.0007159634932856078 * w ^ (-13) + 0.0032821300323626736 * w ^ (-11) + -0.001907488898700853 * w ^ (-9) + 0.012591630667616198 * w ^ (-7) + -0.02371053103573988 * w ^ (-5) + -0.016629567619072996 * w ^ (-3) + -0.18519029619384528 * w ^ (-1) + -0.20476480667058175 * w ^ (1) + -0.5374170003848409 * w ^ (3) + -0.2638332742746844 * w ^ (5) + 0.21212227077358078 * w ^ (7) + 0.27748656462486654 * w ^ (9) + -0.34978762979573563 * w ^ (11) + 0.17888109364117422 * w ^ (13) + -0.07649917245226195 * w ^ (15) + 0.03551004671728361 * w ^ (17) + -0.011926352140796492 * w ^ (19) + 0.001997855069006951 * w ^ (21)
Laurent Pprime poly 0.00018304548956104657 * w ^ (-21) + -0.0006928713729505227 * w ^ (-19) + 0.001759963320411373 * w ^ (-17) + -0.0049609546280004226 * w ^ (-15) + 0.01122516349627657 * w ^ (-13) + -0.021759515686640796 * w ^ (-11) + 0.0458417226179909 * w ^ (-9) + -0.07063815240797505 * w ^ (-7) + 0.10960947071114842 * w ^ (-5) + -0.2139537376684645 * w ^ (-3) + 0.08624396346826982 * w ^ (-1) + 0.06666945299153335 * w ^ (1) + -0.7347411704342324 * w ^ (3) + -0.1305132725277961 * w ^ (5) + 0.12889248769798953 * w ^ (7) + 0.3252357761415583 * w ^ (9) + -0.3748292755147391 * w ^ (11) + 0.1908222206307364 * w ^ (13) + -0.0816474378784115 * w ^ (15) + 0.037504483560420285 * w ^ (17) + -0.012615756466824325 * w ^ (19) + 0.0022087474720948828 * w ^ (21)
[QuantumSignalProcessingWxPhases] Error in reconstruction from QSP angles = 0.4862234440482484
QSP angles = [0.11177647198529908, 0.321286291382733, -2.449109005349844, 0.5475854935639934, -0.23068701778999645, 1.852723853663707, -0.1638598380129409, -0.031620025684050396, -0.3018780297835489, -1.8740083256355018, 0.6314873861312269, -0.036881744534211114, 1.300525766122238, 0.3073180685644643, -0.23974313142278042, 0.09536754986888613, -1.188342341212731, -0.6092537117168764, 0.4850840958947118, 0.8800119831449805, 0.4163653216360098, 2.480415494993986]
```

The sign function is also useful to approximate, e.g. for oblivious amplitude amplification.  Running
```
pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign poly
```
produces the QSP phase angles for a degree 19 polynomial approximation, using the error function of kappa * a, where kappa is 10, with a response function as shown in this plot:

![Example QSP response function approximating the sign function](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-sign-kappa-10-degree-19.png?raw=true)

A threshold function is useful, for example, for distinguishing eigenvalues and singular values.  Running
```
pyqsp --plot-real-only --plot --polyargs=20,20 --polyname poly_thresh poly
```
produces the QSP phase angles for a degree 20 polynomial approximation, using two error functions kappa 20, with a response function as shown in this plot:

![Example QSP response function approximating a threshold function](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-threshold-polynomial-degree-20-kappa-20.png?raw=true)


## Code design

* `angle_sequence.py` is the main module of the algorithm.
* `LPoly.py` defines two classes `LPoly` and `LAlg`, representing Laurent polynomials and Low algebra elements respectively.
* `completion.py` describes the completion algorithm: Given a Laurent polynomial element $F(\tilde{w})$, find its counterpart $G(\tilde{w})$ such that $F(\tilde{w})+G(\tilde{w})*iX$ is a unitary element.
* `decomposition.py` describes the halving algorithm: Given a unitary parity Low algebra element $V(\tilde{w})$, decompose it as a unique product of degree-0 rotations $\exp\{i\theta X\}$ and degree-1 monomials $w$.
* `ham_sim.py` shows an example of how the angle sequence for Hamiltonian simulation can be found.
* `response.py` computes QSP response functions and generates plots
* `poly.py` provides some utility polynomials, namely the approximation of 1/a using a linear combination of Chebyshev polynomials
* `main.py` is the main entry point for command line usage

A set of unit tests is also provided.

To find the QSP angle sequence corresponding to a real Laurent polynomial $A(\tilde{w}) = \sum_{i=-n}\^n a_i\tilde{w}^i$, simply run:

    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    ang_seq = QuantumSignalProcessingPhases([a_{-n}, a_{-n+2}, ..., a_n], model="Wz")
    print(ang_seq)

To find the QSP angle sequence corresponding to a real (non-Laurent) polynomial $A(x) = \sum_{i=0}\^n a_i x^i$, simply run:

    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    ang_seq = QuantumSignalProcessingPhases([a_{0}, a_{1}, ..., a_n], model="Wx")
    print(ang_seq)

## Command line usage

```
usage: pyqsp [-h] [-v] [-o OUTPUT] [--model MODEL] [--plot] [--hide-plot] [--return-angles] [--poly POLY] [--tau TAU] [--kappa KAPPA] [--degree DEGREE]
             [--epsilon EPSILON] [--seqname SEQNAME] [--seqargs SEQARGS] [--polyname POLYNAME] [--polyargs POLYARGS] [--align-first-point-phase]
             [--plot-magnitude] [--plot-real-only] [--plot-positive-only] [--plot-npts PLOT_NPTS] [--niter NITER] [--tolerance TOLERANCE]
             cmd

usage: pyqsp [options] cmd

Version: 0.0.3
Commands:

    poly2angles - compute QSP phase angles for the specified polynomial (use --poly)
    hamsim      - compute QSP phase angles for Hamiltonian simulation using the Jacobi-Anger expansion of exp(-i tau sin(2 theta))
    invert      - compute QSP phase angles for matrix inversion, i.e. a polynomial approximation to 1/a, for given kappa and epsilon parameter values
    angles      - generate QSP phase angles for the specified --seqname and --seqargs
    poly        - generate QSP phase angles for the specified --polyname and --polyargs, e.g. sign and threshold polynomials

Examples:

    pyqsp --poly=-1,0,2 poly2angles
    pyqsp --poly=-1,0,2 --plot --align-first-point-phase poly2angles
    pyqsp --model=Wz --poly=0,0,0,1 --plot  poly2angles
    pyqsp --plot --tau 10 hamsim
    pyqsp --plot --tolerance=0.01 invert
    pyqsp --plot-npts=4000 --plot-positive-only --plot-magnitude --plot --seqargs=1000,1.0e-20 --seqname fpsearch angles
    pyqsp --plot-npts=100 --plot-magnitude --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-npts=100 --plot-positive-only --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-real-only --plot --polyargs=20,20 --polyname poly_thresh poly
    pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign poly

positional arguments:
  cmd                   command

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity (add more -v to increase versbosity)
  -o OUTPUT, --output OUTPUT
                        output filename
  --model MODEL         QSP sequence model, either Wx (signal is X rotations) or Wz (signal is Z rotations)
  --plot                generate QSP response plot
  --hide-plot           do not show plot (but it may be saved to a file if --output is specified)
  --return-angles       return QSP phase angles to caller
  --poly POLY           comma delimited list of floating-point coeficients for polynomial, as const, a, a^2, ...
  --tau TAU             time value for Hamiltonian simulation (hamsim command)
  --kappa KAPPA         parameter for polynomial approximation to 1/a, valid in the regions 1/kappa < a < 1 and -1 < a < -1/kappa
  --degree DEGREE       parameter for polynomial approximation to erf(kappa*x)
  --epsilon EPSILON     parameter for polynomial approximation to 1/a, giving bound on error
  --seqname SEQNAME     name of QSP phase angle sequence to generate using the 'angles' command, e.g. fpsearch
  --seqargs SEQARGS     arguments to the phase angles generated by seqname (e.g. length,delta,gamma for fpsearch)
  --polyname POLYNAME   name of polynomial generate using the 'poly' command, e.g. 'sign'
  --polyargs POLYARGS   arguments to the polynomial generated by poly (e.g. degree,kappa for 'sign')
  --align-first-point-phase
                        when plotting change overall complex phase such that the first point has zero phase
  --plot-magnitude      when plotting only show magnitude, instead of separate imaginary and real components
  --plot-real-only      when plotting only real component, and not imaginary
  --plot-positive-only  when plotting only a-values (x-axis) from 0 to +1, instead of from -1 to +1 
  --plot-npts PLOT_NPTS
                        number of points to use in plotting
  --niter NITER         number of iterations to use in trying to compute phase angles
  --tolerance TOLERANCE
                        error tolerance for phase angle optimizer```
