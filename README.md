# Quantum Signal Processing

## Introduction

[Quantum signal processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) is a framework for quantum algorithms including Hamiltonian simulation, quantum linear system solving, amplitude amplification, etc. 

Quantum signal processing performs spectral transformation of any unitary $U$, given access to an ancilla qubit, a controlled version of $U$ and single-qubit rotations on the ancilla qubit. It first truncates an arbitrary spectral transformation function into a Laurent polynomial, then finds a way to decompose the Laurent polynomial into a sequence of products of controlled-$U$ and single qubit rotations on the ancilla. Such routines achieve optimal gate complexity for many of the quantum algorithmic tasks mentioned above.

This python package genrates QSP phase angles using the code based on [Finding Angles for Quantum Signal Processing with Machine Precision](https://arxiv.org/abs/2003.02831), and extending the original code for QSP phase angle calculation, at https://github.com/alibaba-edu/angle-sequence.  Two QSP model conventions are used in the literature: Wz, where the signal is a Z-rotation and QSP phase shifts are X-rotations, and Wx, where the signal is a X-rotation and QSP phase shifts are Z-rotations.  Specifically, in the Wx convention, the QSP operation sequence is:

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

This package can generate QSP phase angles for both conventions (whereas the original code only handles the Wz convention).  The challenge is that if one wants a certain polynomial  <!-- P_x(a) = \langle 0|U_x|0\rangle --> <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20P_x%28a%29%20%3D%20%5Clangle%200%7CU_x%7C0%5Crangle"/> in the Wx convention, one cannot just use the phases generated for this polynomial in the Wz convention.  Instead, first the Q_x(a) corresponding to P_x(a) is needed to complete the full U_x.  This then gives <!-- P_z(a) = \langle 0|U_z|0\rangle = P_x(a) + Q_x(a) --> <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20P_z%28a%29%20%3D%20%5Clangle%200%7CU_z%7C0%5Crangle%20%3D%20P_x%28a%29%20&plus;%20Q_x%28a%29" />.  Computing the QSP phases for P_z(a) in the Wz convention then gives the desired QSP phases for P_x(a) in the Wx convention.

This package also plots the QSP response function, and can be run from the command line.

![Example QSP response function for 1/a](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-one-over-x-kappa3.png?raw=true)

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
usage: pyqsp [-h] [-v] [-o OUTPUT] [--model MODEL] [--plot] [--hide-plot] [--return-angles] [--poly POLY] [--tau TAU] [--kappa KAPPA] [--epsilon EPSILON]
             [--align-first-point-phase]
             cmd

usage: pyqsp [options] cmd

Version: 0.0.1
Commands:

    poly2angles - compute QSP phase angles for the specified polynomial (use --poly)
    hamsim      - compute QSP phase angles for Hamiltonian simulation using the Jacobi-Anger expansion of exp(-i tau sin(2 theta))
    invert      - compute QSP phase angles for matrix inversion, i.e. a polynomial approximation to 1/a, for given kappa and epsilon parameter values

Examples:

    pyqsp --poly=-1,0,2 poly2angles
    pyqsp --poly=-1,0,2 --plot --align-first-point-phase poly2angles
    pyqsp --model=Wz --poly=0,0,0,1 --plot  poly2angles
    pyqsp --plot --tau 10 hamsim

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
  --poly POLY           comma delimited list of floating-point coeficients for polynomial, as const,a,a^2,...
  --tau TAU             time value for Hamiltonian simulation (hamsim command)
  --kappa KAPPA         parameter for polynomial approximation to 1/a, valid in the regions 1/kappa < a < 1 and -1 < a < -1/kappa
  --epsilon EPSILON     parameter for polynomial approximation to 1/a, giving bound on error
  --align-first-point-phase
                        when plotting change overall complex phase such that the first point has zero phase
```
