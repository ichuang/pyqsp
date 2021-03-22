# Quantum Signal Processing

## Introduction

[Quantum signal processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) is a framework for quantum algorithms including Hamiltonian simulation, quantum linear system solving, amplitude amplification, etc. 

Quantum signal processing performs spectral transformation of any unitary $U$, given access to an ancilla qubit, a controlled version of $U$ and single-qubit rotations on the ancilla qubit. It first truncates an arbitrary spectral transformation function into a Laurent polynomial, then finds a way to decompose the Laurent polynomial into a sequence of products of controlled-$U$ and single qubit rotations on the ancilla. Such routines achieve optimal gate complexity for many of the quantum algorithmic tasks mentioned above.

This python package genrates QSP phase angles using the code based on [Finding Angles for Quantum Signal Processing with Machine Precision](https://arxiv.org/abs/2003.02831), in two different QSP model conventions: Wz, where the signal is a Z-rotation and QSP phase shifts are X-rotations, and Wx, where the signal is a X-rotation and QSP phase shifts are Z-rotations.

The package also plots the QSP response function, and can be run from the command line.

This extends the original code for QSP phase angle calculation, at https://github.com/alibaba-edu/angle-sequence

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
