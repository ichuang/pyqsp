# :hammer_and_wrench: `pyQSP`
## A Python Package for Quantum Signal Processing

![test workflow](https://github.com/ichuang/pyqsp/actions/workflows/run_tests.yml/badge.svg)

<!-- *TODO: add some symbols for the main headings, add a title that matches the title of the repo, and overall remove extraneous lists of outputs and inputs outside of a dedicated example section.* -->

## Recent news and updates

> :round_pushpin: The codebase has recently been updated to work internally almost entirely with polynomials in the Chebyshev, rather than the monomial, basis. This improves the numerical stability of the `laurent` method substantially, and should present almost no visible differences for an end-user. Currently the `poly2angles` command-line function is the only major input for monomial basis targets, while the main `QuantumSignalProcessingPhases` method and various helper and plotting methods all accept only Chebyshev basis inputs with internal checks. For conversion between bases, see [below](#computing-qsp-phases-even-faster-with-the-sym-qsp-method).

## Introduction

[Quantum signal processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) (QSP) is a flexible quantum algorithm subsuming Hamiltonian simulation, quantum linear systems solvers, amplitude amplification, and many other common quantum algorithms. Moreover, for each of these applications, QSP often exhibits state-of-the-art space and time complexity, with comparatively simple analysis. QSP and its related algorithms, e.g., [Quantum singular value transformation](https://arxiv.org/abs/1806.01838) (QSVT), using basic alternating circuit ansätze originally inspired by composite pulse techniques in NMR, permit one to *transform the spectrum of linear operators by near arbitrary polynomial functions*, with the aforementioned good numerical properties and simple analytic treatment.

In their most basic forms, QSP/QSVT give a recipe for a desired spectral transformation of a unitary[^1] $U$, requiring access to an auxiliary qubit, a controlled version of $U$, and single-qubit rotations on the auxiliary qubit. A standard application of QSP/QSVT might look like the following:
- Given a function one wants to apply to the spectrum of a unitary, classically generate a (Laurent) polynomial that suitably approximates this function over a desired spectral range.
- Having computed a good polynomial approximation, and checking that it obeys certain mild conditions, use one among many efficient *classical* algorithms to compute the sequence of single-qubit rotations (the *QSP phases*) interspersing applications of the controlled unitary (the *QSP signal*) corresponding to the polynomial approximation.
- Run the corresponding sequence of gates as a quantum circuit, interleaving QSP signals and phases, followed by a measurement in a chosen basis.

> :round_pushpin: The theory of QSP is not only under active development, but comprises multiple subtly different conventions, each of which can use different terminology compared the barebones outline given here. These included conventions for how the *signal* is encoded, how the *phases* are applied, the basis to measure in, whether one desires to transform eigenvalues or singular values, whether the classical algorithm to find these phases is exact or iterative, and so on.
>
> Regardless, the basic scheme of QSP and QSVT is relatively fixed: given a specific circuit ansatz and a theory for the polynomial transformations achievable for that ansatz, generate those conditions and algorithms relating the *achieved function* and *circuit parameterization*. Understanding the bidirectional map between phases and polynomial transforms, as well as the efficiency of loading linear systems into quantum processes, constitutes most of the theory of these algorithms.

This package provides such conditions and algorithms, and automatically treats a few common conventions, with options to modify the code in basic ways to encompass others. These conventions are enumerated in the recent pedagogical work [A Grand Unification of Quantum Algorithms](https://arxiv.org/abs/2105.02859), and the QSP phase-finding algorithms we treat can be broken roughly into three types:
- :hammer: The `laurent` method employs techniques originated in [Finding Angles for Quantum Signal Processing with Machine Precision](https://arxiv.org/abs/2003.02831), and extends code from [its attached repository](https://github.com/alibaba-edu/angle-sequence.). This method exactly computes phases by studying the properties of the desired polynomials using a divide-and-conquer approach.
- :sparkles: The `tf` method employs TensorFlow + Keras, and finds QSP phase angles using straightforward (but sometimes slow) optimization techniques.
- :key: The `sym_qsp` method employs an iterative, quasi-Newton method to find QSP phases for a special, lightly-restricted sub-class of QSP protocols whose phases are symmetric. In comparison with the above two techniques, this method is almost invariably quick, numerically stable, and should suit most applications. It is based off work in [Efficient phase-factor evaluation in quantum signal processing](https://arxiv.org/abs/2002.11649), and Matlab implementations in the [QSPPACK repository](https://github.com/qsppack/QSPPACK). See the affiliated `CITATION` file for a full list of references.

> :warning: As methods for numerically handling QSP protocols have been refined, we have tried to update this repository to reflect leading methods. Along the way, we have also had to lightly deprecate older methods (which may still be in use by others who use this repository). In the sections that follow, we try to give special attention to where a new user might begin to use this repository in the most frictionless way.

***

[^1]: In general, QSVT and related algorithms can compute matrix polynomials in wide classes of linear operators, from normal operators to non-square operators to infinite-dimensional operators, each with corresponding limitations and a related circuit form. For our purposes, treating normal operators will nearly always be sufficient, and captures most of the essence of QSP/QSVT.

### An overview of the QSP ansatz and conventions

As organized in [this introductory pedagogical overview](https://arxiv.org/abs/2105.02859), there are two major conventions used in QSP literature: $W_x$ convention, where the signal $W(a)$ is an $X$-rotation and the QSP phases correspond to $Z$-rotations, and the $W_z$ convention, where the signal $W(a)$ is a $Z$-rotation and the QSP phases correspond to $X$-rotations.

Concretely, in the $W_x$ convention, the overall unitary for some list of QSP phases $\Phi \in \mathbb{R}^{n + 1}$ is:
```math
    U_x = e^{i\phi_0 Z} \prod_{k=1}^n W(a) e^{i\phi_k Z} \;\;\;\;\text{where}\;\;\;\; W(x)= \begin{bmatrix} a & i\sqrt{1-a^2} \\ i\sqrt{1-a^2} & a \end{bmatrix}
```
while in the $W_z$ convention, the overall unitary is:
```math
    U_z = e^{i\phi_0 X} \prod_{k=1}^n W(a) e^{i\phi_k X} \;\;\;\;\text{where}\;\;\;\; W(a)=\begin{bmatrix} a + i\sqrt{1-a^2} & 0 \\ 0 & a - i\sqrt{1-a^2} \end{bmatrix}
```
As one might guess, these conventions are related by a Hadamard transform:
```math
    U_x = H U_z H
```
The $W_z$ convention is convenient for and employed in [Laurent polynomial formulations of QSP](https://arxiv.org/abs/2003.02831), while the $W_x$ convention is older and perhaps more widespread currently, e.g. as employed in [quantum singular value transform](https://arxiv.org/abs/1806.01838).

In the first convention, the resulting QSP unitary is also given a standard form (which can be shown by induction), namely that
```math
    U_x = e^{i\phi_0 Z} \prod_{k=1}^n W(a) e^{i\phi_k Z} = \begin{bmatrix} P_x(a) & iQ_x(a)\sqrt{1-a^2} \\ i Q_x^*(a)\sqrt{1-a^2} & P_x^*(a) \end{bmatrix},
```
where $P$ and $Q$ are polynomials of degree $n$ and $n-1$ respectively, with definite parity, and satisfying the condition $|P|^2 + |Q|^2(1 - a^2) = 1$ for all $a \in [-1,1]$. Evidently in a different basis the parity and degree of these polynomials may change, and moreover sometimes the substitution $a \rightarrow (b + 1/b)/2$ is sometimes made, moving from polynomials over $[-1,1]$ to Laurent polynomials over the unit circle in the complex plane. Each of these choices has benefits and limitations, but for most initial presentations, the $U_x$ convention with polynomials over $[-1,1]$ is used.

As stated, this package can generate QSP phases in both conventions. The impediment to immediately freely working between both conventions is that if one wants a certain polynomial $P_x(a) = \langle 0|U_x|0\rangle$ in the $W_x$ convention, one cannot just use the phases generated for this polynomial in the $W_z$ convention. Instead, first the $Q_x(a)$ corresponding to $P_x(a)$ is needed to complete the full $U_x$. From this one can compute $P_z(a) = \langle 0|U_z|0\rangle = P_x(a) + Q_x(a)$. Computing the QSP phases for *this* $P_z(a)$ in the $W_z$ convention yields the desired QSP phases for $P_x(a)$ in the $W_x$ convention.

> :warning: In addition to specifying the signal unitary and the signal processing phase rotations, the implicit measurement basis must also be specified. In this codebase the default basis is $|\pm\rangle$ (i.e., the $X$ Pauli's eigenbasis), though the code also allows for the computational basis (e.g., when using `tf` optimization method to generate the phase angles). For information on this, see the `--measurement` option below.

The guiding principle to take away from the discussion above is the following: the choice of circuit convention can change the conditions required of the achieved polynomial transforms (e.g., their parity, degree, boundary conditions, etc.). That said, the *classical* subroutines used to find good polynomial approximations remain mostly unchanged, and for most applications the restrictions on achieved functions are not crucial for performance.

## A few quick-and-dirty examples

In this section we aim to teach basic use-cases for this package through a series of related examples. Given the heterogeneous research efforts in QSP, there are multiple ways to do most of the tasks laid out here. Toward this end, we break this quick-start guide into two parts: (1) a *very* short guide-within-a-guide that shows what we believe will be the most common pattern desired by a user, and (2) a more extended series of examples covering different conventions, phase-finding methods, and visualization tools illustrating the depth of the package. A full list of options and arguments can be found internal to the package, or listed in [later section of the document](#command-line-usage).

### A guide within a guide

For many people tinkering with QSP, especially in experimental contexts, their workflow typically involves some ideal, analytically-expressible target function for which they desire QSP phases to some moderate degree. A minimal, self-contained instance of this technique (also contained in `pyqsp/sym_qsp_min_example.py`) has the following structure:
1. Import required modules and methods.
2. Specify analytic target function and desired QSP protocol length.
3. According to above, compute good polynomial approximant.[^2]
4. Using the `sym_qsp` method, compute QSP phases whose unitary has the desired function as $\Im(\langle 0|U|0 \rangle)$.
5. Plot target and achieved functions to visualize result.

[^2]: For our purpose we are actually computing a *Chebyshev interpolant*, not an approximation or Taylor series. This functionality is folded into the `PolyTaylorSeries` class by enabling the `chebyshev_basis=True` flag. While for high-precision applications the difference is important, for most use-cases, and piecewise-continuous target functions, the same degree approximation will have uniform error between cases differing by a constant. For a pleasant, engaging introduction to this topic, see [Approximation Theory and Approximation Practice](https://epubs.siam.org/doi/10.1137/1.9781611975949), which can be found as a PDF in various locations.

```python
# Import relevant modules and methods.
import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import (polynomial_generators, PolyTaylorSeries)

# Specify definite-parity target function for QSP.
func = lambda x: np.cos(3*x)
polydeg = 12 # Desired QSP protocol length.
max_scale = 0.9 # Maximum norm (<1) for rescaling.
true_func = lambda x: max_scale * func(x) # For error, include scale.

"""
With PolyTaylorSeries class, compute Chebyshev interpolant to degree
'polydeg' (using twice as many Chebyshev nodes to prevent aliasing).
"""
poly = PolyTaylorSeries().taylor_series(
    func=func,
    degree=polydeg,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)

# Compute full phases (and reduced phases, parity) using symmetric QSP.
(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly,
    method='sym_qsp',
    chebyshev_basis=True)

"""
Plot response according to full phases.
Note that `pcoefs` are coefficients of the approximating polynomial,
while `target` is the true function (rescaled) being approximated.
"""
response.PlotQSPResponse(
    phiset,
    pcoefs=poly,
    target=true_func,
    sym_qsp=True,
    simul_error_plot=True)
```

The method specified above works well up to above `polydeg=100`, and is limited mainly by the out-of-the-box method `numpy` uses to compute Chebyshev interpolants. If the `Chebyshev` object fed into `QuantumSignalProcessingPhases` has had its coefficients computed some other way (e.g., analytically as in the approximation of the inverse function), this method can work well into the thousands of phases.

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_min_example_cosine.png" alt="Example QSP response function approximating a cosine function" width="75%"/>
</p>

### A mélange of methods for phase finding and plotting

This package includes various tools for plotting aspects of the computed QSP unitaries, many of which can be run from the command line. As an example, in the chart below the dashed line shows the target ideal polynomial QSP *response function* approximating a scaled version of $1/a$ over a sub-interval of $[-1,1]$. The blue line shows the imaginary part of an actual response function, i.e., the matrix element $P_x(a)$, achieved by a QSP circuit with computed phases, while the red line in the bottom plot shows the error between achieved and desired functions on a logarithmic plot. Here `n = 147` (the length of the QSP phase list less one).

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_qsp_inversion.png" alt="QSP response function for the inverse function 1/a" width="75%"/>
</p>

This was generated by running `pyqsp --plot --tolerance=0.01 --seqargs 6 --method sym_qsp invert`, which also spits out the the following verbose text:

```python
b=147, j0=35
[PolyOneOverX] minimum [-7.75401864] is at [-0.09251427]: normalizing
[pyqsp.PolyOneOverX] pcoefs=[
    0.00000000e+00  2.21344679e-01  0.00000000e+00 -1.99904501e-01
    0.00000000e+00  1.78896005e-01  0.00000000e+00 -1.58587792e-01
    0.00000000e+00  1.39221019e-01  0.00000000e+00 -1.21000963e-01
    ...
    0.00000000e+00  6.32566663e-85  0.00000000e+00 -4.30307535e-87
    0.00000000e+00  1.45866961e-89]
```

The sign function is also often useful to implement, e.g. for oblivious amplitude amplification. Running instead
```python
pyqsp --plot --seqargs=19,10 --method sym_qsp poly_sign
```
yields QSP phases for a degree `19` polynomial approximation, using the error function applied to `kappa * a`, where `kappa` is `10`. This method also uses the `sym_qsp` method, which is more numerically stable and faster than the default `laurent` method. This also gives a plotted response function:

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_amplitude_amplification_sym_qsp.png" alt="Example QSP response function approximating the sign function" width="75%"/>
</p>

A threshold function further generalizes on the sign function, e.g., as used in distinguishing eigenvalues or singular values through windowing. Running
```python
pyqsp --plot --seqargs=20,20 --method sym_qsp poly_thresh
```
yields QSP phases for a degree `20` polynomial approximation, using two error functions applied to `kappa * 20`, with a plotted response function:

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_poly_thresh_sym_qsp.png" alt="Example QSP response function approximating a threshold function" width="75%"/>
</p>

In addition to approximations to piecewise continuous functions, the smooth trigonometric functions sine and cosine functions also often appear, e.g., in Hamiltonian simulation. Running the following:
```python
pyqsp --plot --func "np.cos(3*x)" --polydeg 6 sym_qsp_func
```
produces QSP phases for a degree `6` polynomial approximation of `cos(3*x)`, with the plotted response function:

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_qsp_symbolic_fit.png" alt="Example QSP response function approximating a cosine function" width="75%"/>
</p>

This last example also shows how an arbitrary function can be specified (using a `numpy` expression) as a string, and fit using an arbitrary order polynomial (with parity automatically detected, and the fit rejected in the case of indefinite parity), using optimization internal to `numpy`.

> :construction: The above examples are all run using the `laurent` method introduced earlier.[^3] Recent additions to the codebase (as of `09/01/2024`) have allowed a third method, `sym_qsp` to be specified, as in the [previous full example](#a-guide-within-a-guide), which also comprises new classical approximation subroutines that search for the best Chebyshev expansion matching a desired piecewise continuous function at the *Chebyshev nodes* of a given order. This method generally outperforms the `laurent` and `tf` methods discussed above and in the footnote referenced at the opening of this paragraph, at the cost of achieving the polynomial as only the imaginary part of a single matrix element rather than as an amplitude directly.

[^3]: In earlier version of this repository, direct optimizations over QSP protocols using TensorFlow and Keras could be performed by specifying `--method tf` within the command line. While functions related to this method still work when called (e.g., the plotting option `--plot-qsp-model`), their tests have been currently silenced and the dependent packages moved from `base_requirements.txt` to `tf_requirements.txt`. Overall, while generally effective, such methods are often slower and more prone to mis-optimization than `sym_qsp`, and so for a new user we recommend the latter. For a guide of sorts to using the `tf` methods, see the deprecated test file `pyqsp/test/test_qsp_models.py`.

## An overview of codebase structure

- `angle_sequence.py` is the main module of the algorithm, containing the omnibus method `QuantumSignalProcessingPhases` which takes a polynomial and a series of options, and returns either the `phiset`, or the `(phiset, reduced_phiset, parity)` in the case of `method=sym_qsp`.
- `LPoly.py` defines two classes `LPoly` and `LAlg`, representing Laurent polynomials and Low algebra elements respectively.
- `completion.py` describes the completion algorithm: Given a Laurent polynomial element $F(\tilde{w})$, find its counterpart $G(\tilde{w})$ such that $F(\tilde{w})+G(\tilde{w})*iX$ is a unitary element.
- `decomposition.py` describes the halving algorithm: Given a unitary parity Low algebra element $V(\tilde{w})$, decompose it as a unique product of degree-0 rotations $\exp\{i\theta X\}$ and degree-1 monomials $w$.
- `ham_sim.py` shows an example of how the angle sequence for Hamiltonian simulation can be found.
- `response.py` computes QSP response functions and generates plots
- `poly.py` provides some utility polynomials, namely the approximation of 1/a using a linear combination of Chebyshev polynomials
- `main.py` is the main entry point for command line usage
- `qsp_model` is the submodule providing generation of QSP phase angles using tensorflow + keras
- `sym_qsp_opt` contains the optimization methods used for optimizing over symmetric QSP objects, which are also defined and handled in this module.

> :construction: Recent additions to this codebase exist in the same folder as the above main `*.py` files, i.e., the abovementioned `sym_qsp_opt.py` and the currently exploratory `sym_qsp_plotting.py` and `sym_qsp_phase_gen.py`, as well as tests for these new files in `test/test_sym_qsp_optimization.py` which is automatically run with the other tests, and might serve as a good best-practices read-through for the curious. An executable full example as referenced in []() is also given in `sym_qsp_min_example.py`.
>
> The Chebyshev interpolation methods mentioned above have been added as internal arguments throughout `poly.py` using the `chebyshev_basis` (Boolean) flag and `cheb_samples` (positive integer) argument. When calling `sym_qsp` methods from the command line, these flags and arguments are automatically set.

> :warning: The code is structured such that *TensorFlow* is not imported by default, as its dependencies, size, and overall use have become cumbersome for most applications. If `qsp_model` and its derived methods are used, then TensorFlow is required. Currently tests for this module have also been silenced, and *TensorFlow* dependent functionality is not being actively maintained, though has not been disabled.

## Package requirements

This package can be run entirely without TensorFlow if the `qsp_model` code is not used.  If `qsp_model` is desired, then also install the requirements specified in [tf_requirements.txt](https://github.com/ichuang/pyqsp/blob/master/tf_requirements.txt). Otherwise, the requirements given in [base_requirements.txt](https://github.com/ichuang/pyqsp/blob/master/base_requirements.txt) are sufficient.

## Unit tests

A series of unit tests is also provided. These can be run using `pytest pyqsp/test/`.

> :warning: As the `qsp_model` code depends on having TensorFlow installed, the unit tests for this code take a while; as such they are turned off by default. To enable unit tests for this code, un-comment the corresponding file `test/test_qsp_models.py` after running `export PYQSP_TEST_QSP_MODELS=1`. The `qsp_model` code unit tests can be run by themselves, after un-commenting the file, using `python setup.py test -s pyqsp.test.test_qsp_models`.

## Programmatic usage

To find the QSP angle sequence corresponding to a real Laurent polynomial $A(\tilde{w}) = \sum_{i=-n}\^n a_i\tilde{w}^i$, we can run:

```python
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
ang_seq = QuantumSignalProcessingPhases([a_{-n}, a_{-n+2}, ..., a_n], signal_operator="Wz")
print(ang_seq)
```

To find the QSP angle sequence corresponding to a real (non-Laurent) polynomial $A(x) = \sum_{i=0}\^n a_i x^i$, we can run:

```python
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
ang_seq = QuantumSignalProcessingPhases([a_{0}, a_{1}, ..., a_n], signal_operator="Wx")
print(ang_seq)
```

By default, `QuantumSignalProcessingPhases` uses the `laurent` method, which is typically quite fast, but can become unstable for high-degree polynomials due to roundoff errors, requiring some randomization. `QuantumSignalProcessingPhases` can also be instructed to use the `tf` method, which employs TensorFlow with a Keras model to find phases by optimization. This stably finds very high-quality solutions, but can be slow, particularly compared with the `laurent` method. We can run this method using:

```python
ang_seq = QuantumSignalProcessingPhases(poly, signal_operator="Wx", method="tf")
```

Note that with the `tf` method, only the `Wx` signal_operator convention is supported. With this method, the polynomial can be a numpy Polynomial instance, or an instance of `pyqsp.poly.StringPolynomial`, e.g.

    poly = StringPolynomial("np.cos(3*x)", 6)
    ang_seq = QuantumSignalProcessingPhases(poly, method="tf")

We can also plot the response given by a given QSP angle sequence, e.g. using:

```python
pyqsp.response.PlotQSPResponse(ang_seq, target=poly, signal_operator="Wx")
```

### Computing QSP phases even faster with the `sym_qsp` method

In the same paradigm as the above `laurent` and `tf` methods, we have also introduced the iterative phase finding algorithm discussed earlier under the `sym_qsp` method. Using this method is compatible with the original `QuantumSignalProcessingPhases` method by passing the name of the method, and setting `chebyshev_basis=True`.

```python
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
(full_phi, reduced_phi, parity) = QuantumSignalProcessingPhases(poly, method="sym_qsp", chebyshev_basis=True)
```

Note that here, unlike in the standard case, we return the full phases (analogous to `ang_seq` in the above examples), the reduced phases as used in the original reference [Efficient phase-factor evaluation in quantum signal processing](https://arxiv.org/abs/2002.11649), and the polynomial parity. When using the `sym_qsp_func` option in the command line, just the full phases are returned.

> :warning: Throughout the codebase, there are multiple conventions in which polynomials can be represented; (1) polynomials in the standard basis, (2) polynomials in the Chebyshev basis, and (3) Laurent polynomials in the standard basis.
>
> The `numpy` package can natively handle polynomials in the standard basis and Chebyshev basis through the `numpy.polynomial.polynomial` and `numpy.polynomial.chebyshev` packages, which can also be cast to one another with standard methods, e.g.,
> ```python
> ccoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)
> pcoefs = np.polynomial.chebyshev.cheb2poly(ccoefs)
> ```
> For numerical stability, however, casting between `Polynomial` and `Chebyshev` objects at high degrees is not recommended. As such, recent updates have forced almost all methods to work in the Chebyshev basis by default; notably, in our convention, this will also exactly correspond to the Laurent basis up to copying some of the coefficients.
>
> As it stands, the `sym_qsp` and most of the `laurent` method now work exclusively in the Chebyshev basis, performing internal checks on passed objects.

Plotting is also supported with the `sym_qsp` method, again calling

```python
pyqsp.response.PlotQSPResponse(ang_seq, sym_qsp=True)
```

Note that the `sym_qsp` method is currently only supported with one oracle convention (`signal_operator="Wx"`), and the default measurement basis, as the achieved amplitude appears as the imaginary part of the top-left matrix element of the resulting unitary.

### Overview of more recent updates to phase-finding methods (09/2024)

> :construction: Here we provide some discussion of the recently added (as of `09/01/2024`) method for computing QSP phases using iterative methods for symmetric QSP protocols.

Newly added methods related to the theory of symmetric quantum signal processing allow one to quickly determine, by an iterative quasi-Newton method, the phases corresponding to useful classes of functions entirely subsuming those discussed previously. These methods are double-precision limited, numerically stable, and fast even for high-degree polynomials. Currently these `sym_qsp` methods are contained in `pyqsp/sym_qsp_opt.py`, and we have prepared a temporary plotting module within `pyqsp/sym_qsp_plotting.py` which can be run by itself with `python sym_qsp_plotting.py` to generate some common examples and illustrate common calling/plotting patterns.

For instance, the current file returns approximations to cosine, sine, and a step function, of which we reproduce the first and third plots below.

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_cosine_approximation.png" alt="QSP response function approximating trigonometric cosine" width="75%"/>
</p>

As the quality of the approximation is quite high, causing the three intended plots to superpose, we include a logarithmic plot of the pairwise difference between the plotted values, indicating near-machine-precision limited performance.

Other benefits of the `sym_qsp` method appear when we approximate a scaled version of $1/x$. This is implemented in `pyqsp/sym_qsp_opt.py` for the choices `kappa = 5` (specifying the domain of valid approximation) and `epsilon = 0.01` (the uniform approximation error on the domain). We plot this approximation, indicating the region of validity from `[-1, -1/kappa]`-union-`[1/kappa, 1]` in gray, clearly showing that the approximation error bound and QSP phase error bounds are satisfied in the valid region.

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_qsp_inverse_approx.png" alt="QSP response function approximating scaled 1/x" width="75%"/>
</p>

> :round_pushpin: Previously the computation of QSP phases to approximate 1/x was limited by two factors: (1) the instability of direct polynomial completion methods in the `laurent` approach, and (2) integer overflow errors resulting from computing coefficients of the analytic polynomial approximation in a naïve way. The plot above has been generated in a way which avoids both issues, and its degree can be easily pushed into the hundreds. The plot above uses `d = 155`.

Finally, we can move away from functions for which we have analytic descriptions of their Chebyshev expansions to general piecewise continuous functions for which we numerically compute Chebyshev interpolants. One such example is the step function, plotted analogously below.

<p align="center">
    <img src="https://github.com/ichuang/pyqsp/blob/master/docs/ex_step_approximation.png" alt="QSP response function approximating a step function" width="75%"/>
</p>

As in the case of trigonometric cosine and inverse, the step function's approximation is also excellent within the specified region, and far more forgiving in its generation than with the earlier `laurent` method.

> :round_pushpin: The final plot given above is generated in a fairly simple way, but relies on a few, user-programmable inputs which we discuss with a subsection of the code given in `pyqsp/sym_qsp_plotting.py` below. The code given here is prefaced in the original file by proper imports, and the QSP phases generated by the Newton solver are used to generate the plots with further methods.
>
> :warning: Note that the arguments `chebyshev_basis` (a Boolean) and `cheb_samples` are both used here. The `chebyshev_basis` flag ensures that the behind-the-scenes optimization methods used in finding polynomial approximations work in the more stable Chebyshev basis, and that Chebyshev basis coefficients are returned. The `cheb_samples` argument chooses the number of Chebyshev node interpolation points with which this classical fit is computed; to prevent aliasing issues *it is best to choose `cheb_samples` to be greater than `degree`, which specifies the degree of the polynomial approximant*. Here `delta` scales as the approximate inverse gap between the regions where the desired function takes the extreme values 1 and -1.

```python
# Generate Chebyshev coefs for approx sign function.
pg = poly.PolySign()
pcoefs = pg.generate(
        degree=161,
        delta=25,
        chebyshev_basis=True,
        cheb_samples=250)
pcoefs = pcoefs.coef

# As an example, generate polynomial approximation as a callable function using above-computed coefs.
sign_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)

# Slice out non-trivial coefficients; here degree=161, which has parity 1.
parity = 1
coef = pcoefs[parity::2]

# Iteratively optimize QSP phases using Newton solver, which takes parity-reduced Chebyshev coefs.
(phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_solver(coef, parity, crit=1e-12)
```

This as well as many other polynomial families given in `pyqsp/poly.py` allow for the same `chebyshev_basis` and `cheb_samples` arguments, and can generate the same `pcoefs` results, which can be sliced according to parity and fed as an optimization target into `sym_qsp_opt.newton_solver`, which implicitly generates and optimizes a `SymmetricQSPProtocol` object, up to the desired `crit` uniform maximum error.

## Command line usage

A wide selection of the functionalities provided by this package can also be run from the command line using a series of arguments and flags. We detail these below, noting that there exist new methods under active development not covered here, though these changes will be backwards compatible to the methods given below.

```bash
usage: pyqsp [-h] [-v] [-o OUTPUT] [--signal_operator SIGNAL_OPERATOR] [--plot] [--hide-plot] [--return-angles] [--poly POLY] [--func FUNC] [--polydeg POLYDEG] [--scale SCALE]
             [--seqname SEQNAME] [--seqargs SEQARGS] [--polyname POLYNAME] [--polyargs POLYARGS] [--plot-magnitude] [--plot-probability] [--plot-real-only] [--title TITLE]
             [--measurement MEASUREMENT] [--output-json] [--plot-positive-only] [--plot-tight-y] [--plot-npts PLOT_NPTS] [--tolerance TOLERANCE] [--method METHOD]
             [--plot-qsp-model] [--phiset PHISET] [--nepochs NEPOCHS] [--npts-theta NPTS_THETA]
             cmd

usage: pyqsp [options] cmd

Version: 0.1.6
Commands:

    poly2angles - compute QSP phase angles for the specified polynomial (use --poly). Currently, if using the `laurent` method, this polynomial is expected in monomial basis, while for `sym_qsp` method the Chebyshev basis is expected. Eventually a new flag will be added to specify basis for use with any method.
    hamsim      - compute QSP phase angles for Hamiltonian simulation using the Jacobi-Anger expansion of exp(-i tau sin(2 theta))
    invert      - compute QSP phase angles for matrix inversion, i.e. a polynomial approximation to 1/a, for given delta and epsilon parameter values
    angles      - generate QSP phase angles for the specified --seqname and --seqargs
    poly        - generate QSP phase angles for the specified --polyname and --polyargs, e.g. sign and threshold polynomials
    polyfunc    - generate QSP phase angles for the specified --func and --polydeg using tensorflow + keras optimization method (--tf). Note that the function appears as the top left element of the resulting unitary.
    sym_qsp_func    - generate QSP phase angles for the specified --func and --polydeg using symmetric QSP iterative method. Note that the desired polynomial is automatically rescaled (unless a scalar --scale less than one is specified), and appears as the imaginary part of the top-left unitary matrix element in the standard basis.
    response    - generate QSP polynomial response functions for the QSP phase angles specified by --phiset

Examples:

    # Simple examples for determining QSP phases from poly coefs.
    pyqsp --poly=-1,0,2 poly2angles
    pyqsp --poly=-1,0,2 --plot poly2angles
    pyqsp --signal_operator=Wz --poly=0,0,0,1 --plot poly2angles

    # Note examples using the 'sym_qsp' method.
    pyqsp --plot --seqargs=10,0.1 --method sym_qsp hamsim
    pyqsp --plot --seqargs=19,10 --method sym_qsp poly_sign
    pyqsp --plot --seqargs=19,0.25 --method sym_qsp poly_linear_amp
    pyqsp --plot --seqargs=68,20 --method sym_qsp poly_phase
    pyqsp --plot --seqargs=20,0.6,15 --method sym_qsp relu
    pyqsp --plot --seqargs=3,0.1 --method sym_qsp invert
    pyqsp --plot --func "np.sign(x)" --polydeg 151 --scale 0.5 sym_qsp_func
    pyqsp --plot --func "np.sin(10*x)" --polydeg 31 sym_qsp_func
    pyqsp --plot --func "np.sign(x - 0.5) + np.sign(-1*x - 0.5)" --polydeg 151 --scale 0.9 sym_qsp_func

    # Note older examples using the 'laurent' method.
    pyqsp --plot --tolerance=0.1 --seqargs 4 invert
    pyqsp --plot --seqargs=10,0.1 hamsim
    pyqsp --plot-npts=4000 --plot-positive-only --plot-magnitude --plot --seqargs=1000,1.0e-20 --seqname fpsearch angles
    pyqsp --plot-npts=100 --plot-magnitude --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-npts=100 --plot-positive-only --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-real-only --plot --polyargs=20,20 --polyname poly_thresh poly
    pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign poly
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,3.5 --polyname gibbs poly
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,0.2,0.9 --polyname efilter poly

    # Note older examples using the deprecated 'tf' method.
    pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign --method tf poly
    pyqsp --plot --func "np.cos(3*x)" --polydeg 6 polyfunc
    pyqsp --plot --func "np.cos(3*x)" --polydeg 6 --plot-qsp-model polyfunc
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,3.5 --polyname gibbs --plot-qsp-model poly
    pyqsp --polydeg 16 --measurement="z" --func="-1+np.sign(1/np.sqrt(2)-x)+ np.sign(1/np.sqrt(2)+x)" --plot polyfunc

positional arguments:
  cmd                   command

options:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity (add more -v to increase versbosity)
  -o OUTPUT, --output OUTPUT
                        output filename
  --signal_operator SIGNAL_OPERATOR
                        QSP sequence signal_operator, either Wx (signal is X rotations) or Wz (signal is Z rotations)
  --plot                generate QSP response plot
  --hide-plot           do not show plot (but it may be saved to a file if --output is specified)
  --return-angles       return QSP phase angles to caller
  --poly POLY           comma delimited list of floating-point coefficients for polynomial, as const, a, a^2, ...
  --func FUNC           for tf method, numpy expression specifying ideal function (of x) to be approximated by a polynomial, e.g. 'np.cos(3 * x)'
  --polydeg POLYDEG     for tf method, degree of polynomial to use in generating approximation of specified function (see --func)
  --scale SCALE         Within 'sym_qsp' method, specifies a float to which the extreme point of the approximating polynomial is rescaled in absolute value. For highly oscillatory functions, try decreasing this parameter toward zero.
  --seqname SEQNAME     name of QSP phase angle sequence to generate using the 'angles' command, e.g. fpsearch
  --seqargs SEQARGS     arguments to the phase angles generated by seqname (e.g. length,delta,gamma for fpsearch)
  --polyname POLYNAME   name of polynomial generated using the 'poly' command, e.g. 'sign'
  --polyargs POLYARGS   arguments to the polynomial generated by poly (e.g. degree,kappa for 'sign')
  --plot-magnitude      when plotting only show magnitude, instead of separate imaginary and real components
  --plot-probability    when plotting only show squared magnitude, instead of separate imaginary and real components
  --plot-real-only      when plotting only real component, and not imaginary
  --title TITLE         plot title
  --measurement MEASUREMENT
                        measurement basis if using the polyfunc argument
  --output-json         output QSP phase angles in JSON format
  --plot-positive-only  when plotting only a-values (x-axis) from 0 to +1, instead of from -1 to +1
  --plot-tight-y        when plotting scale y-axis tightly to real part of data
  --plot-npts PLOT_NPTS
                        number of points to use in plotting
  --tolerance TOLERANCE
                        error tolerance for phase angle optimizer
  --method METHOD       method to use for qsp phase angle generation, either 'laurent' (default), 'sym_qsp' for iterative methods involving symmetric QSP, or 'tf' (for tensorflow + keras)
  --plot-qsp-model      show qsp_model version of response plot instead of the default plot
  --phiset PHISET       comma delimited list of QSP phase angles, to be used in the 'response' command
  --nepochs NEPOCHS     number of epochs to use in tensorflow optimization
  --npts-theta NPTS_THETA
                        number of discretized values of theta to use in TensorFlow optimization
```

<!-- *TODO: INCLUDE A SORT OF GALLERY OF COMMON EXAMPLES, ALONG WITH EXPECTED PARAMETERS AND OUTPUT PLOTS, ETC.* -->

<!-- ### Example: plot response polynomial functions for sin(a) approximation

    pyqsp --plot-qsp-model --phiset="[-1.63276817 0.20550406 -0.84198335  0.39732059 -0.26820613 2.41324245  0.04662674 -2.02847501 1.11311765  0.04662674 -0.72835021 -0.26820613 0.39732059 -0.84198335  0.20550406 -0.06197184]" response -->

## Citing this repository

To cite this repository please include a reference to [our paper](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203), [Chao et al.](https://github.com/alibaba-edu/angle-sequence), and [Efficient phase-factor evaluation in quantum signal processing](https://arxiv.org/abs/2002.11649), or reference the file linked below.

> [!TIP]
> A full, bibTeX-formatted list of references can be found [in this plaintext file](https://github.com/ichuang/pyqsp/blob/master/CITATION).

## Repository version history

<!-- *TODO: Update version number once pushed major changes.* -->

- v0.0.3: initial version, with phase angle generation entirely done using https://arxiv.org/abs/2003.02831.
- v0.1.0: added generation of phase angles using optimization via tensorflow (qsp_model code by Jordan Docter and Zane Rossi).
- v0.1.1: add tf unit tests to test_main; readme updates.
- v0.1.2: fixed bug in qsp_model plotting (Re[q] wasn't being correctly computed for the qsp_model plot); made tf an optional requirement.
- v0.1.3: fixed bug in qsp_model.qsp_layers - Re[q] is actually proportional to Imag[u[0,1]]; allow --nepochs and --npts-theta to be specified.
- v0.1.4: add measurement basis option for qsp_models; add phase estimation polynomial.
