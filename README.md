# :hammer_and_wrench: `pyQSP`: a Python Package for Quantum Signal Processing

![test workflow](https://github.com/ichuang/pyqsp/actions/workflows/run_tests.yml/badge.svg)

<!-- *TODO: add some symbols for the main headings, add a title that matches the title of the repo, and overall remove extraneous lists of outputs and inputs outside of a dedicated example section.* -->

## Introduction

[Quantum signal processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) (QSP) is a flexible quantum algorithm subsuming Hamiltonian simulation, quantum linear systems solvers, amplitude amplification, and many other common quantum algorithms. Moreover, for each of these applications, QSP often exhibits state-of-the-art space and time complexity, with comparatively simple analysis. QSP and its related algorithms, e.g., [Quantum singular value transformation](PAPERLINKHERE) (QSVT), using basic alternating circuit ansätze originally inspired by composite pulse techniques in NMR, permit one to *transform the spectrum of linear operators by near arbitrary polynomial functions*, with the aforementioned good numerical properties and simple analytic treatment.

In their most basic forms, QSP/QSVT give a recipe for a desired spectral transformation of a unitary[^1] $U$, requiring access to an auxiliary qubit, a controlled version of $U$, and single-qubit rotations on the auxiliary qubit. A standard application of QSP/QSVT might look like the following:
- Given a function one wants to apply to the spectrum of a unitary, classically generate a (Laurent) polynomial that suitably approximates this function over a desired spectral range.
- Having computed a good polynomial approximation, and checking that it obeys certain mild conditions, use one among many efficient *classical* algorithms to compute the sequence of single-qubit rotations (the *QSP phases*) interspersing applications of the controlled unitary (the *QSP signal*) corresponding to the polynomial approximation.
- Run the corresponding sequence of gates as a quantum circuit, interleaving QSP signals and phases, followed by a measurement in a chosen basis.

> :warning: The theory of QSP is not only under active development, but comprises multiple subtly different conventions, each of which can use different terminology compared the barebones outline given here. These included conventions for how the *signal* is encoded, how the *phases* are applied, the basis to measure in, whether one desires to transform eigenvalues or singular values, whether the classical algorithm to find these phases is exact or iterative, and so on.
> 
> Regardless, the basic scheme of QSP and QSVT is relatively fixed: given a specific circuit ansatz and a theory for the polynomial transformations achievable for that ansatz, generate those conditions and algorithms relating the *achieved function* and *circuit parameterization*. Understanding the bidirectional map between phases and polynomial transforms, as well as the efficiency of loading linear systems into quantum processes, constitutes most of the theory of these algorithms.

This package provides such conditions and algorithms, and automatically treats a few common conventions, with options to modify the code in basic ways to encompass others. These conventions are enumerated in the recent pedagogical work [A Grand Unification of Quantum Algorithms](https://arxiv.org/abs/2105.02859), and the QSP phase-finding algorithms we treat can be broken roughly into three types:
- :hammer: The `laurent` method employs techniques originated in [Finding Angles for Quantum Signal Processing with Machine Precision](https://arxiv.org/abs/2003.02831), and extends code from [its attached repository](https://github.com/alibaba-edu/angle-sequence.). This method exactly computes phases by studying the properties of the desired polynomials using a divide-and-conquer approach.
- :sparkles: The `tf` method employs TensorFlow + Keras, and finds QSP phase angles using straightforward (but sometimes slow) optimization techniques.
- :key: The `symmetric_qsp` method employs an iterative, quasi-Newton technique to find QSP phases for a special, lightly-restricted sub-class of protocols. In comparison to the above two techniques, this method is almost invariably quick, numerically stable, and should suit nearly all near-term application needs. It is based off work from [Efficient phase-factor evaluation in quantum signal processing](https://arxiv.org/abs/2002.11649), and Matlab implementations in the [QSPPACK repository](https://github.com/qsppack/QSPPACK).

> :warning: As methods for numerically handling QSP protocols have been refined, we have tried to update this repository to reflect leading methods. Along the way, we have also had to lightly deprecate older methods (which may still be in use for others). In the sections that follow, we try to give special attention to where a new user might enter to find the repository most useful.

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
where $P$ and $Q$ are polynomials of degree $n$ and $n-1$ respectively, with definite parity, and satisfying the condition $|p|^2 + |Q|^2(1 - a^2) = 1$ for all $a$. Evidently in a different basis the parity and degree of these polynomials may change, and moreover sometimes the substitution $a \rightarrow (b + 1/b)/2$ is made, in which case we move from polynomials over $[-1,1]$ to Laurent polynomials over the unit circle in the complex plane. Each of these choices have benefits and drawbacks, but for most initial presentations, the $U_x$ convention with polynomials over $[-1,1]$ is used.

As stated, this package can generate QSP phases in both conventions. The impediment to immediately freely working between both conventions is that if one wants a certain polynomial $P_x(a) = \langle 0|U_x|0\rangle$ in the $W_x$ convention, one cannot just use the phases generated for this polynomial in the $W_z$ convention. Instead, first the $Q_x(a)$ corresponding to $P_x(a)$ is needed to complete the full $U_x$. From this one can compute $P_z(a) = \langle 0|U_z|0\rangle = P_x(a) + Q_x(a)$. Computing the QSP phases for *this* $P_z(a)$ in the $W_z$ convention yields the desired QSP phases for $P_x(a)$ in the $W_x$ convention.

> :warning: In addition to specifying the signal unitary and the signal processing phase rotations, the implicit measurement basis must also be specified. In this codebase the default basis is $|\pm\rangle$ (i.e., the $X$ Pauli's eigenbasis), though the code also allows for the computational basis (e.g., when using `tf` optimization method to generate the phase angles). For information on this, see the `--measurement` option below.

The guiding principle to take away from the discussion above is the following: the choice of circuit convention can change the conditions required of the achieved polynomial transforms (e.g., their parity, degree, boundary conditions, etc.). That said, the *classical* subroutines used to find good polynomial approximations remain mostly unchanged, and for most applications the restrictions on achieved functions are not crucial for performance.

## A few quick-and-dirty examples

This package includes various tools for plotting aspects of the computed QSP unitaries, many of which can be run from the command line. As an example, in the chart below the blue line shows the target ideal polynomial QSP *response function* approximating a scaled version of $1/a$ over a sub-interval of $[-1,1]$. The red line shows the real part of an actual response function (the matrix element $P_x(a)$) achieved by a QSP circuit with computed phases, while the green line shows the imaginary part of the QSP response, with `n = 20` (the length of the QSP phase list less one).

![QSP response function for the inverse function 1/a](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-one-over-x-kappa3.png?raw=true)

This was generated by running `pyqsp --plot invert`, which also spits out the the following verbose text:

```
    b=20, j0=10
    [PolyOneOverX] minimum [-2.90002589] is at [-0.25174082]: normalizing
    Laurent P poly 0.0002108924030879319 * w ^ (-21) + -0.0006894043260278334 * w ^ (-19) + 0.001994436843136674 * w ^ (-17) + -0.005148265426149545 * w ^ (-15) + 0.011941126989562179 * w ^ (-13) + -0.02504164571900347 * w ^ (-11) + 0.04774921151669176 * w ^ (-9) + -0.08322978307559126 * w ^ (-7) + 0.1333200017468883 * w ^ (-5) + -0.1973241700493915 * w ^ (-3) + 0.2714342596621151 * w ^ (-1) + 0.2714342596621151 * w ^ (1) + -0.1973241700493915 * w ^ (3) + 0.1333200017468883 * w ^ (5) + -0.08322978307559126 * w ^ (7) + 0.04774921151669176 * w ^ (9) + -0.02504164571900347 * w ^ (11) + 0.011941126989562179 * w ^ (13) + -0.005148265426149545 * w ^ (15) + 0.001994436843136674 * w ^ (17) + -0.0006894043260278334 * w ^ (19) + 0.0002108924030879319 * w ^ (21)
    Laurent Q poly -2.784691352688532e-05 * w ^ (-21) + -3.467046922689296e-06 * w ^ (-19) + -0.00023447352272530086 * w ^ (-17) + 0.00018731079814912242 * w ^ (-15) + -0.0007159634932856078 * w ^ (-13) + 0.0032821300323626736 * w ^ (-11) + -0.001907488898700853 * w ^ (-9) + 0.012591630667616198 * w ^ (-7) + -0.02371053103573988 * w ^ (-5) + -0.016629567619072996 * w ^ (-3) + -0.18519029619384528 * w ^ (-1) + -0.20476480667058175 * w ^ (1) + -0.5374170003848409 * w ^ (3) + -0.2638332742746844 * w ^ (5) + 0.21212227077358078 * w ^ (7) + 0.27748656462486654 * w ^ (9) + -0.34978762979573563 * w ^ (11) + 0.17888109364117422 * w ^ (13) + -0.07649917245226195 * w ^ (15) + 0.03551004671728361 * w ^ (17) + -0.011926352140796492 * w ^ (19) + 0.001997855069006951 * w ^ (21)
    Laurent Pprime poly 0.00018304548956104657 * w ^ (-21) + -0.0006928713729505227 * w ^ (-19) + 0.001759963320411373 * w ^ (-17) + -0.0049609546280004226 * w ^ (-15) + 0.01122516349627657 * w ^ (-13) + -0.021759515686640796 * w ^ (-11) + 0.0458417226179909 * w ^ (-9) + -0.07063815240797505 * w ^ (-7) + 0.10960947071114842 * w ^ (-5) + -0.2139537376684645 * w ^ (-3) + 0.08624396346826982 * w ^ (-1) + 0.06666945299153335 * w ^ (1) + -0.7347411704342324 * w ^ (3) + -0.1305132725277961 * w ^ (5) + 0.12889248769798953 * w ^ (7) + 0.3252357761415583 * w ^ (9) + -0.3748292755147391 * w ^ (11) + 0.1908222206307364 * w ^ (13) + -0.0816474378784115 * w ^ (15) + 0.037504483560420285 * w ^ (17) + -0.012615756466824325 * w ^ (19) + 0.0022087474720948828 * w ^ (21)
    [QuantumSignalProcessingWxPhases] Error in reconstruction from QSP angles = 0.4862234440482484
    QSP angles = [0.11177647198529908, 0.321286291382733, -2.449109005349844, 0.5475854935639934, -0.23068701778999645, 1.852723853663707, -0.1638598380129409, -0.031620025684050396, -0.3018780297835489, -1.8740083256355018, 0.6314873861312269, -0.036881744534211114, 1.300525766122238, 0.3073180685644643, -0.23974313142278042, 0.09536754986888613, -1.188342341212731, -0.6092537117168764, 0.4850840958947118, 0.8800119831449805, 0.4163653216360098, 2.480415494993986]
```

The sign function is also often useful to implement, e.g. for oblivious amplitude amplification. Running instead
```
pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign poly
```
yields QSP phases for a degree `19` polynomial approximation, using the error function applied to `kappa * a`, where `kappa` is `10`. This also gives a plotted response function:

![Example QSP response function approximating the sign function](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-sign-kappa-10-degree-19.png?raw=true)

A threshold function further generalizes on the sign function, e.g., as used in distinguishing eigenvalues or singular values through windowing. Running
```
pyqsp --plot-real-only --plot --polyargs=20,20 --polyname poly_thresh poly
```
yields QSP phases for a degree `20` polynomial approximation, using two error functions applied to `kappa * 20`, with a plotted response function:

![Example QSP response function approximating a threshold function](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-threshold-polynomial-degree-20-kappa-20.png?raw=true)

In addition to approximations to piecewise continuous functions, the smooth trigonometric functions sine and cosine functions also often appear, e.g., in Hamiltonian simulation. Running:
```
pyqsp --plot --func "np.cos(3*x)" --polydeg 6 --plot-qsp-model polyfunc
```
produces QSP phases for a degree `6` polynomial approximation of `cos(3*x)`, with the plotted response function:

![Example QSP response function approximating a cosine function](https://github.com/ichuang/pyqsp/blob/master/docs/IMAGE-sample-qsp-response-for-cos-using-tf-order-6.png?raw=true)

This last example also shows how an arbitrary function can be specified (using a `numpy` expression) as a string, and fit using an arbitrary order polynomial (may need to be even or odd, to match the function), using optimization via tensorflow, and a keras model.  The example also shows an alternative style of plot, produced using the `--plot-qsp-model` flag.

> :construction: The above examples are all run using either the `laurent` or `tf` methods as introduced earlier. Recent additions to the codebase (as of `09/01/2024`) have allowed a third method, `symmetric_qsp` to be used, which also comprises new classical approximation subroutines, which search for the best Chebyshev expansion matching a desired piecewise continuous function at the *Chebyshev points* of a given order. A special section will be devoted to discussing best practices for this method, which generally outperforms the first two.

## An overview of codebase structure



- `angle_sequence.py` is the main module of the algorithm.
- `LPoly.py` defines two classes `LPoly` and `LAlg`, representing Laurent polynomials and Low algebra elements respectively.
- `completion.py` describes the completion algorithm: Given a Laurent polynomial element $F(\tilde{w})$, find its counterpart $G(\tilde{w})$ such that $F(\tilde{w})+G(\tilde{w})*iX$ is a unitary element.
- `decomposition.py` describes the halving algorithm: Given a unitary parity Low algebra element $V(\tilde{w})$, decompose it as a unique product of degree-0 rotations $\exp\{i\theta X\}$ and degree-1 monomials $w$.
- `ham_sim.py` shows an example of how the angle sequence for Hamiltonian simulation can be found.
- `response.py` computes QSP response functions and generates plots
- `poly.py` provides some utility polynomials, namely the approximation of 1/a using a linear combination of Chebyshev polynomials
- `main.py` is the main entry point for command line usage
- `qsp_model` is the submodule providing generation of QSP phase angles using tensorflow + keras

> :construction: Recent additions to this codebase exist in the same folder as the above main `*.py` files, named `sym_qsp_opt.py` and the currently exploratory `sym_qsp_plotting.py`, as well as tests for these new files in `test/test_sym_qsp_optimization.py` which is automatically run with the other tests, and might serve as a good best-practices read-through for the curious. The Chebyshev interpolation methods mentioned above have been added as internal arguments throughout `poly.py` using the `chebyshev_basis` (Boolean) flag and `cheb_samples` (positive integer) argument.

> :warning: The code is structured such that TensorFlow is not imported by default, as its dependencies, size, and overall use have become cumbersome for most applications. If `qsp_model` and its derived methods are used, then TensorFlow is required. Currently tests for this module have also been silenced, and TensorFlow dependent functionality is not being actively maintained.

### Package requirements

This package can be run entirely without TensorFlow if the `qsp_model` code is not used.  If `qsp_model` is desired, then also install the requirements specified in [tf_requirements.txt](https://github.com/ichuang/pyqsp/blob/master/tf_requirements.txt). Otherwise, the requirements given in [base_requirements.txt](https://github.com/ichuang/pyqsp/blob/master/base_requirements.txt) are sufficient.

### Unit tests

A series of unit tests is also provided. These can be run using `python setup.py test`.

As the `qsp_model` code depends on having TensorFlow installed, the unit tests for this code take a while; as such they are turned off by default. To enable unit tests for this code, un-comment the corresponding file `test/test_qsp_models.py` after running `export PYQSP_TEST_QSP_MODELS=1`. The `qsp_model` code unit tests can be run by themselves, after un-commenting the file, using `python setup.py test -s pyqsp.test.test_qsp_models`.

### Programmatic usage

To find the QSP angle sequence corresponding to a real Laurent polynomial $A(\tilde{w}) = \sum_{i=-n}\^n a_i\tilde{w}^i$, we can run:

    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    ang_seq = QuantumSignalProcessingPhases([a_{-n}, a_{-n+2}, ..., a_n], signal_operator="Wz")
    print(ang_seq)

To find the QSP angle sequence corresponding to a real (non-Laurent) polynomial $A(x) = \sum_{i=0}\^n a_i x^i$, we can run:

    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    ang_seq = QuantumSignalProcessingPhases([a_{0}, a_{1}, ..., a_n], signal_operator="Wx")
    print(ang_seq)

By default, `QuantumSignalProcessingPhases` uses the `laurent` method, which is typically quite fast, but can become unstable for high-degree polynomials due to roundoff errors, requiring some randomization when completing the polynomials. `QuantumSignalProcessingPhases` can also be instructed to use the `tf` method, which employs TensorFlow with a Keras model to find phases by optimization. This stably finds very high-quality solutions, but can be slow, particularly compared with the `laurent` method. We can run this method using:

    ang_seq = QuantumSignalProcessingPhases(poly, signal_operator="Wx", method="tf")

Note that with the `tf` method, only the `Wx` signal_operator convention is supported. With this method, the polynomial can be a numpy Polynomial instance, or an instance of `pyqsp.poly.StringPolynomial`, e.g.

    poly = StringPolynomial("np.cos(3*x)", 6)
    ang_seq = QuantumSignalProcessingPhases(poly, method="tf")

We can also plot the response given by a given QSP angle sequence, e.g. using:

    pyqsp.response.PlotQSPResponse(ang_seq, target=poly, signal_operator="Wx")

> :construction: Here we provide some discussion of the recently added (as of `09/01/2024`) method for computing QSP phases using iterative methods for symmetric QSP protocols.
>
> Chebyshev interpolation (how to choose a good number of points) and chebyshev basis choice reasons.

## Command line usage

A wide selection of the functionalities provided by this package can also be run from the command line using a series of arguments and flags. We detail these below, noting that there exist new methods under active development not covered here, though these changes will be backwards compatible to the methods given below.

```
usage: pyqsp [-h] [-v] [-o OUTPUT] [--signal_operator SIGNAL_OPERATOR] [--plot] [--hide-plot] [--return-angles] [--poly POLY] [--func FUNC]
             [--polydeg POLYDEG] [--tau TAU] [--epsilon EPSILON] [--seqname SEQNAME] [--seqargs SEQARGS] [--polyname POLYNAME] [--polyargs POLYARGS]
             [--plot-magnitude] [--plot-probability] [--plot-real-only] [--title TITLE] [--measurement MEASUREMENT] [--output-json]
             [--plot-positive-only] [--plot-tight-y] [--plot-npts PLOT_NPTS] [--tolerance TOLERANCE] [--method METHOD] [--plot-qsp-model]
             [--phiset PHISET] [--nepochs NEPOCHS] [--npts-theta NPTS_THETA]
             cmd

usage: pyqsp [options] cmd

Version: 0.1.4
Commands:

    poly2angles - compute QSP phase angles for the specified polynomial (use --poly)
    hamsim      - compute QSP phase angles for Hamiltonian simulation using the Jacobi-Anger expansion of exp(-i tau sin(2 theta))
    invert      - compute QSP phase angles for matrix inversion, i.e. a polynomial approximation to 1/a, for given delta and epsilon parameter values
    angles      - generate QSP phase angles for the specified --seqname and --seqargs
    poly        - generate QSP phase angles for the specified --polyname and --polyargs, e.g. sign and threshold polynomials
    polyfunc    - generate QSP phase angles for the specified --func and --polydeg using tensorflow + keras optimization method (--tf)
    response    - generate QSP polynomial response functions for the QSP phase angles specified by --phiset

Examples:

    pyqsp --poly=-1,0,2 poly2angles
    pyqsp --poly=-1,0,2 --plot poly2angles
    pyqsp --signal_operator=Wz --poly=0,0,0,1 --plot  poly2angles
    pyqsp --plot --tau 10 hamsim
    pyqsp --plot --tolerance=0.01 --seqargs 3 invert
    pyqsp --plot-npts=4000 --plot-positive-only --plot-magnitude --plot --seqargs=1000,1.0e-20 --seqname fpsearch angles
    pyqsp --plot-npts=100 --plot-magnitude --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-npts=100 --plot-positive-only --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-real-only --plot --polyargs=20,20 --polyname poly_thresh poly
    pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign poly
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,3.5 --polyname gibbs poly
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,0.2,0.9 --polyname efilter poly
    pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign --method tf poly
    pyqsp --plot --func "np.cos(3*x)" --polydeg 6 polyfunc
    pyqsp --plot --func "np.cos(3*x)" --polydeg 6 --plot-qsp-model polyfunc
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,3.5 --polyname gibbs --plot-qsp-model poly
    pyqsp --polydeg 16 --measurement="z" --func="-1+np.sign(1/np.sqrt(2)-x)+ np.sign(1/np.sqrt(2)+x)" --plot polyfunc

positional arguments:
  cmd                   command

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity (add more -v to increase versbosity)
  -o OUTPUT, --output OUTPUT
                        output filename
  --signal_operator SIGNAL_OPERATOR
                        QSP sequence signal_operator, either Wx (signal is X rotations) or Wz (signal is Z rotations)
  --plot                generate QSP response plot
  --hide-plot           do not show plot (but it may be saved to a file if --output is specified)
  --return-angles       return QSP phase angles to caller
  --poly POLY           comma delimited list of floating-point coeficients for polynomial, as const, a, a^2, ...
  --func FUNC           for tf method, numpy expression specifying ideal function (of x) to be approximated by a polynomial, e.g. 'np.cos(3 * x)'
  --polydeg POLYDEG     for tf method, degree of polynomial to use in generating approximation of specified function (see --func)
  --tau TAU             time value for Hamiltonian simulation (hamsim command)
  --epsilon EPSILON     parameter for polynomial approximation to 1/a, giving bound on error
  --seqname SEQNAME     name of QSP phase angle sequence to generate using the 'angles' command, e.g. fpsearch
  --seqargs SEQARGS     arguments to the phase angles generated by seqname (e.g. length,delta,gamma for fpsearch)
  --polyname POLYNAME   name of polynomial generate using the 'poly' command, e.g. 'sign'
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
  --method METHOD       method to use for qsp phase angle generation, either 'laurent' (default) or 'tf' (for tensorflow + keras)
  --plot-qsp-model      show qsp_model version of response plot instead of the default plot
  --phiset PHISET       comma delimited list of QSP phase angles, to be used in the 'response' command
  --nepochs NEPOCHS     number of epochs to use in tensorflow optimization
  --npts-theta NPTS_THETA
                        number of discretized values of theta to use in tensorflow optimization

```

<!-- *TODO: INCLUDE A SORT OF GALLERY OF COMMON EXAMPLES, ALONG WITH EXPECTED PARAMETERS AND OUTPUT PLOTS, ETC.* -->

<!-- ### Example: plot response polynomial functions for sin(a) approximation

    pyqsp --plot-qsp-model --phiset="[-1.63276817 0.20550406 -0.84198335  0.39732059 -0.26820613 2.41324245  0.04662674 -2.02847501 1.11311765  0.04662674 -0.72835021 -0.26820613 0.39732059 -0.84198335  0.20550406 -0.06197184]" response -->

## Citing this repository

To cite this repository please include a reference to [our paper](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203), [Chao et al.](https://github.com/alibaba-edu/angle-sequence), and [Efficient phase-factor evaluation in quantum signal processing
(https://arxiv.org/abs/2002.11649).

> :round_pushpin: A full, bibTeX-formatted list of references can be found [in this plaintext file](https://github.com/ichuang/pyqsp/blob/master/CITATION).

## Repository version history

<!-- *TODO: Update version number once pushed major changes.* -->

- v0.0.3: initial version, with phase angle generation entirely done using https://arxiv.org/abs/2003.02831.
- v0.1.0: added generation of phase angles using optimization via tensorflow (qsp_model code by Jordan Docter and Zane Rossi).
- v0.1.1: add tf unit tests to test_main; readme updates.
- v0.1.2: fixed bug in qsp_model plotting (Re[q] wasn't being correctly computed for the qsp_model plot); made tf an optional requirement.
- v0.1.3: fixed bug in qsp_model.qsp_layers - Re[q] is actually proportional to Imag[u[0,1]]; allow --nepochs and --npts-theta to be specified.
- v0.1.4: add measurement basis option for qsp_models; add phase estimation polynomial.
