import argparse
import json
import sys

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev

import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.phases import phase_generators

from pyqsp.poly import (StringPolynomial, TargetPolynomial,
                        polynomial_generators, PolyTaylorSeries)

# -----------------------------------------------------------------------------

class VAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        curval = getattr(args, self.dest, 0) or 0
        values = values.count('v') + 1
        setattr(args, self.dest, values + curval)

# -----------------------------------------------------------------------------

def CommandLine(args=None, arglist=None):
    '''
    Main command line.  Accepts args, to allow for simple unit testing.
    '''
    import pkg_resources  # part of setuptools

    version = pkg_resources.require("pyqsp")[0].version
    help_text = """usage: pyqsp [options] cmd

Version: {}
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
    pyqsp --plot --func "np.sign(np.sin(2*np.pi*x))" --polydeg 101 --scale 0.9 sym_qsp_func

    # Note older examples using the 'laurent' method.
    pyqsp --plot --plot-real-only --tolerance=0.1 --seqargs 5 invert
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

""".format(version)

    parser = argparse.ArgumentParser(
        description=help_text,
        formatter_class=argparse.RawTextHelpFormatter)

    def float_list(value):
        try:
            if not ',' in value and value.startswith("[") and value.endswith("]"):
                fstrset = value[1:-1].split(" ")
                fstrset = [x for x in fstrset if x]
                flist = list(map(float, fstrset))
                return flist
            return list(map(float, value.split(",")))
        except Exception as err:
            print(
                f"[pyqsp.float_list] failed to parse float list, err={err} from {value}")
            raise

    parser.add_argument("cmd", help="command")
    parser.add_argument(
        '-v',
        "--verbose",
        nargs=0,
        help="increase output verbosity (add more -v to increase versbosity)",
        action=VAction,
        dest='verbose')
    parser.add_argument("-o", "--output", help="output filename", default="")
    parser.add_argument(
        "--signal_operator",
        help="QSP sequence signal_operator, either Wx (signal is X rotations) or Wz (signal is Z rotations)",
        type=str,
        default="Wx")
    parser.add_argument(
        "--plot",
        help="generate QSP response plot",
        action="store_true")
    parser.add_argument(
        "--hide-plot",
        help="do not show plot (but it may be saved to a file if --output is specified)",
        action="store_true")
    parser.add_argument(
        "--return-angles",
        help="return QSP phase angles to caller",
        action="store_true")
    parser.add_argument(
        "--poly",
        help="comma delimited list of floating-point coefficients for polynomial, as const, a, a^2, ...",
        action="store",
        type=float_list)
    parser.add_argument(
        "--func",
        help="for tf method, numpy expression specifying ideal function (of x) to be approximated by a polynomial, e.g. 'np.cos(3 * x)'",
        type=str)
    parser.add_argument(
        "--polydeg",
        help="for tf method, degree of polynomial to use in generating approximation of specified function (see --func)",
        type=int)
    parser.add_argument(
        "--scale",
        help="Within 'sym_qsp' method, specifies a float to which the extreme point of the approximating polynomial is rescaled in absolute value. For highly oscillatory functions, try decreasing this parameter toward zero.",
        type=float,
        default=0.9)
    parser.add_argument(
        "--seqname",
        help="name of QSP phase angle sequence to generate using the 'angles' command, e.g. fpsearch",
        type=str,
        default=None)
    parser.add_argument(
        "--seqargs",
        help="arguments to the phase angles generated by seqname (e.g. length,delta,gamma for fpsearch)",
        action="store",
        type=float_list)
    parser.add_argument(
        "--polyname",
        help="name of polynomial generated using the 'poly' command, e.g. 'sign'",
        type=str,
        default=None)
    parser.add_argument(
        "--polyargs",
        help="arguments to the polynomial generated by poly (e.g. degree,kappa for 'sign')",
        action="store",
        type=float_list)
    parser.add_argument(
        "--plot-magnitude",
        help="when plotting only show magnitude, instead of separate imaginary and real components",
        action="store_true")
    parser.add_argument(
        "--plot-probability",
        help="when plotting only show squared magnitude, instead of separate imaginary and real components",
        action="store_true")
    parser.add_argument(
        "--plot-real-only",
        help="when plotting only real component, and not imaginary",
        action="store_true")
    parser.add_argument(
        "--title",
        help="plot title",
        type=str,
        default=None)
    parser.add_argument(
        "--measurement",
        help="measurement basis if using the polyfunc argument",
        type=str,
        default=None)
    parser.add_argument(
        "--output-json",
        help="output QSP phase angles in JSON format",
        action="store_true")
    parser.add_argument(
        "--plot-positive-only",
        help="when plotting only a-values (x-axis) from 0 to +1, instead of from -1 to +1 ",
        action="store_true")
    parser.add_argument(
        "--plot-tight-y",
        help="when plotting scale y-axis tightly to real part of data",
        action="store_true")
    parser.add_argument(
        "--plot-npts",
        help="number of points to use in plotting",
        type=int,
        default=400)
    parser.add_argument(
        "--tolerance",
        help="error tolerance for phase angle optimizer",
        type=float,
        default=0.1)
    parser.add_argument(
        "--method",
        help="method to use for qsp phase angle generation, either 'laurent' (default), 'sym_qsp' for iterative methods involving symmetric QSP, or 'tf' (for tensorflow + keras)",
        type=str,
        default='laurent')
    parser.add_argument(
        "--plot-qsp-model",
        help="show qsp_model version of response plot instead of the default plot",
        action="store_true")
    parser.add_argument(
        "--phiset",
        help="comma delimited list of QSP phase angles, to be used in the 'response' command",
        action="store",
        type=float_list)
    parser.add_argument(
        "--nepochs",
        help="number of epochs to use in tensorflow optimization",
        type=int,
        default=5000)
    parser.add_argument(
        "--npts-theta",
        help="number of discretized values of theta to use in TensorFlow optimization",
        type=int,
        default=30)

    if not args:
        args = parser.parse_args(arglist)

    qspp_args = dict(signal_operator=args.signal_operator,
                     method=args.method,
                     tolerance=args.tolerance,
                     nepochs=args.nepochs,
                     npts_theta=args.npts_theta,
                     )

    # For symmetric QSP method, add additional dictionary entry to switch entirely to Chebyshev basis; entries are also added to plot_args.
    if args.method == "sym_qsp" or (args.cmd == "sym_qsp_func"):
        qspp_args["chebyshev_basis"] = True
        is_sym_qsp = True
    else:
        qspp_args["chebyshev_basis"] = False
        is_sym_qsp = False

    phiset = None
    plot_args = dict(plot_magnitude=args.plot_magnitude,
                     plot_probability=args.plot_probability,
                     plot_positive_only=args.plot_positive_only,
                     plot_real_only=args.plot_real_only,
                     plot_tight_y=args.plot_tight_y,
                     npts=args.plot_npts,
                     show=(not args.hide_plot),
                     show_qsp_model_plot=args.plot_qsp_model,
                     sym_qsp=is_sym_qsp,
                     simul_error_plot=is_sym_qsp
                     )

    if args.cmd == "poly2angles":
        coefs = args.poly
        if not coefs:
            print(
                f"[pyqsp.main] must specify polynomial coefficients using --poly, e.g. --poly -1,0,2")
            sys.exit(0)
        if isinstance(coefs, str):
            coefs = list(map(float, coefs.split(",")))

        basis = "Chebyshev" if is_sym_qsp else "Monomial"
        print(f"[pyqsp] {basis} polynomial coefficients={coefs}")
        print(f"[CHECK] Coefficients expected in {basis} basis.")

        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            coefs, **qspp_args)
        else:
            # QuantumSignalProcessingPhases now expects input in Chebyshev basis in all cases, so we cast (assumed monomial basis) input.
            cheb_coefs = np.polynomial.chebyshev.poly2cheb(coefs)
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            cheb_coefs, **qspp_args)
        if args.plot:
            # Note: while the input to 'poly2angles' is expected in the monomial basis, PlotQSPResponse, like all other internal methods, expects input in the Chebyshev basis.
            cheb_coefs = np.polynomial.chebyshev.poly2cheb(coefs)
            response.PlotQSPResponse(
                phiset,
                pcoefs=cheb_coefs,
                signal_operator=args.signal_operator,
                **plot_args)

    elif args.cmd == "hamsim":
        pg = pyqsp.poly.PolyCosineTX()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp) # Support for sym_qsp.

        # Return types among two methods differ
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.cos(args.seqargs[0] * x),
                signal_operator="Wx",
                title="Hamiltonian Simulation (Cosine)",
                **plot_args)

        pg = pyqsp.poly.PolySineTX()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp) # Support for sym_qsp.
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.sin(args.seqargs[0] * x),
                signal_operator="Wx",
                title="Hamiltonian Simulation (Sine)",
                **plot_args)

    elif args.cmd == "fpsearch":
        pg = pyqsp.phases.FPSearch()
        phiset = pg.generate(*args.seqargs)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                signal_operator="Wx",
                measurement="z",
                title="Oblivious amplification",
                **plot_args)

    elif args.cmd == "invert":
        pg = pyqsp.poly.PolyOneOverX()
        # Note that in either case, approximating polynomial is returned in Chebyshev basis.
        pcoefs, scale = pg.generate(
            *args.seqargs,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * 1 / x,
                signal_operator="Wx",
                measurement="z",
                title="Inversion",
                **plot_args)

    elif args.cmd == "gibbs":
        pg = pyqsp.poly.PolyGibbs()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.exp(-args.seqargs[1] * x),
                signal_operator="Wx",
                title="Gibbs distribution",
                **plot_args)

    elif args.cmd == "efilter":
        pg = pyqsp.poly.PolyEigenstateFiltering()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            delta = args.seqargs[1] / 2.
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale *
                (np.sign(x + delta) - np.sign(x - delta)) / 2,
                signal_operator="Wx",
                title="Eigenstate filtering",
                **plot_args)

    elif args.cmd == "relu":
        pg = pyqsp.poly.PolySoftPlus()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.maximum(x - args.seqargs[1], 0.),
                signal_operator="Wx",
                title="ReLU Function",
                **plot_args)

    elif args.cmd == "poly_sign":
        pg = pyqsp.poly.PolySign()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.sign(x),
                signal_operator="Wx",
                title="Sign Function",
                **plot_args)

    elif args.cmd == "poly_thresh":
        pg = pyqsp.poly.PolyThreshold()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale *
                (np.sign(x + 0.5) - np.sign(x - 0.5)) / 2,
                signal_operator="Wx",
                title="Threshold Function",
                **plot_args)

    elif args.cmd == "poly_phase":
        pg = pyqsp.poly.PolyPhaseEstimation()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale *
                (-1 + np.sign(1/np.sqrt(2) - x) + np.sign(1/np.sqrt(2) + x)),
                signal_operator="Wx",
                title="Phase Estimation Polynomial",
                **plot_args)

    elif args.cmd == "poly_rect":
        pg = pyqsp.poly.PolyRect()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale *
                (1 - (np.sign(x + 3 / (4 * args.seqargs[2])) -
                     np.sign(x - 3 / (4 * args.seqargs[2]))) / 2),
                signal_operator="Wx",
                title="Rect Function",
                **plot_args)

    elif args.cmd == "invert_rect":
        pg = pyqsp.poly.PolyOneOverXRect()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * (1 / x) *
                (1 - (np.sign(x + 1 / args.seqargs[2]) -
                     np.sign(x - 1 / args.seqargs[2])) / 2),
                signal_operator="Wx",
                title="Poly Rect * 1/x",
                **plot_args)

    elif args.cmd == "poly_linear_amp":
        pg = pyqsp.poly.PolyLinearAmplification()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True,
            chebyshev_basis=is_sym_qsp)
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * x / (2 * args.seqargs[1]),
                signal_operator="Wx",
                title="Linear Amplification Polynomial",
                **plot_args)

    elif args.cmd == "poly":
        if not args.polyname or args.polyname not in polynomial_generators:
            print(
                f'Known polynomial generators: {",".join(polynomial_generators.keys())}')
            return
        # If one uses the poly argument, rather than a specific generator, then the name fed specifies the generator, and arguments fed are not seqargs but polyargs (which are in many cases the same format).
        pg = polynomial_generators[args.polyname]()
        if not args.polyargs:
            print(pg.help())
            return
        pcoefs, scale = pg.generate(*args.polyargs, return_scale=True, chebyshev_basis=is_sym_qsp)
        print(f"[pyqsp] polynomial coefs = {pcoefs}")
        if is_sym_qsp:
            (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        else:
            phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            target = None
            if isinstance(pcoefs, TargetPolynomial):
                target = pcoefs.target
            response.PlotQSPResponse(
                phiset,
                pcoefs=pcoefs,
                target=target,
                signal_operator="Wx",
                **plot_args)

    elif args.cmd == "angles":
        if not args.seqname or args.seqname not in phase_generators:
            print(
                f'Known phase generators: {",".join(phase_generators.keys())}')
            return
        pg = phase_generators[args.seqname]()
        if not args.seqargs:
            print(pg.help())
            return
        phiset = pg.generate(*args.seqargs)
        print(f"[pysqp] phiset={phiset}")
        if args.plot:
            response.PlotQSPResponse(phiset, signal_operator="Wx", **plot_args)

    # New --sym_qsp_func option added to support general Chebyshev interpolation, in parallel with usage of --polyfunc
    elif args.cmd == "sym_qsp_func":
        if (not args.func) or (not args.polydeg):
            print(f"Must specify --func and --polydeg")
            return

        # Try to cast string as function; if no immediate error thrown at 0.5, generate anonymous function for later use.
        try:
            ret = eval(args.func, globals(), {'x': 0.5})
        except Exception as err:
            raise ValueError(
                f"Invalid function specifciation, failed to evaluate at x=0.5, err={err}")

        # Note that the polynomial is renormalized once according to its maximum magnitude (both up and down), and then again according to specified scale in a multiplicative way. We ought to be able to specify this, or at least return the metadata.

        # The parameter scale < 1 allows rescaling only within sym_qsp method.
        if args.scale:
            if np.abs(args.scale >= 1) or (args.scale <= 0):
                raise ValueError(
                    f"Invalid scale specification (scale = {args.scale}); must be positive and less than 1.")
            else:
                # Note, this is currently the only use-path for --scale argument; ideally it might also be useful to allow within --polyfunc of 'laurent' method, which may also yield Runge.
                max_scale = args.scale
        else:
            max_scale = 0.9

        # Generate anonymous function evaluated at x.
        func = lambda x: eval(args.func, globals(), {'x': x})

        # Note that QuantumSignalProcessingPhases will determine if the parity is not definite, allowing us to use below method raw.
        base_poly = PolyTaylorSeries()
        poly, scale = base_poly.taylor_series(
            func=func,
            degree=args.polydeg,
            ensure_bounded=True,
            return_scale=True,
            max_scale=max_scale,
            chebyshev_basis=is_sym_qsp,
            cheb_samples=2*args.polydeg) # Set larger than polydeg to prevent aliasing.

        # Modify choice of method globally.
        qspp_args['method'] = 'sym_qsp'

        # Compute phases and derived objects from symmetric QSP method.
        (phiset, reduced_phases, parity) = angle_sequence.QuantumSignalProcessingPhases(
            poly,
            **qspp_args)

        # Given polynomial approximation scale factor, generate new possibly rescaled target function for plotting purposes.
        plot_func = lambda x: scale * eval(args.func, globals(), {'x': x})

        # Finally, if plotting called for, plot while passing 'sym_qsp' flag.
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=plot_func, # Plot against ideal function, not poly approx
                title=args.func, # Support to show function plotted.
                signal_operator="Wx",
                **plot_args)

    elif args.cmd == "polyfunc":
        if (not args.func) or (not args.polydeg):
            print(f"Must specify --func and --polydeg")
            return
        qspp_args['method'] = 'tf'
        poly = StringPolynomial(args.func, args.polydeg)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            poly,
            measurement=args.measurement,
            **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=poly,
                signal_operator="Wx",
                measurement=args.measurement,
                title=args.title,
                **plot_args)

    elif args.cmd == "response":
        if not args.phiset:
            print("Must specify --phiset")
        phiset = args.phiset
        response.PlotQSPResponse(
            phiset, signal_operator=args.signal_operator, **plot_args)

    else:
        print(f"[pyqsp.main] Unknown command {args.cmd}")
        print(help_text)

    if (phiset is not None):
        if args.return_angles:
            return phiset
        if args.output_json:
            print(
                f"QSP Phase angles (for signal_operator={args.signal_operator}) in JSON format:")
            if not isinstance(phiset, list):
                phiset = phiset.tolist()
            print(json.dumps(phiset))
