import argparse
import json
import sys

import numpy as np

import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.phases import phase_generators
from pyqsp.poly import (StringPolynomial, TargetPolynomial,
                        polynomial_generators)

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
        help="comma delimited list of floating-point coeficients for polynomial, as const, a, a^2, ...",
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
        "--tau",
        help="time value for Hamiltonian simulation (hamsim command)",
        type=float,
        default=100)
    parser.add_argument(
        "--epsilon",
        help="parameter for polynomial approximation to 1/a, giving bound on error",
        type=float,
        default=0.3)
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
        help="name of polynomial generate using the 'poly' command, e.g. 'sign'",
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
        help="method to use for qsp phase angle generation, either 'laurent' (default) or 'tf' (for tensorflow + keras)",
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
        help="number of discretized values of theta to use in tensorflow optimization",
        type=int,
        default=30)

    if not args:
        args = parser.parse_args(arglist)

    phiset = None
    plot_args = dict(plot_magnitude=args.plot_magnitude,
                     plot_probability=args.plot_probability,
                     plot_positive_only=args.plot_positive_only,
                     plot_real_only=args.plot_real_only,
                     plot_tight_y=args.plot_tight_y,
                     npts=args.plot_npts,
                     show=(not args.hide_plot),
                     show_qsp_model_plot=args.plot_qsp_model,
                     )

    qspp_args = dict(signal_operator=args.signal_operator,
                     method=args.method,
                     tolerance=args.tolerance,
                     nepochs=args.nepochs,
                     npts_theta=args.npts_theta,
                     )

    if args.cmd == "poly2angles":
        coefs = args.poly
        if not coefs:
            print(
                f"[pyqsp.main] must specify polynomial coeffients using --poly, e.g. --poly -1,0,2")
            sys.exit(0)
        if isinstance(coefs, str):
            coefs = list(map(float, coefs.split(",")))
        print(f"[pyqsp] polynomial coefficients={coefs}")
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            coefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                pcoefs=coefs,
                signal_operator=args.signal_operator,
                **plot_args)

    elif args.cmd == "hamsim":
        pg = pyqsp.poly.PolyCosineTX()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.cos(args.seqargs[0] * x),
                signal_operator="Wx",
                title="Hamiltonian Simultation (Cosine)",
                **plot_args)

        pg = pyqsp.poly.PolySineTX()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.sin(args.seqargs[0] * x),
                signal_operator="Wx",
                title="Hamiltonian Simultation (Sine)",
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
        pcoefs, scale = pg.generate(
            *args.seqargs,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True)
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
            return_scale=True)
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
            return_scale=True)
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
            return_scale=True)
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
            return_scale=True)
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
            return_scale=True)
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
            return_scale=True)
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
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale *
                1 - (np.sign(x + 1 / args.kappa) -
                     np.sign(x - 1 / args.kappa)) / 2,
                signal_operator="Wx",
                title="Rect Function",
                **plot_args)

    elif args.cmd == "invert_rect":
        pg = pyqsp.poly.PolyOneOverXRect()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, **qspp_args)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * 1 / x,
                signal_operator="Wx",
                title="Poly Rect * 1/x",
                **plot_args)

    elif args.cmd == "poly_linear_amp":
        pg = pyqsp.poly.PolyLinearAmplification()
        pcoefs, scale = pg.generate(
            *args.seqargs,
            ensure_bounded=True,
            return_scale=True)
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
        pg = polynomial_generators[args.polyname]()
        if not args.polyargs:
            print(pg.help())
            return
        pcoefs = pg.generate(*args.polyargs)
        print(f"[pyqsp] polynomial coefs = {pcoefs}")
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
