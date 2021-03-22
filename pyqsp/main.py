import os
import sys
import pyqsp
import argparse
from pyqsp import ham_sim
from pyqsp import response
from pyqsp import angle_sequence

#-----------------------------------------------------------------------------

class VAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        curval = getattr(args, self.dest, 0) or 0
        values=values.count('v')+1
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
    invert      - compute QSP phase angles for matrix inversion, i.e. a polynomial approximation to 1/a, for given kappa and epsilon parameter values

Examples:

    pyqsp --poly=-1,0,2 poly2angles
    pyqsp --poly=-1,0,2 --plot --align-first-point-phase poly2angles
    pyqsp --model=Wz --poly=0,0,0,1 --plot  poly2angles
    pyqsp --plot --tau 10 hamsim
    pyqsp --plot invert

""".format(version)

    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

    def float_list(value):
        return list(map(float, value.split(",")))

    parser.add_argument("cmd", help="command")
    parser.add_argument('-v', "--verbose", nargs=0, help="increase output verbosity (add more -v to increase versbosity)", action=VAction, dest='verbose')
    parser.add_argument("-o", "--output", help="output filename", default="")
    parser.add_argument("--model", help="QSP sequence model, either Wx (signal is X rotations) or Wz (signal is Z rotations)", type=str, default="Wx")
    parser.add_argument("--plot", help="generate QSP response plot", action="store_true")
    parser.add_argument("--hide-plot", help="do not show plot (but it may be saved to a file if --output is specified)", action="store_true")
    parser.add_argument("--return-angles", help="return QSP phase angles to caller", action="store_true")
    parser.add_argument("--poly", help="comma delimited list of floating-point coeficients for polynomial, as const,a,a^2,...", action="store", type=float_list)
    parser.add_argument("--tau", help="time value for Hamiltonian simulation (hamsim command)", type=float, default=100)
    parser.add_argument("--kappa", help="parameter for polynomial approximation to 1/a, valid in the regions 1/kappa < a < 1 and -1 < a < -1/kappa", type=float, default=3)
    parser.add_argument("--epsilon", help="parameter for polynomial approximation to 1/a, giving bound on error", type=float, default=0.3)
    parser.add_argument("--align-first-point-phase", help="when plotting change overall complex phase such that the first point has zero phase", action="store_true")

    if not args:
        args = parser.parse_args(arglist)

    phiset = None

    if args.cmd=="poly2angles":
        coefs = args.poly
        if not coefs:
            print(f"[pyqsp.main] must specify polynomial coeffients using --poly, e.g. --poly -1,0,2")
            sys.exit(0)
        if type(coefs)==str:
            coefs = list(map(float, coefs.split(",")))
        print(f"[pyqsp] polynomial coefficients={coefs}")
        phiset = angle_sequence.QuantumSignalProcessingPhases(coefs, model=args.model)        
        if args.plot:
            response.PlotQSPResponse(phiset, pcoefs=coefs, model=args.model, align_first_point_phase=args.align_first_point_phase, show=(not args.hide_plot))

    elif args.cmd=="hamsim":
        phiset, telapsed = ham_sim.ham_sim(args.tau, 1.0e-4, 1 - 1.0e-4)
        if args.plot:
            response.PlotQSPResponse(phiset, model="Wz", align_first_point_phase=args.align_first_point_phase, show=(not args.hide_plot))

    elif args.cmd=="invert":
        pcoefs = pyqsp.poly.PolyOneOverX(args.kappa, args.epsilon, return_coef=True, ensure_bounded=True)
        phiset = angle_sequence.QuantumSignalProcessingWxPhases(pcoefs)
        if args.plot:
            response.PlotQSPResponse(phiset, pcoefs=pcoefs, model="Wx", align_first_point_phase=args.align_first_point_phase, show=(not args.hide_plot))

    else:
        print(f"[pyqsp.main] Unknown command {cmd}")
        print(help_text)

    if (phiset is not None) and args.return_angles:
        return phiset
