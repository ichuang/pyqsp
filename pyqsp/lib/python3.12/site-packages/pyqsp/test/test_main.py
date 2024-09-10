import os
import unittest

from pyqsp import main

# -----------------------------------------------------------------------------
# unit tests

test_cmds = [
    "--return-angles --poly=-1,0,2 poly2angles",
    "--return-angles --poly=-1,0,2 --plot --hide-plot poly2angles",
    "--plot-positive-only --plot-probability --plot-tight-y --plot-npts=400 --seqargs=10,0.5 fpsearch",
    "--plot-real-only --plot-npts=400 --seqargs=19,10 poly_sign",
    "--plot-real-only --plot-npts=400 --seqargs=3,0.3 invert",
    "--plot-real-only --plot-npts=400 --seqargs=10,0.1 hamsim",
    "--plot-real-only --plot-npts=400 --seqargs=18,10 poly_thresh",
    "--plot-real-only --plot-npts=400 --seqargs=19,0.25 poly_linear_amp",
    "--plot-real-only --plot-npts=400 --seqargs=18,10 poly_phase",
    "--plot-positive-only --plot-real-only --seqargs 30,0.3 efilter",
    "--plot-positive-only --plot-real-only --seqargs=20,3.5 gibbs",
    "--plot-real-only --seqargs=20,0.6,15 relu",
]


test_cmds_tf = [
    '--return-angles --func np.cos(3*x) --polydeg 6 polyfunc',
    '--plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign --method tf poly',
]


class Test_main(unittest.TestCase):
    def test_main(self):
        for i, cmd in enumerate(test_cmds):
            with self.subTest(i=i):
                print(f"[pyqsp.test_main testing '{cmd}'")
                phiset = main.CommandLine(arglist=cmd.split(" "))

    def test_main_tf(self):
        '''
        These are slow tests, and run only if PYQSP_TEST_QSP_MODELS is set in the environment
        '''
        enabled = 'PYQSP_TEST_QSP_MODELS' in os.environ
        if not enabled:
            print(
                "[pyqsp.test] Skipping qsp_model tests: export PYQSP_TEST_QSP_MODELS=1 to enable these tests")
            return

        for i, cmd in enumerate(test_cmds_tf):
            with self.subTest(i=i):
                print(f"[pyqsp.test_main testing '{cmd}'")
                phiset = main.CommandLine(arglist=cmd.split(" "))
