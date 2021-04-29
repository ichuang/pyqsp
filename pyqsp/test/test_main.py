import unittest

from pyqsp import main

# -----------------------------------------------------------------------------
# unit tests
test_cmds = [
    "--return-angles --poly=-1,0,2 poly2angles",
    "--return-angles --poly=-1,0,2 --plot --hide-plot poly2angles",
    "--plot-positive-only --plot-magnitude --plot-npts=400 --seqargs=10,0.5 fpsearch",
    "--plot-real-only --plot-npts=400 --seqargs=19,10 poly_sign",
    "--plot-real-only --plot-npts=400 --seqargs=3,0.3 invert",
    "--plot-real-only --plot-npts=400 --seqargs=10,0.1 hamsim",
    "--plot-real-only --plot-npts=400 --seqargs=18,10 poly_thresh",
    "--plot-positive-only --plot-real-only --seqargs 30,0.3 efilter",
    "--plot-positive-only --plot-real-only --seqargs=20,3.5 gibbs",
    "--plot-real-only --seqargs=20,0.6,15 relu",
]


class Test_main(unittest.TestCase):
    def test_main(self):
        for i, cmd in enumerate(test_cmds):
            with self.subTest(i=i):
                phiset = main.CommandLine(arglist=cmd.split(" "))
