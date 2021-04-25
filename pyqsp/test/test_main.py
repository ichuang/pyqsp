import os
import unittest

import numpy as np

from pyqsp import main

# -----------------------------------------------------------------------------
# unit tests


class Test_main(unittest.TestCase):

    def test_main1(self):
        cmdline = "--return-angles --poly=-1,0,2 poly2angles"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main2(self):
        cmdline = "--return-angles --poly=-1,0,2 --plot --hide-plot poly2angles"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main3(self):
        cmdline = "--return-angles --hide-plot --signal_operator=Wz --poly=0,0,0,1 --plot poly2angles"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main4(self):
        cmdline = "--return-angles --hide-plot --tau 10 hamsim"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main5(self):
        cmdline = "--return-angles --hide-plot invert"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main6(self):
        cmdline = "--return-angles --hide-plot --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign poly"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main7(self):
        cmdline = "--return-angles --hide-plot --plot-real-only --plot --polyargs=20,20 --polyname poly_thresh poly"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main8(self):
        cmdline = "--return-angles --hide-plot --plot-real-only --plot --polyargs=20,3.5 --polyname gibbs poly"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main9(self):
        cmdline = "--return-angles --hide-plot --plot-positive-only --plot-real-only --plot --polyargs 20,0.2,0.9 --polyname efilter poly"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)
