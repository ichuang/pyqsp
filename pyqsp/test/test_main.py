import os
import numpy as np
from pyqsp import main

#-----------------------------------------------------------------------------
# unit tests

import unittest

class Test_main(unittest.TestCase):

    def test_main1(self):
        cmdline = "--return-angles --poly=-1,0,2 poly2angles"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main2(self):
        cmdline = "--return-angles --poly=-1,0,2 --plot --hide-plot --align-first-point-phase poly2angles"
        phiset = main.CommandLine(arglist=cmdline.split(" "))
        assert len(phiset)

    def test_main3(self):
        cmdline = "--return-angles --hide-plot --model=Wz --poly=0,0,0,1 --plot poly2angles"
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
