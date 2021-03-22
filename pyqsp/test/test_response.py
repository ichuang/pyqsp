import os
import numpy as np
from pyqsp import LPoly
from pyqsp import response
from pyqsp import angle_sequence

#-----------------------------------------------------------------------------
# unit tests

import unittest

class Test_response(unittest.TestCase):

    def test_rsponse1(self):
        response.PlotQSPResponse([0,0,0,0,0,0,0,0,0,0], show=False)
        assert True

    def test_rsponse2(self):
        #coefs = [0, -3, 0, 4]
        # coefs = [-1, 0, 2]
        coefs = [1, 0, -8, 0, 8]
        phiset = angle_sequence.QuantumSignalProcessingWxPhases(coefs)
        print(f"QSP angles = {phiset / np.pi} * pi")
        response.PlotQSPResponse(phiset, show=False)
        assert True

    def test_response3(self):
        laurent_coefs = [-0.000040560361327379724,0,   0.00015641720852954677,0,   -0.0005355850721002753,0,   0.0016414913408482334,0,   -0.004533861582189047,0,0.011351591436778108,   0,-0.02589608179323477,0,   0.054076031858869555,0,-0.10380535550410741,   0,0.18392482137699062,0,-0.3019956131896606,   0,0.4613911821367651,0,-0.6587380770236564,0,   0.8829959121223965,0,0.8829959121223965,0,   -0.6587380770236564,0,0.4613911821367651,0,   -0.3019956131896606,0,0.18392482137699062,0,   -0.10380535550410741,0,0.054076031858869555,   0,-0.02589608179323477,0,   0.011351591436778108,0,-0.004533861582189047,   0,0.0016414913408482334,0,   -0.0005355850721002753,0,   0.00015641720852954677,0,   -0.000040560361327379724]
        laurent_coefs = np.array(laurent_coefs)/5
        print(f"laurent coefs={laurent_coefs}")
        phiset = angle_sequence.angle_sequence(laurent_coefs, eps=1.0e-3, suc=0.99)
        print(f"QSP angles = {phiset}")
        response.PlotQSPResponse(phiset, model="Wz", show=False)
        assert True

    def test_response4(self):
        laurent_coefs = [0, 0, 0, 0, 0, 0, 1]
        print(f"laurent coefs={laurent_coefs}")
        phiset = angle_sequence.angle_sequence(laurent_coefs)
        print(f"QSP angles = {phiset}")
        response.PlotQSPResponse(phiset, model="Wz", show=False)
        assert True

    
