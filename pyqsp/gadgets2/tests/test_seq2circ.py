import unittest
from pyqsp.gadgets2 import *
# from gadget_assemblage import *

"""
Note these tests can be run with 'python -m unittest tests.test_gadget_assemblage' from outside the folder tests
"""

class TestGadgetSeq2circ(unittest.TestCase):

    def setUp(self):
        self.g0 = Gadget(1, 2, "g0")
        self.g1 = Gadget(1, 2, "g1")
        self.a0 = self.g0.wrap_gadget()

    def test_seq2circ1(self):
        g0 = AtomicGadget(1, 1, "g0", [[0, 0]], [[0]])

        g1 = AtomicGadget(1, 1, "g1", [[0.5, -0.5]], [[0]])
        # Generate assemblages of atomic gadgets.
        a0 = g0.wrap_atomic_gadget()
        a1 = g1.wrap_atomic_gadget()
        a2 = a0.link_assemblage(a1, [(("g0", 0), ("g1", 0))])
        seq = a2.sequence[0]

        print(f"seq = {seq}")
        for idx, so in enumerate(seq):
            print(f"  {idx:02d}: {so}")

        circ = seq2circ(seq)
        assert circ is not None

    def test_seq2circ2(self):
        th1 = 0.5
        th2 = 0.9
        seq = [ XGate(th1), ZGate(th2) ]
        circ = seq2circ(seq)
        U = circ.get_unitary()
        print(U.dim)
        assert U.dim==(2,2)
        assert U is not None
        qasm = circ.circ.qasm()
        print(f"qasm: ", qasm)
        assert "rx(0.5) q[0]" in qasm
        assert "rz(0.9) q[0]" in qasm

    def otest_get_assemblage_circuit(self):
        '''
        Exercise get_assemblage_circuit
        '''
        g0 = AtomicGadget(2, 2, "g0", [[1, 2, 3],[4, 5, 6]], [[0, 1],[1, 0]])
        g1 = AtomicGadget(2, 2, "g1", [[1, 2, 3],[4, 5, 6]], [[0, 1],[1, 0]])
        
        a0 = g0.wrap_atomic_gadget()

        circ = a0.get_assemblage_circuit(1, 1, 0)
        print(circ)
        print(f"circuit size is {circ.circ.size()}")
        assert circ is not None
        assert circ.circ.size()==5

if __name__ == '__main__':
    unittest.main()
        
