import unittest
from pyqsp.gadgets2 import *

class TestGadgetSeq2circ(unittest.TestCase):

    def setUp(self):
        self.g0 = Gadget(1, 2, "g0")
        self.g1 = Gadget(1, 2, "g1")
        self.a0 = self.g0.wrap_gadget()

    def test_seq2circ1(self):
        g0 = AtomicGadget(1, 1, "g0", [[0, 0]], [[0]])

        g1 = AtomicGadget(1, 1, "g1", [[0.5, -0.5]], [[0]])
        # Generate assemblages of atomic gadgets.
        a0 = g0.wrap_gadget()
        a1 = g1.wrap_gadget()
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
        qasm = circ.circ.qasm()
        print(f"qasm: ", qasm)
        U = circ.get_unitary()
        print(U.dim)
        assert U.dim==(2,2)
        assert U is not None
        assert "rx(0.5) main[0]" in qasm
        assert "rz(0.9) main[0]" in qasm

    def test_get_assemblage_circuit(self):
        '''
        Exercise get_assemblage_circuit
        '''
        g0 = AtomicGadget(2, 2, "g0", [[1, 2, 3],[4, 5, 6]], [[0, 1],[1, 0]])
        g1 = AtomicGadget(2, 2, "g1", [[1, 2, 3],[4, 5, 6]], [[0, 1],[1, 0]])
        
        a0 = g0.wrap_gadget()

        sqcirc = a0.get_assemblage_circuit()
        print(sqcirc)
        print(f"circuit size is {sqcirc.size()}")
        assert sqcirc is not None
        assert sqcirc.size()==10

    def test_sequence_circuit_size1(self):
        '''
        Test computation of sequence circuit size
        '''
        sequence = [ XGate(0.1, target=0, controls=None) ]
        csinfo = sequence_circuit_size(sequence)
        assert csinfo['nqubits_main'] == 1
        assert csinfo['nqubits_ancillae'] == 0

    def test_sequence_circuit_size2(self):
        sequence = [ XGate(0.1, target=0, controls=[1, 2]) ]
        csinfo = sequence_circuit_size(sequence)
        print(f"[test_sequence_circuit_size2] seq={sequence} csinfo={csinfo}")
        assert csinfo['nqubits_main'] == 1
        assert csinfo['nqubits_ancillae'] == 2

    def test_circuit_from_assemblage_full_sequence1(self):
        g0 = AtomicGadget(2, 2, "g0", [[1, 2, 3],[4, 5, 6]], [[0, 1],[1, 0]])
        g1 = AtomicGadget(2, 2, "g1", [[7, 8, 9],[10, 11, 12]], [[0, 1],[1, 0]])
        g2 = AtomicGadget(2, 2, "g2", [[7, 8, 9],[10, 11, 12]], [[0, 1],[1, 0]])
        # Generate assemblages of atomic gadgets.
        a0 = g0.wrap_gadget()
        a1 = g1.wrap_gadget()
        a2 = g2.wrap_gadget()
        
        a3 = a0.link_assemblage(a1, [(("g0", 0),("g1", 0))])
        full_seq = a3.sequence
        sqcirc = seq2circ(full_seq)
        assert sqcirc is not None
        assert sqcirc.nqubits==a3.required_ancillae + len(full_seq)
        ngates = sqcirc.circ.size()
        print("Created circuit with {circ.nqubits} qubits and {ngates} gates")
        sqcirc.draw(output='mpl', filename="test_circuit_from_assemblage_full_sequence1.png")

if __name__ == '__main__':
    unittest.main()
        
