import unittest
import numpy as np
from pyqsp.gadgets import *

# Temporary matplotlib import
from matplotlib import pyplot as plt

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
        th1 = 0.5 / 2		# see TEMPORARY doubling in seq2circ.py
        th2 = 0.9 / 2		# see TEMPORARY doubling in seq2circ.py
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

    def test_get_unitary3(self):
        seq = [ XGate(0.5, controls=[1], target=0) ]
        sqcirc2 = seq2circ(seq)
        umat = sqcirc2.get_unitary()
        assert umat is not None
        assert umat.dim==(4,4)

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
        # # Used to draw full circuit to local png.
        # sqcirc.draw(output='mpl', filename="test_circuit_from_assemblage_full_sequence1.png")

    def test_parameters1(self):
        '''
        Test the setting of parameters for signal operators.
        '''
        seq = [ SignalGate(0, target=0) ]
        sqcirc = seq2circ(seq)
        for th in np.linspace(-4, 4, 10):
            sqcirc.bind_parameters([th])
            umat = sqcirc.get_unitary()
            assert umat.dim==(2,2)
            assert abs(umat.data[1,0]-(-1j*np.sin(th/2))) < 1.0e-3

    def test_response_function1(self):
        '''
        test response function method of SeqeuenceQuantumCircuit
        '''
        # try a pi/4 gadget
        ag = AtomicGadget(1,1,"QSP",[ [ 0,np.pi/4, -np.pi/4, 0]], [[0, 0, 0]])	# see TEMPORARY doubling in seq2circ.py
        seq = ag.get_gadget_sequence()
        qc = seq2circ(seq, verbose=False)
        print(f"U = ", qc.get_unitary(values=[0]).data)
        # qc.draw('mpl')
        X, Y = qc.one_dim_response_function(npts=100)
        # plt.plot(X[:,0], Y)
        assert abs(X[0,0] - (-1)) < 1.0e-3
        assert abs(X[-1,0] - (1)) < 1.0e-3
        assert abs(Y[0] - (-1)) < 1.0e-3
        assert abs(Y[-1] - (1)) < 1.0e-3

    def test_gadget_unitary(self):
        '''
        test get_gadget_unitary method of AtomicGadget
        '''
        # try a pi/4 gadget
        ag = AtomicGadget(1,1,"QSP",[ [ 0,np.pi/2, -np.pi/2, 0]], [[0, 0, 0]])
        umat = ag.get_gadget_unitary(signal_values=[0.2])
        assert umat.data.shape==(2,2)

    def test_z_correction(self):
        '''
        test response function for z-correction using pi/4 gadget
        '''
        # try a pi/4 gadget
        ag = AtomicGadget(1,1,"QSP",[[0, np.pi/4, -np.pi/4, 0]], [[0, 0, 0]])
        seq = ag.get_gadget_sequence()
       
        # Manually get first leg and correct it.
        leg_0 = seq[0]
        seq_corrected = get_twice_z_correction(leg_0)
        
        qc = seq2circ(seq_corrected, verbose=False)
        print(f"U = ", qc.get_unitary(values=[0]).data)
        # qc.draw('mpl')
        X, Y = qc.one_dim_response_function(npts=100)

        # Chart the absolute value (near one) and phase.
        Y_abs = np.abs(Y) 
        Y_ang = np.angle(Y)

        # average error between actual Z phase and that generated by the extraction procedure
        qc_orig = seq2circ(seq, verbose=False)
        X, Y_orig = qc_orig.one_dim_response_function(uindex=(0,1), npts=100)
        Y_orig_ang = np.angle(Y_orig)

        # only compute this error away from edges
        idx = np.where(abs(X[:,0]) < 0.8)
        Y_ang = Y_ang[idx]
        Y_orig_ang = Y_orig_ang[idx]
        error = abs(Y_orig_ang - (Y_ang + np.pi/2)).mean()	# note offset by pi/2
        print(error)
        assert error < 1.0e-3

        # compare with theoretical prediction
        Yth = -np.arctan(2*(X[idx, 0]**2))
        err_th = abs(Y_orig_ang - Yth).mean()
        print(f"Error compared with analytical theory: {err_th}")
        assert err_th < 1.0e-3

    def test_linked_trivial_gadgets_1(self):
        '''
        Take two length 1 atomic gadgets, and connect them.
        '''
        # Both gadgets have trivial phases.
        ag0 = AtomicGadget(1,1,"QSP0",[[0, 0]], [[0]])
        ag1 = AtomicGadget(1,1,"QSP1",[[0, 0]], [[0]])

        a0 = ag0.wrap_gadget()
        a1 = ag1.wrap_gadget()
        a2 = a0.link_assemblage(a1, [(("QSP0", 0),("QSP1", 0))])

        # This assertion will fail when single variable protocols are optimized.
        assert a2.required_ancillae == 1

        # Retrieve sequence and get first and only leg.
        seq = a2.sequence
        leg_0 = seq[0]
        # Retrieve circuit.
        qc = seq2circ(leg_0, verbose=False)
        X, Y = qc.one_dim_response_function(npts=100)

        X = X[:,0]
        diff_sum = sum([(X[k] - Y[k])**2 for k in range(len(X))])
        assert abs(diff_sum) < 1.0e-3
        
        # plt.close()
        # plt.figure()
        # plt.plot(X[:,0], Y)
        # plt.show()

    def test_linked_trivial_gadgets_2(self):
        '''
        Take two length 2 atomic gadgets, and connect them.
        '''
        # Both gadgets have trivial phases.
        ag0 = AtomicGadget(1,1,"QSP0",[[0, 0, 0]], [[0, 0]])
        ag1 = AtomicGadget(1,1,"QSP1",[[0, 0, 0]], [[0, 0]])

        a0 = ag0.wrap_gadget()
        a1 = ag1.wrap_gadget()
        a2 = a0.link_assemblage(a1, [(("QSP0", 0),("QSP1", 0))])

        # This assertion will fail when single variable protocols are optimized.
        assert a2.required_ancillae == 1

        # Retrieve sequence and get first and only leg.
        seq = a2.sequence
        leg_0 = seq[0]
        # Retrieve circuit.
        qc = seq2circ(leg_0, verbose=False)
        X, Y = qc.one_dim_response_function(npts=100)

        X = X[:,0]
        X_cheb = 1 - 8*X**2 + 8*X**4
        # Check that achieved function is the fourth Chebyshev polynomial
        diff_sum = sum([(X_cheb[k] - Y[k])**2 for k in range(len(X))])
        assert abs(diff_sum) < 1.0e-3
        
        # plt.close()
        # plt.figure()
        # plt.plot(X[:,0], Y)
        # plt.show()

if __name__ == '__main__':
    unittest.main()
        
