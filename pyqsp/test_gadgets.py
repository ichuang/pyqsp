from pyqsp.gadgets import *
import pennylane as qml

import unittest

###############################################################################################

class Test_gadgets(unittest.TestCase):
    '''
    unit tests for pyqsp gadgets
    '''
    def test_iX_Gate_simple1(self):
        g = iX_Gate()
        assert g is not None

    def test_two_atomic_gadgets(self):
        '''
        create two atomic gadgets and link them together to create a CompositeAtomicGadget
        (soon to be GadgetAssembleage)
        '''
        Xi_1 = [np.array([0, 1, 2, -2, 1, 0])]
        S_1 = [[0, 1, 0, 1, 0]]
        G = AtomicGadget(Xi_1, S_1, label="G")

        Xi_2 = np.array([[np.pi/3, np.pi/2, 0, -0, -np.pi/2, -np.pi/3]])
        S_2 = [[0, 1, 0, 1, 0]]
        G_tilde = AtomicGadget(Xi_2, S_2, label="G_tilde")

        # Performs an interlink of the G gadget with the extraction gadget. Note that 4 is the 
        # degree of the polynomial used in the correction. If it were instead "None" no correction
        # would be applied
        
        G_interlink = G.interlink(G_tilde, [
            (('G1', 0), ('G_tilde', 0), 4)
        ])        
        assert G_interlink is not None
        assert hasattr(G_interlink, 'get_sequence')
        print("In legs = {}".format(G_interlink.in_labels))
        print("Out legs = {}".format(G_interlink.out_labels))
        assert len(G_interlink.in_labels)==3

        # Gets the sequence of a leg of the gadget interlink
        x = G_interlink.get_sequence(('G_tilde', 0))
        assert len(x)==200

    def test_extraction_gadget1(self):
        n = 24

        L = ExtractionGadget(n, 'G').get_sequence(('G', 0))
        L_prime = ExtractionGadget(n, 'G').get_sequence(('G', 0))
        
        Phi, Phi_prime = [], []

        for op in L:
            if isinstance(op, QSP_Rotation):
                Phi.append(op.theta)
        
        for op in L_prime:
            if isinstance(op, QSP_Signal):
                Phi_prime.append(op)

        Phi = Phi[::-1]
        Phi_prime = Phi_prime[::-1]    
        assert len(Phi_prime)==25

        # Testing the protocol!

        dev = qml.device('default.qubit', wires=[0, 1, 2])

        def V():
            qml.CSWAP(wires=[0, 1, 2])
            qml.RX(0.4, wires=2)
            qml.CSWAP(wires=[0, 1, 2])

        @qml.qnode(dev)
        def func():
            for p in range(len(Phi[0:len(Phi)-2])):
                qml.RZ(Phi[p], wires=2)        
        
        dev = qml.device('default.qubit', wires=[0, 1, 2])

        @qml.qnode(dev)
        def func():
            qml.RZ(-0.8, wires=2)
            qml.CSWAP(wires=[0, 1, 2])
            qml.RZ(0.8, wires=1)
            qml.CSWAP(wires=[0, 1, 2])
            return qml.state()

        m = Corrective_CSWAP(1).matrix([1])        
        assert m is not None

        # Construct atomic gadgets
        Xi_1 = np.array([[0, 0]])
        S_1 = [[0]]
        G1 = AtomicGadget(Xi_1, S_1, label="G1")
        
        Xi_2 = np.array([[1, 2, -2, -1]])
        S_2 = [[0, 1, 0]]
        G2 = AtomicGadget(Xi_2, S_2, label="G2")        

        # Construct the interlink between the gadgets
        G = G1.interlink(G2, [(('G1', 0), ('G2', 0), 20)])

        assert len(G1.get_sequence(('G1', 0), correction=8))==40

        # Gets the QSP unitary
        U = lambda x : G1.get_qsp_unitary(('G1', 0), correction=20, rot={('G1', 0):0.4})( {('G1', 0) : x})
        m = U(0.3) @ np.kron(np.kron(np.array([1, 0]), np.array([0, 1])), np.array([1, 0]))
        assert(abs(m[2].imag - np.sin(0.8)) < 1.0e-7)
        assert(abs(m[2] - ( -(np.cos(0.8) - 1j * np.sin(0.8)) * (1j * np.sqrt(1 - 0.1 ** 2)) * -1j )) < 1e-2)
        
