from pyqsp.gadgets import *
import pennylane as qml

import unittest

###############################################################################################

class Test_gadgets(unittest.TestCase):
    '''
    unit tests for pyqsp gadgets
    '''
    def test_iX_Gate_simple1(self):
        '''
        Create an iX gate
        '''
        g = iX_Gate()
        assert g is not None

    def test_cubic_gadget(self):
        '''
        Test cubic polynomial p(x) = x**3 as a gadget
        '''
        G_cubic = AtomicGadget([[0, np.pi/3, -np.pi/3, 0]], [[0, 0, 0]], label="G_cubic")
        G_cheb = AtomicGadget([[0, 0, 0]], [[0, 0]], label="G_cheb")
        # X, Y = G.get_response()
        fn_2 = lambda x : G_cubic.get_qsp_unitary(('G_cheb', 0))({('G_cubic', 0) : x})[0][0]
        X = np.linspace(-1, 1, 200)
        Y= [fn_2(x) for x in X]
        # plt.plot(X, Y)
        assert abs((Y - X**3).mean()) < 1.0e-10        

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
            (('G', 0), ('G_tilde', 0), 4)
        ])        
        assert G_interlink is not None
        assert hasattr(G_interlink, 'get_sequence')
        print("In legs = {}".format(G_interlink.in_labels))
        print("Out legs = {}".format(G_interlink.out_labels))
        assert len(G_interlink.in_labels)==3

        # Gets the sequence of a leg of the gadget interlink
        x = G_interlink.get_sequence(('G_tilde', 0))
        assert len(x)==47	# really should be 200?  Right now sequence not flat: it contains other composite sequences

    def test_extraction_gadget1(self):
        '''
        Extraction gets the z-phase shift of a QSP sequence, such that
        it can be removed and the QSP sequence corrected to become just an x-rotation.
        '''
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

        assert len(G.get_sequence(('G2', 0), correction=8))==17	# really should be 40?  sequences not flat yet

        # Gets the QSP unitary
        U = lambda x : G1.get_qsp_unitary(('G1', 0), correction=20, rot={('G1', 0):0.4})( {('G1', 0) : x})
        m = U(0.3) @ np.kron(np.kron(np.array([1, 0]), np.array([0, 1])), np.array([1, 0]))
        print(f"m[2] = {m[2]}")
        assert(abs(m[2].imag - np.sin(0.8)) < 1.0e-7)
        assert(abs(m[2] - ( -(np.cos(0.8) - 1j * np.sin(0.8)) * (1j * np.sqrt(1 - 0.1 ** 2)) * -1j )) < 1e-2)
        
    def test_sqrt_gadget1(self):
        '''
        Test the SQRT gadget, which is used to perform a sqrt of a unitary.
        Needed for the correction procedure for gadgets.
        '''
        # Defines two atomic gadget

        Xi_1 = [np.array([0, 1, 2, -2, 1, 0])]
        S_1 = [[0, 1, 0, 1, 0]]
        G = AtomicGadget(Xi_1, S_1, label="G")
        
        Xi_2 = np.array([[np.pi/3, np.pi/2, 0, -0, -np.pi/2, -np.pi/3]])
        S_2 = [[0, 1, 0, 1, 0]]
        G_tilde = AtomicGadget(Xi_2, S_2, label="G_tilde")

        # Performs an interlink of the G gadget with the extraction gadget. Note that 20 is the 
        # degree of the polynomial used in the correction. If it were instead "None" no correction
        # would be applied

        G_interlink = G.interlink(G_tilde, [
            (('G1', 0), ('G_tilde', 0), 4)
        ])

        print("In legs = {}".format(G_interlink.in_labels))
        print("Out legs = {}".format(G_interlink.out_labels))

        # Gets the sequence of a leg of the gadget interlink
        G_interlink.get_sequence(('G', 0))

        # Performs tests of the extraction and sqrt gadgets
        G_extraction = ExtractionGadget(29, 'G_ext')
        G_sqrt = SqrtGadget(40, 0.05, 'G_sqrt')

        # check the response function of the extraction gadget
        X, Y = G_extraction.get_response()
        
        # should be near zero for all of abs(X) < 0.7
        # diverges to +1 for X near 1 and to -1 for X near -1
        assert (Y[np.where(X>0.95)] > 0.25).all()
        assert (Y[np.where(X<-0.95)] < -0.25).all()
        assert (abs(Y[np.where(abs(X)<0.8)]) < 0.1).all()        
        assert (abs(Y[np.where(abs(X)<0.7)]) < 0.01).all()        

        # Gets the response function of the sqrt gadget
        X, Y = G_sqrt.get_response()
        y2 = abs(X)*0.25 + 0.75
        assert abs((Y - y2).mean()) < 0.02
        
    def test_abs_gadget1(self):
        '''
        Test the ABS gadget (not yet a class)
        '''
        G_sqrt = SqrtGadget(40, 0.06, 'G_sqrt')
        G_cheb = AtomicGadget([[0, 0, 0]], [[0, 0]], label="G_cheb")
        G_id = G_sqrt.interlink(G_cheb, [
            (('G_sqrt', 0), ('G_cheb', 0), None)
        ])

        fn_2 = lambda x : G_id.get_qsp_unitary(('G_cheb', 0))({('G_sqrt', 0) : x})[0][0]
        X, Y = G_sqrt.get_response()
        data = [fn_2(x) for x in X]
        error = (abs(data - np.abs(X))).mean()
        assert error < 0.08
        
    def otest_abs_gadget2(self):
        '''
        Test the ABS gadget (not yet a class) on a cubic polynomial
        '''
        G_cubic = AtomicGadget([[0, np.pi/3, -np.pi/3, 0]], [[0, 0, 0]], label="G_cubic")
        G_cheb = AtomicGadget([[0, 0, 0]], [[0, 0]], label="G_cheb")
        G_sqrt = SqrtGadget(40, 0.06, 'G_sqrt')

        G_id_1 = G_cubic.interlink(G_sqrt, [
            (('G_cubic', 0), ('G_sqrt', 0), None)
        ])

        G_id_2 = G_id_1.interlink(G_cheb, [
            (('G_sqrt', 0), ('G_cheb', 0), None)
        ])
        
        fn_2 = lambda x : G_id_2.get_qsp_unitary(('G_cheb', 0))({('G_cubic', 0) : x})[0][0]

        X, Y = G_sqrt.get_response()
        data = [fn_2(x) for x in X]
        expected = np.abs(X**3)
        idx = np.where(abs(X)> 0.3)
        error = (abs(data - expected))[idx].mean()
        assert error < 0.2

    def test_addition_gadget1(self):
        '''
        Addition of two gadgets
        '''
        # Defines two atomic gadget

        Xi_1 = [np.array([0, np.pi/3, -np.pi/3, 0])]
        S_1 = [[0, 1, 0]]
        G = AtomicGadget(Xi_1, S_1, label="G")
        
        Xi_2 = np.array([[np.pi/5, np.pi/6, np.pi/6, -np.pi/4]])
        S_2 = [[1, 0, 1]]
        G_tilde = AtomicGadget(Xi_2, S_2, label="G_tilde")
        
        # Performs an interlink of the G gadget with the extraction gadget. Note that deg is the 
        # degree of the polynomial used in the correction. If it were instead "None" no correction
        # would be applied
        deg = 20
        
        G_interlink = G.interlink(G_tilde, [
            (('G', 0), ('G_tilde', 0), deg)
               ])
        AdditionGadget("G_add").get_qsp_unitary(("G_add", 0))({("G_add", 0): 0.3, ("G_add", 1):0.1})
        # Tests the addition gadget
        
        fn = lambda x, y : AdditionGadget("G_add").get_qsp_unitary(("G_add", 0))({("G_add", 0): x, ("G_add", 1):y})[0][0]
        X = np.linspace(-1, 1, 200)
        Y = X
        Z = [[fn(x, y) for x in X ] for y in Y]
        
        # Fourth Chebyshev polynomial
        T_4 = lambda x : 8 * (x ** 4) - 8 * (x ** 2) + 1
        T4_composite = lambda x, y: 0.5 * (T_4(x) + T_4(y))
        Z2 = ([[T4_composite(x, y) for x in X] for y in Y])
        
        err = abs(np.array(Z)-np.array(Z2)).mean()
        assert err < 1.0e-8
