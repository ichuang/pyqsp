from pyqsp.gadgets import *

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

        
