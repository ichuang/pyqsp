'''
pyqsp/gadgets.py

Subroutines for implementation of modular M-QSP protocols and gadgets

Functionality that we want
    - We want the ability to define arrays of gadgets, encoding particular polynomials.
    - We want the ability, for a given collection of gadgets, to output the M-QSP sequence which will yield the desired functional output
    - We want the ability to determine the true function that is implemented, and to compare it to the idealized function
    - We want the ability to compute the cost of applying a particular network of gadgets
    - We want the ability to draw gadget networks
    - Determine whether the output of a gadget is half-twisted embeddable?
'''
import numpy as np


class Gadget:
    """
    Class for (a, b) gadgets

    Args:
        a (int): Number of input legs to gadget
        b (int): Number of output legs to gadget 
    Kwargs:
        var_labels (list)
    """
    def __init__(self, a, b, label):
        self.a, self.b = a, b

        # Assigns generic labels the input legs of a gadget
        self.label = label
        self.in_labels, self.out_labels = [(label, j) for j in range(a)], [(label, j) for j in range(b)]

    @classmethod
    def unitary(self):
        """
        The unitary mapping effectuated by a gadget
        """
        raise NotImplementedError


class AtomicGadget(Gadget):
    """
    Class for not-quite atomic gadgets: to be precise, this class encompasses the closure of all gadgets which 
    can be achieved via interlinking atomic gadgets and correction protocols, arbitrarily.
    """
    def __init__(self, Xi, S, label):
        self.Xi = Xi
        self.S = S
        self.ancillas = []

        a, b = len(set([tuple(s) for s in S])), len(Xi)
        super().__init__(a, b, label)
    
    def unitary(self, leg):
        """
        Returns a function which takes input unitaries, and outputs unitaries corresponding to the 
        gadget
        """
        return lambda U : compute_mqsp_unitary(U, self.Xi[leg], self.S[leg])

    def get_polynomial(self, leg):
        """
        Generates and returns the polynomial corresponding to a particular output leg
        of an atomic gadget.
        """
        input_unitaries = lambda vars : [W(vars[self.var_labels[i]]) for i in range(len(self.var_labels))]
        return lambda vars : self.unitary(leg)(input_unitaries(vars))[0][0]
    
    def get_sequence(self, label):
        """
        Gets the sequence
        """
        leg = self.out_labels.index(label)
        return self.Xi[leg], [(self.label, s) for s in self.S[leg]]

    def interlink(self, gadget, interlink):
        """
        Performs an interlink of an atomic gadget with a gadget
        """
        return CompositeAtomicGadget(self, gadget, interlink)


def _seq_extend(seq, extension):
    """Extends a sequence"""
    if len(seq) > 0:
        print(seq)
        print(extension)
        extension[0] = seq[len(seq)-1] + extension[0]
        seq = list(seq[:-1]) + list(extension)
    else:
        seq = extension
    return seq


class CompositeAtomicGadget(Gadget):
    """
    A particular class of gadget arising from interlinking of two gadgets AtomicGadget instances, which 
    tracks the internal structure of each gadget being linked.
    """
    def __init__(self, gadget_1, gadget_2, interlink):
        self.gadget_1, self.gadget_2 = gadget_1, gadget_2
        self.gadget_labels = {gadget_1.label : gadget_1, gadget_2.label : gadget_2}
        
        # Interlink parameters
        self.interlink = interlink
        self.B, self.C = list(np.array(interlink).T[0]), list(np.array(interlink).T[1])

        # Gets the input and output legs
        self.a, self.b = gadget_1.a + (gadget_2.a - len(interlink)), gadget_2.b + (gadget_1.b - len(interlink))
        self.in_labels = gadget_1.in_labels + list(filter(lambda x : x[1] not in self.C, gadget_2.in_labels))
        self.out_labels = gadget_2.out_labels + list(filter(lambda x : x[1] not in self.B, gadget_1.out_labels))

    def get_sequence(self, label):
        """
        Gets the composite sequence arising from gadget composition. Both gadget involved in 
        the composition must be atomic gadgets, with defined M-QSP sequences.
        """
        if label[0] == self.gadget_2.label:
            external_Xi, external_S = self.gadget_2.get_sequence(label) # Gets the output gadget's sequence
            Phi_seq, S_seq = [external_Xi[0]], []

            for j in range(len(external_S)):
                if external_S[j][0] == self.gadget_2.label and external_S[j][1] in self.C:
                    new_leg = self.B[self.C.index(external_S[j][1])]
                    internal_Xi, internal_S = self.gadget_1.get_sequence((self.gadget_1.label, new_leg))
                    S_seq.extend(internal_S)

                    internal_Xi[len(internal_Xi)-1] = external_Xi[j + 1] + internal_Xi[len(internal_Xi)-1]
                    Phi_seq = _seq_extend(Phi_seq, internal_Xi)
                else:
                    Phi_seq.append(external_Xi[j + 1])
                    S_seq.append(external_S[j])
        if label[0] == self.gadget_1.label:
            Phi_seq, S_seq = self.gadget_1.get_sequence(label)
        return Phi_seq, S_seq

 
################################################################################
# Operations on gadgets and collections of gadgets
# STILL NEED TO DEFINE SEQUENCE FOR CORRECTION

def correction(gadget, legs):
    """
    Applies the correction protocol to an atomic gadget
    """
    pass

def pin(gadget, legs, vals):
    """
    Pins a gadget
    """

def permute(gadget, permutations):
    """
    Permutes the legs of a gadget
    """

def multiply_gadget_legs(gadget1, gadget2, leg1, leg2):
    """
    Multiply the legs of gadgets
    """
    pass

def sum_gadget_legs(gadget1, gadget2, leg1, leg2):
    """
    Sums the legs of gadgets
    """
    pass

########################################################################################
# Not really sure how we'll incorporate this, warrants discussion

class GadgetAssemblage:
    """
    Global gadget assemblage

    Keeps a records of gadgets and interlinks
    """
    def __init__(self, width):
         self.width = width
         self.gadgets = []
         self.interlinks = []
    
    def queue(self, operation):
        """
        Appends gadgets and interlinks for a gadget assemblage
        """
        if isinstance(operation, Gadget):
            self.gadgets.append(operation)
        if isinstance(operation, Interlink):
            self.interlinks.append(operation, Interlink)


###################### Instances of gadgets ######################

class MultiplicationGadget(AtomicGadget):
    """
    An multiplication gadget
    """
    def __init__(self, phi=-np.pi/4):
        self.Phi, self.S = np.array([phi, np.pi/4, -np.pi/4, -phi]), np.array([0, 1, 0])
        self.Xi = [self.Phi, self.S] # Defines the gadget phase sequence


class AdditionGadget(AtomicGadget):
    """
    An addition gadget
    """
    def __init__(self, phi=-np.pi/4):
        self.Phi = np.array([phi, -phi])
        self.S = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        self.Xi = [self.Phi, self.S] # Defines the gadget phase sequence 


class InvChebyshevGadget(AtomicGadget):
    """
    An inverse Chebyshev polynomial gadget
    """
    def __init__(self, n, phi=-np.pi/4):
        pass


################################################################################################

def Rz(phi):
    """
    sigma_z-rotation
    """
    return np.array([
        [np.exp(1j * phi), 0],
        [0, np.exp(-1j * phi)]
    ])


def W(x):
    """
    Standard sigma_x signal operator
    """
    return np.array([
        [x, 1j * np.sqrt(1 - x ** 2)],
        [1j * np.sqrt(1 - x ** 2), x]
    ])


def compute_mqsp_unitary(U, Phi, s):
    """
    Computes an M-QSP unitary
    """
    output_U = Rz(Phi[0])
    for i in range(len(s)):
        output_U = output_U @ U[s[i]] @ Rz(Phi[i])
    return output_U
