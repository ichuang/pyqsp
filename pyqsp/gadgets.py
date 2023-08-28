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
import copy
from .phases import ExtractionSequence
import itertools

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
        self.labels = [label]

        a, b = len(set(list(itertools.chain.from_iterable(S)))), len(Xi)
        super().__init__(a, b, label)

        self.depth = 1
    
    def get_unitary(self, label):
        """
        Returns a function which takes input unitaries, and outputs unitaries corresponding to the 
        gadget
        """
        Xi, S = self.get_sequence(label)
        return lambda U : compute_mqsp_unitary(U, Xi, S) 

    def get_qsp_unitary(self, label):
        """
        Generates and returns the polynomial corresponding to a particular output leg
        of an atomic gadget.

        Args:
            vars (dict): A dictionary assigning values to input legs
        """
        input_unitaries = lambda vars : {l : W(vars[l]) for l in self.in_labels}
        return lambda vars : self.get_unitary(label)(input_unitaries(vars)) # Returns top-left entry  
    
    def get_sequence(self, label, correction=None):
        """
        Gets the sequence
        """
        leg = self.out_labels.index(label)
        Phi, S = self.Xi[leg], [(self.label, s) for s in self.S[leg]]
        if correction is not None:
            Phi, S = corrected_sequence(Phi, S, self.depth, correction)
        return Phi, S

    def get_corrected_gadget(self):
        """
        Gets corrected sequence
        """
        pass

    def interlink(self, gadget, interlink):
        """
        Performs an interlink of an atomic gadget with a gadget
        """
        return CompositeAtomicGadget(self, gadget, interlink)


def _seq_extend(seq, extension):
    """Extends a sequence"""
    if len(seq) > 0:
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
        self.label = "{} - {}".format(gadget_1.label, gadget_2.label)
        self.labels = gadget_1.labels + gadget_2.labels

        self.depth = gadget_1.depth + gadget_2.depth
        # Interlink parameters
        self.B, self.C, self.correction = [x[0] for x in interlink], [x[1] for x in interlink], [x[2] for x in interlink]

        # Gets the input and output legs
        self.a, self.b = gadget_1.a + (gadget_2.a - len(interlink)), gadget_2.b + (gadget_1.b - len(interlink))
        self.in_labels = gadget_1.in_labels + list(filter(lambda x : x not in self.C, gadget_2.in_labels))
        self.out_labels = gadget_2.out_labels + list(filter(lambda x : x not in self.B, gadget_1.out_labels))

    def get_sequence(self, label, correction=None):
        """
        Gets the composite sequence arising from gadget composition. Both gadget involved in 
        the composition must be atomic gadgets, with defined M-QSP sequences.
        """
        if label[0] in self.gadget_2.labels:
            external_Xi, external_S = self.gadget_2.get_sequence(label) # Gets the output gadget's sequence
            Phi_seq, S_seq = [external_Xi[0]], []

            for j in range(len(external_S)):
                if external_S[j] in self.C:
                    new_leg_ind = self.C.index(external_S[j])
                    new_leg, corr = self.B[new_leg_ind], self.correction[new_leg_ind]
                    internal_Xi, internal_S = self.gadget_1.get_sequence(new_leg, correction=corr)
                    internal_Xi, internal_S = copy.deepcopy(internal_Xi), copy.deepcopy(internal_S) # Copies

                    S_seq.extend(internal_S)

                    internal_Xi[len(internal_Xi)-1] = external_Xi[j + 1] + internal_Xi[len(internal_Xi)-1]
                    Phi_seq = _seq_extend(Phi_seq, internal_Xi)
                else:
                    Phi_seq.append(external_Xi[j + 1])
                    S_seq.append(external_S[j])
        if label[0] in self.gadget_1.labels:
            Phi_seq, S_seq = self.gadget_1.get_sequence(label)
        if correction is not None:
            Phi_seq, S_seq = corrected_sequence(Phi_seq, S_seq, self.depth, correction) 
        return Phi_seq, S_seq
    
    def get_unitary(self, label):
        """
        Returns a function which takes input unitaries, and outputs unitaries corresponding to the 
        gadget
        """
        Xi, S = self.get_sequence(label)
        return lambda U : compute_mqsp_unitary(U, Xi, S)

    def get_qsp_unitary(self, label):
        """
        Generates and returns the polynomial corresponding to a particular output leg
        of an atomic gadget.

        Args:
            vars (dict): A dictionary assigning values to input legs
        """
        input_unitaries = lambda vars : {l : W(vars[l]) for l in self.in_labels}
        return lambda vars : self.get_unitary(label)(input_unitaries(vars)) # Returns top-left entry 

    def interlink(self, gadget, interlink):
        """
        Performs an interlink of an atomic gadget with a gadget
        """
        return CompositeAtomicGadget(self, gadget, interlink)

 
################################################################################
# Operations on gadgets and collections of gadgets
# STILL NEED TO DEFINE SEQUENCE FOR CORRECTION

def corrected_sequence(Phi, S, ancilla, deg):
    """
    Applies the correction protocol to a gadget leg, gets the resulting sequence
    Steps to resolving this issue: 
    """
    extraction_gadget = ExtractionGadget(deg, "G_ext")
    temporary_gadget = AtomicGadget([Phi], [S], "G")

    gadget_phase = temporary_gadget.interlink(extraction_gadget, [(("G", 0), ("G_ext", 0), None)])
    phase_Phi, phase_S = gadget_phase.get_sequence(("G_ext", 0))

    cswap_identifier = ("CSWAP", ancilla)

    new_Phi = [0, cswap_identifier] + list(phase_Phi) + [cswap_identifier] + list(Phi) + [cswap_identifier] + list(-1 * np.array(phase_Phi)[::-1]) + [cswap_identifier, 0]
    new_S = [cswap_identifier] + phase_S + [cswap_identifier] + S + [cswap_identifier] + phase_S[::-1] + [cswap_identifier]
    return new_Phi, new_S


def pin(gadget, legs, vals):
    """
    Pins a gadget
    """
    pass

def permute(gadget, permutations):
    """
    Permutes the legs of a gadget
    """
    pass

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

###################### Instances of gadgets ######################

class MultiplicationGadget(AtomicGadget):
    """
    An multiplication gadget
    """
    def __init__(self, phi=-np.pi/4):
        self.Phi, self.S = np.array([phi, np.pi/4, -np.pi/4, -phi]), np.array([0, 1, 0])
        self.Xi = [[self.Phi], [self.S]] # Defines the gadget phase sequence


class AdditionGadget(AtomicGadget):
    """
    An addition gadget
    """
    def __init__(self, phi=-np.pi/4):
        self.Phi = np.array([phi, -phi])
        self.S = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.Xi = [[self.Phi], [self.S]] # Defines the gadget phase sequence 


class InvChebyshevGadget(AtomicGadget):
    """
    An inverse Chebyshev polynomial gadget
    """
    def __init__(self, n, phi=-np.pi/4):
        pass


class ExtractionGadget(AtomicGadget):
    """
    The gadget which corresponds to the extraction protocol. This protocol has both a specific 
    P and Q polynomial associated with it, for arbitrary precisions, and is therefore easy to 
    implement, without performing a QSP completion.
    """
    def __init__(self, deg, label):
        self.a, self.b = 1, 1
        self.deg = deg
        phi = ExtractionSequence().generate(deg)
        Xi, S = [phi], [[0 for _ in range(len(phi)-1)]]
        super().__init__(Xi, S, label)

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
    for i in range(0, len(s)):
        output_U = output_U @ U[s[i]] @ Rz(Phi[i + 1])
    return output_U


def get_corrective_phi(eps, delta):
    """
    Generates the corrective Phi sequence, for a certain error tolerance
    """
    pass

###############################################################################################