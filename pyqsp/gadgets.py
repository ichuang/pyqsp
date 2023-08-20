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
    def __init__(self, a, b, labels=None):
        self.a, self.b = a, b
        self.shape = (a, b)

        # Labels the input legs of a gadget
        self.labels = labels

    @classmethod
    def unitary(self):
        """
        The unitary mapping effectuated by a gadget
        """
        raise NotImplementedError


class AtomicGadget(Gadget):
    """
    Class for atomic gadgets
    """
    def __init__(self, Xi, S, labels=None):
        self.labels = labels
        self.Xi = Xi
        self.S = S

        a, b = len(set([tuple(s) for s in S])), len(Xi)
        super().__init__(a, b, labels=labels)
    
    def unitary(self, label):
        """
        Returns a function which takes input unitaries, and outputs unitaries corresponding to the 
        gadget
        """
        leg = self.labels.index(label)
        return lambda U : compute_mqsp_unitary(U, self.Xi[leg], self.S[leg])

    def get_polynomial(self, label):
        """
        Generates and returns the polynomial corresponding to a particular output leg
        of an atomic gadget.
        """
        leg = self.labels.index(label)
        input_unitaries = lambda vars : [W(vars[self.var_labels[i]]) for i in range(len(self.var_labels))]
        return lambda vars : self.unitary(leg)(input_unitaries(vars))[0][0]

    def interlink(self, gadget, interlink):
        """
        Performs an interlink of an atomic gadget with a gadget
        """
        a, b = gadget.a + (self.a - len(interlink)), self.b + (gadget.b - len(interlink))
        return CompositeGadget


class CompositeGadget(Gadget):
    """
    A particular class of gadget arising from interlinking of gadgets, which 
    tracks the internal structure of each gadget being linked.
    """
    def __init__(self, gadgets, interlinks):
        self.gadgets = gadgets
        self.interlinks = interlinks

        self.a, self.b = None, None
        super().__init__(self.a, self.b)


class Interlink:
    """
    A class for representing a gadget interlink
    """
    def __init__(self, gadgets, interlink_params):
        self.B, self.C, self.J = interlink_params
        self.gadgets = gadgets
    
################################################################################
# Operations on gadgets and collections of gadgets

def correction(gadget, legs):
    """
    Applies the correction protocol to a gadget
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
