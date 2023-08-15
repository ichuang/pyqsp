'''
pyqsp/gadgets.py

Subroutines for implementation of modular M-QSP protocols and gadgets
'''
import numpy as np
import sympy


class Gadget:
    """
    Class for (a, b) gadgets

    Args:
        a (int): Number of input legs to gadget
        b (int): Number of output legs to gadget
    Kwargs:
        F (func): Function of type R^a --> R^b
        F_ideal (func): Function of type R^a --> R^b which the gadget is approximating
        domain (dict[list]): The domain on which the variable corresponding to each input leg is valid
        epsilon (float): The approximation error |F - F_ideal| on the domain of validity
    """
    def __init__(self, a, b, F=None, F_ideal=None, domain=None, epsilon=None):
        self.a, self.b = a, b
        self.shape = (a, b)
        #################################

        self.F = F
        self.F_ideal = F_ideal
        self.domain = {j : [-1, 1] for j in range(a)} if domain is None else domain
        self.epsilon = epsilon
        self.tape = [self] # A tape recorder


class AtomicGadget(Gadget):
    """
    Class for atomic gadgets
    """
    def __init__(self, Xi, S, F_ideal=None, domain=None, epsilon=None):
        self.Xi = Xi
        self.S = S
        a, b = len(set(S)), len(Xi)
        super().__init__(a, b)


class CompositeGadget(Gadget):
    """
    A particular class of gadget arising from interlinking of gadgets, which 
    tracks the internal structure of each gadget being linked.
    """
    def __init__(self):
        pass

    def compute_leg_tape(self, k):
        """
        Computes the tape of a given output leg, which allows for realization as a quantum circuit
        """ 
        pass


# Operations on gadgets and collections of gadgets

def correction(gadget, legs):
    """
    Applies the correction protocol 
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


def interlink(gadget1, gadget2, interlink):
    """
    Defines a new gadget from the interlink of two gadgets
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

###################### Instances of gadgets ######################