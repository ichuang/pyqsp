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
        var_label (str): A string which defines the base variable symbol assigned to the functions induced by 
        the gadget
    Kwargs:
        F (func): Function of type R^a --> R^b
        F_ideal (func): Function of type R^a --> R^b which the gadget is approximating
        domain (dict[list]): The domain on which the variable corresponding to each input leg is valid
        epsilon (float): The approximation error |F - F_ideal| on the domain of validity
    """
    def __init__(self, a, b, var_label, label=None, F=None, F_ideal=None, domain=None, epsilon=None):
        self.a, self.b = a, b
        self.shape = (a, b)
        self.var_label = var_label
        #################################

        self.vars = [sympy.Symbol("{}_{}".format(var_label, j)) for j in range(a)]
        self.label = hash(self) if label is None else label
        self.F = sympy.Function("f_{}".format(self.label)) if F is None else F # Creates a dummy function in the case when no function is assigned
        self.F_ideal = F_ideal
        self.domain = {j : [-1, 1] for j in range(a)} if domain is None else domain
        self.epsilon = epsilon
        self.tape = [self] # A tape recorder


class AtomicGadget(Gadget):
    """
    Class for atomic gadgets
    """
    def __init__(self, Xi, S, var_label, label=None, F_ideal=None, domain=None, epsilon=None):
        self.Xi = Xi
        self.S = S
        a, b = len(set(S)), len(Xi)
        super().__init__(a, b)


class CompositeGadget(Gadget):
    """
    A particular class of gadget arising from interlinking of gadgets, which 
    tracks the internal structure of each gadget being linked.
    """
    def __init__(self, a, b):
        super().__init__(a, b)

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
    interlink_arr = np.array(interlink).T
    a, b = gadget1.a + (gadget2.a - len(interlink)), gadget2.b + (gadget1.b - len(interlink))

    # Gets the input legs
    fn = lambda x : True if gadget2.vars.index(x) not in interlink_arr[1] else False
    input_vars_1, input_vars_2 = gadget1.vars, list(filter(fn, gadget2.vars))

    # Defines the new function
    def F(x):
        input_1 = gadget1.F(x[0:a])

    #gadget = CompositeGadget(a, b)
    return None


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