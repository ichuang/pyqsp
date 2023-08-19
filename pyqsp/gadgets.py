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
import sympy


class Gadget:
    """
    Class for (a, b) gadgets

    Args:
        a (int): Number of input legs to gadget
        b (int): Number of output legs to gadget 
    Kwargs:
        var_label (str): A string which defines the base variable symbol assigned to the functions induced by 
        the gadget
        F (func): Function of type R^a --> R^b encoded by the gadget
        F_ideal (func): Function of type R^a --> R^b which the gadget is approximating
        domain (dict[list]): The domain on which the variable corresponding to each input leg is valid
        epsilon (float): The approximation error |F - F_ideal| on the domain of validity
    """
    def __init__(self, a, b, label=None, var_label=None, F=None, F_ideal=None, domain=None, epsilon=None):
        self.a, self.b = a, b
        self.shape = (a, b)
        #################################
        # Sets the variables to be tracked implicitly

        self.label = hash(self) if label is None else label
        self.var_label = "x_{}".format(self.label) if var_label is None else var_label
        self.F = F 

        ##################################
        # Extra information

        self.F_ideal = F_ideal
        self.domain = {j : [-1, 1] for j in range(a)} if domain is None else domain
        self.epsilon = epsilon


class AtomicGadget(Gadget):
    """
    Class for atomic gadgets
    """
    def __init__(self, Xi, S, var_label, label=None, F_ideal=None, domain=None, epsilon=None):
        self.Xi = Xi
        self.S = S
        a, b = len(set(S)), len(Xi)
        super().__init__(a, b)
    
    def get_polynomial(self, leg, symbolic=False):
        """
        Generates and returns the polynomial corresponding to a particular output leg
        of an atomic gadget.
        """
        pass


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


def interlink(gadget1, gadget2, interlink):
    """
    Defines a new gadget from the interlink of two gadgets

    The idea behind the interlink function is to 
    """
    interlink_arr = np.array(interlink).T

    # Gets the new leg dimensions of the composite gadget
    a, b = gadget1.a + (gadget2.a - len(interlink)), gadget2.b + (gadget1.b - len(interlink))

    # Gets the input legs
    fn_in = lambda x : True if gadget2.vars.index(x) not in interlink_arr[1] else False
    #fn_out = lambda x : True if gadget1.vars.

    input_vars_1, input_vars_2 = gadget1.vars, list(filter(fn, gadget2.vars))

    # Defines the new function
    def F(x):
        input_1 = gadget1.F(x[0:a])
        return input_1

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
        self.Phi = np.array([phi, np.pi/4, -np.pi/4, -phi])
        self.S = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        self.Xi = [self.Phi, self.S] # Defines the gadget phase sequence 
