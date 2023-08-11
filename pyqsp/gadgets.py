'''
pyqsp/gadgets.py

Subroutines for implementation of modular M-QSP protocols and gadgets
'''

import numpy as np


class Gadget:
    """
    Top-level abstract class for gadgets
    """
    def __init__(self):
        pass


class CompositeGadget(Gadget):
    """
    The instance of the gadget class created upon performing an interlink between gadgets
    """
    def __init__(self):
        self.gadgets = []


class CircuitGadget(Gadget):
    """
    Circuit gadget
    """
    def __init__(self):
        pass

class AtomicGadget(CircuitGadget):
    """
    Atomic gadget
    """
    def __init__(self):
        pass


def interlink(gadget1, gadget2):
    """
    Defines a new gadget from the interlink of two gadgets
    """

###################### Instances of gadgets ######################

class Correction(Gadget):
    """
    The correction gadget
    """
    def __init__(self, ancilla_free=True):
        self.ancilla_free = True
    
    