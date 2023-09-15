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

from pyqsp import qsp_models
from .phases import ExtractionSequence, SqrtSequence, FourthRootSequence
import itertools
import pennylane as pl
from .response import ComputeQSPResponse


##############################################################################

class Gate:
    """
    A generic quantum gate
    """
    def matrix(self, *args, **kwargs):
        """
        Returns the matrix corresponding to the gate
        """
        raise NotImplementedError


class QSP_Rotation(Gate):
    """
    A QSP rotation/phase gate
    """
    def __init__(self, theta):
        self.ancilla, self.ancilla_reg = None, None
        self.theta = theta # Angle corresponding to the phase gate
    
    def matrix(self):
        return Rz(self.theta)


class QSP_Signal(Gate):
    """
    A QSP signal gate
    """
    def __init__(self, label):
        self.ancilla, self.ancilla_reg = None, None
        self.inv = False
        self.label = label
    
    def matrix(self, U):
        return U if self.inv else np.conj(U).T


class Corrective_CSWAP(Gate):
    """
    A corrective CSWAP gate, for use in the correction protocol
    """
    def __init__(self, ancilla):
        self.ancilla = ancilla

    def matrix(self, ancilla_qubits):
        """Gets the matrix"""
        dev = pl.device('default.qubit', wires=[0] + [(0, q) for q in ancilla_qubits] + [(1, q) for q in ancilla_qubits])
        @pl.qnode(dev)
        def circ():
            pl.PauliX(wires=[(0, self.ancilla)])
            pl.CSWAP(wires=[(0, self.ancilla), (1, self.ancilla), 0])
            pl.PauliX(wires=[(0, self.ancilla)])
            return pl.state()
        return pl.matrix(circ)() 


class iX_Gate(Gate):
    """
    iX Gate
    """
    def __init__(self, inv=False, phi=np.pi):
        self.inv = inv
        self.phi = phi
    
    def matrix(self):
        """Gets the matrix"""
        if self.inv:
            return pl.RX(-self.phi, wires=0).matrix()
        else:
            return pl.RX(self.phi, wires=0).matrix()


class Corrective_SWAP(Gate):
    """
    A corrective SWAP gate, for use in the correction protocol
    """
    def __init__(self, ancilla):
        self.ancilla = ancilla

    def matrix(self, ancilla_qubits):
        """Gets the matrix"""
        dev = pl.device('default.qubit', wires=[0] + [("a", q) for q in ancilla_qubits])
        @pl.qnode(dev)
        def circ():
            pl.SWAP(wires=[("a", self.ancilla), 0])
            return pl.state()
        return pl.matrix(circ)()


class Corrective_Z(Gate):
    """
    Corrective sigma_z-rotation
    """
    def __init__(self, seq, ancilla):
        self.ancilla = ancilla
        self.unitary = None
        self.seq = seq

    def matrix(self, U, ancilla_reg):
        if self.unitary is not None:
            return self.unitary
        else:
            # Calculates the unitary
            restricted_ancillas = []
            for r in ancilla_reg:
                if r != self.ancilla:
                    restricted_ancillas.append(r)
            target_unitary = _get_unitary(self.seq, restricted_ancillas)(U)

            # Constructs the controlled variant of the unitary
            dev = pl.device('default.qubit', wires=[0] + [("a", q) for q in ancilla_reg])
            @pl.qnode(dev)
            def circ():
                pl.SWAP(wires=[("a", self.ancilla), 0])
                pl.ControlledQubitUnitary(target_unitary, control_wires=[("a", self.ancilla)], wires=[0] + restricted_ancillas)
                pl.SWAP(wires=[("a", self.ancilla), 0])
                return pl.state()
            self.unitary = pl.matrix(circ)()
            return self.unitary


###########################################################################


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
        self.pinned_inputs = dict()

    @classmethod
    def unitary(self):
        """
        The unitary mapping effectuated by a gadget
        """
        raise NotImplementedError
    
    @property
    def pinned_in_labels(self):
        """
        Pinned input legs
        """
        return self.pinned_inputs.keys()

    
    def pin(self, label, unitary):
        """
        Pins an input leg
        """
        self.pinned_inputs[label] = unitary


class AtomicGadget(Gadget):
    """
    Class for not-quite atomic gadgets: to be precise, this class encompasses the closure of all gadgets which 
    can be achieved via interlinking atomic gadgets and correction protocols, arbitrarily.
    """
    def __init__(self, Xi, S, label):
        self.Xi = Xi
        self.ancillas = []
        self.labels = [label]

        a, b = len(set(list(itertools.chain.from_iterable(S)))), len(Xi)
        self.S = [[(label, s) for s in S[leg]] for leg in range(len(S))]

        super().__init__(a, b, label)

        self.depth = 1 
    
    def get_unitary(self, label, correction=None):
        """
        Returns a function which takes input unitaries, and outputs unitaries corresponding to the 
        gadget
        """
        seq = self.get_sequence(label, correction=correction)
        ancilla_reg = list(range(self.depth))
        N = 2 * len(ancilla_reg) + 1
        dim = 2 ** N

        def func(U):
            U = {**U, **self.pinned_inputs} 

            mat = np.eye(dim)
            for s in seq:
                if isinstance(s, QSP_Signal):
                    mat = mat @ _extend_dim(s.matrix(U[s.label]), N)
                elif isinstance(s, QSP_Rotation):
                    mat = mat @ _extend_dim(s.matrix(), N)
            return mat
        return func

    def get_qsp_unitary(self, label, correction=None, rot=None):
        """
        Generates and returns the polynomial corresponding to a particular output leg
        of an atomic gadget.

        Args:
            vars (dict): A dictionary assigning values to input legs
        """
        free_in_labels = []
        for l in self.in_labels:
            if l not in self.pinned_in_labels:
                free_in_labels.append(l)
        if rot is not None:
            input_unitaries = lambda vars : {l : W(vars[l], rot=rot[l]) for l in free_in_labels}
        else:
            input_unitaries = lambda vars : {l : W(vars[l]) for l in free_in_labels} 
        return lambda vars : self.get_unitary(label, correction=correction)(input_unitaries(vars)) # Returns top-left entry  

    def get_sequence(self, label, correction=None):
        """
        Gets the gate sequence corresponding to an output leg
        """
        S_sub = self.S[label[1]] # Gets the S sequence of the leg
        Xi_sub = self.Xi[label[1]] # Gets the Phi sequence of the leg
        seq = [QSP_Rotation(Xi_sub[0])]

        for j in range(len(S_sub)):
            seq.append(QSP_Signal(S_sub[j]))
            seq.append(QSP_Rotation(Xi_sub[j + 1]))
        if correction is not None:
            seq = corrected_sequence(seq, self.depth-1, correction)
        return seq

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

        # This will break
        self.depth = gadget_1.depth + gadget_2.depth
        #######################

        self.ancilla_register = list(range(self.depth-1))
        # Interlink parameters
        self.B, self.C, self.correction = [x[0] for x in interlink], [x[1] for x in interlink], [x[2] for x in interlink]

        # Gets the input and output legs
        self.a, self.b = gadget_1.a + (gadget_2.a - len(interlink)), gadget_2.b + (gadget_1.b - len(interlink))
        self.in_labels = gadget_1.in_labels + list(filter(lambda x : x not in self.C, gadget_2.in_labels))
        self.out_labels = gadget_2.out_labels + list(filter(lambda x : x not in self.B, gadget_1.out_labels))

    def get_sequence(self, label, correction=None):
        if label[0] in self.gadget_2.labels:
            external_seq = self.gadget_2.get_sequence(label) # Gets the output gadget's sequence
            seq = []

            for j in range(len(external_seq)):
                if isinstance(external_seq[j], QSP_Signal):
                    if external_seq[j].label in self.C:
                        new_leg_ind = self.C.index(external_seq[j].label)
                        new_leg, corr = self.B[new_leg_ind], self.correction[new_leg_ind]
                        internal_seq = self.gadget_1.get_sequence(new_leg, correction=corr)
                        seq.extend(internal_seq)
                    else:
                        seq.append(external_seq[j])
                else:
                    seq.append(external_seq[j])
        if label[0] in self.gadget_1.labels:
            seq = self.gadget_1.get_sequence(label)
        if correction is not None:
            seq = corrected_sequence(seq, self.depth-1, correction)
        return seq
    
    def get_unitary(self, label, correction=None):
        """
        Returns a function which takes input unitaries, and outputs unitaries corresponding to the 
        gadget
        """
        seq = self.get_sequence(label, correction=correction)
        ancilla_reg = list(range(self.depth-1))

        N = len(ancilla_reg) + 1
        dim = 2 ** N

        def func(U):
            mat = np.eye(dim)
            for s in seq:
                if isinstance(s, QSP_Signal):
                    mat = mat @ _extend_dim(s.matrix(U[s.label]), N)
                elif isinstance(s, QSP_Rotation):
                    mat = mat @ _extend_dim(s.matrix(), N)
                elif isinstance(s, Corrective_Z):
                    mat = mat @ s.matrix(U, ancilla_reg)
                elif isinstance(s, iX_Gate):
                    mat = mat @ _extend_dim(s.matrix(), N)

            return mat
        return func

    def get_qsp_unitary(self, label, correction=None, rot=None, shift=None):
        """
        Generates and returns the polynomial corresponding to a particular output leg
        of an atomic gadget.

        Args:
            vars (dict): A dictionary assigning values to input legs
        """
        if rot is not None and shift is None:
            input_unitaries = lambda vars : {l : W(vars[l], rot=rot[l]) for l in self.in_labels}
        elif rot is None and shift is not None:
            input_unitaries = lambda vars : {l : W(vars[l], shift=shift[l]) for l in self.in_labels} 
        elif rot is not None and shift is not None:
            input_unitaries = lambda vars : {l : W(vars[l], shift=shift[l], rot=rot[l]) for l in self.in_labels} 
        else:
            input_unitaries = lambda vars : {l : W(vars[l]) for l in self.in_labels} 
        return lambda vars : self.get_unitary(label, correction=correction)(input_unitaries(vars)) # Returns top-left entry  

    def interlink(self, gadget, interlink):
        """
        Performs an interlink of an atomic gadget with a gadget
        """
        return CompositeAtomicGadget(self, gadget, interlink)

 
################################################################################
# Operations on gadgets and collections of gadgets

def corrected_sequence(ext_seq, ancilla, deg):
    """
    Applies the correction protocol to a gadget leg, gets the resulting sequence
    Steps to resolving this issue: 
    """
    extraction_gadget = ExtractionGadget(deg, "G_ext").get_sequence(("G_ext", 0))
    seq = [iX_Gate()] + _nested_seq(extraction_gadget, ext_seq)

    seq_conj = []
    for s in seq:
        s_copy = copy.deepcopy(s)
        if isinstance(s, QSP_Signal):
            s_copy.inv = True
        if isinstance(s, QSP_Rotation):
            s_copy.theta = -s_copy.theta
        seq_conj.append(s_copy)
    seq_conj = seq_conj[::-1] + [iX_Gate(inv=True)]

    corr, corr_dagger = Corrective_Z(seq, ancilla), Corrective_Z(seq_conj, ancilla)
    new_seq = [corr] + ext_seq + [corr_dagger] 

    return new_seq


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


def _nested_seq(external_seq, internal_seq):
    """
    Utility function for nesting composition of gadgets
    """
    seq = []

    for j in range(len(external_seq)):
        if isinstance(external_seq[j], QSP_Signal):
            seq.extend(internal_seq)
        else:
            seq.append(external_seq[j])
    return seq

def _extend_dim(U, dim):
    """
    Utility function for extending the dimension of quantum gates
    """
    return np.kron(U, np.eye(2 ** (dim - 1)))


def _get_unitary(seq, ancilla_reg):
    N = 2 * len(ancilla_reg) + 1
    dim = 2 ** N

    def func(U):
        mat = np.eye(dim)
        for s in seq:
            if isinstance(s, QSP_Signal):
                mat = mat @ _extend_dim(s.matrix(U[s.label]), N)
            elif isinstance(s, QSP_Rotation) or isinstance(s, iX_Gate):
                mat = mat @ _extend_dim(s.matrix(), N)
            elif isinstance(s, Corrective_Z) or isinstance(s, Corrective_SWAP):
                mat = mat @ _extend_dim(s.matrix(ancilla_reg), N)
        return mat
    return func

###################### Instances of gadgets ######################

class MultiplicationGadget(AtomicGadget):
    """
    An multiplication gadget
    """
    def __init__(self, label, phi=-np.pi/4):
        Xi, S = [np.array([phi, np.pi/4, -np.pi/4, -phi])], [np.array([0, 1, 0])]
        super().__init__(Xi, S, label)


class AdditionGadget(AtomicGadget):
    """
    An addition gadget
    """
    def __init__(self, label, phi=-np.pi/4):
        Xi = np.array([phi, 0, np.pi/4, np.pi/2, -np.pi/2, np.pi/2, -3 * np.pi/4, 0, -phi])
        S = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        super().__init__([Xi], [S], label) 

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
    
    def get_response(self):
        adat = np.linspace(-1, 1, 500)
        response = ComputeQSPResponse(adat, np.array(self.Xi[0]), signal_operator="Wx", measurement="z")["pdat"]
        return adat, response


class SqrtGadget(AtomicGadget):
    """
    Implements the square root/inverse Chebyshev gadget
    """
    def __init__(self, deg, delta, label):
        self.a, self.b = 1, 1
        self.deg = deg
        phi = SqrtSequence().generate(deg, delta)
        Xi, S = [phi], [[0 for _ in range(len(phi)-1)]]
        super().__init__(Xi, S, label) 
    
    def get_response(self):
        adat = np.linspace(-1, 1, 500)
        response = ComputeQSPResponse(adat, np.array(self.Xi[0]), signal_operator="Wz", measurement="z")["pdat"]
        return adat, response


class FourthRootGadget(AtomicGadget):
    """
    Implements the square root/inverse Chebyshev gadget
    """
    def __init__(self, deg, delta, label):
        self.a, self.b = 1, 1
        self.deg = deg
        phi = FourthRootSequence().generate(deg, delta)
        Xi, S = [phi], [[0 for _ in range(len(phi)-1)]]
        super().__init__(Xi, S, label) 
    
    def get_response(self):
        adat = np.linspace(-1, 1, 500)
        response = ComputeQSPResponse(adat, np.array(self.Xi[0]), signal_operator="Wz", measurement="z")["pdat"]
        return adat, response


################################################################################################

def Rz(phi):
    """
    sigma_z-rotation
    """
    return np.array([
        [np.exp(1j * phi), 0],
        [0, np.exp(-1j * phi)]
    ])


def Rx(phi):
    """
    sigma_x-rotation
    """
    return np.array([
        [np.cos(phi), 1j * np.sin(phi)],
        [1j * np.sin(phi), np.cos(phi)]
    ]) 


def W(x, rot=None, shift=None):
    """
    Standard sigma_x signal operator
    """
    X = np.array([
        [x, 1j * np.sqrt(1 - x ** 2)],
        [1j * np.sqrt(1 - x ** 2), x]
    ])
    if shift is not None:
        X = Rx(shift) @ X
    if rot is not None:
        X = Rz(rot) @ X @ Rz(-rot)
    return X

def compute_mqsp_unitary(U, Phi, s, rot_gate="z"):
    """
    Computes an M-QSP unitary
    """
    if rot_gate == "z":
        output_U = Rz(Phi[0])
        for i in range(0, len(s)):
            output_U = output_U @ U[s[i]] @ Rz(Phi[i + 1])
        return output_U
    elif rot_gate == "x":
        output_U = Rx(Phi[0])
        for i in range(0, len(s)):
            output_U = output_U @ U[s[i]] @ Rx(Phi[i + 1])
        return output_U 

