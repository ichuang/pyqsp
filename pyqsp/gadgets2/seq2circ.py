import qiskit.circuit.library as qcl
from qiskit import QuantumCircuit
from pyqsp.gadgets2.sequences import *

class QuantumCircuit:
    '''
    Represent a quantum circuit for pyqsp
    '''
    def __init__(self):
        return

# mapping between sequence objects and qiskit circuit library elements
SequenceMap = {
    XGate: {'gate': qcl.XGate, 'nqubits': 1},
    YGate: {'gate': qcl.YGate, 'nqubits': 1},
    ZGate: {'gate': qcl.ZGate, 'nqubits': 1},
    SignalGate: {'gate': qcl.RXGate, 'nqubits': 1},
    SwapGate: {'gate': qcl.SwapGate, 'nqubits': 2},
}

def seq2circ(sequence):
    return
