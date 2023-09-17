import qiskit
import qiskit_aer
import qiskit.circuit.library as qcl

from qiskit import QuantumCircuit
from pyqsp.gadgets2.sequences import *

class SequenceQuantumCircuit:
    '''
    Represent a quantum circuit for pyqsp
    '''
    def __init__(self, nqubits, qubit_indices, verbose=True):
        '''
        nqubits = (int) number of qubits acted upon in the circuit
        qubit_indices = (list) ordered unique integer indices (from 0) of qubits in the circuit
        verbose = (bool) output verbose debug messages if True
        '''
        self.nqubits = nqubits
        self.qubit_indices = qubit_indices
        self.circ = QuantumCircuit(nqubits)
        self.verbose = verbose
        self.seqnum = 0
        if self.verbose:
            print(f"['pyqsp.gadgets.seq2circ.QuantumCircuit'] Creating sequence quantum circuit on {nqubits} qubits with indices {qubit_indices}")
        return

    def add_gate(self, gate, *args, **kwargs):
        '''
        add the given gate to the circuit

        gate = qiskit gate object
        '''
        try:
            self.circ.append(gate, *args, **kwargs)
        except Exception as err:
            print(f"['pyqsp.gadgets.seq2circ.QuantumCircuit'] Error in adding gate {gate} with sequence number {self.seqnum}, err={err}")
            raise
        if self.verbose:
            print(f"['pyqsp.gadgets.seq2circ.QuantumCircuit'] Added gate {gate} at sequence number {self.seqnum}")
        self.seqnum += 1
        
    def get_unitary(self, decimals=3):
        '''
        Return unitary transform matrix performed by this circuit
        '''
        if 0:
            backend = qiskit.Aer.get_backend("unitary_simulator")
            job = qiskit.execute(self.circ, backend)
            result = job.result()
        else:
            backend = qiskit_aer.Aer.get_backend('aer_simulator')
            circ2 = self.circ.copy()	# copy circuit and add save_unitary instruction to end
            circ2.save_unitary()
            result = backend.run(circ2).result()            
        U = result.get_unitary(self.circ, decimals=decimals)
        if self.verbose:
            print(U)
        return U

# mapping between sequence objects and qiskit circuit library elements
SequenceMap = {
    XGate: {'gate': qcl.RXGate, 'nqubits': 1, 'arg': "angle"},
    YGate: {'gate': qcl.RYGate, 'nqubits': 1, 'arg': "angle"},
    ZGate: {'gate': qcl.RZGate, 'nqubits': 1, 'arg': "angle"},
    SignalGate: {'gate': qcl.RXGate, 'nqubits': 1, 'arg': "signal"},
    SwapGate: {'gate': qcl.SwapGate, 'nqubits': 2, 'arg': "angle"},
}

def seq2circ(sequence, signal_value=0):
    '''
    Return an instance of SequenceQuantumCircuit corresponding to the provided sequence.

    sequence: (list) list of SequenceObject's, each of which is to be transformed into a gate
    signal_value: (floar) value to use for signal, if it appears in the sequence
    '''
    all_qubit_indices = []
    for seq in sequence:
        if seq.controls is not None:
            assert isinstance(seq.controls, list)
            all_qubit_indices += seq.controls
        if seq.target is not None:
            assert isinstance(seq.target, int)
            all_qubit_indices += [seq.target]
    if not all_qubit_indices:
        all_qubit_indices = [0]		# default to just one qubit, indexed 0, if all sequences had None for controls and target
    qubit_indices = sorted(list(set(all_qubit_indices)))	# sorted list of unique qubit indices in sequence
    nqubits = max(qubit_indices) - min(qubit_indices) + 1

    qcirc = SequenceQuantumCircuit(nqubits, qubit_indices)
    
    for seq in sequence:
        ginfo = SequenceMap.get(seq.__class__)
        if not ginfo:
            raise Exception(f"[pyqsp.gadgets.seq2circ] Error! Sequence element {seq} ({seq.__class__.__name__}) has no known corresponding quantum circuit gate element!")
        nqubits = ginfo['nqubits']
        target = qubit_indices[seq.target or 0]		# note remapping of qubit indices
        arg = ginfo['arg']
        if arg=='signal':
            argv = signal_value				# use signal_value argument to seq2qcirc call
        elif arg:
            argv = getattr(seq, arg)
        try:
            gate = ginfo['gate'](argv)
        except Exception as err:
            print(f"[pyqsp.gadgets.seq2circ] Error! Could not create gate for {seq}, gate={gate}, argv={argv}, err={err}")
            raise
        if nqubits==1:
            qcirc.add_gate(gate, [target] )
        elif nqubits==2:
            control = qubit_indices[seq.controls[0]]
            qcirc.add_gate(gate, [control, target] )

    return qcirc
