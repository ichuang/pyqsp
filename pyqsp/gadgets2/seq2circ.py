import qiskit
import qiskit_aer
import qiskit.circuit.library as qcl

from qiskit import QuantumCircuit
from pyqsp.gadgets2.sequences import *

class SequenceQuantumCircuit:
    '''
    Represent a quantum circuit for pyqsp

    TODO: use qiskit circuit parameters to allow changing of circuit parameter values after circuit creation
    '''
    def __init__(self, nqubits_main, nqubits_ancillae, qubit_indices_main, qubit_indices_ancillae, verbose=True):
        '''
        nqubits_main = (int) number of qubits acted upon in the circuit as targets
        nqubits_ancillae = (int) number of qubits used as controls
        qubit_indices_main = (list) ordered unique integer indices (from 0) of main qubits in the circuit
        qubit_indices_ancillae = (list) ordered unique integer indices (from 0) of ancillae qubits in the circuit
        verbose = (bool) output verbose debug messages if True
        '''
        self.nqubits_main = nqubits_main
        self.nqubits_ancillae = nqubits_ancillae
        self.nqubits = nqubits_main + nqubits_ancillae
        self.qubit_indices_main = qubit_indices_main
        self.qubit_indices_ancillae = qubit_indices_ancillae
        self.verbose = verbose
        self.seqnum = 0

        # construct qiskit quantum circuit with separate main and ancillae registers
        self.q_main = qiskit.QuantumRegister(self.nqubits_main, name='main')
        self.q_ancillae = qiskit.QuantumRegister(self.nqubits_ancillae, name='ancillae')
        self.circ = QuantumCircuit(self.q_main, self.q_ancillae)

        if self.verbose:
            print(f"['pyqsp.gadgets.seq2circ.QuantumCircuit'] Creating sequence quantum circuit on {self.nqubits} qubits with indices {qubit_indices_main}:{qubit_indices_ancillae}")
        return

    def draw(self, *args, **kwargs):
        '''
        Draw circuit - using qiskit
        '''
        return self.circ.draw(*args, **kwargs)

    def size(self, *args, **kwargs):
        '''
        return circuit size - using qiskit
        '''
        return self.circ.size()

    def circuit_qubit_index_to_register(self, idx):
        '''
        Map a circuit qubit index (0 to nqubits-1) to a qubit in the correct register, either main or ancillae,
        using the convention that the qubits are numbered linearly starting from 0, and any circuit qubit index
        past self.nqubits_main - 1 is to be treated as an ancilla qubit.

        idx: (int) circuit qubit index

        Returns corresponding element of the qubit register (self.q_main or self.q_ancillae).
        '''
        assert idx < self.nqubits
        if idx > self.nqubits_main - 1:
            return self.q_ancillae[idx - self.nqubits_main]
        return self.q_main[idx]

    def add_gate(self, gate, controls, targets, *args, **kwargs):
        '''
        add the given gate to the circuit

        gate = qiskit gate object
        controls = (list) circuit indices of control qubits (in the ancilla register)
        targets = (list) circuit indices of the target qubits (in the main register)

        circuit qubits are split into main and ancillae registers. each register is
        numbered starting from 0.  But we use the convention of mapping qubit indices
        larger than the size of the main register, into qubits of the ancillae register.
        '''
        assert isinstance(controls, list)
        assert isinstance(targets, list)
        if not all([ x > self.nqubits_main - 1 for x in controls ]):
            raise Exception(f"[pyqsp.gadgets.seq2circ.QuantumCircuit] control qubit index not an ancilla qibit?  controls={controls}, nqubits_main={self.nqubits_main}")
        try:
            control_qubits = [self.circuit_qubit_index_to_register(x) for x in controls]
            target_qubits = [self.circuit_qubit_index_to_register(x) for x in targets]
        except Exception as err:
            print(f"[pyqsp.gadgets.seq2circ.QuantumCircuit] Error in adding gate {gate} with sequence number {self.seqnum}, failed to map controls={controls} and targets={targets} to their registers {self.q_ancillae}, {self.q_main}, err={err}")
            raise
        register_qubits = control_qubits + target_qubits
        try:
            self.circ.append(gate, register_qubits, *args, **kwargs)
        except Exception as err:
            print(f"[pyqsp.gadgets.seq2circ.QuantumCircuit] Error in adding gate {gate} on control {controls}={control_qubits} and target {targets}={target_qubits} with sequence number {self.seqnum}, err={err}")
            raise
        if self.verbose:
            print(f"[pyqsp.gadgets.seq2circ.QuantumCircuit] Added gate {gate} on register qubits {register_qubits} at sequence number {self.seqnum}")
        self.seqnum += 1
        
    def get_unitary(self, decimals=3):
        '''
        Return unitary transform matrix performed by this circuit
        '''
        if 0:
            # deprecated
            backend = qiskit.Aer.get_backend("unitary_simulator")
            job = qiskit.execute(self.circ, backend)
            result = job.result()
        else:
            # good as of Fall 2023
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
    SwapGate: {'gate': qcl.SwapGate, 'nqubits': 2, 'arg': None, 'qubits': ['index_0', 'index_1']},
}

def sequence_circuit_size(sequence):
    '''
    Return circuit size information (number of main qubits and ancilla qubits, qubit mapping)
    for the provided sequence.

    Returns dict with:
    	nqubits_main: (int) number of qubits acted upon as targets
        nqubits_ancillae: (int) number of qubits used as controls
        qubit_indices_main: (list) list of integers giving indices of target qubits
        qubit_indices_ancillae: (list) list of integers giving indices of control qubits
        assemblage2circuit_qubit: (dict) mapping of { assemblage_qubit_index : circuit_qubit_index }
    
    sequence: (list) list of SequenceObject's, each of which is to be transformed into a gate
              or list of lists of SequenceObjects, for a full assemblage
    '''
    if sequence and isinstance (sequence[0], list):
        # full assemblage
        all_targets = [ (so.target if so.target is not None else 0) for seq in sequence for so in seq ]
        all_controls_lists = [ so.controls for seq in sequence for so in seq if so.controls is not None ]
    else:
        # single sequence
        all_targets = [ (so.target if so.target is not None else 0) for so in sequence ]	# note remapping None -> 0
        all_controls_lists = [ so.controls for so in sequence if so.controls is not None ]
    all_controls = [ x for clist in all_controls_lists for x in clist ]
    
    # determine nqubits_main from targets, and nqubits_ancillae from controls 
    assert all([ isinstance(x, int) for x in all_targets])
    assert all([ isinstance(x, int) for x in all_controls])

    all_targets = sorted(list(set(all_targets)))
    all_controls = sorted(list(set(all_controls)))

    nqubits_main = len(all_targets)
    nqubits_ancillae = len(all_controls)

    # construct a map from Assemblage qubit indices to quantum circuit indices
    # this is needed because a sequence may only act on a subset of qubits, and
    # the quantum circuit may just be for a portion of the quantum circuit for
    # a full assemblage.
    #
    # assemblage qubits are numbered with main qubits starting from 0, and
    # ancilla qubits afterwards.  The main qubits are the target qubits.
    #
    # circuit qubits are split into main and ancillae registers. each register is
    # numbered starting from 0.  But we use the convention of mapping qubit indices
    # larger than the size of the main register, into qubits of the ancillae register.

    if all_controls:
        assert min(all_controls) > max(all_targets)	# ensure control qubits are numbered after all target qubits
    all_qubits = all_targets + all_controls	# concatenate lists of control and target qubits
    assemblage2circuit_qubit = { aindex: cindex for (cindex, aindex) in enumerate(all_qubits) }

    return {'nqubits_main': nqubits_main,
            'nqubits_ancillae': nqubits_ancillae,
            'qubit_indices_main': all_targets,
            'qubit_indices_ancillae': all_controls,
            'assemblage2circuit_qubit': assemblage2circuit_qubit,
            }

def seq2circ(sequence, signal_value=0):
    '''
    Return an instance of SequenceQuantumCircuit corresponding to the provided sequence.

    sequence: (list) list of SequenceObject's, each of which is to be transformed into a gate
              or list of lists of SequenceObjects, for a full assemblage
    signal_value: (floar) value to use for signal, if it appears in the sequence
    '''
    csinfo = sequence_circuit_size(sequence)
    nqubits_main = csinfo['nqubits_main']
    nqubits_ancillae = csinfo['nqubits_ancillae']
    qubit_indices_main = csinfo['qubit_indices_main']
    qubit_indices_ancillae = csinfo['qubit_indices_ancillae']
    assemblage2circuit_qubit = csinfo['assemblage2circuit_qubit']

    qcirc = SequenceQuantumCircuit(nqubits_main, nqubits_ancillae, qubit_indices_main, qubit_indices_ancillae)
    
    def add_gate_for_seq_obj(seq):
        '''
        seq = SequenceObject instance
        '''
        ginfo = SequenceMap.get(seq.__class__)
        if not ginfo:
            raise Exception(f"[pyqsp.gadgets.seq2circ] Error! Sequence element {seq} ({seq.__class__.__name__}) has no known corresponding quantum circuit gate element!")
        nqubits = ginfo['nqubits']
        try:
            target = assemblage2circuit_qubit[seq.target or 0]	# note remapping of qubit indices
        except Exception as err:
            raise Exception(f"[pyqsp.gadgets.seq2circ] Error! For seq {seq} Cannot remap target {seq.target or 0} using quit_indices_main={qubit_indices_main}, err={err}")
        if seq.controls:
            controls = [ assemblage2circuit_qubit[x] for x in seq.controls ]	# control qubits are in the ancillae qubit register
        else:
            controls = []
        arg = ginfo['arg']
        argv = None					# swap gate has no argument, for example
        qubits = ginfo.get('qubits')			# optional spec of which qubits to act upon, e.g. for SWAP
        if arg=='signal':
            argv = signal_value				# use signal_value argument to seq2qcirc call
        elif arg:
            argv = getattr(seq, arg)
        try:
            if argv is not None:
                gate = ginfo['gate'](argv)
            else:
                gate = ginfo['gate']()
        except Exception as err:
            print(f"[pyqsp.gadgets.seq2circ] Error! Could not create gate for {seq}, gate={gate}, argv={argv}, err={err}")
            raise
        if controls:
            gate = gate.control(len(controls))		# make controlled gate
        if nqubits==1:
            qcirc.add_gate(gate, controls, [target])
        elif nqubits==2:
            if qubits:
                try:
                    qubit_list = [getattr(seq, q) for q in qubits]
                except Exception as err:
                    print(f"[pyqsp.gadgets.seq2circ] Error! Could not create gate for {seq}, gate={gate}, argv={argv}, failed to get qubits {qubits}, err={err}")
                    raise
                try:
                    targets = [ assemblage2circuit_qubit[x] for x in qubit_list ]
                except Exception as err:
                    raise Exception(f"[pyqsp.gadgets.seq2circ] Error! Could not get target qubit indices for {seq}, gate={gate}, qubit_list=={qubit_list}, ginfo={ginfo}, qubit_indices_main={qubit_indices_main}, assemblage2circuit_qubit={assemblage2circuit_qubit}, qubits={qubits}")
                if not len(set(targets))==len(targets):
                    raise Exception(f"[pyqsp.gadgets.seq2circ] Error! Duplicate targets for {seq}, gate={gate}, qubit_list=={qubit_list}, targets={targets} ginfo={ginfo}, qubits={qubits}")
            else:
                targets = [target]
            qcirc.add_gate(gate, controls, targets)

    if sequence and isinstance (sequence[0], list):
        # full assemblage
        for seqlist in sequence:
            for seq in seqlist:
                add_gate_for_seq_obj(seq)
    else:
        # single sequence
        for seq in sequence:
            add_gate_for_seq_obj(seq)

    return qcirc
