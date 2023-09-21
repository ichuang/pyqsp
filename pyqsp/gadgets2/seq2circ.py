import qiskit
import qiskit_aer
import qiskit.circuit.library as qcl
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter	# used for signal parameters
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
        self.signal_parameters = {}	# dict with {signal_label: qiskit_parameter, ...} # QUESTION: where is this initialized?

        # construct qiskit quantum circuit with separate main and ancillae registers
        self.q_main = qiskit.QuantumRegister(self.nqubits_main, name='main')
        self.q_ancillae = qiskit.QuantumRegister(self.nqubits_ancillae, name='ancillae')
        self.circ = QuantumCircuit(self.q_main, self.q_ancillae)
        self.bound_circ = None									# circuit with bound parameters

        if self.verbose:
            print(f"['pyqsp.gadgets.seq2circ.QuantumCircuit'] Creating sequence quantum circuit on {self.nqubits} qubits with indices {qubit_indices_main}:{qubit_indices_ancillae}")
        return

    GATE_COLORS = {
        'signal_0': ('#60A160', '#EEEE12'),
        'signal_1': ('#80B110', '#DEEE12'),
        'signal_2': ('#20A1A0', '#CEEE12'),
        'signal_3': ('#80A1F0', '#BEEE12'),
    }

    def get_parameter(self, signal_label):
        '''
        Return qiskit circuit parameter corresponding to the specified signal label.
        Creates the parameter if it doesn't already exist

        signal_label: (int) identifier for the signal; starts at 0 by convention
        '''
        if signal_label not in self.signal_parameters:
            self.signal_parameters[signal_label] = Parameter(chr(0x3b8) + str(signal_label))	# unicode 0x3b8 = theta
        return self.signal_parameters[signal_label]

    def bind_parameters(self, values):
        '''
        Set signal parameters to the specified values.
        This should be done before evaluating the circuit, or getting its unitary.

        values: either ordered list of floats, for signal_0, signal_1, ...
                or dict with { 0: float, 1: float, ... }
        '''
        if isinstance(values, list) or isinstance(values, np.ndarray):
            vdict = {self.signal_parameters[x]: values[x]  for x in sorted(self.signal_parameters.keys())}
        elif isinstance(values, dict):
            vdict = {self.signal_parameters[x]: values[x]  for x in self.signal_parameters if x in values}
        else:
            raise Exception(f"[pyqsp.gadgets.seq2circ.QuantumCircuit] bind_parameters called withn type(values)={type(values)} unknown - should be list, np.array, or dict")
        try:
            self.bound_circ = self.circ.bind_parameters(vdict)
            if self.verbose:
                print(f"[pyqsp.gadgets.seq2circ.QuantumCircuit] bound parameters with vdict={vdict}")
        except Exception as err:
            raise Exception(f"[pyqsp.gadgets.seq2circ.QuantumCircuit] failed to bind parameters with values={values} and vdict={vdict}, err={err}")

    def one_dim_response_function(self, start_values=None, end_values=None, uindex=None, npts=200):
        '''
        Compute one-dimensional response function for the circuit, returning two 
        numpy arrays X, Y each of which has npts points.  The response is computed along a vector
        starting from start_values and ending at end_values.  By default, these are [-1] and [+1]
        (with zeros appended for higher dimensional inputs).  The response is given by the 
        matrix element U[uindex], where uindex defaults to (0,0) if not specified.

        start_values: (list) starting value for inputs, defaults to [-1, 0...], of length = number of signals
        end_values: (list) ending value for inputs, defaults to [+1, 0...], of length = number of signals
        uindex: (tuple) two-dimensional index for the element of the unitary to take as the response output;
                defaults to (0,0)
        npts: (int) number of points to sample uniformly along the vector from start to end

        Returns:

        X : (np.ndarray) npts x Nsig dimensional array, where Nsig = number of signals
            For Nsig=1, a one-dimensional array of X points can be obtained using X[:, 0]
        Y : (np.ndarray) one-dimensional array of length npts, giving the [uindex] element of the circuit's unitary
        '''

        # QUESTION: if we want to take the same slice across all inputs, do we replace 0s with 1s below?
        # QUESTION: it looks like signal_parameters is not returning the expected size on a two-input gadget (see otest_simple_2_1_gadget_composition in test_gadget_assemblage.py)? Does it need to be initialized first? 
        dim_inputs = len(self.signal_parameters)

        start_values = np.array(start_values or [-1] + [0]*(dim_inputs-1))
        end_values = np.array(end_values or [1] + [0]*(dim_inputs-1))
        vec = end_values - start_values
        xs = np.linspace(0, 1, npts)
        uindex = uindex or (0,0)
        X = []
        Y = []
        for xv in xs:
            xp = start_values + xv * vec 
            xpacos = 2*np.arccos(xp)
            umat = self.get_unitary(values=xpacos).data
            Y.append(umat[uindex])
            X.append(xp)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def draw(self, *args, **kwargs):
        '''
        Draw circuit - using qiskit
        '''
        if 'style' not in kwargs:
            kwargs['style'] = {'displaycolor': self.GATE_COLORS}
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
        
    def get_unitary(self, values=None, decimals=3):
        '''
        Return unitary transform matrix performed by this circuit.
        if values is not None then first use those to bind parameters.

        values: either ordered list of floats, for signal_0, signal_1, ...
                or dict with { 0: float, 1: float, ... }
        '''
        if values is not None:
            self.bind_parameters(values)

        if self.signal_parameters and (self.bound_circ is None):
            raise Exception(f"[pyqsp.gadgets.seq2circ] Error! Cannot get unitary for circuit without providing values for the {len(self.signal_parameters)} signal parameters")

        if self.bound_circ is not None:
            circ2 = self.bound_circ.copy()
        else:
            circ2 = self.circ.copy()

        if 1:
            # deprecated
            backend = qiskit.Aer.get_backend("unitary_simulator")
            job = qiskit.execute(circ2, backend)
            result = job.result()
        else:
            # good as of Fall 2023
            # BUT controlled-Rx failing with qiskit_aer.aererror.AerError: 'unknown instruction: crx'
            backend = qiskit_aer.Aer.get_backend('aer_simulator')
            circ2.save_unitary()		# add save_unitary instruction to copied circuit
            result = backend.run(circ2).result()            
        U = result.get_unitary(circ2, decimals=decimals)
        if self.verbose:
            print(U)
        return U

# mapping between sequence objects and qiskit circuit library elements
SequenceMap = {
    XGate: {'gate': qcl.RXGate, 'nqubits': 1, 'arg': "angle"},
    YGate: {'gate': qcl.RYGate, 'nqubits': 1, 'arg': "angle"},
    ZGate: {'gate': qcl.RZGate, 'nqubits': 1, 'arg': "angle"},
    SignalGate: {'gate': qcl.RXGate, 'nqubits': 1, 'arg': "signal", 'color_index': "label", 'color_class': "signal"},
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

def seq2circ(sequence, verbose=False):
    '''
    Return an instance of SequenceQuantumCircuit corresponding to the provided sequence.

    sequence: (list) list of SequenceObject's, each of which is to be transformed into a gate
              or list of lists of SequenceObjects, for a full assemblage
    verbose: (bool) output debugging messages if True (passed on to SequenceQuantumCircuit
    '''
    csinfo = sequence_circuit_size(sequence)
    nqubits_main = csinfo['nqubits_main']
    nqubits_ancillae = csinfo['nqubits_ancillae']
    qubit_indices_main = csinfo['qubit_indices_main']
    qubit_indices_ancillae = csinfo['qubit_indices_ancillae']
    assemblage2circuit_qubit = csinfo['assemblage2circuit_qubit']

    qcirc = SequenceQuantumCircuit(nqubits_main, nqubits_ancillae, qubit_indices_main, qubit_indices_ancillae, verbose=verbose)
    
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
            # argv = signal_value			# OLD: use signal_value argument to seq2qcirc call
            argv = qcirc.get_parameter(seq.label)	# NEW: use qiskit Parameter for signal
        elif arg:
            argv = getattr(seq, arg) * 2		# TEMPORARY doubling of rotation angles to match GSLW conventions - fix after changing gate primitives
        kwargs = {}
        if 'color_class' in ginfo:
            color_index = getattr(seq, ginfo.get('color_index', 'label'), 0)
            kwargs['label'] = f"{ginfo['color_class']}_{color_index}"
        try:
            if argv is not None:
                gate = ginfo['gate'](argv, **kwargs)
            else:
                gate = ginfo['gate'](**kwargs)
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
