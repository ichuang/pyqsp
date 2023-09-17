class SequenceObject:
    """
    A class to represent a single-qubit (possibly controlled) circuit element.
    
    ...

    Attributes
    ----------
    target : int, optional
        The label of the qubit of the main register on which the single qubit portion of the object acts. Both target and controls are non-overlapping, absolute indices, starting at zero.
    controls : list of int, optional
        A list of labels of qubits in the ancilla register on which the single qubit portion of the object is controlled. Both target and controls are non-overlapping, absolute indices, starting at zero.

    Methods
    -------
    None
    """
    def __init__(self, target=None, controls=None):
        self.target = target
        self.controls = controls

class XGate(SequenceObject):
    """
    A class to represent a single-qubit X rotation.
    
    ...

    Attributes
    ----------
    angle : float
        A real number representing the angle of X rotation, in the form matexp(1j * angle * sig_x) for sig_x the X Pauli matrix.
    target : int, optional
        The label of the qubit of the main register on which the single qubit portion of the object acts.
    controls : list of int, optional
        A list of labels of qubits in the ancilla register on which the single qubit portion of the object is controlled.

    Methods
    -------
    None
    """
    def __init__(self, angle, target=None, controls=None):
        self.angle = angle
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[X: %0.3f, t=%s, c=%s]" % (self.angle, str(self.target), str(self.controls))
        return string

class YGate(SequenceObject):
    """
    A class to represent a single-qubit Y rotation.
    
    ...

    Attributes
    ----------
    angle : float
        A real number representing the angle of Y rotation, in the form matexp(1j * angle * sig_y) for sig_y the Y Pauli matrix.
    target : int, optional
        The label of the qubit of the main register on which the single qubit portion of the object acts.
    controls : list of int, optional
        A list of labels of qubits in the ancilla register on which the single qubit portion of the object is controlled.

    Methods
    -------
    None
    """
    def __init__(self, angle, target=None, controls=None):
        self.angle = angle
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[Y: %0.3f, t=%s, c=%s]" % (self.angle, str(self.target), str(self.controls))
        return string

class ZGate(SequenceObject):
    """
    A class to represent a single-qubit Z rotation.
    
    ...

    Attributes
    ----------
    angle : float
        A real number representing the angle of Y rotation, in the form matexp(1j * angle * sig_z) for sig_z the Z Pauli matrix.
    target : int, optional
        The label of the qubit of the main register on which the single qubit portion of the object acts.
    controls : list of int, optional
        A list of labels of qubits in the ancilla register on which the single qubit portion of the object is controlled.

    Methods
    -------
    None
    """
    def __init__(self, angle, target=None, controls=None):
        self.angle = angle
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[Z: %0.3f, t=%s, c=%s]" % (self.angle, str(self.target), str(self.controls))
        return string

class SignalGate(SequenceObject):
    """
    A class to represent a single qubit oracle unitary.
    
    ...

    Attributes
    ----------
    label : int
        An integer label for an unspecified oracle.
    target : int, optional
        The label of the qubit of the main register on which the single qubit portion of the object acts.
    controls : list of int, optional
        A list of labels of qubits in the ancilla register on which the single qubit portion of the object is controlled.

    Methods
    -------
    None
    """
    def __init__(self, label, target=None, controls=None):
        # TODO: check this label is an integer.
        self.label = label
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[SIG: %d, t=%s, c=%s]" % (self.label, str(self.target), str(self.controls))
        return string

class SwapGate(SequenceObject):
    """
    A class to represent a swap gate between two indices.
    
    ...

    Attributes
    ----------
    index_0 : int
        An integer label for a qubit, to be swapped with index_1.
    index_1 : int
        An integer label for a qubit, to be swapped with index_0.
    target : int, optional
        The label of the qubit of the main register on which the single qubit portion of the object acts.
    controls : list of int, optional
        A list of labels of qubits in the ancilla register on which the single qubit portion of the object is controlled.

    Methods
    -------
    None
    """

    def __init__(self, index_0, index_1, target=None, controls=None):
        self.index_0 = index_0
        self.index_1 = index_1
        super().__init__(target=target, controls=controls)
    
    def __str__(self):
        string = "[SWAP: %d-%d, t=%s, c=%s]" % (self.index_0, self.index_1, str(self.target), str(self.controls))
        return string