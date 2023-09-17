"""
Atomic gadgets can be compiled to a list of SequenceObject objects; all of these are assumed to act on some subset of the b 'working qubits' of a given assemblage.
"""
class SequenceObject:
    def __init__(self, target=None, controls=None):
        self.target = target
        self.controls = controls

class XGate(SequenceObject):
    
    def __init__(self, angle, target=None, controls=None):
        self.angle = angle
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[X: %0.3f, t=%s, c=%s]" % (self.angle, str(self.target), str(self.controls))
        return string

class YGate(SequenceObject):
    
    def __init__(self, angle, target=None, controls=None):
        self.angle = angle
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[Y: %0.3f, t=%s, c=%s]" % (self.angle, str(self.target), str(self.controls))
        return string

class ZGate(SequenceObject):
    
    def __init__(self, angle, target=None, controls=None):
        self.angle = angle
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[Z: %0.3f, t=%s, c=%s]" % (self.angle, str(self.target), str(self.controls))
        return string

class SignalGate(SequenceObject):
    
    def __init__(self, label, target=None, controls=None):
        # This label should be an integer; check this.
        self.label = label
        super().__init__(target=target, controls=controls)

    def __str__(self):
        string = "[SIG: %d, t=%s, c=%s]" % (self.label, str(self.target), str(self.controls))
        return string

class SwapGate(SequenceObject):
    
    def __init__(self, index_0, index_1, target=None, controls=None):
        self.index_0 = index_0
        self.index_1 = index_1
        super().__init__(target=target, controls=controls)
    
    def __str__(self):
        string = "[SWAP: %d-%d, t=%s, c=%s]" % (self.index_0, self.index_1, str(self.target), str(self.controls))
        return string