"""Importable migration shims for retired TensorFlow Quantum layers."""

from pyqsp import raise_tensorflow_retired


class HybridControlledPQC:
    """Retired controlled PQC layer; construction raises a retirement error.

    ``initializer=None`` is an inert placeholder for the former TensorFlow
    initializer object. The shim accepts the historical keywords but never
    imports TensorFlow or constructs a model.
    """

    def __init__(self,
                 model_circuit,
                 operators,
                 *,
                 controlled_symbol_names=None,
                 native_symbol_names=None,
                 repetitions=None,
                 backend=None,
                 initializer=None,
                 regularizer=None,
                 constraint=None,
                 differentiator=None,
                 **kwargs):
        raise_tensorflow_retired("tfq_qsp_layers.HybridControlledPQC")


class Unitary:
    """Retired TFQ unitary layer; use QSPCircuit for Cirq evaluation."""

    def __init__(self, **kwargs):
        raise_tensorflow_retired("tfq_qsp_layers.Unitary")


class QSP:
    """Retired TFQ QSP layer; construction raises a retirement error."""

    def __init__(self, poly_deg=0, **kwargs):
        raise_tensorflow_retired("tfq_qsp_layers.QSP")
