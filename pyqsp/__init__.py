"""Polynomial approximation and quantum signal processing."""


class TensorFlowRetiredError(RuntimeError):
    """Raised when a retired TensorFlow or TensorFlow Quantum API is used."""


def raise_tensorflow_retired(entry_point):
    """Explain the synthesis alternatives without loading optional backends."""
    raise TensorFlowRetiredError(
        "{} has been retired with TensorFlow/TFQ training. For polynomial "
        "synthesis, use QuantumSignalProcessingPhases(..., method='laurent') "
        "or method='sym_qsp' with chebyshev_basis=True; these methods use "
        "different response conventions; see README.md for migration guidance. "
        "Keras training, gradients, models, history, and training-related "
        "return_all=True results have no drop-in replacement. "
        .format(entry_point)
    )


from . import poly
