from .qsp_layers import QSP, construct_qsp_model
from .viz_tools import plot_loss, plot_qsp_response, compute_qsp_response

try:
    from .qsp_circuit import QSPCircuit
except ImportError as error:
    if error.name != "cirq":
        raise

    class QSPCircuit:
        """Deferred optional circuit constructor when Cirq is unavailable."""

        def __new__(cls, *args, **kwargs):
            from .qsp_circuit import QSPCircuit as circuit_type
            return circuit_type(*args, **kwargs)
