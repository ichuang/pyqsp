"""Importable migration shims for the retired TensorFlow training layers."""

from pyqsp import raise_tensorflow_retired


class QSP:
    """Retired Keras QSP layer; construction raises TensorFlowRetiredError."""

    def __init__(self, poly_deg=0, measurement="z"):
        raise_tensorflow_retired("qsp_layers.QSP")


def construct_qsp_model(poly_deg, measurement="z"):
    """Retired model builder; raises without importing TensorFlow."""
    raise_tensorflow_retired("construct_qsp_model")
