"""Training retirement and retained evaluation contracts, without TensorFlow."""
import inspect
import subprocess
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyqsp import angle_sequence, main, qsp_models, response
from pyqsp.qsp_models import qsp_layers, tfq_qsp_layers
from pyqsp import TensorFlowRetiredError


@pytest.mark.parametrize("call", [
    lambda: angle_sequence.QuantumSignalProcessingPhases([0, .4], method="tf"),
    lambda: angle_sequence.QuantumSignalProcessingPhases(
        [0, .4], method="tf", signal_operator="Wz", chebyshev_basis=True),
    lambda: angle_sequence.QuantumSignalProcessingPhasesWithTensorflow(
        None, 30, 5000, 0, "z", True),
    lambda: qsp_models.QSP(3, "x"),
    lambda: qsp_layers.QSP(3, "z"),
    lambda: qsp_models.construct_qsp_model(3, "x"),
    lambda: qsp_layers.construct_qsp_model(3, "z"),
    lambda: tfq_qsp_layers.HybridControlledPQC(None, None),
    lambda: tfq_qsp_layers.Unitary(name="old_model"),
    lambda: tfq_qsp_layers.QSP(3, name="old_layer"),
    lambda: main.CommandLine(arglist=["polyfunc"]),
])
def test_training_entry_points_raise_retirement(call):
    with pytest.raises(TensorFlowRetiredError) as caught:
        call()
    message = str(caught.value)
    assert "retired" in message
    assert "laurent" in message
    assert "sym_qsp" in message
    assert "history" in message
    assert "migration" in message


def test_named_training_helper_preserves_positional_signature():
    parameters = inspect.signature(
        angle_sequence.QuantumSignalProcessingPhasesWithTensorflow).parameters
    assert list(parameters) == ["poly", "npts_theta", "nepochs", "verbose",
                                "measurement", "return_all"]
    assert [parameter.default for parameter in list(parameters.values())[1:]] == [
        30, 5000, 0, "z", False]


def test_import_and_loss_plot_without_optional_backends():
    # A separate interpreter also checks that no previous test warmed the imports.
    source = '''
import builtins
original_import = builtins.__import__
optional_imports = []
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('tensorflow', 'tensorflow_quantum', 'cirq'):
        optional_imports.append(name)
        raise ModuleNotFoundError("optional backend blocked", name=name.split('.')[0])
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from types import SimpleNamespace
import pyqsp
from pyqsp import angle_sequence, main, TensorFlowRetiredError
assert not optional_imports, optional_imports
from pyqsp import qsp_models
from pyqsp.qsp_models import qsp_layers, tfq_qsp_layers, viz_tools
qsp_models.plot_loss(SimpleNamespace(history={'loss': [1., .2]}))
try:
    qsp_models.QSP(2)
except TensorFlowRetiredError:
    pass
else:
    raise AssertionError('training did not retire')
try:
    qsp_models.QSPCircuit([0., 0.])
except ImportError as error:
    assert 'Cirq' in str(error)
else:
    raise AssertionError('circuit should require Cirq')
'''
    subprocess.run([sys.executable, "-c", source], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   universal_newlines=True)


def test_plot_loss_retains_history_protocol(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.figure()
    try:
        qsp_models.plot_loss(SimpleNamespace(history={"loss": [1., .4, .1]}))
        np.testing.assert_array_equal(plt.gca().lines[0].get_ydata(), [1., .4, .1])
    finally:
        plt.close("all")


def circuit_unitaries(thetas, phis):
    """Independent chronological gate products for the retained Cirq circuit."""
    matrices = []
    for theta in thetas:
        signal = np.array([[np.cos(theta), 1j*np.sin(theta)],
                           [1j*np.sin(theta), np.cos(theta)]])
        matrix = np.diag(np.exp(1j*np.array([phis[0], -phis[0]])))
        for phi in phis[1:]:
            matrix = np.diag(np.exp(1j*np.array([phi, -phi]))) @ signal @ matrix
        matrices.append(matrix)
    return np.array(matrices)


def test_optional_cirq_class_and_signed_complex_response():
    cirq = pytest.importorskip("cirq")
    from pyqsp.qsp_models.qsp_circuit import QSPCircuit
    assert qsp_models.QSPCircuit is QSPCircuit
    assert issubclass(QSPCircuit, cirq.Circuit)
    phis = [-.2, .3, .1]
    thetas = np.array([.2, .8, 1.7, 2.9])
    circuit = QSPCircuit(phis)
    expected = circuit_unitaries(thetas, phis)
    np.testing.assert_allclose(circuit.eval_px(thetas), expected[:, 0, 0], atol=1e-12)
    np.testing.assert_allclose(circuit.eval_qx(thetas),
                               expected[:, 0, 1]/(1j*np.sin(thetas)), atol=1e-12)
    selected = np.sum(expected, axis=(1, 2))/2
    np.testing.assert_allclose(circuit.qsp_response(thetas), selected, atol=1e-12)


def test_optional_phase_and_existing_model_evaluation():
    pytest.importorskip("cirq")
    phis = np.array([-.2, .3, .1])
    model = SimpleNamespace(trainable_weights=[SimpleNamespace(numpy=lambda: phis)])
    values, thetas, px, qx = qsp_models.compute_qsp_response(
        phis=phis, return_all=True)
    expected = circuit_unitaries(thetas, phis)
    np.testing.assert_allclose(values, np.sum(expected, axis=(1, 2))/2, atol=1e-12)
    np.testing.assert_allclose(px, expected[:, 0, 0], atol=1e-12)
    np.testing.assert_allclose(qx[1:], expected[1:, 0, 1]/(1j*np.sin(thetas[1:])),
                               atol=1e-12)
    np.testing.assert_allclose(qsp_models.compute_qsp_response(model=model), values,
                               atol=1e-12)


def test_optional_plot_routes(monkeypatch):
    pytest.importorskip("cirq")
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        response.PlotQSPResponse([-.2, .3, .1], target=lambda x: .4*x*x,
                                show_qsp_model_plot=True)
        assert len(plt.gca().lines) == 5
        plt.close("all")
        main.CommandLine(arglist=["--plot", "--plot-qsp-model", "--phiset=.1,.2",
                                  "response"])
        assert len(plt.gca().lines) == 4
    finally:
        plt.close("all")


@pytest.mark.parametrize("arguments", [
    ["--method", "tf", "--seqname", "fpsearch", "--seqargs", "3,.1", "angles"],
    ["--method", "tf", "--phiset=.1,.2", "response"],
    ["--method", "tf", "--func", "np.cos(3*x)", "--polydeg", "12", "sym_qsp_func"],
])
def test_unused_tf_method_flag_does_not_retire_other_operations(arguments, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        phases = main.CommandLine(arglist=["--return-angles"] + arguments)
        assert phases is not None
        assert np.all(np.isfinite(phases))
    finally:
        plt.close("all")


def test_retirement_does_not_import_tensorflow():
    source = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('tensorflow', 'tensorflow_quantum'):
        raise AssertionError('retired backend was imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from pyqsp import TensorFlowRetiredError, angle_sequence, main, qsp_models
from pyqsp.qsp_models import qsp_layers, tfq_qsp_layers
calls = [
    lambda: angle_sequence.QuantumSignalProcessingPhases(None, method='tf'),
    lambda: angle_sequence.QuantumSignalProcessingPhasesWithTensorflow(None),
    lambda: qsp_models.QSP(),
    lambda: qsp_models.construct_qsp_model(2),
    lambda: qsp_layers.QSP(),
    lambda: tfq_qsp_layers.HybridControlledPQC(None, None),
    lambda: tfq_qsp_layers.Unitary(),
    lambda: tfq_qsp_layers.QSP(),
    lambda: main.CommandLine(arglist=['polyfunc']),
]
for call in calls:
    try:
        call()
    except TensorFlowRetiredError:
        pass
    else:
        raise AssertionError('training did not retire')
"""
    subprocess.run([sys.executable, "-c", source], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   universal_newlines=True)
