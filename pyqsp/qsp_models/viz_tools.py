# visualization tools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from . import QSPCircuit


def compute_qsp_response(
        model=None,
        phis=None,
        return_all=False,
        show_svg=False):
    """Compute the QSP response againts the desired function response.

    Params
    ------
    f : function float --> float
            the desired function to be implemented by the QSP sequence
    model : Keras `Model` with `QSP` layer
            model trained to approximate f
    phis: numpy array of qsp phase angles (may be supplied instead of model)
    """
    all_th = np.arange(0, np.pi, np.pi / 300)

    # construct circuit
    if phis is None:
        phis = model.trainable_weights[0].numpy()
    qsp_circuit = QSPCircuit(phis)
    if show_svg:
        qsp_circuit.svg()
    circuit_px = qsp_circuit.eval_px(all_th)
    circuit_qx = qsp_circuit.eval_qx(all_th)
    qsp_response = qsp_circuit.qsp_response(all_th)
    if return_all:
        return qsp_response, all_th, circuit_px, circuit_qx
    return qsp_response


def plot_qsp_response(f, model=None, phis=None, title="QSP Response"):
    """Plot the QSP response againts the desired function response.

    Params
    ------
    f : function float --> float
            the desired function to be implemented by the QSP sequence
    model : Keras `Model` with `QSP` layer
            model trained to approximate f
    phis: numpy array of qsp phase angles (may be supplied instead of model)
    title: plot title (str)
    """
    qsp_response, all_th, circuit_px, circuit_qx = compute_qsp_response(
        model=model, phis=phis, return_all=True)

    pdata = {
        "x": np.cos(all_th),
        "Imag[p(x)]": np.imag(circuit_px),
        "Real[p(x)]": np.real(circuit_px),
        "Real[q(x)]": np.real(circuit_qx),
        "QSP Response": np.real(qsp_response)}
    if f is not None:
        pdata["desired f"] = f(np.cos(all_th))
    df = pd.DataFrame(pdata)
    df = df.melt("x", var_name="src", value_name="value")
    sns.lineplot(x="x", y="value", hue="src", data=df).set_title(title)
    plt.show()


def plot_loss(history):
    """Plot the error of a trained QSP model.

    Params
    ------
    history : tensorflow `History` object
    """
    plt.plot(history.history['loss'])
    plt.title("Learning QSP Angles")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
