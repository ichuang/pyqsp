"""Independent ordinary-QSP matrix products shared by contract tests."""

import numpy as np


def qsp_matrix(phases, x, signal="Wx"):
    """Evaluate physical Wx/Wz matrices without synthesis or response helpers."""
    root = np.sqrt(1 - x*x)
    if signal == "Wx":
        query = np.array([[x, 1j*root], [1j*root, x]])

        def rotation(phi):
            return np.diag([np.exp(1j*phi), np.exp(-1j*phi)])
    else:
        query = np.diag([x + 1j*root, x - 1j*root])

        def rotation(phi):
            return np.array([[np.cos(phi), 1j*np.sin(phi)],
                             [1j*np.sin(phi), np.cos(phi)]])

    unitary = rotation(phases[0])
    for phase in phases[1:]:
        unitary = unitary @ query @ rotation(phase)
    return unitary


def qsp_amplitude(phases, points, signal="Wx", measurement=None):
    """Return the selected complex amplitude, including its sign and phase."""
    measurement = measurement or {"Wx": "x", "Wz": "z"}[signal]
    state = np.array([1., 1.])/np.sqrt(2) if measurement == "x" else np.array([1., 0.])
    return np.array([state @ qsp_matrix(phases, x, signal) @ state for x in points])
