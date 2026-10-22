"""Complex readout and iterator contracts for the shared response product."""

import numpy as np
import pytest

from pyqsp import response
from pyqsp.test import qsp_matrix


@pytest.mark.parametrize("signal, measurement, symmetric", [
    ("Wz", "x", False),
    ("Wx", "x", True), ("Wx", "z", True),
    ("Wz", "x", True), ("Wz", "z", True),
])
def test_response_retains_complex_readout(signal, measurement, symmetric):
    phases, points = [.17, -.31, .29], [-1., -.4, .3, 1.]
    matrices = [qsp_matrix(phases, x, signal) for x in points]
    state = np.array([1., 1.])/np.sqrt(2) if measurement == "x" else np.array([1., 0.])
    expected = [matrix[0, 0] if symmetric else state @ matrix @ state
                for matrix in matrices]
    result = response.ComputeQSPResponse(points, phases, signal, measurement,
                                        sym_qsp=symmetric)
    np.testing.assert_allclose(result["pdat"], expected, rtol=0, atol=1e-14)
    assert result["pdat"].dtype == np.complex128
    assert set(result) == {"adat", "pdat", "model", "phiset"}
    assert result["adat"] is points and result["phiset"] is phases
    assert result["model"] == (signal, measurement)


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("points", [[], [-.3, .4]], ids=["empty", "nonempty"])
def test_symmetry_flag_is_captured_between_phase_and_sample_iteration(
        symmetric, points):
    events = []

    class Phase:
        def __rmul__(self, factor):
            events.append(("phase-multiply", factor))
            return factor*.2

    class Flag:
        calls = 0

        def __bool__(self):
            events.append("flag")
            self.calls += 1
            return symmetric if self.calls == 1 else not symmetric

    def phases():
        events.append("phases-start")
        yield Phase()
        events.append("phases-end")

    def samples():
        events.append("samples-start")
        for point in points:
            events.append(("sample", point))
            yield point
        events.append("samples-end")

    phase_iterator, sample_iterator, flag = phases(), samples(), Flag()
    result = response.ComputeQSPResponse(sample_iterator, phase_iterator,
                                        sym_qsp=flag)
    assert flag.calls == 1
    assert events == ["phases-start", ("phase-multiply", 1j),
                      ("phase-multiply", -1j), "phases-end", "flag",
                      "samples-start"] + [("sample", point) for point in points] + ["samples-end"]
    expected = np.exp(.2j) if symmetric else np.cos(.2)
    np.testing.assert_allclose(result["pdat"], [expected]*len(points),
                               rtol=0, atol=1e-15)
    assert result["adat"] is sample_iterator
    assert result["phiset"] is phase_iterator
    assert result["pdat"].dtype == np.complex128


@pytest.mark.parametrize("phases", [[], [.2]])
@pytest.mark.parametrize("symmetric", [False, True])
def test_empty_response_keeps_complex_array(phases, symmetric):
    points = []
    result = response.ComputeQSPResponse(points, phases, sym_qsp=symmetric)
    assert result["pdat"].shape == (0,)
    assert result["pdat"].dtype == np.complex128
    assert result["adat"] is points and result["phiset"] is phases


def test_ambiguous_symmetry_flag_fails_before_empty_sample_iteration():
    events = []

    class Samples:
        def __iter__(self):
            events.append("samples")
            return iter([])

    def phases():
        events.append("phases")
        yield .2
        events.append("phases-finished")

    with pytest.raises(ValueError, match="truth value"):
        response.ComputeQSPResponse(Samples(), phases(), sym_qsp=np.array([0, 1]))
    assert events == ["phases", "phases-finished"]


@pytest.mark.parametrize("symmetric", [False, True])
def test_signal_evaluation_precedes_missing_phase_index(symmetric):
    class Sample:
        def __pow__(self, exponent):
            raise RuntimeError("signal input evaluated")

    with pytest.raises(RuntimeError, match="signal input evaluated"):
        response.ComputeQSPResponse([Sample()], [], sym_qsp=symmetric)
