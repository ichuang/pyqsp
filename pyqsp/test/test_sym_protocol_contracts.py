"""Ownership and callback contracts for symmetric protocol maintenance."""

import numpy as np
import pytest

from pyqsp import sym_qsp_opt
from pyqsp.test import qsp_matrix


@pytest.mark.parametrize("parity, values, initial, updated", [
    (0, [.2], [.4], [.8]),
    (1, [.2], [.2, .2], [.4, .4]),
    (0, [.2, .3], [.3, .4, .3], [.5, .8, .5]),
    (1, [.2, .3], [.3, .2, .2, .3], [.5, .4, .4, .5]),
])
def test_phase_expansion_copies_and_preserves_subclass_dispatch(
        parity, values, initial, updated):
    events, updates = [], []

    class InputArray(np.ndarray):
        pass

    class Protocol(sym_qsp_opt.SymmetricQSPProtocol):
        def __setattr__(self, name, value):
            events.append(name)
            super().__setattr__(name, value)

        def update_reduced_phases(self, values):
            updates.append(1)
            super().update_reduced_phases(values)

        def _update_full_phases(self):
            raise AssertionError("Initialization and base updates must bypass overrides")

    source = np.array(values).view(InputArray)
    target = object()
    protocol = Protocol(source, parity, target)
    assert events == ["reduced_phases", "parity", "full_phases", "poly_deg", "target_poly"]
    assert updates == [] and protocol.target_poly is target
    assert type(protocol.reduced_phases) is np.ndarray
    assert not np.shares_memory(source, protocol.reduced_phases)
    assert not np.shares_memory(protocol.reduced_phases, protocol.full_phases)
    source[:] = 0
    np.testing.assert_array_equal(protocol.reduced_phases, values)
    np.testing.assert_array_equal(protocol.full_phases, initial)
    assert protocol.poly_deg == len(initial) - 1
    samples = [-1., -.3, 0., .7, 1.]
    np.testing.assert_allclose(protocol.gen_unitary(samples),
                               [qsp_matrix(initial, x) for x in samples],
                               rtol=0, atol=1e-15)

    new_values = [.4] if len(values) == 1 else [.4, .5]
    source = np.array(new_values).view(InputArray)
    events.clear()
    protocol.update_reduced_phases(source)
    assert events == ["reduced_phases", "full_phases", "poly_deg"]
    assert updates == [1] and protocol.target_poly is target
    assert not np.shares_memory(source, protocol.reduced_phases)
    assert not np.shares_memory(protocol.reduced_phases, protocol.full_phases)
    source[:] = 0
    np.testing.assert_array_equal(protocol.reduced_phases, new_values)
    np.testing.assert_array_equal(protocol.full_phases, updated)
    assert protocol.poly_deg == len(updated) - 1
    protocol.full_phases[:] = 0
    np.testing.assert_array_equal(protocol.reduced_phases, new_values)


def test_constant_protocol_never_calls_signal():
    phases = []

    class Protocol(sym_qsp_opt.SymmetricQSPProtocol):
        def phase(self, value):
            phases.append(value)
            return super().phase(value)

        def signal(self, value):
            raise AssertionError("A constant protocol must not evaluate signals")

    protocol = Protocol([.2], 0)
    actual = protocol.gen_unitary(iter([2., np.nan, None]))
    expected = np.diag(np.exp([.4j, -.4j]))
    np.testing.assert_allclose(actual, [expected]*3, rtol=0, atol=1e-15)
    assert phases == [.4]


@pytest.mark.parametrize("mutation", [None, "phase", "signal"])
def test_unitary_preserves_callback_order_and_product_grouping(mutation):
    events = []

    class Product:
        def __init__(self, expression):
            self.expression = expression

        def __matmul__(self, other):
            return Product("(" + self.expression + "@" + other.expression + ")")

    class Protocol(sym_qsp_opt.SymmetricQSPProtocol):
        def phase(self, value):
            index = len(events)
            events.append(("phase", value))
            if mutation == "phase" and index == 2:
                self.full_phases = np.array([0.])
            return Product("p" + str(index))

        def signal(self, value):
            events.append(("signal", value))
            if mutation == "signal":
                self.full_phases = np.array([0.])
            return Product("s" + str(value))

    def samples():
        for value in (0, 1):
            events.append(("sample", value))
            yield value

    actual = Protocol([.2, .3], 0).gen_unitary(samples())
    expected_events = [("phase", .3), ("phase", .4), ("phase", .3)]
    expected = []
    for value in (0, 1):
        expected_events.append(("sample", value))
        if mutation == "phase":
            expected.append("p0")
        else:
            expected_events.append(("signal", value))
            expected.append("((((p0@s{0})@p1)@s{0})@p2)".format(value))
    assert events == expected_events
    assert [matrix.expression for matrix in actual] == expected


def test_nonconstant_signal_callback_precedes_missing_phase_index():
    calls = []

    class Protocol(sym_qsp_opt.SymmetricQSPProtocol):
        def signal(self, value):
            calls.append(value)
            return np.eye(2)

    protocol = Protocol([.2], 1)
    protocol.full_phases = []
    with pytest.raises(IndexError):
        protocol.gen_unitary([.3])
    assert calls == [.3]


def test_empty_samples_still_construct_phase_matrices():
    calls = []

    class Protocol(sym_qsp_opt.SymmetricQSPProtocol):
        def phase(self, value):
            calls.append(value)
            return super().phase(value)

        def signal(self, value):
            raise AssertionError("Empty samples must not evaluate signals")

    result = Protocol([.2], 1).gen_unitary([])
    assert result.shape == (0,)
    assert calls == [.2, .2]
