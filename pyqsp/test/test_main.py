import unittest
import runpy

import numpy as np
import pytest

from pyqsp import main, response, TensorFlowRetiredError

# -----------------------------------------------------------------------------
# unit tests

test_cmds = [
    "--return-angles --poly=-1,0,2 poly2angles",
    "--return-angles --poly=-1,0,2 --plot --hide-plot poly2angles",
    "--plot-positive-only --plot-probability --plot-tight-y --plot-npts=400 --seqargs=10,0.5 fpsearch",
    "--plot-real-only --plot-npts=400 --seqargs=19,10 poly_sign",
    "--plot-real-only --plot-npts=400 --seqargs=10,0.1 hamsim",
    "--plot-real-only --plot-npts=400 --seqargs=18,10 poly_thresh",
    "--plot-real-only --plot-npts=400 --seqargs=19,0.25 poly_linear_amp",
    "--plot-real-only --plot-npts=400 --seqargs=18,10 poly_phase",
    "--plot-positive-only --plot-real-only --seqargs 20,0.3 efilter",
    "--plot-positive-only --plot-real-only --seqargs=20,3.5 gibbs",
    "--plot-real-only --seqargs=20,0.6,15 relu",
    "--return-angles --func np.cos(3*x) --polydeg 41 sym_qsp_func",
]

"""
Removed "--plot-real-only --plot-npts=400 --seqargs=3,0.3 invert"
from above, given instability of completion; updated method of computing
coefficients of inverse-approximating polynomial in a stable way now
pushes completion methods outside realm of good performance. However
such QSP phases are now easily computable with symmetric QSP iteration.
"""

test_cmds_tf = [
    '--return-angles --func np.cos(3*x) --polydeg 6 polyfunc',
    '--plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign --method tf poly',
]


class Test_main(unittest.TestCase):
    def test_main(self):
        for i, cmd in enumerate(test_cmds):
            with self.subTest(i=i):
                print(f"[pyqsp.test_main testing '{cmd}'")
                phiset = main.CommandLine(arglist=cmd.split(" "))

    def test_main_tf(self):
        for cmd in test_cmds_tf:
            with self.subTest(cmd=cmd):
                with self.assertRaises(TensorFlowRetiredError):
                    main.CommandLine(arglist=cmd.split(" "))


@pytest.mark.parametrize("method", ["laurent", "sym_qsp"])
def test_cli_plot_uses_the_synthesized_target_basis(method, monkeypatch):
    captured = {}
    monkeypatch.setattr(response, "PlotQSPResponse",
                        lambda phases, **kwargs: captured.update(kwargs))
    phases = main.CommandLine(arglist=[
        "--poly=.1,0,.2", "--method=" + method, "--return-angles",
        "--plot", "--hide-plot", "poly2angles"])
    points = np.array([-1., -.7, 0., .3, 1.])
    expected = .1 + .2*(2*points**2-1) if method == "sym_qsp" else .1+.2*points**2
    plotted = np.polynomial.chebyshev.chebval(points, captured["pcoefs"])
    np.testing.assert_allclose(plotted, expected, atol=1e-15, rtol=0)
    assert captured["sym_qsp"] == (method == "sym_qsp")
    measured = response.ComputeQSPResponse(points, phases,
                                          sym_qsp=(method == "sym_qsp"))["pdat"]
    if method == "sym_qsp":
        np.testing.assert_allclose(measured.imag, expected, atol=1e-12, rtol=0)
    else:
        effective = (1-1e-4)*(expected+.5e-4*(2*points**2-1))
        np.testing.assert_allclose(measured, effective, atol=1e-12, rtol=0)


def test_symmetric_example_uses_returned_scale(monkeypatch):
    captured = {}
    monkeypatch.setattr(response, "PlotQSPResponse",
                        lambda phases, **kwargs: captured.update(kwargs))
    example = runpy.run_module("pyqsp.sym_qsp_min_example", run_name="__main__")
    points = np.array([-1., -.7, 0., .3, 1.])
    expected = example["scale"]*np.cos(3*points)
    np.testing.assert_allclose(captured["target"](points), expected, atol=0, rtol=0)
    # The returned multiplier differs from the requested estimated magnitude.
    assert abs(example["scale"]-example["max_scale"]) > 1e-10
    measured = response.ComputeQSPResponse(
        points, example["phiset"], sym_qsp=True)["pdat"].imag
    np.testing.assert_allclose(measured, example["poly"](points), atol=1e-12, rtol=0)


@pytest.mark.parametrize("method", ["laurent", "sym_qsp"])
@pytest.mark.parametrize("command, generator, title, expected", [
    ("invert", "PolyOneOverX", "Inversion", [-.8, 1.6, .4/.75]),
    ("gibbs", "PolyGibbs", "Gibbs distribution", .4*np.exp([.25, -.125, -.375])),
    ("efilter", "PolyEigenstateFiltering", "Eigenstate filtering", [0, .2, 0]),
    ("relu", "PolySoftPlus", "ReLU Function", [0, 0, .1]),
    ("poly_sign", "PolySign", "Sign Function", [-.4, .4, .4]),
    ("poly_thresh", "PolyThreshold", "Threshold Function", [.2, .4, 0]),
    ("poly_phase", "PolyPhaseEstimation", "Phase Estimation Polynomial", [.4, .4, -.4]),
    ("poly_rect", "PolyRect", "Rect Function", [.4, 0, .4]),
    ("invert_rect", "PolyOneOverXRect", "Poly Rect * 1/x", [-.4, 0, .4/.75]),
    ("poly_linear_amp", "PolyLinearAmplification", "Linear Amplification Polynomial", [-.2, .1, .3]),
])
def test_named_command_generation_and_plot_contract(
        command, generator, title, expected, method, monkeypatch):
    symmetric = method == "sym_qsp"
    phases = np.array([.1, .2, .3]) if symmetric else [.1, .2, .3]
    coefficients, captured = [.2, 0, .1], {}

    def generate(self, *values, **options):
        assert values == (4, .5, 2)
        expected_options = dict(ensure_bounded=True, return_scale=True,
                                chebyshev_basis=symmetric)
        if command == "invert":
            expected_options["return_coef"] = True
        assert options == expected_options
        return coefficients, .4

    def synthesize(actual, **options):
        assert actual is coefficients
        assert options == dict(signal_operator="Wz", method=method, tolerance=.1,
                               nepochs=5000, npts_theta=30, chebyshev_basis=symmetric)
        return (phases, [], 0) if symmetric else phases

    def plot(actual, **options):
        assert actual is phases
        captured.update(options)

    monkeypatch.setattr(getattr(main.pyqsp.poly, generator), "generate", generate)
    monkeypatch.setattr(main.angle_sequence, "QuantumSignalProcessingPhases", synthesize)
    monkeypatch.setattr(main.response, "PlotQSPResponse", plot)
    actual = main.CommandLine(arglist=[command, "--seqargs=4,.5,2", "--method", method,
                                      "--signal_operator=Wz", "--measurement=x",
                                      "--plot", "--hide-plot", "--return-angles"])
    assert actual is phases
    np.testing.assert_allclose(captured.pop("target")(np.array([-.5, .25, .75])), expected,
                               atol=1e-15, rtol=0)
    assert captured["signal_operator"] == "Wx" and captured["title"] == title
    assert captured.get("measurement") == ("z" if command == "invert" else None)
    assert captured["sym_qsp"] is symmetric and captured["show"] is False


def test_named_command_resolves_rebound_generator_classes(monkeypatch):
    calls = []
    phases = [.1, .2, .3]

    def synthesize(coefficients, **options):
        calls.append(coefficients)
        assert options['method'] == 'laurent'
        return phases

    monkeypatch.setattr(main.angle_sequence, 'QuantumSignalProcessingPhases', synthesize)
    for coefficients in ([0, .2], [0, -.3]):
        class ReboundSign:
            def generate(self, *values, **options):
                assert values == (3, .5)
                return coefficients, .4

        monkeypatch.setattr(main.pyqsp.poly, 'PolySign', ReboundSign)
        actual = main.CommandLine(arglist=[
            'poly_sign', '--seqargs=3,.5', '--method=laurent', '--return-angles'])
        assert actual is phases
        assert calls[-1] is coefficients
    assert len(calls) == 2


@pytest.mark.parametrize('method', ['laurent', 'sym_qsp'])
def test_custom_registry_preserves_target_polynomial_metadata(method, monkeypatch):
    symmetric = method == 'sym_qsp'
    phases = np.array([.1, .2, .3]) if symmetric else [.1, .2, .3]
    target = lambda x: .2*x*x - .1
    metadata = object()
    coefficients = main.TargetPolynomial([.2, 0, .1], target=target, scale=metadata)
    captured = {}

    class CustomRecipe:
        def generate(self, *values, **options):
            assert values == (2, .25)
            assert options == dict(return_scale=True, chebyshev_basis=symmetric)
            return coefficients, .7

    def synthesize(actual, **options):
        assert actual is coefficients
        assert options['method'] == method
        return (phases, [], 0) if symmetric else phases

    def plot(actual, **options):
        assert actual is phases
        captured.update(options)

    assert main.polynomial_generators is main.pyqsp.poly.polynomial_generators
    monkeypatch.setitem(main.polynomial_generators, 'custom_contract', CustomRecipe)
    monkeypatch.setattr(main.angle_sequence, 'QuantumSignalProcessingPhases', synthesize)
    monkeypatch.setattr(main.response, 'PlotQSPResponse', plot)
    actual = main.CommandLine(arglist=[
        'poly', '--polyname=custom_contract', '--polyargs=2,.25', '--method', method,
        '--signal_operator=Wz', '--plot', '--hide-plot', '--return-angles'])
    assert actual is phases
    assert coefficients.scale is metadata
    assert captured['pcoefs'] is coefficients
    assert captured['target'] is target
    assert captured['signal_operator'] == 'Wx'
    assert captured['sym_qsp'] is symmetric
