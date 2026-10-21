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
