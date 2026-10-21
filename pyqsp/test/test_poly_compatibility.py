"""Legacy recipe extension contracts, independent of the fitted coefficients."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyqsp import poly


def erf(values):
    return np.vectorize(math.erf)(values)


def filter_chebyshev(values):
    transformed = -1 + 2 * (values**2 - .2**2) / (1 - .2**2)
    return 2 * transformed**2 - 1


RECIPES = [
    pytest.param(poly.PolySign, 3, lambda x: erf(2*x), id="sign"),
    pytest.param(poly.PolyThreshold, 2,
                 lambda x: (erf(2*(x + .5)) - erf(2*(x - .5))) / 2,
                 id="threshold"),
    pytest.param(poly.PolyPhaseEstimation, 2,
                 lambda x: -1 + erf(2*(1/np.sqrt(2) - x))
                 + erf(2*(1/np.sqrt(2) + x)), id="phase-estimation"),
    pytest.param(poly.PolyRect, 2,
                 lambda x: 1 + (erf(np.sqrt(np.log(2/(np.pi*.1**2))/2)
                                   * (x - .25))
                                + erf(np.sqrt(np.log(2/(np.pi*.1**2))/2)
                                      * (-x - .25))) / 2,
                 id="rectangle"),
    pytest.param(poly.PolyGibbs, 2, lambda x: np.exp(-2*np.abs(x)), id="gibbs"),
    pytest.param(poly.PolyEigenstateFiltering, 2,
                 lambda x: filter_chebyshev(x) / filter_chebyshev(0),
                 id="eigenstate-filter"),
    pytest.param(poly.PolySoftPlus, 2,
                 lambda x: np.log(1 + np.exp(np.abs(x) - .2)), id="softplus"),
]


@pytest.mark.parametrize("generator, degree, target", RECIPES)
@pytest.mark.parametrize("with_scale", [False, True], ids=["bare", "scaled"])
def test_recipe_overridden_fitter_preserves_objects(generator, degree, target,
                                                    with_scale):
    class Coefficients(np.ndarray):
        pass

    coefficients = np.arange(1., degree + 2).view(Coefficients)
    fitted = SimpleNamespace(coef=coefficients)
    scale = object()
    max_scale, cheb_samples = object(), object()
    calls = []
    samples = np.array([-.7, 0., .4])

    class Recipe(generator):
        def taylor_series(self, function, requested_degree, **options):
            calls.append((self, requested_degree, options))
            np.testing.assert_allclose(function(samples), target(samples),
                                       rtol=0, atol=1e-14)
            np.testing.assert_allclose(function(.4), target(.4),
                                       rtol=0, atol=1e-14)
            return (fitted, scale) if with_scale else fitted

    recipe = Recipe()
    result = recipe.generate(float(degree), ensure_bounded=True,
                             return_scale=with_scale, chebyshev_basis=False,
                             max_scale=max_scale, cheb_samples=cheb_samples)
    assert len(calls) == 1
    instance, requested_degree, options = calls[0]
    assert instance is recipe and type(requested_degree) is int
    assert requested_degree == degree
    expected_options = dict(ensure_bounded=True, return_scale=with_scale,
                            max_scale=max_scale, chebyshev_basis=False,
                            cheb_samples=cheb_samples)
    assert options.keys() == expected_options.keys()
    assert all(options[key] is value for key, value in expected_options.items())
    expected = np.arange(1., degree + 2)
    expected[0 if generator is poly.PolySign else 1::2] = 0
    np.testing.assert_array_equal(coefficients, expected)
    assert type(coefficients) is Coefficients
    if with_scale:
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] is coefficients and result[1] is scale
    elif generator in (poly.PolySign, poly.PolyGibbs):
        assert type(result) is poly.TargetPolynomial
        np.testing.assert_array_equal(result.coef, expected)
        expected_target = np.sign(samples) if generator is poly.PolySign else target(samples)
        np.testing.assert_allclose(result.target(samples), expected_target,
                                   rtol=0, atol=1e-14)
        assert result.scale is None
    else:
        assert result is coefficients


@pytest.mark.parametrize("generator, degree, target", RECIPES)
@pytest.mark.parametrize("change", ["bounded", "return", "short-circuit"])
def test_recipe_fitter_keeps_flag_evaluation_order(generator, degree, target,
                                                  change):
    events = []

    class Flag:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def __bool__(self):
            events.append(self.name)
            return self.value

    bounded = Flag("bounded", change != "short-circuit")
    return_scale = Flag("return", True)
    coefficients = np.arange(1., degree + 2)
    fitted = SimpleNamespace(coef=coefficients)

    class Recipe(generator):
        def taylor_series(self, function, requested_degree, **options):
            events.append("fitter")
            assert options["ensure_bounded"] is bounded
            assert options["return_scale"] is return_scale
            if change == "short-circuit":
                return fitted
            if change == "bounded":
                bounded.value = False
            else:
                return_scale.value = False
            return fitted, object()

    result = Recipe().generate(degree, ensure_bounded=bounded,
                               return_scale=return_scale)
    if change == "short-circuit":
        assert events == ["bounded", "fitter", "bounded"]
    elif change == "bounded":
        assert events == ["bounded", "return", "fitter", "bounded"]
    else:
        assert events == ["bounded", "return", "fitter", "bounded", "return"]
    if generator in (poly.PolySign, poly.PolyGibbs):
        assert type(result) is poly.TargetPolynomial
        np.testing.assert_array_equal(result.coef, coefficients)
    else:
        assert result is coefficients


def test_eigenstate_fitter_callback_retains_scale_rebinding():
    callbacks = []
    scale = np.float64(.25)
    samples = np.array([-.7, 0., .4])

    class Recipe(poly.PolyEigenstateFiltering):
        def taylor_series(self, function, degree, **options):
            callbacks.append(function)
            np.testing.assert_allclose(
                function(samples), filter_chebyshev(samples)/filter_chebyshev(0),
                rtol=0, atol=1e-14)
            return SimpleNamespace(coef=np.array([.2, .3, .4])), scale

    _, returned_scale = Recipe().generate(2, return_scale=True)
    assert returned_scale is scale
    np.testing.assert_allclose(callbacks[0](samples), scale*filter_chebyshev(samples),
                               rtol=0, atol=1e-14)


def test_legacy_fitter_normalization_can_expand_small_target():
    fitted, scale = poly.PolyTaylorSeries().taylor_series(
        lambda x: .1 + .05*x, 3, ensure_bounded=True, return_scale=True,
        max_scale=.9, chebyshev_basis=False, cheb_samples=999)
    assert type(fitted) is np.polynomial.chebyshev.Chebyshev
    np.testing.assert_allclose(scale, 6, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted.coef, [.6, .3, 0, 0], rtol=0, atol=1e-12)
