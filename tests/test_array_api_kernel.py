import numpy as np
import pytest

from structured_linear._solver import _interaction_product


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_interaction_product_preserves_array_api_namespace(dtype):
    xp = pytest.importorskip("array_api_strict", minversion="2.6")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(17, 3)).astype(dtype)
    z = rng.normal(size=(17, 2)).astype(dtype)
    theta = rng.normal(size=(3, 2)).astype(dtype)
    device = xp.Device("CPU_DEVICE")
    arrays = [xp.asarray(a, device=device) for a in (x, z, theta)]
    actual = _interaction_product(*arrays)
    assert actual.__array_namespace__() is xp
    assert actual.device == arrays[0].device
    assert actual.dtype == getattr(xp, dtype)
    expected = np.einsum("ij,ij->i", x, z @ theta.T)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)
