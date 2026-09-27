"""Cross-effect propagation: the Leontief inverse of ``contribution_from``.

An effect that contributes to another — CO2 priced into cost — makes the
effect system a linear one, and ``(I - C)^-1`` is what solves it. The binder
hands the model the chained share it implies; the ledger and the priced
contributions in ``reporting.yaml`` both read that share, so nothing else
inverts it.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def leontief(cf: xr.DataArray) -> xr.DataArray:
    """Compute Leontief inverse (I - C)^-1 from cross-effect coefficients.

    Args:
        cf: Cross-effect coefficients with dims ``(effect, source_effect)``
            and optionally extra batch dims (e.g. ``time``).
    """
    n = cf.sizes['effect']
    batch_dims = [d for d in cf.dims if d not in ('effect', 'source_effect')]
    ordered = [*batch_dims, 'effect', 'source_effect']
    mat = np.eye(n) - cf.transpose(*ordered).values  # (..., n, n)
    return xr.DataArray(_inverse_per_batch(mat), dims=ordered, coords=cf.coords).transpose(*cf.dims)


def _inverse_per_batch(mat: np.ndarray) -> np.ndarray:
    """Invert a batched matrix array ``(..., n, n)``.

    The cross-effect factors are usually constant along ``time``/``period``,
    so when every batch element holds the same matrix it is decomposed once
    and broadcast instead of once per element.

    Raises:
        ValueError: If any matrix is singular (circular contribution_from chains).
    """
    n = mat.shape[-1]
    stack = mat.reshape(-1, n, n)
    if len(stack) > 1 and bool((stack == stack[0]).all()):
        stack = stack[:1]
    if np.any(np.linalg.matrix_rank(stack) < n):
        raise ValueError('Cross-effect matrix (I - C) is singular — check for circular contribution_from chains')
    inverse = np.linalg.inv(stack)
    return np.broadcast_to(inverse[0], mat.shape) if len(inverse) == 1 else inverse.reshape(mat.shape)
