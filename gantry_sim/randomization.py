"""Per-episode domain randomisation of the physical parameters."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .params import EnvConfig, GantryParams, get_path


def randomize_params(key: jax.Array, nominal: GantryParams, cfg: EnvConfig,
                     scale: jax.Array | float = 1.0) -> GantryParams:
    """Multiply each listed parameter by U(1 - frac, 1 + frac).

    `scale` (traceable) shrinks the ranges; scale = 0 gives the nominal plant
    without recompiling.
    """
    if not cfg.randomization:
        return nominal
    keys = jax.random.split(key, len(cfg.randomization))
    p = nominal
    for k, (path, frac) in zip(keys, cfg.randomization):
        u = jax.random.uniform(k, minval=-1.0, maxval=1.0)
        factor = 1.0 + scale * frac * u
        p = eqx.tree_at(lambda t, path=path: get_path(t, path), p, get_path(p, path) * factor)
    return p


def make_shift(nominal: GantryParams, factors: dict[str, float] | None = None) -> GantryParams:
    """A GantryParams-shaped tree of multipliers (ones except the listed paths).

    Passed to `GantryEnv.reset(shift=...)` as a runtime argument, so evaluating
    many out-of-distribution plants needs only one compilation.
    """
    shift = jax.tree.map(jnp.ones_like, nominal)
    for path, factor in (factors or {}).items():
        get_path(nominal, path)  # validate the path
        shift = eqx.tree_at(lambda t, path=path: get_path(t, path), shift,
                            jnp.asarray(float(factor), dtype=get_path(nominal, path).dtype))
    return shift


def apply_shift(p: GantryParams, shift: GantryParams | None) -> GantryParams:
    return p if shift is None else jax.tree.map(lambda a, b: a * b, p, shift)


def param_deviation(p: GantryParams, nominal: GantryParams, cfg: EnvConfig) -> jax.Array:
    """Relative deviation of each randomised parameter from nominal, in [-1, 1].

    Used as privileged critic input: the critic knows the episode's true plant.
    """
    if not cfg.randomization:
        return jnp.zeros((0,))
    devs = [
        (get_path(p, path) / get_path(nominal, path) - 1.0) / frac if frac > 0 else 0.0 * get_path(p, path)
        for path, frac in cfg.randomization
    ]
    return jnp.stack(devs)
