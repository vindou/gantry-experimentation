"""Advance the plant over one control period with the torque held constant (ZOH).

* ``rk4``      fixed-step RK4 with `substeps` steps; cheap and uniform across a
               vmapped batch, so it is the default for RL training.
* ``kvaerno5`` adaptive implicit (stiff-capable) Diffrax solver; reference
               accuracy for validation / testing.
* ``tsit5``    adaptive explicit Diffrax solver.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp

from .dynamics import vector_field
from .params import GantryParams, IntegratorConfig


def _rk4(z: jax.Array, u: jax.Array, p: GantryParams, dt: float, n: int) -> jax.Array:
    h = dt / n

    def body(_, z):
        k1 = vector_field(z, u, p)
        k2 = vector_field(z + 0.5 * h * k1, u, p)
        k3 = vector_field(z + 0.5 * h * k2, u, p)
        k4 = vector_field(z + h * k3, u, p)
        return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return jax.lax.fori_loop(0, n, body, z)


_DIFFRAX_SOLVERS = {
    "kvaerno5": diffrax.Kvaerno5,
    "tsit5": diffrax.Tsit5,
}


def _diffrax(z, u, p, dt, cfg: IntegratorConfig):
    term = diffrax.ODETerm(lambda t, z, args: vector_field(z, args[0], args[1]))
    sol = diffrax.diffeqsolve(
        term,
        _DIFFRAX_SOLVERS[cfg.method](),
        t0=0.0,
        t1=dt,
        dt0=dt / cfg.substeps,
        y0=z,
        args=(u, p),
        stepsize_controller=diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=cfg.max_steps,
        throw=False,  # a failed solve surfaces as non-finite state -> episode terminates
    )
    z_next = sol.ys[-1]
    ok = sol.result == diffrax.RESULTS.successful
    return jnp.where(ok, z_next, jnp.nan)


def step_zoh(
    z: jax.Array, u: jax.Array, p: GantryParams, dt: float, cfg: IntegratorConfig
) -> jax.Array:
    """z(t + dt) given z(t) and constant torque u over the interval."""
    if cfg.method == "rk4":
        return _rk4(z, u, p, dt, cfg.substeps)
    if cfg.method in _DIFFRAX_SOLVERS:
        return _diffrax(z, u, p, dt, cfg)
    raise ValueError(f"unknown integrator {cfg.method!r}; use rk4, kvaerno5 or tsit5")
