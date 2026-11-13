"""Reduced-order, model-based reference planner (the CLF-RL "reference generator").

The planner uses the RIGID-BELT reduced model (belt stretch = 0), the gantry
analogue of the paper's H-LIP model: cheap, analytic, and accurate enough to
produce dynamically feasible references. With theta_A = (x + y) / r_A and
theta_B = (x - y) / r_B, the inverse dynamics are

    F_x = m_x xdd + phi_x(xd),          F_y = m_y ydd + phi_y(yd)
    T_A = (F_x + F_y) / 2,              T_B = (F_x - F_y) / 2
    tau_i = r_i T_i + I_i thdd_i + phi_i(thd_i)

The planner always uses NOMINAL parameters. It does not know the episode's
randomised plant, exactly like a planner running on the real machine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .dynamics import friction
from .params import EnvConfig, GantryParams


def rigid_inverse_dynamics(p: jax.Array, v: jax.Array, a: jax.Array, P: GantryParams) -> jax.Array:
    """Motor torques [tau_A, tau_B] that realise carriage (p, v, a) with rigid belts."""
    del p  # rigid model is position independent
    rA, rB = P.belt_A.r, P.belt_B.r
    thd = jnp.stack([(v[0] + v[1]) / rA, (v[0] - v[1]) / rB])
    thdd = jnp.stack([(a[0] + a[1]) / rA, (a[0] - a[1]) / rB])
    Fx = P.m_x * a[0] + friction(v[0], P.fric_x)
    Fy = P.m_y * a[1] + friction(v[1], P.fric_y)
    TA, TB = 0.5 * (Fx + Fy), 0.5 * (Fx - Fy)
    return jnp.stack([
        rA * TA + P.I_A * thdd[0] + friction(thd[0], P.fric_A),
        rB * TB + P.I_B * thdd[1] + friction(thd[1], P.fric_B),
    ])


# ------------------------------------------------------------- minimum jerk


def min_jerk(p0: jax.Array, p1: jax.Array, D: jax.Array, tau: jax.Array):
    """Quintic rest-to-rest move from p0 to p1 over duration D, at elapsed time tau."""
    s = jnp.clip(tau / D, 0.0, 1.0)
    inside = (tau >= 0) & (tau <= D)
    sig = 10 * s**3 - 15 * s**4 + 6 * s**5
    dsig = (30 * s**2 - 60 * s**3 + 30 * s**4) / D
    ddsig = (60 * s - 180 * s**2 + 120 * s**3) / D**2
    d = p1 - p0
    return p0 + d * sig, d * dsig * inside, d * ddsig * inside


def _peak_torque_ratio(taus: jax.Array, cfg: EnvConfig) -> jax.Array:
    return jnp.max(jnp.abs(taus) / jnp.asarray(cfg.tau_max))


def plan_move_duration(p0: jax.Array, p1: jax.Array, P: GantryParams, cfg: EnvConfig,
                       n_candidates: int = 32, n_samples: int = 33) -> jax.Array:
    """Shortest minimum-jerk duration whose rigid-model torque stays within
    torque_margin * tau_max (falls back to max_move_time)."""
    tc = cfg.task
    Ds = jnp.linspace(tc.min_move_time, tc.max_move_time, n_candidates)
    s = jnp.linspace(0.0, 1.0, n_samples)

    def peak(D):
        p, v, a = jax.vmap(lambda t: min_jerk(p0, p1, D, t))(s * D)
        return _peak_torque_ratio(jax.vmap(rigid_inverse_dynamics, (0, 0, 0, None))(p, v, a, P), cfg)

    feasible = jax.vmap(peak)(Ds) <= tc.torque_margin
    return jnp.min(jnp.where(feasible, Ds, tc.max_move_time))


def sines(center, amp, freq, phase, t):
    """Sum-of-sines reference: (p, v, a) at time t."""
    w = 2 * jnp.pi * freq
    arg = w * t + phase
    p = center + jnp.sum(amp * jnp.sin(arg), axis=-1)
    v = jnp.sum(amp * w * jnp.cos(arg), axis=-1)
    a = -jnp.sum(amp * w**2 * jnp.sin(arg), axis=-1)
    return p, v, a


def feasible_amplitude_scale(center, amp, freq, phase, P: GantryParams, cfg: EnvConfig,
                             n_scales: int = 20, n_samples: int = 256) -> jax.Array:
    """Largest amplitude scale in (0, 1] whose rigid-model torque is within margin."""
    scales = jnp.linspace(1.0, 0.05, n_scales)
    ts = jnp.linspace(0.0, cfg.episode_time, n_samples)

    def peak(k):
        p, v, a = jax.vmap(lambda t: sines(center, k * amp, freq, phase, t))(ts)
        return _peak_torque_ratio(jax.vmap(rigid_inverse_dynamics, (0, 0, 0, None))(p, v, a, P), cfg)

    feasible = jax.vmap(peak)(scales) <= cfg.task.torque_margin
    return jnp.max(jnp.where(feasible, scales, scales[-1]))
