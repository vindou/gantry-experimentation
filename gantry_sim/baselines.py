"""Classical reference controller, using the same encoder-only information as the
RL actor. Used to sanity-check the environment and as a benchmark."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .env import FRAME_SIZE, GantryEnv


def make_pd_policy(env: GantryEnv, kp: float = 6000.0, kd: float = 120.0):
    """Cartesian PD on encoder-estimated position -> belt forces -> torques.

    Reads only the newest actor frame (same information as the RL actor).
    """
    oc = env.cfg.observation
    rA, rB = env.nominal.belt_A.r, env.nominal.belt_B.r
    tau_max = jnp.asarray(env.cfg.tau_max)
    start = (oc.history - 1) * FRAME_SIZE

    def policy(actor_obs: jax.Array) -> jax.Array:
        f = actor_obs[start:start + FRAME_SIZE]
        e_p = f[0:2] * oc.pos_scale
        v = f[4:6] * oc.vel_scale
        v_ref = f[6:8] * oc.vel_scale
        F = kp * e_p + kd * (v_ref - v)  # desired Cartesian force [N]
        # F_x = T_A + T_B, F_y = T_A - T_B
        TA, TB = 0.5 * (F[0] + F[1]), 0.5 * (F[0] - F[1])
        return jnp.stack([rA * TA, rB * TB]) / tau_max

    return policy
