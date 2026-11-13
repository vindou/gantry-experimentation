"""Tasks and their planned references.

Every episode carries a `Task` with a fixed array layout regardless of mode, so
setpoint and trajectory episodes can share one vmapped batch.

* SETPOINT:   a new random goal every `hold_time`; the reference moves to it
              along a minimum-jerk profile whose duration the planner picks
              (shortest feasible on the nominal rigid model).
* TRAJECTORY: sum of sinusoids whose amplitude the planner scales down until
              the nominal rigid model can follow it within the torque margin.

`reference` gives the planned (p_d, v_d, a_d) used by the CLF reward; `command`
gives the raw user command (goal / trajectory) for actors that should not see
the planned reference.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .params import EnvConfig, GantryParams
from .planner import feasible_amplitude_scale, min_jerk, plan_move_duration, sines

SETPOINT, TRAJECTORY = 0, 1


class Task(eqx.Module):
    mode: jax.Array        # int: SETPOINT or TRAJECTORY
    start: jax.Array       # (2,) setpoint mode: position at t = 0
    setpoints: jax.Array   # (K, 2) goals
    durations: jax.Array   # (K,) planned move durations [s]
    center: jax.Array      # (2,)
    amp: jax.Array         # (2, M) already scaled to be feasible
    freq: jax.Array        # (2, M) [Hz]
    phase: jax.Array       # (2, M)


def sample_task(key: jax.Array, cfg: EnvConfig, nominal: GantryParams,
                mode: int | jax.Array | None = None) -> Task:
    """Draw a random task and plan its reference. `mode=None` samples the mode."""
    tc = cfg.task
    k_mode, k_start, k_sp, k_c, k_a, k_f, k_p = jax.random.split(key, 7)
    M, K = tc.n_harmonics, cfg.n_setpoints

    if mode is None:
        mode = jnp.where(jax.random.uniform(k_mode) < tc.p_trajectory, TRAJECTORY, SETPOINT)
    mode = jnp.asarray(mode, dtype=jnp.int32)

    # --- setpoints + planned move durations
    box = dict(minval=-tc.ref_bound, maxval=tc.ref_bound)
    start = jax.random.uniform(k_start, (2,), **box)
    setpoints = jax.random.uniform(k_sp, (K, 2), **box)
    origins = jnp.concatenate([start[None], setpoints[:-1]], axis=0)
    durations = jax.vmap(plan_move_duration, (0, 0, None, None))(origins, setpoints, nominal, cfg)

    # --- sum of sines; |center| + sum(amp) <= ref_bound keeps it in the box
    amp = jax.random.uniform(k_a, (2, M), minval=0.0, maxval=tc.amp_max / M)
    c_lim = tc.ref_bound - tc.amp_max
    center = jax.random.uniform(k_c, (2,), minval=-c_lim, maxval=c_lim)
    freq = jax.random.uniform(k_f, (2, M), minval=tc.freq_min, maxval=tc.freq_max)
    phase = jax.random.uniform(k_p, (2, M), minval=0.0, maxval=2 * jnp.pi)
    amp = amp * feasible_amplitude_scale(center, amp, freq, phase, nominal, cfg)

    return Task(mode=mode, start=start, setpoints=setpoints, durations=durations,
                center=center, amp=amp, freq=freq, phase=phase)


def _segment(task: Task, t: jax.Array, cfg: EnvConfig):
    K = task.setpoints.shape[0]
    k = jnp.clip(jnp.floor(t / cfg.task.hold_time).astype(jnp.int32), 0, K - 1)
    origin = jnp.where(k == 0, task.start, task.setpoints[jnp.maximum(k - 1, 0)])
    return k, origin, t - k * cfg.task.hold_time


def reference(task: Task, t: jax.Array, cfg: EnvConfig):
    """Planned reference (p_d, v_d, a_d) at time t, each (2,)."""
    k, origin, tau = _segment(task, t, cfg)
    sp = min_jerk(origin, task.setpoints[k], task.durations[k], tau)
    tr = sines(task.center, task.amp, task.freq, task.phase, t)
    is_tr = task.mode == TRAJECTORY
    return tuple(jnp.where(is_tr, a, b) for a, b in zip(tr, sp))


def command(task: Task, t: jax.Array, cfg: EnvConfig):
    """Raw command (p_cmd, v_cmd): the current goal for setpoints (zero velocity),
    the trajectory itself for tracking."""
    k, _, _ = _segment(task, t, cfg)
    p_tr, v_tr, _ = sines(task.center, task.amp, task.freq, task.phase, t)
    is_tr = task.mode == TRAJECTORY
    return jnp.where(is_tr, p_tr, task.setpoints[k]), jnp.where(is_tr, v_tr, jnp.zeros(2))
