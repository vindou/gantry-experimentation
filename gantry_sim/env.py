"""Functional, jit/vmap-friendly RL environment for the belt-driven gantry,
with CLF-RL reward shaping.

Asymmetric information:
* actor obs  - built ONLY from (noisy, quantised) encoder readings, the
               reference (planned reference or raw command, see
               `observation.actor_reference`) and the previous action. Carriage
               position is estimated through nominal rigid-belt kinematics, so
               belt stretch and parameter errors are invisible to the actor.
* critic obs - actor obs + true carriage state, CLF tracking error eta and V,
               reference acceleration, belt stretch, episode time and the
               randomised plant parameters (privileged).
* reward     - from the TRUE carriage state: CLF-RL (tracking + decay +
               regularisation) or the heuristic baseline.

API (pure functions, vmap over the batch yourself):
    state, actor_obs, critic_obs = env.reset(key)
    state, actor_obs, critic_obs, reward, done, info = env.step(key, state, action)
`step` auto-resets finished episodes; `step_no_reset` does not (for evaluation).
Actions are in [-1, 1]^2 and scale to [-tau_max, tau_max].
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .clf import clf_rewards, design_clf, lyapunov, tracking_error
from .dynamics import belt_quantities, kinematic_state
from .integrator import step_zoh
from .params import EnvConfig, GantryParams
from .randomization import apply_shift, param_deviation, randomize_params
from .tasks import Task, command, reference, sample_task

FRAME_SIZE = 10      # per-step actor features, see `_frame`
N_PRIVILEGED = 16    # critic-only features besides parameter deviations


class EnvState(eqx.Module):
    z: jax.Array            # (8,) true plant state
    step: jax.Array         # int control-step counter
    params: GantryParams    # this episode's (randomised) plant
    task: Task
    prev_action: jax.Array  # (2,) in [-1, 1]
    enc: jax.Array          # (2,) last encoder measurement [rad]
    frames: jax.Array       # (history, FRAME_SIZE) actor history, newest last
    ep_return: jax.Array
    ep_sq_err: jax.Array    # running sum of squared position error [m^2]


class GantryEnv:
    def __init__(self, nominal: GantryParams, cfg: EnvConfig, task_mode: int | None = None):
        """`task_mode` forces SETPOINT/TRAJECTORY for every episode (evaluation)."""
        self.nominal = nominal
        self.cfg = cfg
        self.task_mode = task_mode
        self.action_size = 2
        oc = cfg.observation
        self.actor_obs_size = oc.history * FRAME_SIZE + 2 * len(oc.preview_steps)
        self.critic_obs_size = self.actor_obs_size + N_PRIVILEGED + len(cfg.randomization)
        self._tau_max = jnp.asarray(cfg.tau_max)
        self.clf = design_clf(cfg.reward.clf)
        self._P = jnp.asarray(self.clf.P)

    # ------------------------------------------------------------ references

    def _time(self, step: jax.Array) -> jax.Array:
        return step * self.cfg.control_dt

    def _ref(self, task: Task, step: jax.Array):
        return reference(task, self._time(step), self.cfg)

    def _actor_ref(self, task: Task, step: jax.Array):
        """What the actor is told to follow."""
        if self.cfg.observation.actor_reference == "command":
            return command(task, self._time(step), self.cfg)
        return self._ref(task, step)[:2]

    def V(self, z: jax.Array, task: Task, step: jax.Array) -> jax.Array:
        p_d, v_d, _ = self._ref(task, step)
        return lyapunov(tracking_error(z, p_d, v_d), self._P)

    # ------------------------------------------------------------ sensing

    def _measure(self, key: jax.Array, z: jax.Array) -> jax.Array:
        oc = self.cfg.observation
        q = 2 * jnp.pi / oc.encoder_cpr
        noisy = z[2:4] + oc.encoder_noise_std * jax.random.normal(key, (2,))
        return jnp.round(noisy / q) * q

    def _enc_to_xy(self, theta: jax.Array) -> jax.Array:
        """Rigid-belt forward kinematics with NOMINAL radii (all the controller knows)."""
        sA = self.nominal.belt_A.r * theta[0]  # = x + y
        sB = self.nominal.belt_B.r * theta[1]  # = x - y
        return jnp.stack([0.5 * (sA + sB), 0.5 * (sA - sB)])

    def _frame(self, enc, enc_vel, step, task, prev_action) -> jax.Array:
        oc = self.cfg.observation
        p_enc = self._enc_to_xy(enc)
        v_enc = self._enc_to_xy(enc_vel)
        p_ref, v_ref = self._actor_ref(task, step)
        return jnp.concatenate([
            (p_ref - p_enc) / oc.pos_scale,
            p_enc / oc.pos_scale,
            v_enc / oc.vel_scale,
            v_ref / oc.vel_scale,
            prev_action,
        ])

    def _actor_obs(self, s: EnvState) -> jax.Array:
        oc = self.cfg.observation
        p_enc = self._enc_to_xy(s.enc)
        previews = [(self._actor_ref(s.task, s.step + k)[0] - p_enc) / oc.pos_scale
                    for k in oc.preview_steps]
        obs = jnp.concatenate([s.frames.reshape(-1), *previews])
        return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _critic_obs(self, s: EnvState, actor_obs: jax.Array) -> jax.Array:
        oc = self.cfg.observation
        z, p = s.z, s.params
        p_d, v_d, a_d = self._ref(s.task, s.step)
        eta = tracking_error(z, p_d, v_d)
        V = lyapunov(eta, self._P)
        bq = belt_quantities(z, p)
        priv = jnp.concatenate([
            eta[0:2] / oc.pos_scale,                                   # 2
            eta[2:4] / oc.vel_scale,                                   # 2
            (V / self.clf.sigma_v)[None],                              # 1
            a_d / oc.acc_scale,                                        # 2
            z[0:2] / oc.pos_scale,                                     # 2
            z[4:6] / oc.vel_scale,                                     # 2
            jnp.stack([p.belt_A.r * z[6], p.belt_B.r * z[7]]) / oc.vel_scale,  # 2
            jnp.stack([bq["eA"], bq["eB"]]) / oc.stretch_scale,        # 2
            (s.step / self.cfg.episode_steps)[None],                   # 1 (time limit -> Markov)
            param_deviation(p, self.nominal, self.cfg),
        ])
        obs = jnp.concatenate([actor_obs, jnp.clip(priv, -100.0, 100.0)])
        return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def observe(self, s: EnvState) -> tuple[jax.Array, jax.Array]:
        a = self._actor_obs(s)
        return a, self._critic_obs(s, a)

    # ------------------------------------------------------------ rewards

    def _reward(self, s: EnvState, z1, step1, a):
        """Returns (reward, V_t, V_{t+1}, Vdot) from the true state."""
        cfg = self.cfg
        oc = cfg.observation
        a_rate = jnp.sum((a - s.prev_action) ** 2)
        V_t = self.V(s.z, s.task, s.step)
        p_d, v_d, _ = self._ref(s.task, step1)
        eta1 = tracking_error(z1, p_d, v_d)
        V_1 = lyapunov(eta1, self._P)

        if cfg.reward.mode == "clf":
            c = cfg.reward.clf
            r_v, r_vdot, Vdot = clf_rewards(V_t, V_1, cfg.control_dt, self.clf, c)
            r = r_v + r_vdot - c.w_torque * jnp.sum(a**2) - c.w_action_rate * a_rate
        else:
            h = cfg.reward.heuristic
            err = jnp.linalg.norm(eta1[0:2])
            vel_err = jnp.linalg.norm(eta1[2:4])
            r = (h.w_pos * jnp.exp(-err / h.sigma_pos)
                 + h.w_precision * jnp.exp(-((err / h.sigma_precision) ** 2))
                 - h.w_vel * (vel_err / oc.vel_scale) ** 2
                 - h.w_torque * jnp.sum(a**2) - h.w_action_rate * a_rate)
            Vdot = (V_1 - V_t) / cfg.control_dt
        return r, V_t, V_1, Vdot

    # ------------------------------------------------------------ reset / step

    def reset(self, key: jax.Array, task_mode=None, shift: GantryParams | None = None,
              rand_scale: jax.Array | float = 1.0) -> tuple[EnvState, jax.Array, jax.Array]:
        """Optional runtime (traceable) arguments, used by evaluation:
        task_mode  SETPOINT / TRAJECTORY (default: the constructor's, else random)
        shift      multiplier tree from `randomization.make_shift`: changes the TRUE
                   plant only; planner and actor kinematics stay nominal
        rand_scale 0 disables domain randomisation, 1 = training ranges
        """
        k_p, k_t, k_x, k_m = jax.random.split(key, 4)
        params = randomize_params(k_p, self.nominal, self.cfg, rand_scale)
        params = apply_shift(params, shift)
        mode = self.task_mode if task_mode is None else task_mode
        task = sample_task(k_t, self.cfg, self.nominal, mode)
        step = jnp.asarray(0, dtype=jnp.int32)
        # start on the planned reference (eta ~ 0), plus a small position offset
        p_d, v_d, _ = self._ref(task, step)
        p0 = p_d + self.cfg.task.init_noise * jax.random.normal(k_x, (2,))
        z = kinematic_state(p0, v_d, params)
        enc = self._measure(k_m, z)
        prev_action = jnp.zeros(2)
        enc_vel0 = jnp.stack([(v_d[0] + v_d[1]) / self.nominal.belt_A.r,
                              (v_d[0] - v_d[1]) / self.nominal.belt_B.r])
        frame = self._frame(enc, enc_vel0, step, task, prev_action)
        s = EnvState(
            z=z, step=step, params=params, task=task, prev_action=prev_action, enc=enc,
            frames=jnp.tile(frame, (self.cfg.observation.history, 1)),
            ep_return=jnp.asarray(0.0), ep_sq_err=jnp.asarray(0.0),
        )
        return (s, *self.observe(s))

    def step_no_reset(self, key: jax.Array, s: EnvState, action: jax.Array):
        cfg = self.cfg
        a = jnp.clip(action, -1.0, 1.0)
        u = a * self._tau_max

        z1 = step_zoh(s.z, u, s.params, cfg.control_dt, cfg.integrator)
        step1 = s.step + 1
        finite = jnp.all(jnp.isfinite(z1))

        reward, V_t, V_1, Vdot = self._reward(s, z1, step1, a)
        out_of_bounds = jnp.any(jnp.abs(z1[0:2]) > cfg.workspace)
        terminated = out_of_bounds | ~finite
        w_term = cfg.reward.w_terminate
        reward = jnp.where(terminated, -w_term, reward)
        reward = jnp.where(jnp.isfinite(reward), reward, -w_term)
        truncated = step1 >= cfg.episode_steps
        done = terminated | truncated

        # --- sensing
        enc1 = self._measure(key, z1)
        frame = self._frame(enc1, (enc1 - s.enc) / cfg.control_dt, step1, s.task, a)
        frames = jnp.concatenate([s.frames[1:], frame[None]], axis=0)

        p_d = self._ref(s.task, step1)[0]
        pos_err = jnp.linalg.norm(p_d - z1[0:2])
        sq_err = jnp.where(finite, pos_err**2, 0.0)
        s1 = EnvState(
            z=z1, step=step1, params=s.params, task=s.task, prev_action=a, enc=enc1,
            frames=frames, ep_return=s.ep_return + reward, ep_sq_err=s.ep_sq_err + sq_err,
        )
        lam, sig = self.clf.decay_rate, self.clf.sigma_v
        bq = belt_quantities(z1, s.params)
        info = dict(
            p=z1[0:2], p_ref=p_d, p_cmd=command(s.task, self._time(step1), cfg)[0],
            pos_err=pos_err, u=u, TA=bq["TA"], TB=bq["TB"],
            V=V_1, V_norm=V_1 / sig, Vdot=Vdot,
            # decay condition Vdot <= -lambda V violated (ignored when V is negligible)
            clf_violation=(Vdot + lam * V_t > 0) & (V_t > 1e-3 * sig),
            terminated=terminated, truncated=truncated, mode=s.task.mode,
            episode_return=s1.ep_return,
            episode_rms_err=jnp.sqrt(s1.ep_sq_err / step1),
        )
        actor_obs, critic_obs = self.observe(s1)
        return s1, actor_obs, critic_obs, reward, done, info

    def step(self, key: jax.Array, s: EnvState, action: jax.Array, reset=None):
        """Step with auto-reset. `reset` = (state, actor_obs, critic_obs) from a
        pre-computed `reset`; pass it to avoid re-planning a task every step
        (under vmap the reset branch is always evaluated)."""
        k_step, k_reset = jax.random.split(key)
        s1, ao, co, reward, done, info = self.step_no_reset(k_step, s, action)
        s0, ao0, co0 = self.reset(k_reset) if reset is None else reset
        pick = lambda new, old: jax.tree.map(lambda a, b: jnp.where(done, a, b), new, old)
        return pick(s0, s1), pick(ao0, ao), pick(co0, co), reward, done, info

