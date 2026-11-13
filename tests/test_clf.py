import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gantry_sim import SETPOINT, TRAJECTORY, GantryEnv, load_config
from gantry_sim.clf import clf_rewards, design_clf
from gantry_sim.params import IntegratorConfig
from gantry_sim.planner import min_jerk, rigid_inverse_dynamics
from gantry_sim.tasks import reference, sample_task


@pytest.fixture(scope="module")
def setup():
    return load_config("configs/gantry.yaml")


def _double_integrator():
    I2, Z2 = np.eye(2), np.zeros((2, 2))
    return np.block([[Z2, I2], [Z2, Z2]]), np.vstack([Z2, I2])


def test_P_solves_care(setup):
    _, cfg = setup
    c = cfg.reward.clf
    d = design_clf(c)
    A, B = _double_integrator()
    Q = np.diag([c.Q_pos, c.Q_pos, c.Q_vel, c.Q_vel])
    res = A.T @ d.P + d.P @ A - d.P @ B @ B.T @ d.P / c.R + Q
    np.testing.assert_allclose(res, 0.0, atol=1e-6 * np.abs(Q).max())
    assert np.all(np.linalg.eigvalsh(d.P) > 0)


def test_V_is_clf_for_feedback_linearised_output(setup):
    """Prop. 1: with v = K eta (closed loop eta' = (A - B K) eta), V' <= -lambda_cert V."""
    _, cfg = setup
    d = design_clf(cfg.reward.clf)
    A, B = _double_integrator()
    Acl = A - B @ d.K
    rng = np.random.default_rng(0)
    for _ in range(100):
        eta = rng.normal(size=4) * np.array([1e-3, 1e-3, 3e-2, 3e-2])
        V = eta @ d.P @ eta
        Vdot = 2 * eta @ d.P @ Acl @ eta
        assert Vdot <= -d.certified_rate * V + 1e-12


def test_decay_reward_only_penalises_slow_decay(setup):
    _, cfg = setup
    c = cfg.reward.clf
    d = design_clf(c)
    dt, V = 2e-3, 0.5
    fast = V * np.exp(-2 * d.decay_rate * dt)  # decays faster than lambda
    slow = V * 1.01                            # grows
    _, r_fast, _ = clf_rewards(jnp.asarray(V), jnp.asarray(fast), dt, d, c)
    _, r_slow, _ = clf_rewards(jnp.asarray(V), jnp.asarray(slow), dt, d, c)
    assert float(r_fast) == 0.0
    assert -c.w_vdot <= float(r_slow) < 0.0


def test_min_jerk_boundary_conditions():
    p0, p1, D = jnp.array([0.0, 0.1]), jnp.array([0.05, -0.02]), jnp.asarray(0.2)
    for tau, p_exp in ((0.0, p0), (0.2, p1), (0.3, p1)):
        p, v, a = min_jerk(p0, p1, D, jnp.asarray(tau))
        np.testing.assert_allclose(p, p_exp, atol=1e-7)
        np.testing.assert_allclose(v, 0.0, atol=1e-7)


@pytest.mark.parametrize("mode", [SETPOINT, TRAJECTORY])
def test_planned_references_are_feasible_and_continuous(setup, mode):
    params, cfg = setup
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    tasks = jax.vmap(lambda k: sample_task(k, cfg, params, mode))(keys)
    ts = jnp.arange(cfg.episode_steps + 1) * cfg.control_dt
    p, v, a = jax.vmap(lambda task: jax.vmap(lambda t: reference(task, t, cfg))(ts))(tasks)
    tau = jax.vmap(jax.vmap(lambda p, v, a: rigid_inverse_dynamics(p, v, a, params)))(p, v, a)
    assert float((jnp.abs(tau) / jnp.asarray(cfg.tau_max)).max()) <= cfg.task.torque_margin + 1e-3
    assert float(jnp.abs(p).max()) <= cfg.task.ref_bound + 1e-6
    # continuous reference: per-step change bounded by max speed * dt (no setpoint jumps)
    assert float(jnp.abs(jnp.diff(p, axis=1)).max()) <= float(jnp.abs(v).max()) * cfg.control_dt * 1.01


def test_rigid_inverse_dynamics_reproduces_plant(setup):
    """Feeding planned torques open-loop to a stiff, nominal plant follows the reference."""
    params, cfg = setup
    stiff = eqx.tree_at(lambda p: (p.belt_A.kappa, p.belt_B.kappa), params,
                        (params.belt_A.kappa * 100, params.belt_B.kappa * 100))
    cfg = cfg.replace(randomization=(),
                      integrator=IntegratorConfig(method="rk4", substeps=200))
    env = GantryEnv(stiff, cfg, task_mode=TRAJECTORY)
    s, _, _ = env.reset(jax.random.PRNGKey(3))
    errs = []
    for k in range(100):
        p, v, a = reference(s.task, s.step * cfg.control_dt, cfg)
        pn, vn, an = reference(s.task, (s.step + 0.5) * cfg.control_dt, cfg)
        u = rigid_inverse_dynamics(pn, vn, an, stiff)
        s, *_, info = env.step_no_reset(jax.random.PRNGKey(k), s, u / jnp.asarray(cfg.tau_max))
        errs.append(float(info["pos_err"]))
    assert max(errs) < 1e-3  # open-loop within 1 mm over 0.2 s


def test_env_reward_modes(setup):
    params, cfg = setup
    for mode in ("clf", "heuristic"):
        env = GantryEnv(params, cfg.replace(reward=dataclasses.replace(cfg.reward, mode=mode)))
        s, _, _ = env.reset(jax.random.PRNGKey(0))
        *_, r, _, info = env.step_no_reset(jax.random.PRNGKey(1), s, jnp.zeros(2))
        assert np.isfinite(float(r)) and np.isfinite(float(info["V"]))
    # starting on the reference gives near-maximal CLF tracking reward
    env = GantryEnv(params, cfg)
    s, _, _ = env.reset(jax.random.PRNGKey(0))
    assert float(env.V(s.z, s.task, s.step)) / env.clf.sigma_v < 0.5
