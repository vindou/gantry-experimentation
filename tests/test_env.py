import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gantry_sim import SETPOINT, TRAJECTORY, GantryEnv, load_config, rest_state
from gantry_sim.tasks import reference
from rl.ppo import load_ppo_config, make_trainer


@pytest.fixture(scope="module")
def env():
    return GantryEnv(*load_config("configs/gantry.yaml"))


def test_shapes_and_vmap(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    s, ao, co = jax.vmap(env.reset)(keys)
    assert ao.shape == (8, env.actor_obs_size) and co.shape == (8, env.critic_obs_size)
    s, ao, co, r, d, info = jax.jit(jax.vmap(env.step))(keys, s, jnp.zeros((8, 2)))
    assert ao.shape == (8, env.actor_obs_size) and r.shape == (8,) and d.shape == (8,)
    assert jnp.all(jnp.isfinite(ao)) and jnp.all(jnp.isfinite(co)) and jnp.all(jnp.isfinite(r))


def test_actor_obs_ignores_privileged_state(env):
    """Moving the carriage without touching the encoders must not change the actor
    obs, but must change the critic obs and the reward-relevant state."""
    s, ao, co = env.reset(jax.random.PRNGKey(1))
    s2 = eqx.tree_at(lambda s: s.z, s, s.z.at[0].add(0.01).at[4].add(0.2))
    s2 = eqx.tree_at(lambda s: s.params.m_x, s2, s.params.m_x * 1.1)
    ao2, co2 = env.observe(s2)
    np.testing.assert_array_equal(ao, ao2)
    assert not np.allclose(co, co2)


def test_auto_reset_on_truncation(env):
    s, _, _ = env.reset(jax.random.PRNGKey(2))
    s = eqx.tree_at(lambda s: s.step, s, jnp.asarray(env.cfg.episode_steps - 1, jnp.int32))
    s1, _, _, _, done, info = env.step(jax.random.PRNGKey(3), s, jnp.zeros(2))
    assert bool(done) and bool(info["truncated"]) and int(s1.step) == 0


def test_termination_outside_workspace(env):
    s, _, _ = env.reset(jax.random.PRNGKey(4))
    z_out = rest_state(jnp.array([env.cfg.workspace + 0.01, 0.0]), s.params)
    s = eqx.tree_at(lambda s: s.z, s, z_out)
    *_, r, done, info = env.step_no_reset(jax.random.PRNGKey(5), s, jnp.zeros(2))
    assert bool(info["terminated"]) and float(r) == -env.cfg.reward.w_terminate


@pytest.mark.parametrize("mode", [SETPOINT, TRAJECTORY])
def test_references_stay_in_bounds(env, mode):
    e = GantryEnv(env.nominal, env.cfg, task_mode=mode)
    keys = jax.random.split(jax.random.PRNGKey(6), 64)
    ts = jnp.linspace(0, e.cfg.episode_time, 200)
    tasks = jax.vmap(lambda k: e.reset(k)[0].task)(keys)
    p = jax.vmap(lambda task: jax.vmap(lambda t: reference(task, t, e.cfg)[0])(ts))(tasks)
    assert float(jnp.abs(p).max()) <= e.cfg.task.ref_bound + 1e-6


def test_ppo_update_runs(env):
    cfg = load_ppo_config("configs/ppo_cpu_demo.yaml").replace(
        num_envs=8, num_steps=16, num_minibatches=2, update_epochs=1, total_timesteps=256)
    init_fn, update_fn = make_trainer(env, cfg)
    ts, runner = init_fn(jax.random.PRNGKey(0))
    for _ in range(2):
        ts, runner, m = update_fn(ts, runner)
    assert all(np.isfinite(float(v)) for v in m.values())
