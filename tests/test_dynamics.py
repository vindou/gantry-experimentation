import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gantry_sim import energy, load_config, rest_state, step_zoh, vector_field
from gantry_sim.dynamics import belt_quantities, belt_stiffness, friction
from gantry_sim.params import IntegratorConfig

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def params():
    p, _ = load_config("configs/gantry.yaml")
    return jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), p)


def conservative(p):
    """Zero friction and belt damping -> energy is conserved."""
    zero = lambda f: type(f)(F_c=0 * f.F_c, F_s=0 * f.F_s, v_s=f.v_s, v_eps=f.v_eps, b=0 * f.b)
    belt = lambda b: type(b)(r=b.r, kappa=b.kappa, c=0 * b.c, L=b.L, ell0=b.ell0)
    return type(p)(m_x=p.m_x, m_y=p.m_y, I_A=p.I_A, I_B=p.I_B,
                   belt_A=belt(p.belt_A), belt_B=belt(p.belt_B),
                   fric_x=zero(p.fric_x), fric_y=zero(p.fric_y),
                   fric_A=zero(p.fric_A), fric_B=zero(p.fric_B))


def test_rest_state_is_equilibrium(params):
    z = rest_state(jnp.array([0.03, -0.05]), params)
    np.testing.assert_allclose(vector_field(z, jnp.zeros(2), params), 0.0, atol=1e-9)
    bq = belt_quantities(z, params)
    np.testing.assert_allclose([bq["eA"], bq["eB"]], 0.0, atol=1e-15)


def test_stiffness_derivative_matches_autodiff(params):
    for ell in (0.3, 0.6, 1.4):
        k, dk = belt_stiffness(jnp.asarray(ell), params.belt_A)
        dk_ad = jax.grad(lambda l: belt_stiffness(l, params.belt_A)[0])(jnp.asarray(ell))
        np.testing.assert_allclose(dk, dk_ad, rtol=1e-12)


def test_forces_are_gradient_of_belt_energy(params):
    """Elastic generalised forces must equal -dV/dq (checks the k' e^2 / 2 terms)."""
    p = conservative(params)
    q = jnp.array([0.04, -0.02, 10.0, 15.0])

    def V(q):
        z = jnp.concatenate([q, jnp.zeros(4)])
        bq = belt_quantities(z, p)
        return 0.5 * (bq["kA"] * bq["eA"] ** 2 + bq["kB"] * bq["eB"] ** 2)

    z = jnp.concatenate([q, jnp.zeros(4)])
    acc = vector_field(z, jnp.zeros(2), p)[4:]
    M = jnp.array([p.m_x, p.m_y, p.I_A, p.I_B])
    np.testing.assert_allclose(M * acc, -jax.grad(V)(q), rtol=1e-9)


def test_energy_conserved_without_dissipation(params):
    p = conservative(params)
    z0 = rest_state(jnp.array([0.02, 0.01]), p).at[4].set(0.3).at[6].set(5.0)
    cfg = IntegratorConfig(method="rk4", substeps=200)
    z = z0
    for _ in range(50):
        z = step_zoh(z, jnp.zeros(2), p, 2e-3, cfg)
    np.testing.assert_allclose(energy(z, p), energy(z0, p), rtol=1e-6)


def test_friction_is_odd_and_bounded(params):
    v = jnp.linspace(-1, 1, 101)
    phi = jax.vmap(lambda v: friction(v, params.fric_x))(v)
    np.testing.assert_allclose(phi, -phi[::-1], atol=1e-12)
    assert float(friction(jnp.asarray(0.0), params.fric_x)) == 0.0


def test_torque_kinematics(params):
    """Equal torques push x, opposite torques push y (CoreXY)."""
    cfg = IntegratorConfig(method="rk4", substeps=20)
    z0 = rest_state(jnp.zeros(2), params)
    run = lambda u: jax.lax.fori_loop(0, 50, lambda _, z: step_zoh(z, u, params, 2e-3, cfg), z0)
    z_same = run(jnp.array([0.1, 0.1]))
    z_opp = run(jnp.array([0.1, -0.1]))
    assert z_same[0] > 1e-3 and abs(z_same[1]) < 1e-6
    assert z_opp[1] > 1e-3 and abs(z_opp[0]) < 1e-6


@pytest.mark.parametrize("method", ["kvaerno5", "tsit5"])
def test_rk4_matches_adaptive_solver(params, method):
    """The fast training integrator must agree with the reference solver."""
    rk4 = IntegratorConfig(method="rk4", substeps=20)
    ref = IntegratorConfig(method=method, substeps=20, rtol=1e-9, atol=1e-12, max_steps=100_000)
    key = jax.random.PRNGKey(0)
    us = 0.15 * jax.random.uniform(key, (250, 2), minval=-1, maxval=1)

    def roll(cfg):
        def body(z, u):
            z = step_zoh(z, u, params, 2e-3, cfg)
            return z, z
        return jax.lax.scan(body, rest_state(jnp.array([0.01, -0.02]), params), us)[1]

    a, b = jax.jit(lambda: roll(rk4))(), jax.jit(lambda: roll(ref))()
    assert jnp.all(jnp.isfinite(b))
    # RK4 at h = 0.1 ms is not fully converged near velocity reversals (the
    # tanh(v / v_eps) friction is very steep there); measured drift is ~26 µm over
    # 0.5 s. Acceptable for training; evaluate policies with kvaerno5 to check.
    np.testing.assert_allclose(a[:, :2], b[:, :2], atol=5e-5)
