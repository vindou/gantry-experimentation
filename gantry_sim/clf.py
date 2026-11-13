"""Control Lyapunov function reward shaping (CLF-RL).

K. Li, Z. Olkin, Y. Yue, A. D. Ames, "CLF-RL: Control Lyapunov Function Guided
Reinforcement Learning", IEEE RA-L 2025 (arXiv:2508.09354), Sec. III-B.

Output: carriage position y = [x, y]. Tracking error
    eta = [y_d - y ; yd_d - yd]                                   (paper eq. 7)
    V(eta) = eta^T P eta                                          (paper eq. 8)
with P the CARE solution for the feedback-linearised output dynamics, a double
integrator  eta_dot = A eta + B v,  A = [0 I; 0 0], B = [0; I].

Why the double integrator is the right model here (paper Prop. 1 analogue):
with rigid belts the gantry is fully actuated. The two torques map one-to-one
onto the two carriage forces through J = [[1, 1], [1, -1]], so the output
dynamics M_eff yddot = J^T R^-1 tau - friction can be feedback-linearised, and V is
a valid CLF for that reduced-order model. On the flexible-belt plant the
relative degree from torque to carriage position is 4, so V is used as
guidance, which is how the paper applies it on the full robot.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from .params import ClfConfig


@dataclasses.dataclass(frozen=True)
class ClfDesign:
    P: np.ndarray            # (4, 4) CARE solution
    K: np.ndarray            # (2, 4) LQR gain; with eta = desired - actual, v = a_d + K eta
    mu_max: float            # largest eigenvalue of P
    sigma_v: float           # mu_max eta_max^2
    sigma_vdot: float        # 2 ||P|| eta_max eta_dot_max + lambda mu_max eta_max^2
    decay_rate: float        # lambda used in the reward
    certified_rate: float    # lambda_min(Qbar) / lambda_max(P) from Prop. 1 (for reference)


def design_clf(c: ClfConfig) -> ClfDesign:
    I2, Z2 = np.eye(2), np.zeros((2, 2))
    A = np.block([[Z2, I2], [Z2, Z2]])
    B = np.vstack([Z2, I2])
    Q = np.diag([c.Q_pos, c.Q_pos, c.Q_vel, c.Q_vel])
    R = c.R * I2
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    P = 0.5 * (P + P.T)
    K = np.linalg.solve(R, B.T @ P)

    mu_max = float(np.linalg.eigvalsh(P).max())
    norm_P = float(np.linalg.norm(P, 2))
    Qbar = Q + P @ B @ np.linalg.solve(R, B.T @ P)
    certified = float(np.linalg.eigvalsh(Qbar).min() / mu_max)

    lam = c.decay_rate
    sigma_v = mu_max * c.eta_max**2
    sigma_vdot = 2 * norm_P * c.eta_max * c.eta_dot_max + lam * mu_max * c.eta_max**2
    return ClfDesign(
        P=P,
        K=K,
        mu_max=mu_max,
        sigma_v=c.sigma_v if c.sigma_v > 0 else sigma_v,
        sigma_vdot=c.sigma_vdot if c.sigma_vdot > 0 else sigma_vdot,
        decay_rate=lam,
        certified_rate=certified,
    )


def tracking_error(z: jax.Array, p_ref: jax.Array, v_ref: jax.Array) -> jax.Array:
    """eta = [p_d - p ; v_d - v], shape (4,)."""
    return jnp.concatenate([p_ref - z[0:2], v_ref - z[4:6]])


def lyapunov(eta: jax.Array, P: jax.Array) -> jax.Array:
    return eta @ P @ eta


def clf_rewards(V_t: jax.Array, V_next: jax.Array, dt: float, d: ClfDesign, c: ClfConfig):
    """(r_v, r_vdot, Vdot). Vdot by forward finite difference, as in the paper."""
    Vdot = (V_next - V_t) / dt
    r_v = c.w_v * jnp.exp(-V_next / d.sigma_v)
    r_vdot = -c.w_vdot * jnp.clip((Vdot + d.decay_rate * V_t) / d.sigma_vdot, 0.0, 1.0)
    return r_v, r_vdot, Vdot
