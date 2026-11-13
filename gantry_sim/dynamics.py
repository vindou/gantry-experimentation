"""Equations of motion from model.tex.

State   z = [x, y, theta_A, theta_B, xd, yd, theta_Ad, theta_Bd]   (8,)
Input   u = [tau_A, tau_B]                                         (2,)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .params import BeltParams, FrictionParams, GantryParams


# ------------------------------------------------------------------- friction


def friction(v: jax.Array, f: FrictionParams) -> jax.Array:
    """Stribeck + Coulomb (tanh-smoothed) + viscous friction, phi_j(v)."""
    stribeck = f.F_c + (f.F_s - f.F_c) * jnp.exp(-((v / f.v_s) ** 2))
    return stribeck * jnp.tanh(v / f.v_eps) + f.b * v


# ---------------------------------------------------------------------- belts


def belt_stiffness(ell: jax.Array, b: BeltParams) -> tuple[jax.Array, jax.Array]:
    """k(ell) for two spans in parallel, and dk/d(ell)."""
    rest = b.L - ell
    k = b.kappa * (1.0 / ell + 1.0 / rest)
    dk = b.kappa * (-1.0 / ell**2 + 1.0 / rest**2)
    return k, dk


def belt_quantities(z: jax.Array, p: GantryParams) -> dict[str, jax.Array]:
    """Stretch, stretch rate, stiffness, dk/dell and net force for both belts."""
    x, y, thA, thB, xd, yd, thAd, thBd = (z[i] for i in range(8))
    rA, rB = p.belt_A.r, p.belt_B.r

    eA = rA * thA - (x + y)
    eB = rB * thB - (x - y)
    eAd = rA * thAd - (xd + yd)
    eBd = rB * thBd - (xd - yd)

    kA, kpA = belt_stiffness(p.belt_A.ell0 + x + y, p.belt_A)
    kB, kpB = belt_stiffness(p.belt_B.ell0 - x + y, p.belt_B)

    TA = kA * eA + p.belt_A.c * eAd
    TB = kB * eB + p.belt_B.c * eBd
    return dict(eA=eA, eB=eB, eAd=eAd, eBd=eBd, kA=kA, kB=kB, kpA=kpA, kpB=kpB, TA=TA, TB=TB)


# ----------------------------------------------------------- equations of motion


def vector_field(z: jax.Array, u: jax.Array, p: GantryParams) -> jax.Array:
    """dz/dt = f(z, u)."""
    xd, yd, thAd, thBd = z[4], z[5], z[6], z[7]
    bq = belt_quantities(z, p)
    eA, eB, kpA, kpB, TA, TB = bq["eA"], bq["eB"], bq["kpA"], bq["kpB"], bq["TA"], bq["TB"]

    gA = 0.5 * kpA * eA**2  # force from position-dependent stiffness
    gB = 0.5 * kpB * eB**2

    xdd = (TA + TB - gA + gB - friction(xd, p.fric_x)) / p.m_x
    ydd = (TA - TB - gA - gB - friction(yd, p.fric_y)) / p.m_y
    thAdd = (u[0] - p.belt_A.r * TA - friction(thAd, p.fric_A)) / p.I_A
    thBdd = (u[1] - p.belt_B.r * TB - friction(thBd, p.fric_B)) / p.I_B

    return jnp.stack([xd, yd, thAd, thBd, xdd, ydd, thAdd, thBdd])


# -------------------------------------------------------------------- outputs

C_O = jnp.eye(8)[0:2]  # y_o = [x, y]
C_M = jnp.eye(8)[2:4]  # y_m = [theta_A, theta_B]


def controlled_output(z: jax.Array) -> jax.Array:
    return z[0:2]


def measured_output(z: jax.Array) -> jax.Array:
    return z[2:4]


# ------------------------------------------------------------------- helpers


def kinematic_state(xy: jax.Array, vxy: jax.Array, p: GantryParams) -> jax.Array:
    """State at carriage position xy, velocity vxy, with zero belt stretch and stretch rate."""
    x, y = xy[0], xy[1]
    vx, vy = vxy[0], vxy[1]
    return jnp.stack([
        x, y, (x + y) / p.belt_A.r, (x - y) / p.belt_B.r,
        vx, vy, (vx + vy) / p.belt_A.r, (vx - vy) / p.belt_B.r,
    ])


def rest_state(xy: jax.Array, p: GantryParams) -> jax.Array:
    """State at rest at carriage position xy with zero belt stretch."""
    return kinematic_state(xy, jnp.zeros_like(xy), p)


def energy(z: jax.Array, p: GantryParams) -> jax.Array:
    """Kinetic + belt elastic energy (conserved when friction and damping are zero)."""
    xd, yd, thAd, thBd = z[4], z[5], z[6], z[7]
    bq = belt_quantities(z, p)
    T = 0.5 * (p.m_x * xd**2 + p.m_y * yd**2 + p.I_A * thAd**2 + p.I_B * thBd**2)
    V = 0.5 * (bq["kA"] * bq["eA"] ** 2 + bq["kB"] * bq["eB"] ** 2)
    return T + V
