"""Parameter containers and YAML loading.

`GantryParams` holds every *physical* parameter as a JAX array inside an
equinox Module, so it is a pytree: it can be jitted, vmapped (one parameter set
per parallel environment for domain randomisation) and differentiated (system
identification).

`EnvConfig` and friends hold *structural* settings (time step, integrator,
observation layout, ...). They are frozen dataclasses of plain Python values,
hashable, and treated as compile-time constants.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import yaml


# --------------------------------------------------------------------------- physical


def _arr(v: Any) -> jax.Array:
    return jnp.asarray(float(v))


class FrictionParams(eqx.Module):
    """phi(v) = [F_c + (F_s - F_c) exp(-(v/v_s)^2)] tanh(v/v_eps) + b v."""

    F_c: jax.Array
    F_s: jax.Array
    v_s: jax.Array
    v_eps: jax.Array
    b: jax.Array


class BeltParams(eqx.Module):
    r: jax.Array      # signed effective pulley radius [m]
    kappa: jax.Array  # tensile stiffness E*A [N]
    c: jax.Array      # belt damping [N s/m]
    L: jax.Array      # total belt length [m]
    ell0: jax.Array   # short-span length at x = y = 0 [m]


class GantryParams(eqx.Module):
    m_x: jax.Array
    m_y: jax.Array
    I_A: jax.Array
    I_B: jax.Array
    belt_A: BeltParams
    belt_B: BeltParams
    fric_x: FrictionParams
    fric_y: FrictionParams
    fric_A: FrictionParams
    fric_B: FrictionParams


def _friction(d: dict) -> FrictionParams:
    return FrictionParams(**{k: _arr(d[k]) for k in ("F_c", "F_s", "v_s", "v_eps", "b")})


def _belt(d: dict) -> BeltParams:
    return BeltParams(**{k: _arr(d[k]) for k in ("r", "kappa", "c", "L", "ell0")})


def gantry_params_from_dict(model: dict) -> GantryParams:
    inertia, belts, fric = model["inertia"], model["belts"], model["friction"]
    p = GantryParams(
        m_x=_arr(inertia["m_x"]),
        m_y=_arr(inertia["m_y"]),
        I_A=_arr(inertia["I_A"]),
        I_B=_arr(inertia["I_B"]),
        belt_A=_belt(belts["A"]),
        belt_B=_belt(belts["B"]),
        fric_x=_friction(fric["x"]),
        fric_y=_friction(fric["y"]),
        fric_A=_friction(fric["A"]),
        fric_B=_friction(fric["B"]),
    )
    _validate_params(p)
    return p


def _validate_params(p: GantryParams) -> None:
    for name in ("m_x", "m_y", "I_A", "I_B"):
        if float(getattr(p, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("belt_A", "belt_B"):
        b = getattr(p, name)
        if float(b.r) == 0:
            raise ValueError(f"{name}.r must be nonzero")
        if not 0 < float(b.ell0) < float(b.L):
            raise ValueError(f"{name}: need 0 < ell0 < L")
        if float(b.kappa) <= 0 or float(b.c) < 0:
            raise ValueError(f"{name}: need kappa > 0 and c >= 0")
    for name in ("fric_x", "fric_y", "fric_A", "fric_B"):
        f = getattr(p, name)
        if float(f.v_s) <= 0 or float(f.v_eps) <= 0:
            raise ValueError(f"{name}: v_s and v_eps must be positive")


def get_path(tree: Any, path: str) -> Any:
    """`get_path(params, "belt_A.kappa")` -> params.belt_A.kappa."""
    for attr in path.split("."):
        tree = getattr(tree, attr)
    return tree


# --------------------------------------------------------------------------- structural


@dataclasses.dataclass(frozen=True)
class IntegratorConfig:
    method: str = "rk4"
    substeps: int = 20
    rtol: float = 1e-6
    atol: float = 1e-9
    max_steps: int = 4096


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    p_trajectory: float = 0.5
    ref_bound: float = 0.12
    hold_time: float = 0.5
    n_harmonics: int = 2
    freq_min: float = 0.2
    freq_max: float = 1.5
    amp_max: float = 0.08
    min_move_time: float = 0.05
    max_move_time: float = 0.4
    init_noise: float = 0.0005
    torque_margin: float = 0.6  # planner may use this fraction of tau_max


@dataclasses.dataclass(frozen=True)
class ObservationConfig:
    history: int = 4
    preview_steps: tuple[int, ...] = (5, 20)
    encoder_cpr: int = 16384
    encoder_noise_std: float = 2e-4
    pos_scale: float = 0.1
    vel_scale: float = 0.5
    stretch_scale: float = 5e-4
    acc_scale: float = 10.0
    actor_reference: str = "planned"  # planned | command


@dataclasses.dataclass(frozen=True)
class ClfConfig:
    """CLF-RL reward (Li, Olkin, Yue, Ames 2025). See gantry_sim/clf.py."""

    Q_pos: float = 8.1e5
    Q_vel: float = 1.8e3
    R: float = 1.0
    decay_rate: float = 10.0  # lambda [1/s]
    eta_max: float = 5e-3
    eta_dot_max: float = 0.5
    w_v: float = 10.0
    w_vdot: float = 2.0
    w_torque: float = 0.1
    w_action_rate: float = 0.5
    w_terminate: float = 100.0
    sigma_v: float = 0.0     # > 0 overrides mu_max(P) eta_max^2
    sigma_vdot: float = 0.0  # > 0 overrides 2||P|| eta_max eta_dot_max + lambda mu_max eta_max^2


@dataclasses.dataclass(frozen=True)
class HeuristicRewardConfig:
    """Hand-shaped tracking reward (the pre-CLF baseline)."""

    w_pos: float = 1.0
    sigma_pos: float = 0.05
    w_precision: float = 1.0
    sigma_precision: float = 0.003
    w_vel: float = 0.05
    w_torque: float = 0.01
    w_action_rate: float = 0.05
    w_terminate: float = 50.0


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    mode: str = "clf"  # clf | heuristic
    clf: ClfConfig = ClfConfig()
    heuristic: HeuristicRewardConfig = HeuristicRewardConfig()

    @property
    def w_terminate(self) -> float:
        return self.clf.w_terminate if self.mode == "clf" else self.heuristic.w_terminate


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    control_dt: float = 2e-3
    episode_time: float = 1.5
    tau_max: tuple[float, float] = (0.2, 0.2)
    workspace: float = 0.15
    integrator: IntegratorConfig = IntegratorConfig()
    task: TaskConfig = TaskConfig()
    observation: ObservationConfig = ObservationConfig()
    reward: RewardConfig = RewardConfig()
    randomization: tuple[tuple[str, float], ...] = ()  # (attribute path, fraction)

    @property
    def episode_steps(self) -> int:
        return int(round(self.episode_time / self.control_dt))

    @property
    def n_setpoints(self) -> int:
        return max(1, math.ceil(self.episode_time / self.task.hold_time))

    def replace(self, **kw) -> "EnvConfig":
        return dataclasses.replace(self, **kw)


def env_config_from_dict(d: dict) -> EnvConfig:
    env, task, obs, rew = d["env"], d["task"], d["observation"], d["reward"]
    integ = env["integrator"]
    rnd = d.get("randomization", {}) or {}
    randomization: tuple[tuple[str, float], ...] = ()
    if rnd.get("enabled", False):
        randomization = tuple((str(k), float(v)) for k, v in (rnd.get("scale") or {}).items())
        for path, frac in randomization:
            if not 0 <= frac < 1:
                raise ValueError(f"randomization fraction for {path} must be in [0, 1)")

    cfg = EnvConfig(
        control_dt=float(env["control_dt"]),
        episode_time=float(env["episode_time"]),
        tau_max=(float(env["tau_max"][0]), float(env["tau_max"][1])),
        workspace=float(env["workspace"]),
        integrator=IntegratorConfig(
            method=str(integ["method"]),
            substeps=int(integ["substeps"]),
            rtol=float(integ["rtol"]),
            atol=float(integ["atol"]),
            max_steps=int(integ["max_steps"]),
        ),
        task=TaskConfig(
            p_trajectory=float(task["p_trajectory"]),
            ref_bound=float(task["ref_bound"]),
            hold_time=float(task["setpoint"]["hold_time"]),
            n_harmonics=int(task["trajectory"]["n_harmonics"]),
            freq_min=float(task["trajectory"]["freq"][0]),
            freq_max=float(task["trajectory"]["freq"][1]),
            amp_max=float(task["trajectory"]["amp_max"]),
            min_move_time=float(task["setpoint"]["min_move_time"]),
            max_move_time=float(task["setpoint"]["max_move_time"]),
            init_noise=float(task["init_noise"]),
            torque_margin=float(d["planner"]["torque_margin"]),
        ),
        observation=ObservationConfig(
            history=int(obs["history"]),
            preview_steps=tuple(int(s) for s in obs["preview_steps"]),
            encoder_cpr=int(obs["encoder_cpr"]),
            encoder_noise_std=float(obs["encoder_noise_std"]),
            pos_scale=float(obs["pos_scale"]),
            vel_scale=float(obs["vel_scale"]),
            stretch_scale=float(obs["stretch_scale"]),
            acc_scale=float(obs["acc_scale"]),
            actor_reference=str(obs["actor_reference"]),
        ),
        reward=RewardConfig(
            mode=str(rew["mode"]),
            clf=ClfConfig(**{k: 0.0 if v is None else float(v) for k, v in rew["clf"].items()}),
            heuristic=HeuristicRewardConfig(**{k: float(v) for k, v in rew["heuristic"].items()}),
        ),
        randomization=randomization,
    )
    if cfg.task.ref_bound + 1e-9 >= cfg.workspace:
        raise ValueError("task.ref_bound must be smaller than env.workspace")
    if cfg.task.amp_max >= cfg.task.ref_bound:
        raise ValueError("trajectory.amp_max must be smaller than task.ref_bound")
    if not 0 < cfg.task.min_move_time < cfg.task.max_move_time <= cfg.task.hold_time:
        raise ValueError("need 0 < min_move_time < max_move_time <= hold_time")
    if cfg.reward.mode not in ("clf", "heuristic"):
        raise ValueError("reward.mode must be 'clf' or 'heuristic'")
    if cfg.observation.actor_reference not in ("planned", "command"):
        raise ValueError("observation.actor_reference must be 'planned' or 'command'")
    c = cfg.reward.clf
    if min(c.Q_pos, c.Q_vel, c.R, c.eta_max, c.eta_dot_max) <= 0 or c.decay_rate < 0:
        raise ValueError("CLF: Q_pos, Q_vel, R, eta_max, eta_dot_max must be > 0, decay_rate >= 0")
    return cfg


def apply_overrides(d: dict, sets: dict) -> dict:
    """In-place dotted-path overrides on a config dict:
    `{"reward.clf.w_vdot": 0.0}` -> d["reward"]["clf"]["w_vdot"] = 0.0."""
    for path, value in sets.items():
        *parents, leaf = path.split(".")
        node = d
        for k in parents:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"unknown config key {path!r}")
        node[leaf] = value
    return d


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> tuple[GantryParams, EnvConfig]:
    d = load_yaml(path)
    params = gantry_params_from_dict(d["model"])
    cfg = env_config_from_dict(d)
    for p, _ in cfg.randomization:
        get_path(params, p)  # raises AttributeError on typos
    return params, cfg
