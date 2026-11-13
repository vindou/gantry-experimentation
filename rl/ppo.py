"""PPO with an asymmetric actor-critic, written for jit/vmap end to end.

The actor sees `actor_obs` (encoders only); the critic sees `critic_obs`
(privileged). Only the actor is needed at deployment.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import yaml

from gantry_sim.env import GantryEnv
from gantry_sim.tasks import SETPOINT, TRAJECTORY

from .networks import Actor, Critic, RunningNorm, gaussian_entropy, gaussian_logp


# ---------------------------------------------------------------------- config


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    total_timesteps: int = 20_000_000
    num_envs: int = 512
    num_steps: int = 64
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    anneal_lr: bool = True
    reward_scale: float = 0.1
    init_log_std: float = -0.5
    actor_hidden: tuple[int, ...] = (512, 256, 128)  # paper: [512, 256, 128], ELU
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    log_every: int = 10
    checkpoint_every: int = 100

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_updates(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)

    def replace(self, **kw) -> "PPOConfig":
        return dataclasses.replace(self, **kw)


def ppo_config_from_dict(d: dict) -> PPOConfig:
    d = dict(d)
    actor, critic = d.pop("actor", {}) or {}, d.pop("critic", {}) or {}
    defaults = PPOConfig()
    kw: dict[str, Any] = {}
    for k, v in d.items():
        if not hasattr(defaults, k) or k in ("actor_hidden", "critic_hidden"):
            raise KeyError(f"unknown PPO option {k!r}")
        kw[k] = type(getattr(defaults, k))(v)
    if "hidden" in actor:
        kw["actor_hidden"] = tuple(int(h) for h in actor["hidden"])
    if "hidden" in critic:
        kw["critic_hidden"] = tuple(int(h) for h in critic["hidden"])
    cfg = PPOConfig(**kw)
    if cfg.batch_size % cfg.num_minibatches:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    return cfg


def load_ppo_config(path: str | Path) -> PPOConfig:
    with open(path) as f:
        return ppo_config_from_dict(yaml.safe_load(f))


# ----------------------------------------------------------------------- state


class Models(eqx.Module):
    actor: Actor
    critic: Critic
    norm_actor: RunningNorm
    norm_critic: RunningNorm


class TrainState(eqx.Module):
    models: Models
    opt_state: Any


class Runner(NamedTuple):
    env_state: Any
    actor_obs: jax.Array
    critic_obs: jax.Array
    key: jax.Array


class Transition(NamedTuple):
    actor_obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


def init_models(env: GantryEnv, cfg: PPOConfig, key: jax.Array) -> Models:
    k_a, k_c = jax.random.split(key)
    return Models(
        actor=Actor(env.actor_obs_size, env.action_size, cfg.actor_hidden, cfg.activation,
                    cfg.init_log_std, key=k_a),
        critic=Critic(env.critic_obs_size, cfg.critic_hidden, cfg.activation, key=k_c),
        norm_actor=RunningNorm.init(env.actor_obs_size),
        norm_critic=RunningNorm.init(env.critic_obs_size),
    )


def _trainable(models: Models):
    """Only the networks are optimised; the normalisers are updated by statistics."""
    return eqx.filter((models.actor, models.critic), eqx.is_inexact_array)


def make_optimizer(cfg: PPOConfig) -> optax.GradientTransformation:
    n = cfg.num_updates * cfg.update_epochs * cfg.num_minibatches
    lr = optax.linear_schedule(cfg.lr, 0.0, n) if cfg.anneal_lr else cfg.lr
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))


# ------------------------------------------------------------------- training


def make_trainer(env: GantryEnv, cfg: PPOConfig):
    """Returns (init_fn, update_fn). update_fn is jitted: one rollout + PPO update."""
    optimizer = make_optimizer(cfg)
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    def init_fn(key: jax.Array) -> tuple[TrainState, Runner]:
        k_m, k_env, k_run = jax.random.split(key, 3)
        models = init_models(env, cfg, k_m)
        opt_state = optimizer.init(_trainable(models))
        env_state, ao, co = v_reset(jax.random.split(k_env, cfg.num_envs))
        # spread episode phases so resets don't all happen in lock-step
        env_state = eqx.tree_at(
            lambda s: s.step, env_state,
            jax.random.randint(k_env, (cfg.num_envs,), 0, env.cfg.episode_steps),
        )
        return TrainState(models, opt_state), Runner(env_state, ao, co, k_run)

    def rollout(models: Models, runner: Runner):
        # One pool of fresh episodes per rollout: planning a task is far more
        # expensive than a physics step, so don't do it inside every step.
        k_pool, k_run = jax.random.split(runner.key)
        pool = v_reset(jax.random.split(k_pool, cfg.num_envs))
        runner = runner._replace(key=k_run)

        def body(r: Runner, _):
            key, k_a, k_e = jax.random.split(r.key, 3)
            mean, log_std = jax.vmap(models.actor)(models.norm_actor(r.actor_obs))
            action = mean + jnp.exp(log_std) * jax.random.normal(k_a, mean.shape)
            logp = gaussian_logp(action, mean, log_std)
            value = jax.vmap(models.critic)(models.norm_critic(r.critic_obs))
            env_state, ao, co, reward, done, info = v_step(
                jax.random.split(k_e, cfg.num_envs), r.env_state, action, pool)
            log_info = {k: info[k] for k in
                        ("episode_return", "episode_rms_err", "terminated", "mode", "pos_err",
                         "V_norm", "clf_violation")}
            tr = Transition(r.actor_obs, r.critic_obs, action, logp, value,
                            reward * cfg.reward_scale, done.astype(jnp.float32), log_info)
            return Runner(env_state, ao, co, key), tr

        return jax.lax.scan(body, runner, None, length=cfg.num_steps)

    def gae(traj: Transition, last_value: jax.Array):
        def body(carry, tr: Transition):
            adv, next_value = carry
            nonterminal = 1.0 - tr.done
            delta = tr.reward + cfg.gamma * next_value * nonterminal - tr.value
            adv = delta + cfg.gamma * cfg.gae_lambda * nonterminal * adv
            return (adv, tr.value), adv

        _, advantages = jax.lax.scan(
            body, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        return advantages, advantages + traj.value

    def loss_fn(nets, norms, batch):
        actor, critic = nets
        norm_a, norm_c = norms
        tr, adv, ret = batch
        mean, log_std = jax.vmap(actor)(norm_a(tr.actor_obs))
        logp = gaussian_logp(tr.action, mean, log_std)
        ratio = jnp.exp(logp - tr.logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss = -jnp.mean(jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv))
        value = jax.vmap(critic)(norm_c(tr.critic_obs))
        v_loss = 0.5 * jnp.mean((value - ret) ** 2)
        entropy = jnp.mean(gaussian_entropy(log_std))
        total = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
        approx_kl = jnp.mean((ratio - 1) - jnp.log(ratio))
        return total, dict(pg_loss=pg_loss, v_loss=v_loss, entropy=entropy, approx_kl=approx_kl)

    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    @eqx.filter_jit
    def update_fn(ts: TrainState, runner: Runner):
        models = ts.models
        runner, traj = rollout(models, runner)
        last_value = jax.vmap(models.critic)(models.norm_critic(runner.critic_obs))
        advantages, returns = gae(traj, last_value)

        # flatten (T, N, ...) -> (T*N, ...), info isn't needed for the update
        data = (traj._replace(info={}), advantages, returns)
        data = jax.tree.map(lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), data)

        nets = (models.actor, models.critic)
        norms = (models.norm_actor, models.norm_critic)
        nets_dyn, nets_static = eqx.partition(nets, eqx.is_array)

        def epoch(carry, key):
            def minibatch(carry, idx):
                nets_dyn, opt_state = carry
                nets = eqx.combine(nets_dyn, nets_static)
                mb = jax.tree.map(lambda x: x[idx], data)
                (_, aux), grads = grad_fn(nets, norms, mb)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(nets, eqx.is_inexact_array))
                nets = eqx.apply_updates(nets, updates)
                return (eqx.partition(nets, eqx.is_array)[0], opt_state), aux

            perm = jax.random.permutation(key, cfg.batch_size)
            idxs = perm.reshape(cfg.num_minibatches, -1)
            return jax.lax.scan(minibatch, carry, idxs)

        key, k_perm = jax.random.split(runner.key)
        (nets_dyn, opt_state), aux = jax.lax.scan(
            epoch, (nets_dyn, ts.opt_state), jax.random.split(k_perm, cfg.update_epochs))
        actor, critic = eqx.combine(nets_dyn, nets_static)

        # normalisers are updated AFTER the gradient steps so rollout and update
        # used identical statistics
        models = Models(actor, critic,
                        models.norm_actor.update(traj.actor_obs),
                        models.norm_critic.update(traj.critic_obs))

        metrics = {k: v.mean() for k, v in aux.items()}
        metrics.update(_episode_metrics(traj))
        metrics["mean_reward"] = traj.reward.mean() / cfg.reward_scale
        metrics["action_std"] = jnp.exp(actor.log_std).mean()
        return TrainState(models, opt_state), runner._replace(key=key), metrics

    return init_fn, update_fn


def _episode_metrics(traj: Transition) -> dict:
    done = traj.done
    info = traj.info
    n_done = jnp.maximum(done.sum(), 1.0)

    def mean_where(x, mask):
        return jnp.sum(x * mask) / jnp.maximum(mask.sum(), 1.0)

    sp = done * (info["mode"] == SETPOINT)
    tr = done * (info["mode"] == TRAJECTORY)
    return dict(
        episodes=done.sum(),
        episode_return=jnp.sum(info["episode_return"] * done) / n_done,
        rms_err_setpoint_mm=1e3 * mean_where(info["episode_rms_err"], sp),
        rms_err_trajectory_mm=1e3 * mean_where(info["episode_rms_err"], tr),
        termination_rate=jnp.sum(info["terminated"] * done) / n_done,
        V_over_sigma=info["V_norm"].mean(),
        clf_violation_rate=info["clf_violation"].mean(),
    )


# ------------------------------------------------------------------ checkpoints


def save_models(path: str | Path, models: Models) -> None:
    eqx.tree_serialise_leaves(path, models)


def load_models(path: str | Path, env: GantryEnv, cfg: PPOConfig) -> Models:
    """Checkpoints are float32; they load under float64 (x64) too."""
    cast = lambda tree, dt: jax.tree.map(
        lambda x: x.astype(dt) if eqx.is_inexact_array(x) else x, tree)
    skeleton = cast(init_models(env, cfg, jax.random.PRNGKey(0)), jnp.float32)
    return cast(eqx.tree_deserialise_leaves(path, skeleton), jnp.result_type(float))


def make_actor_policy(models: Models, deterministic: bool = True):
    """Deployable policy: encoder-only obs -> action in [-1, 1]^2."""
    def policy(actor_obs: jax.Array, key: jax.Array | None = None) -> jax.Array:
        mean, log_std = models.actor(models.norm_actor(actor_obs))
        if deterministic or key is None:
            return jnp.clip(mean, -1, 1)
        return jnp.clip(mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape), -1, 1)
    return policy
