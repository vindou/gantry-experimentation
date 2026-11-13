"""Actor / critic networks and a running observation normaliser."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


ACTIVATIONS = {"elu": jax.nn.elu, "tanh": jnp.tanh, "relu": jax.nn.relu, "silu": jax.nn.silu}


class MLP(eqx.Module):
    """Feed-forward network with arbitrary hidden sizes, e.g. [512, 256, 128]."""

    layers: tuple[eqx.nn.Linear, ...]
    activation: str = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, hidden: tuple[int, ...], activation: str,
                 *, key: jax.Array):
        sizes = (in_size, *hidden, out_size)
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(eqx.nn.Linear(a, b, key=k) for a, b, k in zip(sizes[:-1], sizes[1:], keys))
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        act = ACTIVATIONS[self.activation]
        for layer in self.layers[:-1]:
            x = act(layer(x))
        return self.layers[-1](x)


class Actor(eqx.Module):
    """Gaussian policy with state-independent log-std. Sees encoder-only obs."""

    mlp: MLP
    log_std: jax.Array

    def __init__(self, obs_size: int, action_size: int, hidden: tuple[int, ...], activation: str,
                 init_log_std: float, *, key: jax.Array):
        mlp = MLP(obs_size, action_size, hidden, activation, key=key)
        # small initial actions
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].weight, mlp, mlp.layers[-1].weight * 0.01)
        self.log_std = jnp.full((action_size,), init_log_std)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.mlp(obs), self.log_std


class Critic(eqx.Module):
    """State-value function. Sees privileged obs."""

    mlp: MLP

    def __init__(self, obs_size: int, hidden: tuple[int, ...], activation: str, *, key: jax.Array):
        self.mlp = MLP(obs_size, 1, hidden, activation, key=key)

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.mlp(obs)[0]


def gaussian_logp(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    return jnp.sum(
        -0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1
    )


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)


class RunningNorm(eqx.Module):
    """Running mean/variance (parallel Welford) observation normaliser."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @staticmethod
    def init(size: int) -> "RunningNorm":
        return RunningNorm(jnp.zeros(size), jnp.ones(size), jnp.asarray(1e-4))

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.clip((x - self.mean) / jnp.sqrt(self.var + 1e-8), -10.0, 10.0)

    def update(self, batch: jax.Array) -> "RunningNorm":
        batch = batch.reshape(-1, batch.shape[-1])
        b_mean, b_var, b_n = batch.mean(0), batch.var(0), batch.shape[0]
        delta = b_mean - self.mean
        n = self.count + b_n
        mean = self.mean + delta * b_n / n
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / n
        return RunningNorm(mean, m2 / n, n)
