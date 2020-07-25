import jax.numpy as jnp
from jax import nn
from jax import random
from typing import Tuple

from utils import gaussian_likelihood
import haiku as hk


class DoubleCritic(hk.Module):
    def __call__(
        self, state: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        state_action = jnp.concatenate([state, action], axis=-1)

        q1 = hk.Linear(500)(state_action)
        # q1 = hk.LayerNorm(q1)
        q1 = nn.relu(q1)
        q1 = hk.Linear(500)(q1)
        q1 = nn.relu(q1)
        q1 = hk.Linear(1)(q1)

        # if Q1:
        #     return q1

        q2 = hk.Linear(500)(state_action)
        # q2 = hk.LayerNorm(q2)
        q2 = nn.relu(q2)
        q2 = hk.Linear(500)(q2)
        q2 = nn.relu(q2)
        q2 = hk.Linear(1)(q2)

        return q1, q2


class GaussianPolicy(hk.Module):
    def __init__(
        self,
        action_dim: int,
        max_action,
        mpo=False,
        log_sig_min: float = -20,
        log_sig_max: float = 2,
    ):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.MPO = mpo
        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

    def __call__(self, x, key=None):
        x = hk.Linear(200)(x)
        # x = hk.LayerNorm(x)
        x = nn.relu(x)
        x = hk.Linear(200)(x)
        x = nn.relu(x)
        x = hk.Linear(2 * self.action_dim)(x)

        mu, log_sig = jnp.split(x, 2, axis=-1)
        log_sig = nn.softplus(log_sig)
        log_sig = jnp.clip(log_sig, self.log_sig_min, self.log_sig_max)

        if self.MPO:
            return mu, log_sig

        if key is None:
            return self.max_action * jnp.tanh(mu), log_sig
        else:
            sig = jnp.exp(log_sig)
            pi = mu + random.normal(key, mu.shape) * sig
            log_pi = gaussian_likelihood(pi, mu, log_sig)
            pi = jnp.tanh(pi)
            log_pi -= jnp.sum(jnp.log(nn.relu(1 - pi ** 2) + 1e-6), axis=1)
            return self.max_action * pi, log_pi


class Constant(hk.Module):
    def __call__(self, start_value, dtype=jnp.float32):
        value = hk.get_parameter("value", (1,), init=jnp.ones)
        return start_value * jnp.asarray(value, dtype)


def build_constant_model(start_value, init_rng):
    constant = Constant.partial(start_value=start_value)
    _, init_params = constant.init(init_rng)

    return hk.Model(constant, init_params)


def build_double_critic_model(input_shapes, init_rng):
    critic = DoubleCritic.partial()
    _, init_params = critic.init_by_shape(init_rng, input_shapes)

    return hk.Model(critic, init_params)


def build_gaussian_policy_model(input_shapes, action_dim, max_action, init_rng):
    actor = GaussianPolicy.partial(action_dim=action_dim, max_action=max_action)
    _, init_params = actor.init_by_shape(init_rng, input_shapes)

    return hk.Model(actor, init_params)
