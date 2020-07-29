#! /usr/bin/env python
import gc

from jax.ops import index_update, index
import jax
import jax.nn.initializers as initializers
from memory_profiler import profile
from flax import nn
from haiku import PRNGSequence
from flax import optim

import haiku as hk
import jax.numpy as jnp


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def loss_fn(images, labels):
    mlp = hk.Sequential([hk.Linear(30)])
    logits = mlp(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))


class Net(nn.Module):
    def apply(self, x):
        return nn.Dense(x, features=2000)


@profile
def main():
    loss_obj = hk.transform(loss_fn, apply_rng=True)
    # Initial parameter values are typically random. In JAX you need a key in order
    # to generate random numbers and so Haiku requires you to pass one in.
    rng = PRNGSequence(42)

    # `init` runs your function, as such we need an example input. Typically you can
    # pass "dummy" inputs (e.g. ones of the same shape and dtype) since initialization
    # is not usually data dependent.
    shape = [([1000], float)]

    adam = optim.Adam(learning_rate=0.1)
    partial = Net.partial()
    _, params = partial.init_by_shape(next(rng), shape)
    net = nn.Model(partial, params)

    optimizer = jax.device_put(adam.create(net))
    print(optimizer.target)
    input("waiting")
    _, params = partial.init_by_shape(next(rng), shape)
    net = net.replace(params=params)
    optimizer = jax.device_put(adam.create(net))
    print(optimizer.target)
    input("waiting")


if __name__ == "__main__":
    main()
