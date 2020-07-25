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


@profile
def main():
    loss_obj = hk.transform(loss_fn, apply_rng=True)
    # Initial parameter values are typically random. In JAX you need a key in order
    # to generate random numbers and so Haiku requires you to pass one in.
    rng = jax.random.PRNGKey(42)

    # `init` runs your function, as such we need an example input. Typically you can
    # pass "dummy" inputs (e.g. ones of the same shape and dtype) since initialization
    # is not usually data dependent.
    images, labels = (
        jax.random.normal(rng, shape=[100]),
        jax.random.randint(rng, shape=[1], minval=0, maxval=10),
    )

    # The result of `init` is a nested data structure of all the parameters in your
    # network. You can pass this into `apply`.
    params = loss_obj.init(rng, images, labels)
    print(params)
    rng, rng_input = jax.random.split(rng)
    params = loss_obj.init(rng, images, labels)
    print(params)


if __name__ == "__main__":
    main()
