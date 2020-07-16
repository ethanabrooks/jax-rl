import os
from flax import serialization


def save_model(filename, model):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as fp:
        fp.write(serialization.to_bytes(model))


def load_model(filename, model):
    with open(filename, "rb") as fp:
        return serialization.from_bytes(model, fp.read())
