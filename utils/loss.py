import optax
from jax import numpy as jnp


def mse(apply_fn):
    def loss_fn(params, x, y):
        y_pred = apply_fn({'params': params}, x)
        loss = optax.l2_loss(y_pred, y).sum()
        return loss

    return loss_fn


def hKR(apply_fn, margin, lamb, rho):
    """Compute the hinged Kantorovitch-Rubinstein loss.
    See: https://arxiv.org/pdf/2407.09505.

    Args:
        params: model parameters
        x: coordinates
        y: ground truth values
        margin: error threshold under which misclassification is ignored
        lamb: balance between HR and hinge loss
        rho: probability distribution function
        apply_fn: model apply function
    """

    def loss_fn(params, x, y):
        y_pred = apply_fn({'params': params}, x)
        signed_y = y_pred * jnp.sign(y)
        loss = lamb * jnp.maximum(0.0, margin - signed_y) + jnp.mean(-signed_y)
        return loss * rho(x, y)

    return loss_fn
