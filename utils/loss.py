import jax
import optax
from jax import numpy as jnp


def mse(apply_fn, constants):
    def loss_fn(params, x, y):
        y_pred = apply_fn({'params': params, 'constants': constants}, x)
        loss = optax.l2_loss(y_pred, y)
        return loss.sum()  # batch loss

    return loss_fn


def eikonal(apply_fn, constants, lamb):
    def loss_fn(params, x, y):
        def forward(x):
            return apply_fn({'params': params, 'constants': constants}, x).squeeze()

        y_pred = apply_fn({'params': params, 'constants': constants}, x)
        grad = jax.vmap(jax.grad(forward))(x)
        grad_norm = jnp.linalg.norm(grad, axis=1, keepdims=True)
        loss = optax.l2_loss(y_pred, y) + lamb * optax.l2_loss(
            grad_norm, jnp.full_like(grad_norm, 1.0)
        )
        return loss.sum()  # batch loss

    return loss_fn


def hKR(apply_fn, constants, margin, lamb, rho):
    """Compute the hinged Kantorovitch-Rubinstein loss.
    See: https://arxiv.org/pdf/2407.09505.

    Args:
        apply_fn: model apply function
        margin: error threshold under which misclassification is ignored
        lamb: balance between HR and hinge loss
        rho: probability distribution function
    """

    def loss_fn(params, x, y):
        y_pred = apply_fn({'params': params, 'constants': constants}, x)
        signed_y = y_pred * jnp.sign(y)
        loss = (lamb * jnp.maximum(0.0, margin - signed_y) - signed_y) * rho(x, y)
        return loss.sum()

    return loss_fn
