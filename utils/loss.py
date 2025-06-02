import inspect
from functools import partial

import jax
import optax
from jax import numpy as jnp


def mse(apply_fn, constants):
    def loss_fn(params, x, y):
        y_pred = apply_fn({'params': params, 'constants': constants}, x)
        loss = optax.l2_loss(y_pred, y)
        return loss

    return loss_fn


def eikonal(apply_fn, constants):
    def loss_fn(params, x, _):
        def forward(x):
            x = x.reshape(-1, x.shape[-1])  # [batch_size, in_dim]
            return apply_fn({'params': params, 'constants': constants}, x).squeeze()

        grad = jax.vmap(jax.grad(forward))(x)
        grad_norm = jnp.linalg.norm(grad, axis=1, keepdims=True)
        loss = optax.l2_loss(grad_norm, jnp.full_like(grad_norm, 1.0))
        return loss

    return loss_fn


def heat_loss(apply_fn, constants, lamb):
    """Heat loss introduced in https://arxiv.org/pdf/2411.14628 to avoid gradient discontinuities in Eikonal loss.
    See https://sci-hub.lu/10.1002/cpa.3160200210 for proof of conversion between distance and equilibrium heat field obtained by solving screened Poisson equation.

    Args:
        lamb: absorption coefficient
    """

    def loss_fn(params, x, y):
        def forward(x):
            return apply_fn({'params': params, 'constants': constants}, x).squeeze()

        y_pred, grad = jax.vmap(jax.value_and_grad(forward))(x)
        grad_norm = jnp.linalg.norm(grad, axis=1, keepdims=True)
        loss = jnp.exp(-2.0 * lamb * jnp.abs(y_pred)) * (
            jnp.square(grad_norm) + jnp.full_like(grad_norm, 1.0)
        )
        return loss

    return loss_fn


def hKR(apply_fn, constants, margin, lamb):
    """Compute the hinged Kantorovitch-Rubinstein loss.
    See: https://arxiv.org/pdf/2407.09505.

    Args:
        margin: error threshold under which misclassification is ignored
        lamb: balance between HR and hinge loss
    """

    def loss_fn(params, x, y):
        def rho(coords):
            x0, x1 = coords[:, 0], coords[:, 1]
            # TODO: design a weight function
            return jnp.ones_like(x0)

        y_pred = apply_fn({'params': params, 'constants': constants}, x)
        signed_y = y_pred * jnp.sign(y)
        loss = (lamb * jnp.maximum(0.0, margin - signed_y) - signed_y) * rho(x)
        # loss = (lamb * jnp.maximum(0.0, margin - signed_y) + jnp.exp(-0.1 * signed_y)) * rho(x)
        return loss

    return loss_fn


loss_fn_zoo = {
    'mse': mse,
    'eikonal': eikonal,
    'heat': heat_loss,
    'hkr': hKR,
}


def safe_call(func, args):
    sig = inspect.signature(func)
    valid_keys = sig.parameters.keys()
    return func(**{k: v for k, v in args.items() if k in valid_keys})


def get_loss_fn(config, apply_fn, constants):
    """Get the loss function based on the specified loss types and weights.

    Args:
        config: training configuration.
        apply_fn: Model apply function.
        constants: Model constants.

    Returns:
        function: Combined loss function.
    """
    assert len(config.loss_types) == len(config.loss_weights)

    def combined_loss_fn(params, x, y):
        total_loss = jnp.zeros_like(y)
        for i, loss_type in enumerate(config.loss_types):
            loss_fn = safe_call(
                partial(loss_fn_zoo[loss_type], apply_fn=apply_fn, constants=constants), config.loss
            )
            total_loss += config.loss_weights[i] * loss_fn(params, x, y)
        return total_loss.sum()

    return combined_loss_fn
