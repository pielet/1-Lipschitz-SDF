from typing import Callable
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import normal, lecun_normal, zeros_init


def safe_inv(x, eps=1e-6):
    """Safe inverse to avoid division by zero."""
    return jnp.where(jnp.abs(x) < eps, 0.0, 1.0 / x)


class FrobeniusLinear(nn.Module):
    out_dim: int
    b_disjoint: bool = False  # row-wise spectral normalization, same when out_dim = 1

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', lecun_normal(), (self.out_dim, x.shape[-1]))
        bias = self.param('bias', zeros_init(), (1, self.out_dim))

        if self.b_disjoint:
            weight /= jnp.linalg.norm(weight, axis=1, keepdims=True)
        else:
            weight /= jnp.linalg.norm(weight)

        out = jnp.dot(x, weight.T) + bias
        return out


class SLL(nn.Module):
    """SDP-based Lipschitz Layer. See https://arxiv.org/pdf/2303.03169 Equation 8."""

    hidden_units: int

    @nn.compact
    def __call__(self, x):
        # parameters
        weight = self.param('weight', lecun_normal(), (self.hidden_units, x.shape[-1]))
        bias = self.param('bias', zeros_init(), (1, self.hidden_units))
        q_raw = self.param('q', normal(), (self.hidden_units,))

        # compute t
        q = jnp.exp(q_raw)
        q_inv = jnp.exp(-q_raw)

        # t = diag(q_inv) @ W @ W^T @ diag(q), then sum over rows
        gram = jnp.einsum('i,ik,kj,j->ij', q_inv, weight, weight.T, q)
        t = safe_inv(jnp.abs(gram).sum(axis=1))

        # forward
        res = jnp.dot(x, weight.T) + bias  # linear
        res = nn.relu(res) * t[None, :]  # ReLU + scale
        res = 2 * jnp.dot(res, weight)  # project back
        out = x - res  # final residual
        return out


class SLLNet(nn.Module):
    """multiple layer SLL network"""

    out_dim: int
    hidden_units: int
    hidden_layers: int
    pos_enc: Callable | None

    @nn.compact
    def __call__(self, x):
        _ = self.variable('constants', 'dummy', lambda: jnp.zeros((1, 1), dtype=x.dtype))
        x = self.pos_enc(x) if self.pos_enc is not None else x
        for _ in range(self.hidden_layers):
            x = SLL(self.hidden_units)(x)
        x = FrobeniusLinear(self.out_dim)(x)
        return x
