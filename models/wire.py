from typing import Callable
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import constant


class GaborLayer(nn.Module):
    """Gabor wavelet with learnable parameters as activation function."""

    out_dim: int
    omega_0: float = 2.0
    scale_0: float = 1.0
    trainable: bool = False

    @nn.compact
    def __call__(self, x):
        if self.trainable:
            omega = self.param('omega', constant(self.omega_0), (1,))
            scale = self.param('scale', constant(self.scale_0), (1,))
        else:
            key = self.make_rng('constants')
            scale = self.variable('constants', 'scale', constant(self.omega_0), key, (1,)).value
            omega = self.variable('constants', 'omega', constant(self.scale_0), key, (1,)).value
        x = nn.Dense(self.out_dim)(x)
        return jnp.exp(1j * omega * x - jnp.square(jnp.abs(scale * x)))


class GaborNet(nn.Module):
    """Gabor network with multiple layers."""

    out_dim: int
    hidden_units: int
    hidden_layers: int
    pos_enc: Callable | None

    @nn.compact
    def __call__(self, x):
        _ = self.variable('constants', 'dummy', lambda: jnp.zeros((1, 1), dtype=x.dtype))
        x = self.pos_enc(x) if self.pos_enc is not None else x
        for _ in range(self.hidden_layers):
            x = GaborLayer(self.hidden_units, trainable=True)(x)
        x = nn.Dense(self.out_dim)(x)
        return x.real
