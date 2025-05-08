import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal


class SirenLayer(nn.Module):
    """MLP with sine activation and weight scaling."""

    out_dim: int
    omega: float = 1.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.out_dim, kernel_init=glorot_normal(), use_bias=False)(x)
        bias = self.param('bias', normal(), (self.out_dim,))
        return jnp.sin(self.omega * x + bias)


class SIREN(nn.Module):
    """See: https://arxiv.org/pdf/2006.09661."""

    out_dim: int
    hidden_layers: int
    hidden_units: int

    @nn.compact
    def __call__(self, x):
        x = SirenLayer(out_dim=self.hidden_units, omega=30.0)(x)  # see 3.2
        for _ in range(self.hidden_layers - 1):
            x = SirenLayer(self.hidden_units)(x)
        x = nn.Dense(self.out_dim, kernel_init=glorot_normal())(x)
        return x
