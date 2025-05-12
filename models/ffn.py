import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal

from models.pe import GaussianPE, FourierPE


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
        _ = self.variable('constants', 'dummy', lambda: jnp.zeros((1, 1), dtype=x.dtype))
        x = SirenLayer(out_dim=self.hidden_units, omega=30.0)(
            x
        )  # see 3.2: cover wilder frequency range to get arcsin output distribution
        for _ in range(self.hidden_layers - 1):
            x = SirenLayer(self.hidden_units)(x)
        x = nn.Dense(self.out_dim, kernel_init=glorot_normal())(x)
        return x


class MLP(nn.Module):
    """MLP with optional positional encoding."""

    out_dim: int
    hidden_layers: int
    hidden_units: int

    pe_dim: int = 0
    pe_sigma: float = 1.0
    pe_trainable: bool = False

    @nn.compact
    def __call__(self, x):
        if self.pe_dim > 0:
            # x = FourierPE(self.pe_dim, x)
            # _ = self.variable('constants', 'dummy', lambda: jnp.zeros((1, 1), dtype=x.dtype))
            x = GaussianPE(self.pe_dim, self.pe_sigma, self.pe_trainable)(x)
        else:
            _ = self.variable('constants', 'dummy', lambda: jnp.zeros((1, 1), dtype=x.dtype))
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x
