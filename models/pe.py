import jax
import jax.numpy as jnp
from flax import linen as nn


class GaussianPE(nn.Module):
    """Gaussian positional encoding: https://arxiv.org/pdf/2006.10739."""

    out_dim: int
    sigma: float = 1.0
    trainable: bool = False

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        m = self.out_dim // 2
        if self.trainable:
            B = self.param(
                'B',
                nn.initializers.truncated_normal(
                    stddev=self.sigma, lower=-2.0 * self.sigma, upper=2.0 * self.sigma
                ),
                (m, d),
            )
        else:
            B = self.variable(
                'constants',
                'B',
                lambda: jax.random.truncated_normal(
                    self.make_rng('params'), lower=-2.0, upper=2.0, shape=(m, d), dtype=x.dtype
                )
                * self.sigma,
            )
        proj = jnp.dot(x, B.value.T)
        return jnp.concatenate([jnp.sin(2 * jnp.pi * proj), jnp.cos(2 * jnp.pi * proj)], axis=-1)
