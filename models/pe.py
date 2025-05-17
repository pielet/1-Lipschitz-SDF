import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import constant, truncated_normal


class GaussianFourierPE(nn.Module):
    """Gaussian positional encoding: https://arxiv.org/pdf/2006.10739."""

    emb_size: int
    sigma: float = 1.0
    trainable: bool = False
    spectral_norm: bool = False

    @nn.compact
    def __call__(self, x):
        m, d = self.emb_size // 2, x.shape[-1]
        if self.trainable:
            B = self.param(
                'B',
                truncated_normal(
                    stddev=self.sigma, lower=-2.0 * self.sigma, upper=2.0 * self.sigma
                ),
                (m, d),
            )
        else:
            def init_B():
                key = self.make_rng('constants')
                return truncated_normal(
                    stddev=self.sigma, lower=-2.0 * self.sigma, upper=2.0 * self.sigma
                )(key, (m, d))
            B = self.variable('constants', 'B', init_B).value
        sn = 1.0
        if self.spectral_norm:
            _, s, _ = jnp.linalg.svd(B, full_matrices=False)
            sn = 2 * jnp.pi * jnp.max(s)
        proj = 2 * jnp.pi * jnp.dot(x, B.T)
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1) / sn


def FourierPE(x, emb_size):
    d = x.shape[-1]
    m = emb_size // d // 2
    proj = x[..., None] * 2 ** jnp.arange(m) * jnp.pi
    emb = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1).reshape(x.shape[0], -1)
    return emb


class GaussianGaborPE(nn.Module):
    """Gabor positional encoding."""

    emb_size: int
    sigma: float = 1.0
    omega_0: float = 2.0
    scale_0: float = 1.0
    trainable: bool = False
    spectral_norm: bool = False

    @nn.compact
    def __call__(self, x):
        m, d = self.emb_size // 2, x.shape[-1]
        if self.trainable:
            scale = self.param('scale', constant(self.scale_0), (1,))
            omega = self.param('omega', constant(self.omega_0), (1,))
            B = self.param(
                'B',
                truncated_normal(
                    stddev=self.sigma, lower=-2.0 * self.sigma, upper=2.0 * self.sigma
                ),
                (m, d),
            )
        else:
            key = self.make_rng('constants')
            scale = self.variable('constants', 'scale', constant(self.scale_0), key, (1,)).value
            omega = self.variable('constants', 'omega', constant(self.omega_0), key, (1,)).value
            def init_B():
                key = self.make_rng('constants')
                return truncated_normal(
                    stddev=self.sigma, lower=-2.0 * self.sigma, upper=2.0 * self.sigma
                )(key, (m, d))
            B = self.variable('constants', 'B', init_B).value
        if self.spectral_norm:
            _, s, _ = jnp.linalg.svd(B, full_matrices=False)
        proj = 2 * jnp.pi * jnp.dot(x, B.T)
        exp = jnp.exp(-jnp.square(proj * scale))
        return jnp.concatenate([jnp.cos(omega * proj) * exp, jnp.sin(omega * proj) * exp], axis=-1)


def GaborPE(x, emb_size, omega_0=2.0, scale_0=1.0):
    d = x.shape[-1]
    m = emb_size // d // 2
    proj = x[..., None] * 2 ** jnp.arange(m) * jnp.pi
    exp = jnp.exp(-jnp.square(proj * scale_0))
    emb = jnp.concatenate(
        [jnp.cos(omega_0 * proj) * exp, jnp.sin(omega_0 * proj) * exp], axis=-1
    ).reshape(x.shape[0], -1)
    return emb
