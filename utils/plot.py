import jax
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def render_sdf_2d(model, params, min=0.0, max=1.0, resolution=1000):
    """Renders the 2D signed distance function (SDF) contour and gradient of a given model.

    Args:
        model (flax.linen.Module): nn architecture
        params (flax.core.FrozenDict): current model parameters
        min (float): minimum value of the grid
        max (float): maximum value of the grid
        resolution (int): grid resolution
    """

    def forward(x):
        return model.apply({'params': params}, x)[0]

    X, Y = np.mgrid[min : max : (max - min) / resolution, min : max : (max - min) / resolution]
    coords = np.column_stack((X.ravel(), Y.ravel()))
    sdf, grads = jax.vmap(jax.value_and_grad(forward))(coords)

    sdf_img = sdf.reshape(resolution, resolution)[:, ::-1].T
    sdf_norm = colors.TwoSlopeNorm(vmin=np.min(sdf), vmax=np.max(sdf), vcenter=0)
    grad_img = np.linalg.norm(grads, axis=1).reshape(resolution, resolution)[:, ::-1].T
    grad_norm = colors.TwoSlopeNorm(vmin=np.min(grad_img), vmax=np.max(grad_img), vcenter=1.0)
    # print(f'GRAD NORM INTERVAL {np.min(grad_img):.2f}, {np.max(grad_img):.2f}')

    # contours
    contour_fig, ax = plt.subplots(figsize=(8, 8))
    pos = ax.imshow(sdf_img, cmap='seismic', norm=sdf_norm)
    ax.contour(sdf_img, levels=16, colors='k', linestyles='solid', linewidths=0.3)
    ax.contour(sdf_img, levels=[0.0], colors='k', linestyles='solid', linewidths=0.6)
    contour_fig.colorbar(pos, ax=ax, fraction=0.04)
    ax.axis('off')

    # gradients
    grad_fig, ax = plt.subplots(figsize=(8, 8))
    pos = ax.imshow(grad_img, cmap='seismic', norm=grad_norm)
    ax.contour(sdf_img, levels=[0.0], colors='k', linestyles='solid', linewidths=0.6)
    grad_fig.colorbar(pos, ax=ax, fraction=0.04)
    ax.axis('off')

    return contour_fig, grad_fig
