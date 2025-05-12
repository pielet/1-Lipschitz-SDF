import jax
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorbar_params(field, center, n_ticks=4):
    fmin, fmax = np.min(field), np.max(field)
    norm = colors.TwoSlopeNorm(vmin=fmin, vmax=fmax, vcenter=center)
    ticks = np.stack(
        [np.linspace(fmin, center, n_ticks), np.linspace(center, fmax, n_ticks)]
    ).flatten()
    return norm, ticks


def render_sdf_2d(model, variables, pivot, domain_size, resolution=1000):
    """Renders the 2D signed distance function (SDF) contour and gradient of a given model.

    Args:
        model (flax.linen.Module): nn architecture
        variables (flax.core.FrozenDict): current model parameters
        pivot (tuple): lower left corner of the sampling domain
        domain_size (float): size of the sampling domain
        resolution (int): grid resolution
    """

    def forward(x):
        return model.apply(variables, x)[0]

    X, Y = np.mgrid[
        pivot[0] : pivot[0] + domain_size : domain_size / resolution,
        pivot[1] : pivot[1] + domain_size : domain_size / resolution,
    ]
    coords = np.column_stack((X.ravel(), Y.ravel()))
    sdf, grads = jax.vmap(jax.value_and_grad(forward))(coords)

    sdf_img = sdf.reshape(resolution, resolution)[:, ::-1].T
    grad_img = np.linalg.norm(grads, axis=1).reshape(resolution, resolution)[:, ::-1].T
    print(f'SDF min: {np.min(sdf)}, max: {np.max(sdf)}')
    print(f'|∇f| min: {np.min(grad_img)}, max: {np.max(grad_img)}')

    # contours
    contour_fig, ax = plt.subplots(figsize=(8, 8))
    norm, ticks = colorbar_params(sdf, 0.0)
    pos = ax.imshow(sdf_img, cmap='RdBu', norm=norm)
    ax.contour(sdf_img, levels=16, colors='k', linestyles='solid', linewidths=0.3)
    ax.contour(sdf_img, levels=[0.0], colors='k', linestyles='solid', linewidths=0.6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    contour_fig.colorbar(pos, ax=ax, cax=cax, ticks=ticks, format='%.2f')
    ax.axis('off')

    # gradients
    grad_fig, ax = plt.subplots(figsize=(8, 8))
    norm, ticks = colorbar_params(grad_img, 1.0)
    pos = ax.imshow(grad_img, cmap='RdBu', norm=norm)
    ax.contour(sdf_img, levels=[0.0], colors='k', linestyles='solid', linewidths=0.6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    grad_fig.colorbar(pos, ax=ax, cax=cax, ticks=ticks, format='%.2f',)
    ax.axis('off')

    return contour_fig, grad_fig


# TODO: add gradient?
# but then I need to differentiate through ground truth sdf, I'm too lazy for this
# plus gradient norm should be always 1
def render_ground_truth_2d(f, pivot, domain_size, resolution=1000):
    """Renders the 2D signed distance function (SDF) contour and gradient of a given model.

    Args:
        f (callable): SDF function
        pivot (tuple): sampling domain lower left corner
        domain_size (float): size of the sampling domain
        resolution (int): grid resolution
    """

    X, Y = np.mgrid[
        pivot[0] : pivot[0] + domain_size : (domain_size) / resolution,
        pivot[1] : pivot[1] + domain_size : (domain_size) / resolution,
    ]
    coords = np.column_stack((X.ravel(), Y.ravel()))
    sdf = f(coords)
    print(f'SDF min: {np.min(sdf)}, max: {np.max(sdf)}')

    sdf_img = sdf.reshape(resolution, resolution)[:, ::-1].T
    sdf_norm = colors.TwoSlopeNorm(vmin=np.min(sdf), vmax=np.max(sdf), vcenter=0)

    # contours
    contour_fig, ax = plt.subplots(figsize=(8, 8))
    pos = ax.imshow(sdf_img, cmap='RdBu', norm=sdf_norm)
    ax.contour(sdf_img, levels=16, colors='k', linestyles='solid', linewidths=0.3)
    ax.contour(sdf_img, levels=[0.0], colors='k', linestyles='solid', linewidths=0.6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    contour_fig.colorbar(
        pos,
        ax=ax,
        cax=cax,
        ticks=np.stack([np.linspace(np.min(sdf), 0, 4), np.linspace(0, np.max(sdf), 4)]).flatten(),
        format='%.2f',
    )
    ax.axis('off')

    return contour_fig
