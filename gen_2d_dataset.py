import os
import time

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import OmegaConf
from sdf import image

from data.shapes import koch_snowflake, sierpinski_triangle
from utils.plot import render_ground_truth_2d


def timing(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__} took {end - start:.4f} seconds')
        return result

    return wrapper


def parse_config(config_path):
    with open(config_path) as f:
        config = OmegaConf.load(f)
    return config


def select_sdf(cfg):
    sdf_fn = {
        'koch': koch_snowflake,
        'sierpinski': sierpinski_triangle,
    }

    if cfg.dataset in sdf_fn.keys():  # fractal sdf
        f = sdf_fn[cfg.dataset](cfg.order, cfg.scale)
    elif os.path.exists(f'data/{cfg.dataset}.png'):  # image
        f = image(f'data/{cfg.dataset}.png', cfg.scale, cfg.scale)
    else:
        raise ValueError(f'Unknown dataset: {cfg.dataset}')

    return f


@timing
def generate_2d_dataset_from_sdf(f, n_samples, pivot, domain_size, eps=1e-2):
    # eps here is to avoid sampling on boundary, which has large vaule for image sdf
    # TODO: fix this
    X = np.random.uniform(pivot[0] + eps, pivot[0] + domain_size - eps, n_samples)
    Y = np.random.uniform(pivot[1] + eps, pivot[1] + domain_size - eps, n_samples)
    coords = np.stack((X, Y)).T
    field = f(coords)
    return coords, field


def visualize_samples(X, Y, pivot, domain_size):
    fig, ax = plt.subplots(figsize=(8, 8))
    print(f'Sample min: {np.min(Y)}, max: {np.max(Y)}')
    norm = colors.TwoSlopeNorm(vmin=np.min(Y), vmax=np.max(Y), vcenter=0)
    im = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='RdBu', norm=norm, s=0.1)
    ax.set_xlim(pivot[0], pivot[0] + domain_size)
    ax.set_ylim(pivot[1], pivot[1] + domain_size)
    ax.set_aspect('equal', adjustable='box')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(
        im,
        ax=ax,
        cax=cax,
        ticks=np.stack([np.linspace(np.min(Y), 0, 4), np.linspace(0, np.max(Y), 4)]).flatten(),
        format='%.2f',
    )
    return fig


if __name__ == '__main__':
    import sys

    os.makedirs('input', exist_ok=True)

    cfg = parse_config(sys.argv[1])
    f = select_sdf(cfg)

    X, Y = generate_2d_dataset_from_sdf(f, cfg.n_samples, cfg.domain_pivot, cfg.domain_size)
    np.savez(f'input/{cfg.dataset}.npz', X=X, Y=Y)

    fig = visualize_samples(X, Y, cfg.domain_pivot, cfg.domain_size)
    fig.savefig(f'input/{cfg.dataset}_samples.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    fig = render_ground_truth_2d(f, cfg.domain_pivot, cfg.domain_size)
    fig.savefig(f'input/{cfg.dataset}_ground_truth.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
