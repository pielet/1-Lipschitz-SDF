import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sdf import image

try:
    from data.shapes import koch_snowflake, sierpinski_triangle
except ImportError:
    from shapes import koch_snowflake, sierpinski_triangle


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


def generate_2d_dataset_from_sdf(f, n_samples, min, max, eps=1e-2):
    X = np.random.uniform(min + eps, max - eps, (n_samples, 2))
    Y = f(X)
    return X, Y


def visualize_samples(X, Y, image_path, min, max):
    fig, ax = plt.subplots(figsize=(8, 7))
    print(f'min: {np.min(Y)}, max: {np.max(Y)}')
    norm = colors.TwoSlopeNorm(vmin=np.min(Y), vmax=np.max(Y), vcenter=0)
    im = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='RdBu', norm=norm, s=0.1)
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()


if __name__ == '__main__':
    import sys
    os.makedirs('input', exist_ok=True)

    cfg = parse_config(sys.argv[1])
    f = select_sdf(cfg)
    X, Y = generate_2d_dataset_from_sdf(f, cfg.n_samples, cfg.sample_min, cfg.sample_max)
    np.savez(f'input/{cfg.dataset}.npz', X=X, Y=Y)
    visualize_samples(X, Y, f'input/{cfg.dataset}_samples.png', cfg.sample_min, cfg.sample_max)
