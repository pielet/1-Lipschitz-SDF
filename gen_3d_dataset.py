import os

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sdf import Mesh

from utils.plot import render_ground_truth_slice_3d


def parse_config(config_path):
    with open(config_path) as f:
        config = OmegaConf.load(f)
    return config


def generate_3d_dataset_from_sdf(f, n_samples, bbox_min, bbox_max, eps=1e-2):
    # eps here is to avoid sampling on boundary
    # TODO: add importance sampling decay from surface
    X = np.random.uniform(bbox_min[0] - eps, bbox_max[0] + eps, n_samples)
    Y = np.random.uniform(bbox_min[1] - eps, bbox_max[1] + eps, n_samples)
    Z = np.random.uniform(bbox_min[2] - eps, bbox_max[2] + eps, n_samples)
    coords = np.stack((X, Y, Z)).T
    field = f(coords)
    return coords, field


def visualize_samples_3d(coords, field):
    """
    Visualize 3D samples in a scatter plot.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=field, cmap='RdBu', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(sc, ax=ax, label='SDF Value')
    return fig


if __name__ == '__main__':
    import sys

    os.makedirs('input', exist_ok=True)

    cfg = parse_config(sys.argv[1])
    mesh = Mesh.from_file(os.path.join('data', f'{cfg.dataset}.obj'))
    mesh.scaled(1.0 / max(mesh.size))
    mesh.centered()
    f = mesh.sdf(voxel_size=cfg.voxel_size, half_width=cfg.half_width)

    X, Y = generate_3d_dataset_from_sdf(
        f, cfg.n_samples, mesh.bounding_box[0], mesh.bounding_box[1]
    )
    np.savez(f'input/{cfg.dataset}.npz', X=X, Y=Y)

    fig = visualize_samples_3d(X, Y)
    fig.savefig(os.path.join('input', f'{cfg.dataset}_samples.png'))

    fig = render_ground_truth_slice_3d(f, z=0, bounds=mesh.bounding_box)
    fig.savefig(os.path.join('input', f'{cfg.dataset}_slice_xy.png'))
