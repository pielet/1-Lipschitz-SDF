import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sdf import image

os.makedirs('input', exist_ok=True)


def generate_2d_dataset_from_sdf(f, n_samples, scale):
    # default scale for all 2d sdf is 1.0, which means their bbox is [0, 0] to [1, 1]
    X = np.random.uniform(-0.1 * scale, 1.1 * scale, (n_samples, 2))
    Y = f(X)
    return X, Y


def visualize_samples(X, Y, scale, image_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', s=1)
    plt.colorbar(label='SDF Value')
    plt.title('2D SDF Samples')
    plt.xlim(-0.1 * scale, 1.1 * scale)
    plt.ylim(-0.1 * scale, 1.1 * scale)
    plt.savefig(image_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 2D dataset.')
    parser.add_argument(
        '-n',
        '--n_samples',
        type=int,
        required=True,
        default=1000,
        help='number of points to sample',
    )
    parser.add_argument(
        '-i', '--image', type=str, help='binary image file path to load the shape from'
    )
    parser.add_argument(
        '-f',
        '--sdf',
        choices=['koch_snowflake', 'sierpinski_triangle'],
        help='ground truth sdf',
    )
    parser.add_argument('-o', '--order', type=int, default=0, help='order of the fractal')
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='scale of the fractal')
    args = parser.parse_args()

    if hasattr(args, 'image') and args.image is not None:
        X, Y = generate_2d_dataset_from_sdf(
            image(args.image, args.scale), args.n_samples, args.scale
        )
        visualize_samples(X, Y, args.scale, f'input/{args.image}_samples.png')
        np.save(f'input/{args.image}_X.npy', X)
        np.save(f'input/{args.image}_Y.npy', Y)
    elif hasattr(args, 'sdf') and args.sdf is not None:
        X, Y = generate_2d_dataset_from_sdf(
            globals()[args.sdf](args.order, args.scale), args.n_samples, args.scale
        )
        visualize_samples(X, Y, args.scale, f'input/{args.sdf}_samples.png')
        np.save(f'input/{args.sdf}_X.npy', X)
        np.save(f'input/{args.sdf}_Y.npy', Y)
    else:
        raise ValueError('Please provide either an image or sdf name.')
