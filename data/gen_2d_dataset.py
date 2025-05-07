import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sdf import image

os.makedirs('input', exist_ok=True)


def generate_2d_dataset_from_sdf(f, n_samples, min, max):
    # default scale for all 2d sdf is 1.0, which means their bbox is [0, 0] to [1, 1]
    X = np.random.uniform(min, max, (n_samples, 2))
    Y = f(X)
    return X, Y


def visualize_samples(X, Y, image_path, min, max):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', s=1)
    plt.colorbar(label='SDF Value')
    plt.title('2D SDF Samples')
    plt.xlim(min, max)
    plt.ylim(min, max)
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
        min = -0.5 * args.scale
        max = 0.5 * args.scale
        X, Y = generate_2d_dataset_from_sdf(
            image(args.image, args.scale, args.scale), args.n_samples, min, max
        )
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        visualize_samples(X, Y, f'input/{image_name}_samples.png', min, max)
        np.savez(f'input/{image_name}.npz', X=X, Y=Y)
    elif hasattr(args, 'sdf') and args.sdf is not None:
        min = -0.1 * args.scale
        max = 1.1 * args.scale
        X, Y = generate_2d_dataset_from_sdf(
            globals()[args.sdf](args.order, args.scale), args.n_samples, min, max
        )
        visualize_samples(X, Y, f'input/{args.sdf}_samples.png', min, max)
        np.savez(f'input/{args.sdf}.npz', X=X, Y=Y)
    else:
        raise ValueError('Please provide either an image or sdf name.')
