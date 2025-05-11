import jax
import numpy as np
from scipy.spatial import cKDTree


def evaluate_sdf_2d(model, variables, sdf, pivot, domain_size, resolution=1000, eps=5e-3):
    """Evaluate soft chamfer distance, IoU, and MSE between the predicted and true SDF.

    Args:
        model (flax.linen.Module): nn architecture
        variables (flax.core.FrozenDict): current model parameters
        sdf (callable): ground truth SDF
        pivot (tuple): lower left corner of the sampling domain
        domain_size (float): size of the sampling domain
        resolution (int): grid resolution
        eps (float): soft margin threshold for chamfer distance

    Returns:
        chamfer_dist (float): soft chamfer distance
        iou (float): intersection over union
        mse (float): mean squared error
    """

    def forward(x):
        return model.apply(variables, x)[0]

    X, Y = np.mgrid[
        pivot[0] : pivot[0] + domain_size : domain_size / resolution,
        pivot[1] : pivot[1] + domain_size : domain_size / resolution,
    ]
    coords = np.column_stack((X.ravel(), Y.ravel()))
    sdf_pred = jax.vmap(forward)(coords)
    sdf_true = sdf(coords).squeeze()

    # (soft) chamfer distance
    pts_pred = coords[np.abs(sdf_pred) < eps]
    pts_true = coords[np.abs(sdf_true) < eps]
    tree_pred = cKDTree(pts_pred)
    tree_true = cKDTree(pts_true)
    d1, _ = tree_pred.query(pts_true)
    d2, _ = tree_true.query(pts_pred)
    chamfer_dist = np.mean(d1**2) + np.mean(d2**2)

    # IoU
    pred_in = sdf_pred < 0
    true_in = sdf_true < 0
    iou = np.sum(pred_in & true_in) / np.sum(pred_in | true_in)

    # mse
    mse = np.mean((sdf_pred - sdf_true) ** 2)

    # eikonal
    grad = jax.vmap(jax.grad(forward))(coords)
    grad_norm = np.linalg.norm(grad, axis=1)
    eikonal = np.mean(np.abs(grad_norm - 1.0))

    return chamfer_dist, iou, mse, eikonal
