import numpy as np
from scipy.spatial import cKDTree


def evaluate_sdf(coords, sdf_pred, sdf_true, grad_pred, eps=5e-3):
    """Evaluate soft chamfer distance, IoU, and MSE between the predicted and true SDF.

    Args:
        coords (np.ndarray): coordinates of the points
        sdf_pred (np.ndarray): predicted signed distance function
        sdf_true (np.ndarray): true signed distance function
        grad_pred (np.ndarray): predicted gradient of the signed distance function
        eps (float): threshold for soft chamfer distance

    Returns:
        chamfer_dist (float): soft chamfer distance
        iou (float): intersection over union
        mse (float): mean squared error
        eikonal (float): eikonal loss
    """

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
    grad_norm = np.linalg.norm(grad_pred, axis=1)
    eikonal = np.mean(np.abs(grad_norm - 1.0))

    return chamfer_dist, iou, mse, eikonal
