from dataclasses import dataclass

import igl
import numpy as np
from sdf import SDF2, SDF3, capsule, cylinder, polygon, sdf2, slab, sphere


@dataclass
class Mesh:
    vertices: np.ndarray  # shape: (N, 3)
    faces: np.ndarray  # shape: (F, 3)
    edges: np.ndarray | None = None  # shape: (E, 2)

    def __post_init__(self):
        if self.edges is None:
            self.edges = igl.edges(self.faces)

    @classmethod
    def from_obj(cls, obj_path: str):
        v, f = igl.read_triangle_mesh(obj_path)
        return cls(v, f, v.shape[1], igl.edges(f))

    def normalize(self):
        min_v = np.min(self.vertices, axis=0)
        max_v = np.max(self.vertices, axis=0)
        center = (max_v + min_v) / 2
        scale = max(max_v - min_v)
        self.vertices = (self.vertices - center) / scale
        return self

    def __contains__(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are inside the mesh.
        Args:
            points (np.ndarray): shape (N, 3)
        Returns:
            boolean array of shape (N,)
        """
        return igl.winding_number(self.vertices, self.faces, points) < 0.5

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the signed distance from [points] to the mesh.
        Args:
            points (np.ndarray): shape (N, 3)
        Returns:
            signed distance values of shape (N,)
        """
        return igl.signed_distance(points, self.vertices, self.faces)[0]


def gearlike() -> SDF3:
    "from: https://github.com/fogleman/sdf/blob/main/examples/gearlike.py"
    f = sphere(2) & slab(z0=-0.5, z1=0.5).k(0.1)
    f -= cylinder(1).k(0.1)
    f -= cylinder(0.25).circular_array(16, 2).k(0.1)
    return f


def blobby() -> SDF3:
    "from: https://github.com/fogleman/sdf/blob/main/examples/blobby.py"
    X = np.array((1, 0, 0))
    Y = np.array((0, 1, 0))
    Z = np.array((0, 0, 1))

    s = sphere(0.75)
    s = s.translate(Z * -3) | s.translate(Z * 3)
    s = s.union(capsule(Z * -3, Z * 3, 0.5), k=1)

    f = sphere(1.5).union(s.orient(X), s.orient(Y), s.orient(Z), k=1)
    return f


def koch_snowflake(order, scale=1) -> SDF2:
    """Generate a Koch snowflake fractal.

    Args:
        order (int): The order of the Koch snowflake.
        scale (float): Scaling factor for the snowflake (from equilateral_triangle of size 1).
    """

    def iterate(points):
        new_points = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            dp = (p2 - p1) / 3
            pA = p1 + dp
            pB = p1 + 2 * dp
            angle = np.pi / 3
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            pC = pA + rot @ (dp)
            new_points += [p1, pA, pC, pB]
        return np.array(new_points)

    # initial triangle
    points = np.array([[0.0, np.sqrt(3) / 6], [0.5, 2 * np.sqrt(3) / 3], [1.0, np.sqrt(3) / 6]])
    for _ in range(order):
        points = iterate(points)
    return polygon(points * scale)


@sdf2
def sierpinski_triangle(order, scale=1):
    """Generate a Sierpinski triangle fractal.

    Args:
        order (int): The order of the Sierpinski triangle.
        scale (float): Scaling factor for the triangle (from equilateral_triangle of size 1).
    """

    def sierpinski(order, p0, p1, p2):
        if order == 0:
            return [(p0, p1, p2)]
        else:
            mid01 = (p0 + p1) / 2
            mid12 = (p1 + p2) / 2
            mid20 = (p2 + p0) / 2
            return (
                sierpinski(order - 1, p0, mid01, mid20)
                + sierpinski(order - 1, mid01, p1, mid12)
                + sierpinski(order - 1, mid20, mid12, p2)
            )

    def point_in_triangle(p, a, b, c, eps=1e-6):
        v0, v1, v2 = c - a, b - a, p - a
        d00, d11 = np.dot(v0, v0), np.dot(v1, v1)
        d01, d20, d21 = np.dot(v0, v1), np.dot(v2, v0), np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < eps:
            return False  # degenerate triangle
        # barycentric coordinates
        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom
        return (u >= 0) & (v >= 0) & (u + v <= 1)

    def point_line_distance(p: np.ndarray, a, b) -> np.ndarray:
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        projection = a + np.outer(t, ab)
        return np.linalg.norm(p - projection, axis=1)

    triangles = sierpinski(
        order,
        np.array([0, 0]),
        scale * np.array([0.5, np.sqrt(3) / 2]),
        scale * np.array([1, 0]),
    )

    def f(p):
        signed_dist = np.full((p.shape[0],), np.inf)
        for a, b, c in triangles:
            d = np.min(
                np.array(
                    [
                        point_line_distance(p, a, b),
                        point_line_distance(p, b, c),
                        point_line_distance(p, c, a),
                    ]
                ),
                axis=0,
            )
            b_in = point_in_triangle(p, a, b, c)
            d = np.where(b_in, -d, d)
            signed_dist = np.where(np.abs(d) < np.abs(signed_dist), d, signed_dist)
        return signed_dist

    return f
