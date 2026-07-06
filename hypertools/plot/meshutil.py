"""Pure-geometry mesh utilities for smooth convex-hull surfaces (GH #109).

This module builds smooth 3-D "blob" surfaces and smooth 2-D outlines from
raw point clouds by inflating and refining their convex hulls, plus a small
set of shading/culling helpers used to render those surfaces.

Pipeline (3-D): ``ConvexHull`` -> outward-oriented triangle mesh -> pre-inflate
about the centroid -> interleaved rounds of [midpoint 1->4 subdivision, then
Taubin lambda/mu smoothing]. Interleaving subdivision with smoothing (rather
than subdividing everything up front and smoothing once at the end) avoids
pinched creases at the original hull vertices.

Pipeline (2-D): ``ConvexHull`` vertices (already ordered) -> centripetal
Catmull-Rom spline (alpha=0.5) through the closed polygon.

This module is intentionally dependency-light (numpy + scipy only) and does
not import anything from :mod:`hypertools.plot.plot` or the plot backends --
it is a leaf geometry module that those higher-level modules import from.
"""
import warnings

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

__all__ = [
    "smooth_hull_3d",
    "smooth_hull_2d",
    "face_normals",
    "blinn_phong_colors",
    "backface_cull",
    "points_enclosed",
]


def _orient_hull(points):
    """Compute a convex hull and return a consistently outward-oriented mesh.

    Parameters
    ----------
    points : ndarray of shape (n, 3)
        Input point cloud.

    Returns
    -------
    verts : ndarray of shape (m, 3)
        Hull vertex coordinates (subset of `points`).
    faces : ndarray of shape (f, 3), dtype int
        Triangle indices into `verts`, wound so each face's normal
        (via the right-hand rule) points away from the hull interior.
    """
    hull = ConvexHull(points)
    verts = points[hull.vertices]
    remap = -np.ones(len(points), dtype=int)
    remap[hull.vertices] = np.arange(len(hull.vertices))
    faces = remap[hull.simplices].copy()

    normals = hull.equations[:, :3]
    v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    flip = np.einsum("ij,ij->i", fn, normals) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return verts, faces


def _subdivide(verts, faces):
    """One round of midpoint (1 -> 4) triangle subdivision.

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    faces : ndarray of shape (f, 3), dtype int

    Returns
    -------
    new_verts : ndarray of shape (n + e, 3)
        Original vertices followed by one new midpoint vertex per unique
        edge.
    new_faces : ndarray of shape (4 * f, 3), dtype int
        Each input triangle is replaced by 4 triangles.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    mid = verts[uniq].mean(axis=1)
    mid_idx = len(verts) + inv.reshape(3, -1).T  # per face: m01, m12, m20
    new_verts = np.vstack([verts, mid])
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    m01, m12, m20 = mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2]
    new_faces = np.concatenate(
        [
            np.stack([a, m01, m20], axis=1),
            np.stack([m01, b, m12], axis=1),
            np.stack([m20, m12, c], axis=1),
            np.stack([m01, m12, m20], axis=1),
        ]
    )
    return new_verts, new_faces


def _taubin_smooth(verts, faces, iterations=8, lam=0.5, mu=-0.53):
    """Taubin lambda/mu Laplacian smoothing (shrinkage-resistant).

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    faces : ndarray of shape (f, 3), dtype int
    iterations : int, optional
        Number of lambda/mu passes (each pass is one shrink step + one
        inflate step). Default 8.
    lam : float, optional
        Lambda (shrink) factor. Default 0.5.
    mu : float, optional
        Mu (inflate) factor; should be negative and larger in magnitude
        than `lam` to counteract shrinkage. Default -0.53.

    Returns
    -------
    ndarray of shape (n, 3)
        Smoothed vertex positions.
    """
    n = len(verts)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    deg = np.bincount(i, minlength=n).astype(float)
    deg[deg == 0] = 1.0  # guard against isolated vertices (shouldn't occur)
    v = verts.copy()
    for _ in range(iterations):
        for factor in (lam, mu):
            nb = np.zeros_like(v)
            np.add.at(nb, i, v[j])
            v = v + factor * (nb / deg[:, None] - v)
    return v


def smooth_hull_3d(points, rounds=3, taubin_iters=8, lam=0.5, mu=-0.53, pre_inflate=1.15):
    """Build a smooth, outward-oriented triangle mesh from a 3-D point cloud.

    Computes the convex hull of `points`, pre-inflates it about its centroid,
    then performs `rounds` interleaved rounds of midpoint (1 -> 4) subdivision
    followed by Taubin lambda/mu smoothing. Interleaving (rather than
    subdividing fully and then smoothing once) avoids pinched creases at the
    original hull vertices.

    Parameters
    ----------
    points : array-like of shape (n, 3)
        Input point cloud. Must contain at least 4 non-coplanar points.
    rounds : int, optional
        Number of interleaved [subdivide, smooth] rounds. Face count scales
        as ``4 ** rounds`` relative to the (inflated) hull. Default 3.
    taubin_iters : int, optional
        Number of Taubin lambda/mu passes performed *per round*. Default 8.
    lam : float, optional
        Taubin lambda (shrink) factor. Default 0.5.
    mu : float, optional
        Taubin mu (inflate) factor. Default -0.53.
    pre_inflate : float, optional
        Scale factor applied to hull vertices about the centroid before
        subdivision/smoothing begins, to compensate for the net shrinkage
        introduced by Taubin smoothing and keep (most of) the original
        points inside the final surface. Default 1.15.

    Returns
    -------
    verts : ndarray of shape (m, 3)
        Smoothed mesh vertex coordinates.
    faces : ndarray of shape (f, 3), dtype int
        Triangle indices into `verts`, outward-oriented.

    Raises
    ------
    ValueError
        If `points` does not have shape (n, 3) with n >= 4, or if the
        points are (nearly) coplanar so no 3-D convex hull can be formed.

    Examples
    --------
    >>> import numpy as np
    >>> from hypertools.plot.meshutil import smooth_hull_3d
    >>> pts = np.random.default_rng(0).normal(size=(200, 3))
    >>> verts, faces = smooth_hull_3d(pts)
    >>> verts.shape[1], faces.shape[1]
    (3, 3)
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"`points` must have shape (n, 3); got shape {points.shape}"
        )
    if len(points) < 4:
        raise ValueError(
            f"need at least 4 points to form a 3-D convex hull; got {len(points)}"
        )

    try:
        verts, faces = _orient_hull(points)
    except QhullError as exc:
        raise ValueError(
            "could not compute a 3-D convex hull -- points appear to be "
            "coplanar or otherwise degenerate"
        ) from exc

    centroid = verts.mean(axis=0)
    hull_verts_inflated = centroid + pre_inflate * (verts - centroid)
    verts = hull_verts_inflated

    for _ in range(rounds):
        verts, faces = _subdivide(verts, faces)
        verts = _taubin_smooth(verts, faces, iterations=taubin_iters, lam=lam, mu=mu)

    # GH #109: `pre_inflate` compensates for Taubin smoothing's net
    # shrinkage, but that compensation is tuned against the SAME fixed
    # `taubin_iters`/`rounds` regardless of hull size -- a small, sparse
    # hull (few input points -> few, often widely-spaced hull vertices)
    # loses proportionally much MORE of its "bulge" to the same number of
    # smoothing passes than a large, dense one does, so small point clouds
    # can end up with the final surface pulled far inside the original
    # data (observed: a 5-point cloud's mesh contained as few as ~1/5 of
    # its own input points, with no warning). Post-hoc rescale the mesh
    # (uniformly, about its centroid -- never shrinking it) so it recovers
    # containment of the inflated hull vertices, then verify against the
    # ACTUAL input points and warn if that still falls short.
    verts = _rescale_for_containment(verts, hull_verts_inflated, centroid)

    contained_frac = points_enclosed(points, verts).mean()
    if contained_frac < 0.96:
        warnings.warn(
            "smooth_hull_3d: the smoothed surface contains only "
            f"{100 * contained_frac:.0f}% of the input points (target "
            "96%) even after rescaling for containment; it may visibly "
            "fail to enclose some of the data."
        )

    return verts, faces


def _rescale_for_containment(verts, hull_verts_inflated, centroid, margin=1.02):
    """Uniformly rescale `verts` about `centroid` (growing only, never
    shrinking) so the (already pre-inflated) original hull vertices
    `hull_verts_inflated` are contained within it (GH #109).

    For each inflated hull vertex, finds the final-mesh vertex nearest it
    BY DIRECTION from the centroid -- a cheap proxy for "where the
    smoothed surface lies along that same ray from the centroid" -- and
    computes how much farther out the hull vertex is along that ray. The
    whole mesh is grown by the worst (largest) such ratio across all hull
    vertices, plus a small safety `margin`, so the rescaled surface clears
    every hull vertex rather than just meeting it exactly.
    """
    to_hull = hull_verts_inflated - centroid
    dist_hull = np.linalg.norm(to_hull, axis=1)
    unit_hull = to_hull / (dist_hull[:, None] + 1e-12)

    to_mesh = verts - centroid
    dist_mesh = np.linalg.norm(to_mesh, axis=1)
    unit_mesh = to_mesh / (dist_mesh[:, None] + 1e-12)

    nearest = np.argmax(unit_hull @ unit_mesh.T, axis=1)
    ratios = dist_hull / np.maximum(dist_mesh[nearest], 1e-12)

    scale = max(1.0, ratios.max() * margin)
    if scale > 1.0:
        verts = centroid + scale * (verts - centroid)
    return verts


def _catmull_rom_closed(poly, n_per_seg=20, alpha=0.5):
    """Centripetal Catmull-Rom spline through a closed polygon.

    Parameters
    ----------
    poly : ndarray of shape (n, 2)
        Ordered polygon vertices (the curve passes through each of these).
    n_per_seg : int, optional
        Number of sampled points generated per polygon edge. Default 20.
    alpha : float, optional
        Catmull-Rom parameterization exponent; 0.5 gives the "centripetal"
        variant, which never self-intersects/overshoots. Default 0.5.

    Returns
    -------
    ndarray of shape (n * n_per_seg, 2)
        Sampled closed curve; the curve loops back to (approximately)
        ``poly[0]`` after its last sample, without repeating that point
        explicitly.
    """
    P = np.asarray(poly, dtype=float)
    n = len(P)
    out = []
    for idx in range(n):
        p0, p1, p2, p3 = (
            P[(idx - 1) % n],
            P[idx],
            P[(idx + 1) % n],
            P[(idx + 2) % n],
        )

        def tj(ti, a, b):
            return ti + np.linalg.norm(b - a) ** alpha

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        t = np.linspace(t1, t2, n_per_seg, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        out.append(c)
    return np.vstack(out)


def smooth_hull_2d(points, samples_per_edge=20):
    """Build a smooth closed outline from a 2-D point cloud.

    Computes the convex hull of `points` and threads a centripetal
    Catmull-Rom spline (alpha=0.5) through the (already correctly ordered)
    hull vertices, closing the loop.

    Parameters
    ----------
    points : array-like of shape (n, 2)
        Input point cloud. Must contain at least 3 non-collinear points.
    samples_per_edge : int, optional
        Number of curve samples generated per hull edge. Default 20.

    Returns
    -------
    ndarray of shape (h * samples_per_edge, 2)
        Sampled closed outline, where `h` is the number of hull vertices.
        The outline is topologically closed (its last sample connects
        smoothly back to its first) without an explicit duplicate closing
        point.

    Raises
    ------
    ValueError
        If `points` does not have shape (n, 2) with n >= 3, or if the
        points are (nearly) collinear so no 2-D convex hull can be formed.

    Examples
    --------
    >>> import numpy as np
    >>> from hypertools.plot.meshutil import smooth_hull_2d
    >>> pts = np.random.default_rng(0).normal(size=(200, 2))
    >>> curve = smooth_hull_2d(pts)
    >>> curve.shape[1]
    2
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"`points` must have shape (n, 2); got shape {points.shape}"
        )
    if len(points) < 3:
        raise ValueError(
            f"need at least 3 points to form a 2-D convex hull; got {len(points)}"
        )

    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        raise ValueError(
            "could not compute a 2-D convex hull -- points appear to be "
            "collinear or otherwise degenerate"
        ) from exc

    poly = points[hull.vertices]
    if len(poly) < 3:
        raise ValueError(
            "need at least 3 non-collinear points to form a 2-D convex hull"
        )
    return _catmull_rom_closed(poly, n_per_seg=samples_per_edge, alpha=0.5)


def face_normals(verts, faces):
    """Compute unit outward normals for each triangular face.

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    faces : ndarray of shape (f, 3), dtype int
        Triangle indices into `verts`, assumed consistently wound (e.g.
        as returned by :func:`smooth_hull_3d`).

    Returns
    -------
    ndarray of shape (f, 3)
        Unit-length face normals (right-hand rule on each triangle's
        winding order).
    """
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces)
    v0, v1, v2 = (verts[faces[:, k]] for k in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    return fn


def _view_frame(view):
    """Build an orthonormal (view, up, right) frame for a given view vector."""
    v = np.asarray(view, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(v, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(v, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    return v, up, right


def blinn_phong_colors(
    verts,
    faces,
    base_rgb,
    view,
    lightdir=None,
    ambient=0.45,
    diffuse=0.55,
    fill=0.25,
    specular=0.30,
    shininess=48,
):
    """Shade each face with a two-light Blinn-Phong model.

    Uses a key light (derived from the view direction unless `lightdir` is
    given explicitly) plus a weaker fill light from roughly the opposite
    side, so faces angled away from the key light are not rendered fully
    black.

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    faces : ndarray of shape (f, 3), dtype int
    base_rgb : array-like of shape (3,)
        Base surface color as (r, g, b) in [0, 1].
    view : array-like of shape (3,)
        Direction from the scene towards the camera.
    lightdir : array-like of shape (3,), optional
        Explicit key-light direction. If None (default), a key light is
        derived automatically from `view` (offset above and to the side
        of the camera).
    ambient : float, optional
        Ambient light contribution. Default 0.45.
    diffuse : float, optional
        Key-light diffuse contribution. Default 0.55.
    fill : float, optional
        Fill-light diffuse contribution. Default 0.25.
    specular : float, optional
        Specular highlight contribution. Default 0.30.
    shininess : float, optional
        Specular exponent (higher = tighter highlight). Default 48.

    Returns
    -------
    ndarray of shape (f, 4)
        Per-face RGBA color, values clipped to [0, 1]. Alpha is always 1.0
        (callers that need transparency can overwrite the alpha column).
    """
    faces = np.asarray(faces)
    n = face_normals(verts, faces)
    v, up, right = _view_frame(view)

    if lightdir is None:
        key = v + 0.7 * up - 0.5 * right
    else:
        key = np.asarray(lightdir, dtype=float)
    key = key / (np.linalg.norm(key) + 1e-12)

    fill_dir = v - 0.5 * up + 0.6 * right
    fill_dir = fill_dir / (np.linalg.norm(fill_dir) + 1e-12)

    half = key + v
    half = half / (np.linalg.norm(half) + 1e-12)

    lam_key = np.clip(n @ key, 0.0, 1.0)
    lam_fill = np.clip(n @ fill_dir, 0.0, 1.0)
    spec = np.clip(n @ half, 0.0, 1.0) ** shininess

    base = np.asarray(base_rgb, dtype=float)
    rgb = (
        ambient * base
        + diffuse * lam_key[:, None] * base
        + fill * lam_fill[:, None] * base
        + specular * spec[:, None]
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    alpha = np.ones((len(faces), 1))
    return np.concatenate([rgb, alpha], axis=1)


def points_enclosed(points, verts):
    """Boolean mask of which `points` fall within the volume enclosed by a
    `smooth_hull_3d` mesh's vertex set `verts` (GH #109 rendering-fix).

    Containment is tested via the Delaunay triangulation of `verts` itself
    (points on/inside the convex hull of the mesh's own vertices count as
    "enclosed") -- this is deliberately the same style of exact-containment
    test used by the smoothed-mesh containment tests in
    ``tests/test_meshutil.py``, just exposed as a reusable primitive.

    This is used by the plotly backend to work around two related, upstream
    WebGL rendering defects it cannot otherwise avoid: (1) ``Scatter3d``
    marker points enclosed by an opaque ``Mesh3d`` surface are not always
    correctly depth-composited by plotly's renderer and can visibly "punch
    through" the mesh; and (2) when two datasets' surfaces geometrically
    intersect, the same defect can punch a hole in whichever mesh was drawn
    first, wherever the other mesh's volume encloses it. In both cases the
    enclosed geometry (marker points; the other mesh's overlapping faces)
    is simply not drawn there instead, since it would be hidden by an
    opaque enclosing surface anyway.

    Parameters
    ----------
    points : array-like of shape (n, 3)
    verts : array-like of shape (m, 3)
        Vertex coordinates of the enclosing mesh (e.g. from
        :func:`smooth_hull_3d`).

    Returns
    -------
    ndarray of shape (n,), dtype bool
        True where the corresponding point is enclosed.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    tri = Delaunay(np.asarray(verts, dtype=float))
    return tri.find_simplex(points) >= 0


def backface_cull(verts, faces, view_vector, threshold=-0.05):
    """Compute a boolean keep-mask that culls faces pointing away from the
    viewer.

    Matplotlib's 3-D ``Poly3DCollection`` does not depth-sort/cull faces on
    its own, so translucent surfaces show interior back-faces through the
    front; this mask lets a caller drop (or otherwise flag) those back-faces
    before rendering.

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    faces : ndarray of shape (f, 3), dtype int
    view_vector : array-like of shape (3,)
        Direction from the scene towards the camera.
    threshold : float, optional
        Faces whose normal has a dot product with the (normalized) view
        vector greater than `threshold` are kept. A small negative
        threshold (rather than 0) keeps faces that are exactly edge-on to
        the camera, avoiding thin gaps at silhouette edges. Default -0.05.

    Returns
    -------
    ndarray of shape (f,), dtype bool
        True for faces to keep (front-facing), False for faces to cull.
    """
    fn = face_normals(verts, faces)
    v = np.asarray(view_vector, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12)
    return (fn @ v) > threshold
