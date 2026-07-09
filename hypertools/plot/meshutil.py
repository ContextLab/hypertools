"""Pure-geometry mesh utilities for smooth convex-hull surfaces (GH #109).

This module builds smooth 3-D "blob" surfaces and smooth 2-D outlines from
raw point clouds by inflating and refining their convex hulls, plus a small
set of shading/culling helpers used to render those surfaces.

Pipeline (3-D): ``ConvexHull`` -> outward-oriented triangle mesh -> pre-inflate
about the centroid -> interleaved rounds of [midpoint 1->4 subdivision, then
Taubin lambda/mu smoothing, then a hull-hugging pull-back of any vertex that
smoothing pulled inside the original hull]. Interleaving subdivision with
smoothing (rather than subdividing everything up front and smoothing once at
the end) avoids pinched creases at the original hull vertices; the pull-back
(Task M1b, "tighter hulls") makes the mesh hug the data's own convex hull BY
CONSTRUCTION rather than relying solely on a uniform post-hoc regrow.

Pipeline (2-D): ``ConvexHull`` vertices (already ordered) -> centripetal
Catmull-Rom spline (alpha=0.5) through the closed polygon.

This module is intentionally dependency-light (numpy + scipy only) and does
not import anything from :mod:`hypertools.plot.plot` or the plot backends --
it is a leaf geometry module that those higher-level modules import from.
"""
import warnings

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from scipy.spatial.distance import cdist

__all__ = [
    "smooth_hull_3d",
    "smooth_hull_2d",
    "face_normals",
    "vertex_normals",
    "blinn_phong_colors",
    "blinn_phong_vertex_colors",
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


def smooth_hull_3d(
    points, rounds=3, taubin_iters=8, lam=0.5, mu=-0.53, pre_inflate=1.0,
    hull_blend=0.85,
):
    """Build a smooth, outward-oriented triangle mesh from a 3-D point cloud.

    Computes the convex hull of `points`, optionally pre-inflates it about
    its centroid, then performs `rounds` interleaved rounds of midpoint
    (1 -> 4) subdivision followed by Taubin lambda/mu smoothing. Interleaving
    (rather than subdividing fully and then smoothing once) avoids pinched
    creases at the original hull vertices.

    Task M1b (2026-07-06, maintainer feedback: "the convex hulls need to be
    tighter; they don't hug the observations as closely as they should"):
    after every round's Taubin pass, any mesh vertex that has fallen INSIDE
    the original (unsmoothed) hull of `points` is pulled back out towards
    that hull's surface -- see :func:`_pull_back_to_hull`/`hull_blend`
    below. This makes the mesh hug the data BY CONSTRUCTION, rather than
    (as Task M1 did) shrinking uniformly via Taubin and then uniformly
    re-growing the whole mesh to restore containment, which compensates
    Taubin's corner-shrink by ballooning the already-tight flat faces
    outward instead.

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
        subdivision/smoothing begins. Default 1.0 (no blanket padding) --
        maintainer feedback (2026-07-06) reported that a default of 1.15
        (the previous default) made surfaces visibly balloon past the data
        they were meant to hug, in some cases past the plotted axes cube
        entirely. Any shrinkage Taubin smoothing introduces is instead
        recovered by the hull-hugging pull-back below (and, as a final
        bounded safety net, :func:`_rescale_for_containment`) targeting
        the *actual input points*, not this pre-inflation.
    hull_blend : float, optional
        How aggressively interior mesh vertices are pulled back towards
        the original hull's surface after each smoothing round, in
        ``[0, 1]``. 0 disables pull-back entirely (Task M1 behavior); 1
        snaps pulled vertices exactly onto the original hull surface
        (perfectly tight, but polyhedral/faceted at corners); the default
        0.85 keeps most of the tightness while leaving enough slack for
        Taubin's final light touch-up pass to round corners smoothly.
        Applied progressively (increasing across rounds, lighter on the
        final touch-up pass) so early, coarse rounds don't lock in sharp
        facets that later subdivision could otherwise still soften. See
        :func:`_pull_back_to_hull`.

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
    if pre_inflate != 1.0:
        verts = centroid + pre_inflate * (verts - centroid)

    # Task M1b ("tight hulls, by construction"): the ORIGINAL (unsmoothed,
    # un-inflated) hull of the actual input points is the surface every
    # smoothed vertex should hug -- its halfspace (facet normal/offset)
    # equations are computed once, up front, and reused by every pull-back
    # below (both the per-round ones and the final touch-up), all rooted
    # at this same `centroid`.
    orig_hull_eqs = ConvexHull(points).equations

    for round_idx in range(rounds):
        verts, faces = _subdivide(verts, faces)
        verts = _taubin_smooth(verts, faces, iterations=taubin_iters, lam=lam, mu=mu)
        if hull_blend > 0:
            # ramp `hull_blend` up across rounds: a full-strength pull-back
            # on an early, coarse (barely-subdivided) mesh would lock in
            # sharp facets that later subdivision/smoothing could otherwise
            # still round off, so early rounds pull back lightly and only
            # the final round applies the full requested `hull_blend`.
            round_blend = hull_blend * (round_idx + 1) / rounds
            verts = _pull_back_to_hull(verts, orig_hull_eqs, centroid, round_blend)

    if hull_blend > 0 and rounds > 0:
        # The final round's pull-back can leave faint facets at vertices
        # it snapped onto the hull surface. A short, light Taubin touch-up
        # rounds those off -- but a touch-up alone would let the mesh sink
        # back inside the hull, undoing the tightness just gained, so a
        # second, gentler pull-back (half strength) follows it to restore
        # snugness without re-introducing hard facets.
        verts = _taubin_smooth(verts, faces, iterations=2, lam=lam, mu=mu)
        verts = _pull_back_to_hull(verts, orig_hull_eqs, centroid, 0.5 * hull_blend)

    # GH #109 (maintainer feedback 2026-07-06, "tight hulls"): Taubin
    # smoothing has a net shrinking effect that is NOT proportional across
    # hull sizes -- a small, sparse hull (few input points -> few, often
    # widely-spaced hull vertices) loses proportionally much MORE of its
    # "bulge" to the same fixed `taubin_iters`/`rounds` than a large, dense
    # one does. Rather than compensate with a blanket `pre_inflate` (which
    # over-inflates the common case to cover the rare sparse-hull case --
    # exactly what made surfaces balloon past the data), post-hoc rescale
    # the mesh (uniformly about its centroid, GROW-ONLY, mathematically
    # bounded to at most `_RESCALE_CAP`) so it recovers containment of the
    # ACTUAL input points, not any inflated proxy for them.
    verts = _rescale_for_containment(verts, points, centroid)

    contained_frac = points_enclosed(points, verts).mean()
    if contained_frac < 0.99:
        warnings.warn(
            "smooth_hull_3d: the smoothed surface contains only "
            f"{100 * contained_frac:.0f}% of the input points (target "
            "99%) even after rescaling for containment (capped at "
            f"{_RESCALE_CAP:.0%} growth); it may visibly fail to enclose "
            "some of the data."
        )

    return verts, faces


_RESCALE_CAP = 3.0


def _ray_exit_distance(hull_eqs, centroid, dirs):
    """Exact ray-vs-convex-polytope exit distance, reused by both
    :func:`_rescale_for_containment` (Task M1) and :func:`_pull_back_to_hull`
    (Task M1b).

    For each ray ``x = t * dirs[k]`` (``t >= 0``, coordinates relative to
    `centroid`), returns the `t` at which the ray exits the convex polytope
    described by the halfspace (facet normal/offset) equations `hull_eqs`
    (e.g. ``ConvexHull(...).equations``, shape ``(n_facets, 4)``): the
    smallest ``offset / (normal . dir)`` over every facet the ray is moving
    TOWARDS (``normal . dir > 0``); facets the ray moves away from impose
    no constraint. This is the standard exact formula -- not a proxy -- for
    where a ray leaves a convex body, and matches precisely what a
    Delaunay/ConvexHull containment test (e.g. :func:`points_enclosed`)
    checks.

    Parameters
    ----------
    hull_eqs : ndarray of shape (n_facets, 4)
        Halfspace equations of a convex hull (``ConvexHull.equations``):
        row k is ``[normals[k, 0], normals[k, 1], normals[k, 2], offset_k]``
        such that the hull is ``{x : normals[k] . x + offset_k <= 0}``.
    centroid : ndarray of shape (3,)
        Point the rays originate from (must be strictly inside the hull
        described by `hull_eqs` for the result to be finite/positive).
    dirs : ndarray of shape (n, 3)
        Unit ray directions, relative to `centroid`.

    Returns
    -------
    ndarray of shape (n,)
        Exit distance (``t``) along each ray, floored at ``1e-12`` to avoid
        returning exactly zero (e.g. for a degenerate all-zero direction).
    """
    normals = hull_eqs[:, :3]
    offsets = -hull_eqs[:, 3] - normals @ centroid
    proj = dirs @ normals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        candidates = np.where(proj > 1e-12, offsets[None, :] / proj, np.inf)
    return np.maximum(candidates.min(axis=1), 1e-12)


def _pull_back_to_hull(verts, hull_eqs, centroid, blend):
    """Task M1b ("tight hulls, by construction"): pull mesh vertices that
    fell INSIDE the original hull back out towards that hull's surface.

    For each vertex, casts a ray from `centroid` through the vertex (the
    same exact halfspace ray-exit machinery :func:`_rescale_for_containment`
    uses, via :func:`_ray_exit_distance`) and, if the vertex's own distance
    from `centroid` is currently SHORTER than where that ray would exit the
    original hull described by `hull_eqs`, moves the vertex to
    ``centroid + dir * (blend * d_hull + (1 - blend) * d_current)``. This
    is exactly the "shrink then blend back towards the boundary" that makes
    the mesh hug the hull directly, rather than (as the uniform post-hoc
    rescale alone does) growing the ENTIRE mesh -- including faces that were
    already snug -- to compensate for a few corners Taubin smoothing pulled
    in the most.

    Vertices already outside (or exactly on) the original hull along their
    own ray (``d_current >= d_hull``) are left untouched: pulling them
    "back" would mean pushing them further out, past the hull, which is not
    what this step is for (containment growth, if still needed after this,
    is `_rescale_for_containment`'s job).

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    hull_eqs : ndarray of shape (n_facets, 4)
        Halfspace equations of the ORIGINAL (unsmoothed) hull to hug --
        see :func:`_ray_exit_distance`.
    centroid : ndarray of shape (3,)
    blend : float
        In ``[0, 1]``: 0 leaves every vertex untouched, 1 snaps every
        interior vertex exactly onto the original hull surface.

    Returns
    -------
    ndarray of shape (n, 3)
        Vertex positions after pull-back.
    """
    to_v = verts - centroid
    r = np.linalg.norm(to_v, axis=1)
    nonzero = r > 1e-12
    dirs = np.zeros_like(to_v)
    dirs[nonzero] = to_v[nonzero] / r[nonzero, None]

    d_hull = _ray_exit_distance(hull_eqs, centroid, dirs)
    inside = nonzero & (r < d_hull)
    if not inside.any():
        return verts

    new_r = r.copy()
    new_r[inside] = blend * d_hull[inside] + (1 - blend) * r[inside]
    out = verts.copy()
    out[inside] = centroid + dirs[inside] * new_r[inside, None]
    return out


def _rescale_for_containment(
    verts, points, centroid, target_frac=0.99, tol_frac=0.005, cap=_RESCALE_CAP,
):
    """Uniformly rescale `verts` about `centroid` (growing only, never
    shrinking, and capped at `cap`) so at least `target_frac` of the
    ACTUAL input `points` fall within it (GH #109, "tight hulls").

    For each input point, casts a ray from `centroid` through the point
    and finds exactly where that ray exits the convex hull of `verts` --
    via the hull's halfspace (facet normal/offset) representation, i.e.
    ``min`` over facets the ray is moving towards of ``offset / (normal
    . ray_direction)`` -- the standard, exact ray-vs-convex-polytope exit
    distance. This is precisely the same containment notion
    :func:`points_enclosed` tests (a point is "in" iff it falls inside
    the convex hull of `verts`), so the ratio of the point's own distance
    from the centroid to this exit distance is exactly the uniform scale
    that would place the mesh boundary AT that point along its ray. A
    small absolute margin (`tol_frac` of the cloud's extent) is added on
    top of the point's distance so it ends up genuinely, strictly inside
    -- not merely touching the boundary: :func:`points_enclosed`'s
    Delaunay test is a hard binary in/out check with no notion of "close
    enough", so a point placed EXACTLY on the boundary (zero margin) is a
    floating-point/triangulation-round-off coin flip, while one placed
    `tol_frac` inside is not.

    This replaces two flawed earlier proxies for that same exit distance:
    (1) matching each hull vertex to its nearest-BY-ANGLE mesh vertex and
    using that single vertex's radial distance -- unstable, since two
    directions can be almost perfectly aligned (cosine similarity ~0.999)
    while the nearest-angle vertex still sits well short of the surface's
    true reach along that exact ray; and (2) a per-ray support-function
    value (farthest mesh vertex projected ONTO the ray direction) --
    mathematically tempting but WRONG, because the vertex achieving that
    projection maximum generally lies off to the side of the ray, not on
    it (support-function value >= true ray-exit distance always, for any
    convex body not centered on that vertex), so it systematically
    UNDER-estimates how much growth is truly needed and left points
    falsely "already contained" that the rescaled mesh did not actually
    reach. Both proxies compounded with a blanket `pre_inflate` to
    balloon the mesh far past the data (the "explosion" maintainer
    feedback reported, 2026-07-06) or, in the second proxy's case, to
    silently under-cover. The exact halfspace-based ray exit has neither
    failure mode, and the result is additionally capped at `cap` so a
    single degenerate ray can never blow the whole mesh up unboundedly.

    `cap` = 3.0, not the tighter ~1.1 one might initially expect: measured
    across n=4..2000 (see docs/superpowers/plans/2026-07-06-morph-
    animation.md Task M1 report), well-SAMPLED clouds (n >~ 100, several
    hundred+ hull vertices) need at most ~1.15x-1.25x to hit
    `target_frac` -- this cap essentially never binds for them, and it is
    THIS regime (not the cap) where the "explosion" bug lived (a WRONG,
    overestimated ratio from the flawed proxy above, now fixed). But very
    SPARSE hulls (n=4-20, few/no interior points -- literally every point
    close to a hull vertex) genuinely, unavoidably need much more: Taubin
    smoothing's net shrinkage is proportionally far larger on a coarse,
    few-vertex hull than a dense one (the very asymmetry `pre_inflate`
    used to paper over with a blanket constant) -- a bare (degenerate)
    4-point tetrahedron needs ~2.9x, a 5-point cloud ~2.3x, to recover 99%
    containment. A hard 1.1 cap would silently regress that long-standing
    guarantee (GH #109 round 1; `TestSmoothHull3DTinyCloudContainment`)
    back to the "5-point cloud contains as few as ~1/5 of its own points"
    bug it originally fixed. 3.0 is a generously-calibrated but still
    HARD, FINITE ceiling -- unlike the old code's literally unbounded
    `ratios.max()` -- chosen so genuinely sparse hulls (down to the
    smallest possible 4-point hull) keep their historical containment
    guarantee, while well-sampled data (where the real explosion bug
    lived) is governed almost entirely by the corrected computation
    above, not by where this ceiling happens to sit.

    Only `target_frac` of points need to be covered (rather than all of
    them) so that a single outlier point cannot force an outsized grow;
    remaining shortfall (if any, given the `cap`) is reported by the
    caller's containment warning.
    """
    extent = np.ptp(points, axis=0).mean()
    tol_abs = tol_frac * extent

    to_pts = points - centroid
    r_pts = np.linalg.norm(to_pts, axis=1)
    nonzero = r_pts > 1e-12
    unit_pts = np.zeros_like(to_pts)
    unit_pts[nonzero] = to_pts[nonzero] / r_pts[nonzero, None]

    # Halfspace (facet normal/offset) representation of the mesh's own
    # convex hull, centered on `centroid`: this is exactly the convex body
    # `points_enclosed` tests membership against (via Delaunay(verts)).
    mesh_hull = ConvexHull(verts)
    exit_dist = _ray_exit_distance(mesh_hull.equations, centroid, unit_pts)

    needed_scale = np.ones(len(points))
    needed_scale[nonzero] = (
        (r_pts[nonzero] + tol_abs) / exit_dist[nonzero]
    )

    n = len(points)
    k = min(n, int(np.ceil(target_frac * n)))
    required = np.partition(needed_scale, k - 1)[k - 1]

    scale = min(max(1.0, required), cap)
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


def vertex_normals(verts, faces):
    """Compute unit per-vertex normals as the (unweighted) average of each
    vertex's adjacent face normals.

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    faces : ndarray of shape (f, 3), dtype int
        Triangle indices into `verts`, assumed consistently wound (e.g. as
        returned by :func:`smooth_hull_3d`).

    Returns
    -------
    ndarray of shape (n, 3)
        Unit-length per-vertex normals. Vertices with no adjacent faces
        (shouldn't occur for a real mesh) get an arbitrary unit normal
        rather than a division-by-zero NaN.
    """
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces)
    fn = face_normals(verts, faces)
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    degenerate = (norms < 1e-12).ravel()
    norms[degenerate, 0] = 1.0
    vn[degenerate] = np.array([0.0, 0.0, 1.0])
    return vn / norms


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


def _blinn_phong_shade(
    normals,
    base_rgb,
    view,
    lightdir=None,
    ambient=0.45,
    diffuse=0.55,
    fill=0.25,
    specular=0.30,
    shininess=48,
):
    """Shared two-light Blinn-Phong math for :func:`blinn_phong_colors`
    (per-face) and :func:`blinn_phong_vertex_colors` (per-vertex): given a
    (k, 3) array of unit normals (one per face, or one per vertex), returns
    a (k, 4) RGBA array. See :func:`blinn_phong_colors` for parameter docs.
    """
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

    lam_key = np.clip(normals @ key, 0.0, 1.0)
    lam_fill = np.clip(normals @ fill_dir, 0.0, 1.0)
    spec = np.clip(normals @ half, 0.0, 1.0) ** shininess

    base = np.asarray(base_rgb, dtype=float)
    rgb = (
        ambient * base
        + diffuse * lam_key[:, None] * base
        + fill * lam_fill[:, None] * base
        + specular * spec[:, None]
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    alpha = np.ones((len(normals), 1))
    return np.concatenate([rgb, alpha], axis=1)


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
    n = face_normals(verts, np.asarray(faces))
    return _blinn_phong_shade(
        n, base_rgb, view, lightdir=lightdir, ambient=ambient, diffuse=diffuse,
        fill=fill, specular=specular, shininess=shininess,
    )


def blinn_phong_vertex_colors(
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
    """Per-VERTEX variant of :func:`blinn_phong_colors` (GH #109 round 3).

    Identical two-light Blinn-Phong model, but shaded at each mesh VERTEX
    (using its averaged-adjacent-face normal -- see :func:`vertex_normals`)
    rather than at each face.

    This is what makes plotly's ``Mesh3d`` ``vertexcolor`` workaround for
    the double-winding self-culling fix (see
    ``plotly_backend._mesh3d_trace``) actually work: both windings of a
    doubled face share the SAME three vertex indices, so with per-vertex
    colors they are colored identically. A per-FACE color, by contrast, is
    computed from each face's own (possibly reversed) normal and so differs
    between the two winding copies -- the reversed copy's normal points
    away from the key light wherever the original copy's points towards it,
    rendering it dark/black and producing the large jagged dark patches
    this function fixes.

    Parameters
    ----------
    verts : ndarray of shape (n, 3)
    faces : ndarray of shape (f, 3), dtype int
    base_rgb, view, lightdir, ambient, diffuse, fill, specular, shininess
        See :func:`blinn_phong_colors`.

    Returns
    -------
    ndarray of shape (n, 4)
        Per-vertex RGBA color, values clipped to [0, 1]. Alpha is always
        1.0 (callers that need transparency can overwrite the alpha
        column).
    """
    vn = vertex_normals(verts, faces)
    return _blinn_phong_shade(
        vn, base_rgb, view, lightdir=lightdir, ambient=ambient, diffuse=diffuse,
        fill=fill, specular=specular, shininess=shininess,
    )


def vertex_colors_from_points(verts, points, point_colors, power=2.0, eps=1e-9):
    """Per-vertex base colors as an inverse-distance-weighted average of the
    data points' colors (Shepard's method / IDW).

    For each mesh vertex, its color is a weighted blend of the data
    coordinates' colors, with weight ``1 / distance**power`` -- so the
    NEAREST coordinates dominate a vertex's color and distant ones fall off
    smoothly (with the default ``power=2``). A vertex that coincides with a
    data point takes that point's color exactly. This is what colors a
    ``surface=`` hull to match the hue of the points it encloses, per-vertex,
    instead of painting the whole hull one flat (mean) color.

    Parameters
    ----------
    verts : array-like of shape (V, >=3)
        Mesh vertex coordinates (only the first 3 columns are used).
    points : array-like of shape (P, >=3)
        Data-point coordinates the surface was built from (first 3 columns).
    point_colors : array-like of shape (P, >=3)
        Per-point RGB(A) colors in [0, 1] (only the first 3 channels used).
    power : float, optional
        Inverse-distance exponent (default 2.0; higher = sharper locality).
    eps : float, optional
        Added to ``distance**power`` before inverting so a zero distance
        (a vertex exactly on a point) yields a large-but-finite weight
        rather than a divide-by-zero. Default 1e-9.

    Returns
    -------
    ndarray of shape (V, 3)
        Per-vertex RGB in [0, 1].
    """
    verts = np.asarray(verts, dtype=float)[:, :3]
    points = np.asarray(points, dtype=float)[:, :3]
    colors = np.asarray(point_colors, dtype=float)[:, :3]
    if len(points) == 0 or len(colors) == 0:
        raise ValueError('need at least one data point/color to color a surface')
    d = cdist(verts, points)                       # (V, P) euclidean distances
    w = 1.0 / (d ** power + eps)                    # inverse-distance weights
    w /= w.sum(axis=1, keepdims=True)               # normalize per vertex
    return np.clip(w @ colors, 0.0, 1.0)


def face_colors_from_vertex_colors(vertex_colors, faces):
    """Per-face RGB as the mean of each triangle's three vertex colors -- the
    per-face form the matplotlib backend needs (it shades whole faces), from
    the per-vertex colors :func:`vertex_colors_from_points` produces."""
    vertex_colors = np.asarray(vertex_colors, dtype=float)
    faces = np.asarray(faces)
    return vertex_colors[faces].mean(axis=1)


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
