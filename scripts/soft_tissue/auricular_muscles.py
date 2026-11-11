"""The extrinsic auricular muscles as a thickness map in the scalp's layer 3 (envelope.py).

They lie in the musculoaponeurotic layer (temporoparietal fascia / SMAS), deep to the skin (layer 1)
and fat (layer 2), on the loose areolar plane (layer 4) (user, 2026-11-11):
  anterior   temporal fascia / epicranial aponeurosis -> spine of the helix
  superior   epicranial aponeurosis -> upper cranial surface of the auricle
  posterior  mastoid -> lower cranial concha
and they end at the auricle's base, the ring where the pinna meets the head's skin.

Each footprint is drawn on the envelope's uv grid (the canal-axis projection the temporalis map
uses), from the same fans the physics uses (``auricle_sim.place_fans``, superior turned by
``SUPERIOR_TURN_DEG``): from the fan's origin line to the arc of the base outline nearest its
insertion. The belly is thickest in its middle and thin at both attachments (END_THICKNESS) and
toward its edges; the posterior muscle is two rounded fascicles. Thickness per muscle:
``tissue_params.FANS``. The layers above are then built over it (they drape over the bellies).
Drawing and layering only: the fans' fibres carry the force, from origins outside the scene too.
"""

from __future__ import annotations

import numpy as np

END_THICKNESS = 0.35  # at each attachment, as a fraction of the belly
FASCICLES = {"posterior": 2}
FASCICLE_GAP = 0.12  # fraction of the width between fascicles
SAMPLES = 260  # across and along each footprint (fine against the 0.5 mm grid)


def _arc(base, i0, i1):
    """The shorter way round the closed outline ``base`` from index i0 to i1 (inclusive)."""
    n = len(base)
    fwd = [(i0 + k) % n for k in range((i1 - i0) % n + 1)]
    back = [(i0 - k) % n for k in range((i0 - i1) % n + 1)]
    return base[fwd if len(fwd) <= len(back) else back]


def _resample(poly, m):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(poly[:1], m, axis=0)
    q = np.linspace(0.0, s[-1], m)
    return np.stack([np.interp(q, s, poly[:, c]) for c in range(poly.shape[1])], axis=1)


def _widen(poly, width):
    """``poly`` (an arc) lengthened symmetrically about its middle to at least ``width`` mm,
    straight on past its ends."""
    length = np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()
    if length >= width or len(poly) < 2:
        return poly
    extra = 0.5 * (width - length)
    d0 = poly[0] - poly[1]
    d1 = poly[-1] - poly[-2]
    d0 /= max(np.linalg.norm(d0), 1e-9)
    d1 /= max(np.linalg.norm(d1), 1e-9)
    return np.concatenate([[poly[0] + extra * d0], poly, [poly[-1] + extra * d1]])


def footprints(model, base_outline):
    """Per auricular muscle: (name, origin line (S, 3), insertion arc (S, 3), thickness mm), volume mm."""
    import tissue_params as tp

    base = np.asarray(base_outline, float)
    out = []
    for fs in model.fan_sets:
        name = fs.names[0]
        spec = tp.FANS[name]
        if spec.material != "auricular_muscle":
            continue
        o = np.asarray(fs.origin, float)
        ins = model.mid_v[np.asarray(fs.a)]
        j = np.argmin(np.linalg.norm(base[None] - ins[:, None], axis=2), axis=1)
        arc = _widen(_arc(base, int(j[0]), int(j[-1])), spec.insertion_width_mm.nominal)
        out.append((name, _resample(np.stack([o[0], o[-1]]), SAMPLES), _resample(arc, SAMPLES),
                    spec.thickness_mm.nominal))
    return out


def profile(name, m):
    """(m, m) thickness fraction: across (rows) x along (columns, origin -> insertion)."""
    u = np.linspace(0.0, 1.0, m)
    k = FASCICLES.get(name, 1)
    if k == 1:
        across = 0.35 + 0.65 * np.sqrt(np.clip(1.0 - (2 * u - 1) ** 2, 0.0, 1.0))
    else:
        across = np.zeros(m)
        w = (1.0 - FASCICLE_GAP * (k - 1)) / k
        for f in range(k):
            a = f * (w + FASCICLE_GAP)
            x = (u - a) / w
            inside = (x >= 0) & (x <= 1)
            across[inside] = np.sqrt(np.clip(1.0 - (2 * x[inside] - 1) ** 2, 0.0, 1.0))
    along = END_THICKNESS + (1 - END_THICKNESS) * np.sin(np.pi * u) ** 0.6
    return across[:, None] * along[None, :]


def muscle_map(env, base_outline, model=None, log=print):
    """(thickness map on the envelope's uv grid, its origin): the auricular muscles' bellies."""
    import envelope as ev
    from scipy import ndimage

    if model is None:
        import auricle_sim as asim

        model = asim.AuricleModel(env.vol, layers=1, lobule=True, log=lambda *a: None)
    f = env.vol.frame
    lo = env.near_uv.reshape(-1, 2).min(axis=0) - 1.0
    hi = env.near_uv.reshape(-1, 2).max(axis=0) + 1.0
    n = np.ceil((hi - lo) / ev.UV_MM).astype(int) + 1
    t = np.zeros(n, np.float32)
    for name, origin, arc, h in footprints(model, base_outline):
        m = len(origin)
        v = np.linspace(0.0, 1.0, m)
        pts = origin[:, None, :] * (1 - v)[None, :, None] + arc[:, None, :] * v[None, :, None]  # (across, along, 3)
        uv = f.uv(pts.reshape(-1, 3))
        g = np.round((uv - lo) / ev.UV_MM).astype(int)
        thick = (h * profile(name, m)).ravel()
        ok = (g[:, 0] >= 0) & (g[:, 0] < n[0]) & (g[:, 1] >= 0) & (g[:, 1] < n[1])
        np.maximum.at(t, (g[ok, 0], g[ok, 1]), thick[ok])
        log(f"auricular muscle {name}: {h:g} mm belly, {ok.mean():.0%} of it on the scene's grid")
    t = ndimage.gaussian_filter(t, 0.5 / ev.UV_MM * 0.6)  # soften the raster, not the shape
    return t.astype(np.float32), lo
