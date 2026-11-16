"""Rigid CT -> MRI registration on the temporal bones (torch; GPU when one is free).

What the first pair taught (CT "Bilateral Dehiscence 0.4", MRI "C+ AX 3D FIESTA"):

* Mutual information fails. A bone-kernel CT has no soft-tissue contrast (the brain is
  noise), and FIESTA shows bone, air and brain all dark, so the joint histogram has
  little to work with. Mattes MI, normalized MI and MI with padding each found a
  different wrong pose.
* Both scans are thin axial slabs (52 and 61 mm) tilted ~41 degrees to each other, so
  only part of the CT lies inside the MRI. Any score that is averaged over the overlap
  rewards poses with little overlap.
* Their scanner coordinates are 240 mm apart, so nothing can be assumed about the start.

So the score is normalized gradient fields (Haber & Modersitzki 2006): at the CT's
strongest edges, how well the edge direction (rotated into the MRI) lines up with
the MRI's, sign ignored, since bone is bright on CT and dark on MRI. It is summed
over all CT edge samples and divided by their number, so samples outside the MRI
earn nothing instead of being averaged away. Only edges in a 44 mm box around each
temporal bone count: the labyrinth, IAC, ossicles, air cells and petrous ridge. The
head's outline fits a wrong 30-degree tilt almost as well as the right one; the
inner ear does not.

1. Brute force at 3 mm: every rotation in a grid (pitch +-60 deg in 10 deg steps,
   roll and yaw -10/0/10) times every 3 mm position of the CT's middle in and
   around the MRI. ~24 M poses, ~7 s on an RTX 5090. Without a free GPU: 6 mm steps,
   no roll or yaw in the grid, 8 starts; the first pair took 74 s on the CPU, same pose.
2. The best distinct poses are refined with Powell's method at 3, 2, 1 mm; the best
   at 1 mm is refined at the MRI's resolution (0.5 mm for CISS / FIESTA).

Tested: the FIESTA pair (9 of 16 starts end within 0.1 mm of each other, score 0.41
vs 0.25 for the next pose), the same MRI blurred to 0.9 x 0.9 x 3 mm with T1-like
contrast (same pose within 0.12 mm), and the CT cut to the left ear only.

Pose: angles in degrees (pitch about x, roll about y, yaw about z; LPS axes;
R = Rz @ Ry @ Rx) and the translation t of the CT's middle c:  q = R (p - c) + c + t,
p a CT point, q the same point in the MRI. That is ITK's CT -> MRI resampling transform.
"""
from __future__ import annotations

import time

import numpy as np

EDGE_FRACTION = 0.25  # samples: the strongest quarter of the CT's edges
MIN_OVERLAP = 0.20  # a pose must keep this much of the samples inside the MRI
EAR_HALF_MM = 22.0  # half side of the box around each temporal bone
SEARCH_MM = 3.0
SEARCH_MARGIN_MM = 20.0  # the CT's middle may sit outside a thin MRI slab
PITCH_DEG = np.arange(-60.0, 60.1, 10.0)
SIDE_DEG = (-10.0, 0.0, 10.0)  # roll and yaw
STARTS = 24  # distinct poses out of the search
STARTS_CPU = 8  # refining is ~8 s a start on a CPU (the first pair: 4 min with 24, same answer)
DISTINCT_MM = 4.0
MAX_POINTS = 250000
CHUNK_SAMPLES = 1.2e7  # (pose, edge sample) pairs scored per batch
GPU_MIN_FREE = 0.75 * 1024 ** 3
GPU_FIXED_BYTES = 0.5 * 1024 ** 3  # volumes, CUDA's own use
GPU_BYTES_PER_SAMPLE = 80  # with headroom: 0.87 GB peak at 1.2e7


def rotation(angles_deg):
    """(..., 3) degrees -> (..., 3, 3). R = Rz(yaw) @ Ry(roll) @ Rx(pitch)."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    cx, cy, cz = np.cos(a[..., 0]), np.cos(a[..., 1]), np.cos(a[..., 2])
    sx, sy, sz = np.sin(a[..., 0]), np.sin(a[..., 1]), np.sin(a[..., 2])
    one, zero = np.ones_like(cx), np.zeros_like(cx)
    rx = np.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).reshape(a.shape[:-1] + (3, 3))
    ry = np.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(a.shape[:-1] + (3, 3))
    rz = np.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).reshape(a.shape[:-1] + (3, 3))
    return rz @ ry @ rx


def pick_device(torch):
    """(device, samples scored per batch). The GPU is used whenever ~0.75 GB is free (e.g. the
    lab's RTX 5060 with the simulator running); the batch shrinks to what is free. Measured on
    the first pair: 0.87 GB peak with full batches, of which ~0.3 GB is the MRI's gradients."""
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            free = 0
        if free >= GPU_MIN_FREE:
            room = free - GPU_FIXED_BYTES
            return torch.device("cuda"), int(np.clip(room / GPU_BYTES_PER_SAMPLE, 1e6, CHUNK_SAMPLES))
    return torch.device("cpu"), CHUNK_SAMPLES


def mapped(params, center, points):
    return (points - center) @ rotation(params[:3]).T + center + params[3:]


class Level:
    """CT edge samples and the MRI's gradient at one resolution."""

    def __init__(self, torch, device, ct_hu, ct_valid, mri01, spacing_mm, isotropic, boxes=None,
                 max_points=MAX_POINTS, seed=0, chunk_samples=CHUNK_SAMPLES):
        """ct_hu: CT in HU; ct_valid: 1 where the CT has data; mri01: MRI scaled 0..1;
        boxes: [(center_mm, half_side_mm)] to sample only there; None = the whole CT."""
        import SimpleITK as sitk
        from scipy import ndimage

        self.torch, self.device, self.spacing = torch, device, spacing_mm
        self.chunk_samples = chunk_samples
        fixed = isotropic(ct_hu, spacing_mm, -1000.0)
        valid = sitk.Resample(sitk.Cast(ct_valid, sitk.sitkFloat32), fixed, sitk.Transform(), sitk.sitkLinear, 0.0)
        values = sitk.GetArrayViewFromImage(fixed).astype(np.float32)
        direction = np.array(fixed.GetDirection()).reshape(3, 3)
        gz, gy, gx = np.gradient(values, spacing_mm)
        grad = np.stack([gx, gy, gz], -1) @ direction.T  # physical x, y, z
        magnitude = np.linalg.norm(grad, axis=-1)
        keep = ndimage.binary_erosion(sitk.GetArrayViewFromImage(valid) > 0.99, iterations=2)  # not the rim
        index_zyx = np.argwhere(keep)
        if boxes:
            physical = np.array(fixed.GetOrigin()) + (index_zyx[:, ::-1] * spacing_mm) @ direction.T
            in_box = np.zeros(len(index_zyx), bool)
            for box_center, half in boxes:
                in_box |= (np.abs(physical - np.asarray(box_center)) <= half).all(1)
            index_zyx = index_zyx[in_box]
        strength = magnitude[tuple(index_zyx.T)]
        index_zyx = index_zyx[strength >= np.quantile(strength, 1.0 - EDGE_FRACTION)]
        if len(index_zyx) > max_points:
            rng = np.random.default_rng(seed)
            index_zyx = index_zyx[np.sort(rng.choice(len(index_zyx), max_points, replace=False))]
        points = np.array(fixed.GetOrigin()) + (index_zyx[:, ::-1] * spacing_mm) @ direction.T
        g = grad[tuple(index_zyx.T)]
        self.n = len(points)
        self.points_np = points
        self.points = torch.as_tensor(points, dtype=torch.float32, device=device)
        self.ct_grad = torch.as_tensor(g / np.linalg.norm(g, axis=1, keepdims=True), dtype=torch.float32,
                                       device=device)

        native = float(np.min(mri01.GetSpacing()))
        moving = isotropic(mri01, spacing_mm, 0.0) if spacing_mm > native + 1e-6 else mri01
        array = sitk.GetArrayFromImage(moving).astype(np.float32)
        m_dir = np.array(moving.GetDirection()).reshape(3, 3)
        gz, gy, gx = np.gradient(array, *np.array(moving.GetSpacing())[::-1])
        mgrad = np.stack([gx, gy, gz], -1) @ m_dir.T
        mag = np.linalg.norm(mgrad, axis=-1)
        self.eps2 = float(np.median(mag[array > 0.05]) * 0.5) ** 2  # the MRI's noise floor
        self.volume = torch.as_tensor(np.moveaxis(mgrad, -1, 0).astype(np.float32), device=device)[None]
        to_index = np.linalg.inv(m_dir @ np.diag(moving.GetSpacing()))
        size = np.array(moving.GetSize(), dtype=np.float64)
        a = np.diag(2.0 / np.maximum(size - 1, 1)) @ to_index  # physical -> grid_sample's [-1, 1]
        self.norm_a = torch.as_tensor(a, dtype=torch.float32, device=device)
        self.norm_b = torch.as_tensor(-a @ np.array(moving.GetOrigin()) - 1.0, dtype=torch.float32, device=device)

    def score(self, rotations, translations, center):
        """Score for B poses (rotations (B,3,3), translations (B,3)). Returns (score, overlap)."""
        torch = self.torch
        rot = torch.as_tensor(np.asarray(rotations), dtype=torch.float32, device=self.device)
        tr = torch.as_tensor(np.asarray(translations), dtype=torch.float32, device=self.device)
        c = torch.as_tensor(np.asarray(center), dtype=torch.float32, device=self.device)
        relative = self.points - c
        scores, overlaps = [], []
        step = max(1, int(self.chunk_samples // self.n))
        with torch.no_grad():
            for s in range(0, len(rot), step):
                r, t = rot[s:s + step], tr[s:s + step]
                b = len(r)
                # norm = A (R (p - c) + c + t) + b0
                norm = torch.einsum("bij,nj->bni", self.norm_a @ r, relative) + \
                    ((c + t) @ self.norm_a.T + self.norm_b)[:, None, :]
                inside = (norm.abs() <= 1.0).all(-1)
                g = torch.nn.functional.grid_sample(self.volume, norm.view(1, b, self.n, 1, 3), mode="bilinear",
                                                    padding_mode="border", align_corners=True)
                g = g.view(3, b, self.n).permute(1, 2, 0)
                dot = (torch.einsum("bij,nj->bni", r, self.ct_grad) * g).sum(-1)
                cos2 = dot * dot / ((g * g).sum(-1) + self.eps2)
                overlap = inside.float().mean(1)
                value = (cos2 * inside).sum(1) / float(self.n)
                scores.append(torch.where(overlap >= MIN_OVERLAP, value, torch.zeros_like(value)).cpu())
                overlaps.append(overlap.cpu())
        return torch.cat(scores).numpy(), torch.cat(overlaps).numpy()

    def score_pose(self, params, center):
        params = np.asarray(params, dtype=np.float64)
        value, overlap = self.score(rotation(params[:3])[None], params[None, 3:], center)
        return float(value[0]), float(overlap[0])


def refine(level, params, center, step_deg, step_mm, max_evals=400):
    """Powell's method from params; steps are the first moves in degrees and mm."""
    from scipy.optimize import minimize

    scale = np.array([step_deg] * 3 + [step_mm] * 3)
    start = np.asarray(params, dtype=np.float64)
    result = minimize(lambda x: -level.score_pose(start + x * scale, center)[0], np.zeros(6), method="Powell",
                      options={"xtol": 0.02, "ftol": 1e-5, "maxfev": max_evals, "direc": np.eye(6)})
    return start + result.x * scale, -float(result.fun)


def register(ct_hu, ct_valid, mri01, isotropic, ears, progress=lambda f, t: None, log=print):
    """ct_hu: CT in HU; ct_valid: 1 where the CT has data; mri01: MRI scaled 0..1;
    ears: temporal bone centers in the CT (LPS mm). Returns (params, center, record)."""
    import torch

    device, chunk = pick_device(torch)
    on_gpu = device.type == "cuda"
    log(f"registration on {device}" + (f" ({chunk / 1e6:.1f} M samples a batch)" if on_gpu else ""))
    timings = {}
    t0 = time.time()
    size = np.array(ct_hu.GetSize())
    center = np.array(ct_hu.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2.0).tolist()))
    boxes = [(np.asarray(e, dtype=np.float64), EAR_HALF_MM) for e in ears]
    finest = float(np.clip(np.sort(mri01.GetSpacing())[1], 0.5, 1.5))  # middle spacing: slices may be thick
    spacings = [SEARCH_MM, 2.0, 1.0] + ([finest] if finest < 1.0 - 1e-6 else [])
    levels = {s: Level(torch, device, ct_hu, ct_valid, mri01, s, isotropic, boxes=boxes, chunk_samples=chunk)
              for s in spacings}
    timings["levels_s"] = time.time() - t0

    # ---- brute force
    t1 = time.time()
    level = levels[SEARCH_MM]
    msize = np.array(mri01.GetSize())
    corners = np.array([mri01.TransformContinuousIndexToPhysicalPoint([float(x), float(y), float(z)])
                        for x in (-0.5, msize[0] - 0.5) for y in (-0.5, msize[1] - 0.5)
                        for z in (-0.5, msize[2] - 0.5)])
    step = SEARCH_MM if on_gpu else 2 * SEARCH_MM  # a CPU gets a coarser grid
    sides = SIDE_DEG if on_gpu else (0.0,)
    low, high = corners.min(0) - SEARCH_MARGIN_MM, corners.max(0) + SEARCH_MARGIN_MM
    axes = [np.arange(lo, hi + 1e-6, step) for lo, hi in zip(low, high)]
    targets = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    angles = np.stack(np.meshgrid(PITCH_DEG, sides, sides, indexing="ij"), -1).reshape(-1, 3)
    found = []  # (score, angles, target)
    for n, (a, r) in enumerate(zip(angles, rotation(angles))):
        value, _ = level.score(np.repeat(r[None], len(targets), 0), targets - center, center)
        for i in np.argsort(-value)[:20]:
            if value[i] > 0:
                found.append((float(value[i]), a, targets[i]))
        progress(0.05 + 0.45 * (n + 1) / len(angles), f"Searching: tilt {n + 1} of {len(angles)}")
    poses = len(angles) * len(targets)
    timings["search_s"] = time.time() - t1
    found.sort(key=lambda f: -f[0])
    probes = level.points_np[:: max(1, level.n // 200)]  # where distance matters: the temporal bones
    starts = []
    for value, a, target in found:
        params = np.concatenate([a, target - center])
        if all(np.linalg.norm(mapped(params, center, probes) - mapped(q, center, probes), axis=1).max()
               > DISTINCT_MM for _, q in starts):
            starts.append((value, params))
        if len(starts) == (STARTS if on_gpu else STARTS_CPU):
            break
    log(f"search: {poses} poses in {timings['search_s']:.1f} s; best " +
        ", ".join(f"{v:.3f} (pitch {p[0]:+.0f})" for v, p in starts[:5]))

    # ---- refine
    t2 = time.time()
    candidates = [{"search": v, "params": p} for v, p in starts]
    plan = [(SEARCH_MM, 1.5), (2.0, 1.0), (1.0, 0.5)]
    for n, c in enumerate(candidates):
        progress(0.5 + 0.42 * n / len(candidates), f"Refining pose {n + 1} of {len(candidates)}")
        for s, move in plan:
            c["params"], c[f"{s:g}mm"] = refine(levels[s], c["params"], center, move, move)
    candidates.sort(key=lambda c: -c["1mm"])
    best_params, value = candidates[0]["params"], candidates[0]["1mm"]
    if spacings[-1] < 1.0:
        progress(0.93, f"Refining at {spacings[-1]:g} mm")
        best_params, value = refine(levels[spacings[-1]], best_params, center, 0.25, 0.25)
    timings["refine_s"] = time.time() - t2
    timings["total_s"] = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if on_gpu else None

    def gap(c):
        return float(np.linalg.norm(mapped(c["params"], center, probes) - mapped(best_params, center, probes),
                                    axis=1).max())

    other = next((c for c in candidates if gap(c) >= 2.0), None)
    record = {
        "method": "normalized gradient fields on the temporal bones; brute-force start, Powell",
        "device": str(device),
        "peak_gpu_gib": peak,
        "levels_mm": spacings,
        "search_poses": int(poses),
        "score": value,
        "score_1mm": candidates[0]["1mm"],
        "overlap": levels[spacings[-1]].score_pose(best_params, center)[1],
        "starts_agreeing": sum(1 for c in candidates if gap(c) < 2.0),
        "starts_tried": len(candidates),
        "runner_up_score_1mm": other["1mm"] if other else None,
        "runner_up_mm": gap(other) if other else None,
        "timings": timings,
        "candidates": [{k: ([float(x) for x in v] if k == "params" else v) for k, v in c.items()}
                       for c in candidates],
    }
    return best_params, center, record
