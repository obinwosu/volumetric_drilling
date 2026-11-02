"""The sheet is wider than the jet, runs in world −Z, and stays a mesh."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.impact_film import CELL_MM, WORLD_DOWN, ImpactFilm, surface_normal
from fluid.irrigation_bench import IrrigationBench, quat_aiming_at_bone, screen_down
from fluid.occupancy import VolumeFrame, identity_frame
from fluid.outlet import MM_TO_SIM


FILM_CLASS = ImpactFilm


def _new_film():
    return FILM_CLASS()


def _grid(film, mesh, key="verts"):
    """Mesh values on the full drawn grid, top then floor; NaN where not drawn."""
    mm = film.m * film.m
    values = np.asarray(mesh[key], dtype=np.float64)
    if "vert_ids" not in mesh:
        return values
    ids = mesh["vert_ids"]
    full = np.full((2 * mm, 3), np.nan)
    full[ids] = values[: len(ids)]
    full[mm + ids] = values[len(ids) :]
    return full


def _frame(counts):
    voxel = 0.5 * MM_TO_SIM
    size = np.asarray(counts, dtype=np.float64) * voxel
    return identity_frame(counts, size)


def _slope():
    """Surface falls toward +Y, which is downhill for world −Z. +Y past 44 is air."""
    counts = (48, 56, 28)
    occupied = np.zeros(counts, dtype=bool)
    for y in range(8, 44):
        top = 18 - (y - 8) // 4
        occupied[10:38, y, :top] = True
    return occupied, _frame(counts)


def _pour(film, occupied, frame, hit, jet_mm, spread, depth, seconds, incoming):
    steps = int(round(seconds / (1.0 / 60.0)))
    for _ in range(steps):
        film.step(
            1.0 / 60.0,
            hit,
            jet_mm,
            spread,
            depth,
            80.0,
            frame,
            occupied,
            0,
            incoming=incoming,
        )


def _world_centroid(film):
    mass = film.depth
    total = float(mass.sum())
    if total <= 0.0:
        return None
    bed = np.nan_to_num(film.bed, nan=0.0)
    return (bed * mass[..., None]).sum(axis=(0, 1)) / total


def _pour_life(film, occupied, frame, hit, seconds, life_s):
    steps = int(round(seconds / (1.0 / 60.0)))
    for _ in range(steps):
        film.step(
            1.0 / 60.0,
            hit,
            0.5,
            6.0,
            3.0,
            80.0,
            frame,
            occupied,
            0,
            incoming=(0.0, 0.0, -1.0),
            life_s=life_s,
            smooth=0.8,
        )


def test_longer_life_holds_more_water():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 16.5, 15.5)))
    brief = _new_film()
    held = _new_film()
    _pour_life(brief, occupied, frame, hit, 0.8, 0.4)
    _pour_life(held, occupied, frame, hit, 0.8, 8.0)
    assert held.volume_mm3 > brief.volume_mm3 * 1.5
    assert float(held.depth.max()) > 1.0
    mesh = held.mesh()
    assert mesh is not None
    assert _grid(held, mesh).shape == (2 * held.m * held.m, 3)
    assert int(mesh["tris"].max()) < len(mesh["verts"])


def test_world_down_is_negative_z():
    assert np.allclose(WORLD_DOWN, (0.0, 0.0, -1.0))
    camera = screen_down()
    assert abs(float(np.dot(camera, WORLD_DOWN))) < 0.95


def test_sheet_spreads_wider_than_the_jet_and_runs_downhill():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 12.5, 16.5)))
    incoming = np.array((0.0, 0.0, -1.0))
    film = _new_film()
    _pour(film, occupied, frame, hit, jet_mm=0.5, spread=6.0, depth=0.8, seconds=0.45, incoming=incoming)
    wet = film.depth >= 0.012
    assert int(wet.sum()) > 4
    radius = film._offsets(hit)
    assert float(radius[wet].max()) > 2.0
    assert float(radius[wet].max()) <= 6.0 + CELL_MM
    centroid = _world_centroid(film)
    assert centroid is not None
    # +Y is the falling side of this slope, and it is not the camera's screen-down.
    assert float(centroid[1] - hit[1]) > 1.0 * MM_TO_SIM
    assert film.volume_mm3 <= np.pi * 6.0 * 6.0 * 0.8 + 1e-3
    assert float(film.depth.max()) <= 0.8 + 1e-6


def test_water_drains_off_the_lip():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 30.5, 12.5)))
    incoming = np.array((0.0, 0.0, -1.0))
    film = _new_film()
    _pour(film, occupied, frame, hit, jet_mm=0.5, spread=7.0, depth=0.8, seconds=0.2, incoming=incoming)
    held = film.volume_mm3
    assert held > 0.2
    for _ in range(90):
        film.step(1.0 / 60.0, None, 0.5, 7.0, 0.8, 80.0, frame, occupied, 0, incoming=incoming)
    assert film.volume_mm3 < held * 0.25


def test_closed_pit_does_not_grow_without_bound():
    counts = (40, 40, 24)
    occupied = np.zeros(counts, dtype=bool)
    occupied[8:32, 8:32, :14] = True
    occupied[14:26, 14:26, 10:14] = False
    frame = _frame(counts)
    hit = frame.index_to_world(np.array((20.0, 20.0, 10.2)))
    film = _new_film()
    _pour(film, occupied, frame, hit, jet_mm=0.5, spread=5.0, depth=0.6, seconds=1.0, incoming=(0, 0, -1))
    assert film.volume_mm3 > 0.5
    assert film.volume_mm3 <= np.pi * 5.0 * 5.0 * 0.6 + 1e-2
    assert float(film.depth.max()) <= 0.6 + 1e-6


def test_rim_of_a_drilled_hole_is_not_inside_the_bone():
    counts = (40, 40, 24)
    occupied = np.zeros(counts, dtype=bool)
    occupied[:, :, :16] = True
    occupied[16:24, 16:24, 8:16] = False
    frame = _frame(counts)
    hit = frame.index_to_world(np.array((20.0, 20.0, 8.0)))
    film = _new_film()
    _pour(film, occupied, frame, hit, jet_mm=1.0, spread=10.0, depth=3.0, seconds=0.5, incoming=(0, 0, -1))
    top = float(frame.index_to_world(np.array((0.0, 0.0, 16.0)))[2])
    center = film.n // 2
    rim = film.bed[center + 12, center]
    assert film.on_bone[center + 12, center]
    assert abs(float(rim[2]) - top) <= 0.5 * MM_TO_SIM


def test_sideways_voxel_face_uses_the_jet():
    counts = (32, 32, 32)
    occupied = np.zeros(counts, dtype=bool)
    occupied[:, :, :16] = True
    frame = _frame(counts)
    hit = frame.index_to_world(np.array((16.0, 16.0, 16.0)))
    incoming = np.array((0.0, 1.0, 0.0))
    normal = surface_normal(frame, occupied, hit, incoming)
    assert float(np.dot(normal, -incoming)) > 0.9


def test_mesh_is_a_sheet_with_outward_normals():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 16.5, 15.5)))
    film = _new_film()
    _pour(film, occupied, frame, hit, jet_mm=0.5, spread=6.0, depth=0.8, seconds=0.35, incoming=(0, 0, -1))
    mesh = film.mesh()
    assert mesh is not None
    assert mesh["tris"].shape[0] >= 8
    assert _grid(film, mesh).shape == (2 * film.m * film.m, 3)
    normals = _grid(film, mesh, "normals")[: film.m * film.m].reshape(film.m, film.m, 3)
    wet = film._mask
    mean = normals[wet].mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    assert float(np.dot(mean, film._normal)) > 0.4
    # A second build keeps the vertex count and only rewrites positions.
    before = mesh["verts"]
    again = film.mesh()
    assert again["verts"].shape == before.shape
    if FILM_CLASS is ImpactFilm:
        assert again["verts"] is before


def test_water_is_a_closed_body_with_the_top_facing_out():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 16.5, 15.5)))
    film = _new_film()
    _pour(film, occupied, frame, hit, jet_mm=0.5, spread=6.0, depth=0.8, seconds=0.35, incoming=(0, 0, -1))
    mesh = film.mesh()
    tris = mesh["tris"]
    directed = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(int(u), int(v))] = directed.get((int(u), int(v)), 0) + 1
    # Watertight and consistently wound: every edge once each way.
    assert all(count == 1 for count in directed.values())
    assert all((v, u) in directed for u, v in directed)
    verts = mesh["verts"].astype(np.float64)
    top = tris[: film.n_top_tris]
    face = np.cross(verts[top[:, 1]] - verts[top[:, 0]], verts[top[:, 2]] - verts[top[:, 0]])
    assert float((face @ film._normal).mean()) > 0.0
    floor = _grid(film, mesh)[film.m * film.m :].reshape(film.m, film.m, 3)[film._mask]
    lift = (floor - film.draw_bed[film._mask]) @ film._normal
    assert float(lift.max()) < 0.0


def test_ripples_move_the_highlights_downhill():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 16.5, 15.5)))
    film = _new_film()
    for _ in range(30):
        film.step(
            1.0 / 60.0, hit, 0.5, 6.0, 0.8, 80.0, frame, occupied, 0,
            incoming=(0, 0, -1), life_s=8.0, smooth=0.65, ripple=1.0,
        )
    top = film.m * film.m
    first = _grid(film, film.mesh(), "normals")[:top]
    film.step(
        0.05, hit, 0.5, 6.0, 0.8, 80.0, frame, occupied, 0,
        incoming=(0, 0, -1), life_s=8.0, smooth=0.65, ripple=1.0,
    )
    second = _grid(film, film.mesh(), "normals")[:top]
    wet = film._mask.reshape(-1)
    turn = np.linalg.norm(second[wet] - first[wet], axis=1)
    assert float(turn.max()) > 0.01
    assert np.allclose(np.linalg.norm(second[wet], axis=1), 1.0, atol=1e-3)


def test_rivulets_break_the_front_into_streams():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 12.5, 16.5)))
    footprints = []
    for rivulets in (0.0, 1.0):
        film = _new_film()
        for _ in range(60):
            film.step(
                1.0 / 60.0, hit, 0.5, 10.0, 1.0, 80.0, frame, occupied, 0,
                incoming=(0, 0, -1), life_s=2.0, smooth=0.65, rivulets=rivulets,
            )
        wet = film.depth >= 0.012
        footprints.append((int(wet.sum()), film.volume_mm3))
    (smooth_area, smooth_volume), (stream_area, stream_volume) = footprints
    # Gathered into fewer, thicker streams. Streams also run off faster, so
    # they hold less water than the even sheet.
    assert stream_area < 0.85 * smooth_area
    assert stream_volume / stream_area > 1.2 * smooth_volume / smooth_area


def test_raycast_is_cached_until_pose_or_occupancy_changes():
    occupied = np.zeros((32, 32, 32), dtype=bool)
    occupied[16:24, 8:24, 8:24] = True
    frame = VolumeFrame(
        translation=np.zeros(3),
        rotation=np.eye(3),
        dimensions=np.ones(3),
        counts=np.array([32, 32, 32], dtype=np.float64),
    )
    bench = IrrigationBench(occupied, frame)
    bench.settings["film_ms"] = 0.0
    origin = np.array((-25.0 * MM_TO_SIM, 0.0, 0.0))
    quat = quat_aiming_at_bone(np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)))
    bench.step(0.016, origin, quat, occ_gen=1)
    bench.step(0.016, origin, quat, occ_gen=1)
    assert bench.raycast_queries == 1
    moved = origin + np.array((1.0 * MM_TO_SIM, 0.0, 0.0))
    bench.step(0.016, moved, quat, occ_gen=1)
    assert bench.raycast_queries == 2
    bench.step(0.016, moved, quat, occ_gen=2)
    assert bench.raycast_queries == 3


def test_suction_clears_the_sheet_and_dust_sees_it():
    occupied, frame = _slope()
    hit = frame.index_to_world(np.array((24.5, 16.5, 15.5)))
    film = _new_film()
    _pour(film, occupied, frame, hit, jet_mm=0.5, spread=6.0, depth=0.8, seconds=0.3, incoming=(0, 0, -1))
    assert film.any_wet
    assert bool(film.covers(np.array([hit]))[0])
    held = film.volume_mm3
    film.step(
        1.0 / 60.0, hit, 0.5, 6.0, 0.8, 0.0, frame, occupied, 0,
        incoming=(0, 0, -1), suction_tip=hit, suction_radius_mm=4.0,
    )
    assert film.volume_mm3 < held * 0.5


def test_moving_the_jet_leaves_the_old_water():
    occupied, frame = _slope()
    first = frame.index_to_world(np.array((12.5, 14.5, 16.5)))
    second = frame.index_to_world(np.array((28.5, 14.5, 16.5)))
    film = _new_film()
    _pour_life(film, occupied, frame, first, 1.0, 20.0)
    assert bool(film.covers(np.array([first]))[0])
    _pour_life(film, occupied, frame, second, 0.3, 20.0)
    assert bool(film.covers(np.array([first]))[0])
    assert bool(film.covers(np.array([second]))[0])


def _field():
    """80 mm of rolling bone with a 6 mm deep pit at index (120, 80)."""
    counts = (160, 160, 80)
    occupied = np.zeros(counts, dtype=bool)
    x = np.arange(160)[:, None]
    y = np.arange(160)[None, :]
    top = (40 + 6 * np.sin(x / 20.0) + 4 * np.cos(y / 25.0)).astype(int)
    pit = (x - 120) ** 2 + (y - 80) ** 2 <= 36
    top = np.where(pit, top - 24, top)
    for k in range(counts[2]):
        occupied[:, :, k] = k < top
    return occupied, _frame(counts), top


def _aim(film, occupied, frame, hit, seconds):
    for _ in range(int(round(seconds * 60))):
        film.step(
            1.0 / 60.0, hit, 1.0, 8.0, 3.0, 80.0, frame, occupied, 0,
            incoming=(0.0, 0.0, -1.0), life_s=15.0, smooth=0.65,
        )


def test_two_aims_leave_two_films_that_stay_put():
    occupied, frame, top = _field()
    first = frame.index_to_world(np.array((60.5, 80.5, top[60, 80] - 0.5)))
    second = frame.index_to_world(np.array((100.5, 80.5, top[100, 80] - 0.5)))
    film = _new_film()
    _aim(film, occupied, frame, first, 1.0)
    covered = film.covers(np.array([first]))[0]
    _aim(film, occupied, frame, second, 1.0)
    # 20 mm away: the first film is still there, in the same place on the bone.
    assert bool(covered) and bool(film.covers(np.array([first]))[0])
    assert bool(film.covers(np.array([second]))[0])
    midpoint = 0.5 * (first + second)
    assert not bool(film.covers(np.array([midpoint]))[0])


def test_new_bone_below_the_old_read_still_gets_water():
    occupied, frame, top = _field()
    rim = frame.index_to_world(np.array((104.5, 80.5, top[104, 80] - 0.5)))
    floor = frame.index_to_world(np.array((120.5, 80.5, top[120, 80] - 0.5)))
    film = _new_film()
    _aim(film, occupied, frame, rim, 0.5)
    _aim(film, occupied, frame, floor, 0.5)
    assert bool(film.covers(np.array([floor]))[0])


def test_gpu_mesh_matches_the_numpy_reference():
    occupied, frame, top = _field()
    film = ImpactFilm()
    for x in (90, 110):
        _aim(film, occupied, frame, frame.index_to_world(np.array((x + 0.5, 80.5, top[x, 80] - 0.5))), 2.0)
    gpu = film.mesh()
    verts, normals, mask = gpu["verts"].copy(), gpu["normals"].copy(), film._mask.copy()
    tris = np.sort(np.sort(gpu["tris"], axis=1), axis=0)
    cpu = film.mesh_cpu()
    assert (mask == film._mask).all()
    assert np.array_equal(tris, np.sort(np.sort(cpu["tris"], axis=1), axis=0))
    shown = np.concatenate([mask.reshape(-1), mask.reshape(-1)])
    assert np.abs(verts[shown] - cpu["verts"][shown]).max() < 1e-5 * MM_TO_SIM
    assert np.abs(normals[shown] - cpu["normals"][shown]).max() < 1e-3


def test_gpu_film_follows_the_numpy_film():
    from fluid.film_gpu import GpuFilm

    occupied, frame, top = _field()
    cpu, gpu = ImpactFilm(), GpuFilm()
    gen = 0
    for step in range(240):
        x = 50 + (step // 3) % 70
        if step % 60 == 0:
            occupied[x - 2 : x + 2, 78:82, top[x, 80] - 3 : top[x, 80] + 1] = False
            gen += 1
        column = np.flatnonzero(occupied[x, 80])
        hit = frame.index_to_world(np.array((x + 0.5, 80.5, column.max() + 0.5))) if step % 50 < 45 else None
        tip = frame.index_to_world(np.array((x - 10.5, 80.5, top[x, 80] + 0.5))) if step % 120 > 100 else None
        for film in (cpu, gpu):
            film.step(
                1.0 / 30.0, hit, 1.0, 8.0, 0.6, 80.0, frame, occupied, gen,
                incoming=(0.0, 0.0, -1.0), life_s=8.0, smooth=0.65, ripple=1.0, rivulets=0.5,
                suction_tip=tip, suction_radius_mm=2.0,
            )
    assert cpu._bed_samples == gpu._bed_samples
    assert np.allclose(cpu._origin, gpu._origin)
    assert (cpu.on_bone == gpu.on_bone).all()
    # Float32 against float64; the wet patch and the volume still agree closely.
    wet_cpu, wet_gpu = cpu.depth >= 0.012, gpu.depth >= 0.012
    assert (wet_cpu == wet_gpu).mean() > 0.995
    assert abs(cpu.volume_mm3 - gpu.volume_mm3) <= 0.02 * cpu.volume_mm3 + 1e-3
    mesh = gpu.mesh()
    # Every wet vertex is drawn, at the place the full-grid mesh puts it.
    reference = cpu.mesh_cpu()
    drawn = _grid(gpu, mesh)
    shown = np.concatenate([cpu._mask.reshape(-1), cpu._mask.reshape(-1)])
    shown &= np.concatenate([gpu._mask.reshape(-1), gpu._mask.reshape(-1)])
    # Wet vertices only on quads that bridge a drop are left out of the sheet.
    missing = np.isnan(drawn[:, 0]) & shown
    assert missing.sum() <= 0.01 * shown.sum(), (int(missing.sum()), int(shown.sum()))
    shown &= ~missing
    full = gpu._verts_d.numpy()
    assert np.abs(drawn[shown] - full[shown]).max() < 1e-9
    # The two water edges round a few vertices differently; the rest agree.
    gap = np.linalg.norm(drawn[shown] - reference["verts"][shown], axis=1)
    assert np.percentile(gap, 99) < 0.25 * MM_TO_SIM


def _assert_no_bridges(film, mesh):
    """The floor follows the bone; none of its edges climbs more than a steep slope over one cell."""
    count = film.n_top_tris
    verts, floor = mesh["verts"], mesh["tris"][count : 2 * count]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        edge = np.linalg.norm(verts[floor[:, a]] - verts[floor[:, b]], axis=1) / MM_TO_SIM
        assert float(edge.max()) < 2.0, float(edge.max())


def test_gpu_sheet_does_not_bridge_drilled_walls():
    from fluid.film_gpu import GpuFilm

    occupied, frame, top = _field()
    film = GpuFilm(pipelined=True)
    gen = 0
    for step in range(240):
        x = 40 + (step // 2) % 90
        if step % 40 == 0:
            # Deep holes with vertical walls, which a dry quad would otherwise span.
            occupied[x - 2 : x + 2, 78:82, top[x, 80] - 12 : top[x, 80] + 1] = False
            gen += 1
        column = np.flatnonzero(occupied[x, 80])
        hit = frame.index_to_world(np.array((x + 0.5, 80.5, column.max() + 0.5)))
        film.step(
            1.0 / 60.0, hit, 1.0, 4.5, 4.0, 100.0, frame, occupied, gen,
            incoming=(0.0, 0.0, -1.0), life_s=15.0, smooth=1.0, ripple=1.0,
        )
        mesh = film.mesh()
        if mesh is not None:
            _assert_no_bridges(film, mesh)


def test_gpu_sheet_keeps_its_triangles_while_water_flows():
    from fluid.film_gpu import GpuFilm

    occupied, frame, top = _field()
    film = GpuFilm()
    hit = frame.index_to_world(np.array((80.5, 80.5, top[80, 80] - 0.5)))
    versions = []
    for _ in range(90):
        film.step(
            1.0 / 60.0, hit, 1.0, 8.0, 0.6, 80.0, frame, occupied, 0,
            incoming=(0.0, 0.0, -1.0), life_s=8.0, smooth=0.65, ripple=1.0,
        )
        mesh = film.mesh()
        if mesh is not None:
            versions.append(mesh["topology"])
    # The water keeps spreading, but the triangles change only when it reaches new blocks.
    assert len(versions) > 60
    assert len(set(versions[-45:])) <= 8
    _assert_no_bridges(film, mesh)
    # Dry vertices sit on the floor, so nothing shows outside the water.
    mm = film.m * film.m
    drawn = _grid(film, mesh)
    dry = ~film._mask.reshape(-1) & ~np.isnan(drawn[:mm, 0])
    assert dry.any()
    assert np.allclose(drawn[:mm][dry], drawn[mm:][dry])


TESTS = (
    test_longer_life_holds_more_water,
    test_world_down_is_negative_z,
    test_sheet_spreads_wider_than_the_jet_and_runs_downhill,
    test_water_drains_off_the_lip,
    test_closed_pit_does_not_grow_without_bound,
    test_rim_of_a_drilled_hole_is_not_inside_the_bone,
    test_sideways_voxel_face_uses_the_jet,
    test_mesh_is_a_sheet_with_outward_normals,
    test_water_is_a_closed_body_with_the_top_facing_out,
    test_ripples_move_the_highlights_downhill,
    test_rivulets_break_the_front_into_streams,
    test_raycast_is_cached_until_pose_or_occupancy_changes,
    test_suction_clears_the_sheet_and_dust_sees_it,
    test_moving_the_jet_leaves_the_old_water,
    test_two_aims_leave_two_films_that_stay_put,
    test_new_bone_below_the_old_read_still_gets_water,
)


if __name__ == "__main__":
    from fluid.film_gpu import GpuFilm

    for test in TESTS:
        test()
    test_gpu_mesh_matches_the_numpy_reference()
    print("impact film ok")
    # The same behaviour from the film whose state lives on the GPU.
    FILM_CLASS = GpuFilm
    for test in TESTS:
        test()
    test_gpu_film_follows_the_numpy_film()
    test_gpu_sheet_keeps_its_triangles_while_water_flows()
    test_gpu_sheet_does_not_bridge_drilled_walls()
    print("gpu film ok")
