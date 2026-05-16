"""
Load saved topology optimization fields from VTK and recompute metrics.

Reads rho_bar (DG0), u (CG1 vector), and sigma_m (DG0) from the parallel VTU
files written by save_final_realizations, then evaluates compliance J, volume
fractions V and V_pitch, and sigma_max using the BowspritTopOpt methods.

Run serial (no mpiexec):
    python load_run_results.py
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
import fenics as fs
import fenics_adjoint as fa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beam_configurator_2d import GeometryProperties2d, LoadCase2d, MaterialProperties2d
from bowsprit_topopt import BowspritTopOpt, print_comparison

# ── configuration ─────────────────────────────────────────────────────────────

RUN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "plots", "run_20260510_213824",
)

MESH_SIZE        = (460, 60)
LENGTH, HEIGHT   = 3.80, 0.50
THICKNESS        = 0.05
E_MODULUS        = 70e9
POISSON_RATIO    = 0.33
VOLUME_FRACTION  = 0.25
PITCH_ALPHA      = 2.5
SIGMA_Y          = 40e6

# ── VTU reading ───────────────────────────────────────────────────────────────

def _parse(text, dtype=float):
    return np.array(text.split(), dtype=dtype)


def _read_piece_cell_scalar(vtu_path, field_name):
    """Return (cell_centers_xy, scalar_values) for one VTU piece."""
    root = ET.parse(vtu_path).getroot()
    piece = root.find(".//Piece")

    pts = _parse(piece.find(".//Points/DataArray").text).reshape(-1, 3)[:, :2]

    conn = _parse(piece.find(".//Cells/DataArray[@Name='connectivity']").text, dtype=int)
    offs = _parse(piece.find(".//Cells/DataArray[@Name='offsets']").text, dtype=int)

    centers = np.zeros((len(offs), 2))
    prev = 0
    for i, off in enumerate(offs):
        verts = conn[prev:int(off)]
        centers[i] = pts[verts].mean(axis=0)
        prev = int(off)

    da = piece.find(f".//CellData/DataArray[@Name='{field_name}']")
    if da is None:
        raise KeyError(f"Cell field '{field_name}' not found in {vtu_path}")
    return centers, _parse(da.text)


def _read_piece_point_vector(vtu_path, field_name):
    """Return (node_coords_xy, values_xy) for one VTU piece."""
    root = ET.parse(vtu_path).getroot()
    piece = root.find(".//Piece")

    pts = _parse(piece.find(".//Points/DataArray").text).reshape(-1, 3)[:, :2]

    da = piece.find(f".//PointData/DataArray[@Name='{field_name}']")
    if da is None:
        raise KeyError(f"Point field '{field_name}' not found in {vtu_path}")
    ncomp = int(da.get("NumberOfComponents", 1))
    vals = _parse(da.text).reshape(-1, ncomp)[:, :2]
    return pts, vals


def _read_pvtu_cell_scalar(pvtu_path, field_name):
    """Assemble all pieces. Returns (all_centers, all_values)."""
    d = os.path.dirname(pvtu_path)
    root = ET.parse(pvtu_path).getroot()
    coords, vals = [], []
    for el in root.iter("Piece"):
        c, v = _read_piece_cell_scalar(os.path.join(d, el.get("Source")), field_name)
        coords.append(c)
        vals.append(v)
    return np.vstack(coords), np.concatenate(vals)


def _read_pvtu_point_vector(pvtu_path, field_name):
    """Assemble all pieces. Returns (all_pts, all_values_xy)."""
    d = os.path.dirname(pvtu_path)
    root = ET.parse(pvtu_path).getroot()
    pts, vals = [], []
    for el in root.iter("Piece"):
        p, v = _read_piece_point_vector(os.path.join(d, el.get("Source")), field_name)
        pts.append(p)
        vals.append(v)
    return np.vstack(pts), np.vstack(vals)


# ── FEniCS function loaders ───────────────────────────────────────────────────

def load_dg0(pvtu_path, field_name, V):
    """Load a DG0 scalar from parallel VTU into FEniCS FunctionSpace V."""
    centers_vtk, vals_vtk = _read_pvtu_cell_scalar(pvtu_path, field_name)
    dof_coords = V.tabulate_dof_coordinates()
    _, idx = cKDTree(centers_vtk).query(dof_coords)
    f = fs.Function(V)
    f.vector()[:] = vals_vtk[idx]
    return f


def load_cg1_vector(pvtu_path, field_name, V):
    """Load a CG1 vector from parallel VTU into FEniCS VectorFunctionSpace V."""
    pts_vtk, vals_vtk = _read_pvtu_point_vector(pvtu_path, field_name)
    tree = cKDTree(pts_vtk)

    x_dofs = V.sub(0).dofmap().dofs()
    y_dofs = V.sub(1).dofmap().dofs()
    all_coords = V.tabulate_dof_coordinates()

    _, ix = tree.query(all_coords[x_dofs])
    _, iy = tree.query(all_coords[y_dofs])

    f = fs.Function(V)
    arr = f.vector().get_local()
    arr[x_dofs] = vals_vtk[ix, 0]
    arr[y_dofs] = vals_vtk[iy, 1]
    f.vector().set_local(arr)
    f.vector().apply("insert")
    return f


# ── metric computation ────────────────────────────────────────────────────────

def compute_metrics(beam, task_dir, prefix):
    """
    Load saved fields for one realization and compute J, V, V_pitch, sigma_max.
    sigma_max is in Pa (same units as the saved sigma_m field).
    """
    Vd = beam._filter_project.Vd
    Vu = beam._function_space

    rho_bar = load_dg0(
        os.path.join(task_dir, f"{prefix}_rho_bar000000.pvtu"), "rho_bar", Vd
    )
    u = load_cg1_vector(
        os.path.join(task_dir, f"{prefix}_u000000.pvtu"), "Displacement", Vu
    )

    fa.pause_annotation()
    try:
        J  = float(beam.compliance(u))
        V  = float(beam.volume_fraction_of(rho_bar))
        Vp = float(beam.pitch_weighted_volume_fraction_of(rho_bar))
    finally:
        fa.continue_annotation()

    sigma_max = None
    sm_path = os.path.join(task_dir, f"{prefix}_sigma_m000000.pvtu")
    if os.path.exists(sm_path):
        _, sm_vals = _read_pvtu_cell_scalar(sm_path, "sigma_m")
        sigma_max = float(sm_vals.max())

    return J, V, Vp, sigma_max


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    fs.set_log_level(40)

    beam = BowspritTopOpt(
        geometry_properties=GeometryProperties2d(
            length=LENGTH, height=HEIGHT, thickness=THICKNESS
        ),
        material_properties=MaterialProperties2d(
            e_modulus=E_MODULUS, poisson_ratio=POISSON_RATIO
        ),
        loads=LoadCase2d(),
        mesh_size=MESH_SIZE,
        volume_fraction=VOLUME_FRACTION,
        pitch_weight_alpha=PITCH_ALPHA,
        verbose=False,
    )

    task3_dir = os.path.join(RUN_DIR, "task3")
    task4_dir = os.path.join(RUN_DIR, "task4")

    results3, results4 = {}, {}

    for label in ("dilated", "nominal", "eroded"):
        metrics = compute_metrics(beam, task3_dir, f"task3_{label}")
        results3[label] = metrics
        J, V, Vp, sm = metrics
        sm_str = f", σ_max = {sm/SIGMA_Y:.3f}·σ_y  ({sm/1e6:.1f} MPa)" if sm else ""
        print(f"task3 {label:12s}  J = {J:.4e} N·m  V = {V:.3f}  Vp = {Vp:.3f}{sm_str}")

    for label in ("dilated", "intermediate", "eroded"):
        metrics = compute_metrics(beam, task4_dir, f"task4_{label}")
        results4[label] = metrics
        J, V, Vp, sm = metrics
        sm_str = f", σ_max = {sm/SIGMA_Y:.3f}·σ_y  ({sm/1e6:.1f} MPa)" if sm else ""
        print(f"task4 {label:12s}  J = {J:.4e} N·m  V = {V:.3f}  Vp = {Vp:.3f}{sm_str}")

    print()
    print_comparison(results3, results4)


if __name__ == "__main__":
    main()
