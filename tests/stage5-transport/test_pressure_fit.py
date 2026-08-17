"""Tests for the Stage 5 pressure-fit helpers.

These reuse the real functions from ``fit_vmec_pressure_from_transport_h5.py``, loaded
by path because ``stages/`` has no package ``__init__``, so the tests pin the actual
fit and file-rewrite behavior rather than a copy. The script imports h5py/numpy only,
so it loads with no solver present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution

fit_mod = load_stage_module("stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

polyval = np.polynomial.polynomial.polyval


def _face_grid(n_radial: int = 5, rho_edge: float = 1.0) -> np.ndarray:
    """NEOPAX's face grid: the ``n_radial + 1`` cell boundaries spanning ``[0, rho_edge]``."""
    return np.linspace(0.0, rho_edge, n_radial + 1)


def _center_grid(n_radial: int = 5, rho_edge: float = 1.0) -> np.ndarray:
    """NEOPAX's cell centers: face-grid midpoints, so neither 0 nor ``rho_edge`` is a member."""
    faces = _face_grid(n_radial, rho_edge)
    return 0.5 * (faces[:-1] + faces[1:])


def _profiles(n_species: int = 3, n_rho: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a staggered density/temperature set: ``n_rho`` cell centers and ``n_rho + 1`` faces.

    The face values are deliberately disjoint from the centered ones, so a reader taking the wrong grid's
    arrays cannot pass by coincidence.
    """
    n_face = n_rho + 1
    return (
        np.linspace(1.0, 2.0, n_species * n_rho).reshape(n_species, n_rho),
        np.linspace(2.0, 3.0, n_species * n_rho).reshape(n_species, n_rho),
        np.linspace(5.0, 6.0, n_species * n_face).reshape(n_species, n_face),
        np.linspace(7.0, 8.0, n_species * n_face).reshape(n_species, n_face),
    )


def _write_total_pressure(
    path: Path, *, rho_face: np.ndarray, total_face: np.ndarray, total_center: np.ndarray
) -> Path:
    """Write a one-species solution whose species-summed pressure is the given total on each grid."""
    return write_transport_solution(
        path,
        rho=0.5 * (rho_face[:-1] + rho_face[1:]),
        pressure=np.asarray(total_center, dtype=float)[None, :],
        rho_face=rho_face,
        pressure_face=np.asarray(total_face, dtype=float)[None, :],
    )


def _args(
    h5_path: Path, *, degree: int = 8, time_index: int = -1, final_time: bool = False, drop_axis: bool = False
) -> SimpleNamespace:
    """The argparse namespace ``_fit_from_args`` consumes, built without going through the CLI parser."""
    return SimpleNamespace(
        h5_path=h5_path, degree=degree, time_index=time_index, final_time=final_time, drop_axis=drop_axis
    )


def _pedestal(rho: np.ndarray | float) -> np.ndarray:
    """An edge-pedestal pressure profile that is *not* a low-order polynomial in ``s = rho**2``.

    A profile exactly representable in the fitted basis (``(1 - rho**2)**2``, say) is recovered from any grid,
    so it cannot distinguish one grid from another; this one can.
    """
    return 0.5 * (1.0 - np.tanh((np.asarray(rho, dtype=float) - 0.85) / 0.06)) + 0.05


# A round-trip test for the polynomial fitter. It builds pressure data from known coefficients [3, -2, 0.5], fits a
# degree-2 power series back to that data, and asserts the recovered coefficients match the originals. Recovering a
# known analytical answer is a strong correctness check.
def test_fit_power_series_recovers_known_polynomial() -> None:
    rho = np.linspace(0.0, 1.0, 6)
    s = rho**2
    coeffs_true = np.array([3.0, -2.0, 0.5])  # low-order first: 3 - 2 s + 0.5 s^2
    p = coeffs_true[0] + coeffs_true[1] * s + coeffs_true[2] * s**2
    coeffs = fit_mod._fit_power_series(s, p, degree=2)
    assert_allclose(coeffs, coeffs_true, atol=1e-9)


# Total pressure can be supplied directly, or reconstructed from temperature x density. This writes one file each way
# (with matching inputs) and asserts `_load_total_pressure` returns the same total for both, that it equals the
# species-summed *face* pressure, and that a static profile reports no time index. Confirms the two supported input
# forms agree and that both read the face grid.
def test_load_total_pressure_temperature_density_matches_pressure(tmp_path: Path) -> None:
    rho, rho_face = _center_grid(), _face_grid()
    density, temperature, density_face, temperature_face = _profiles()
    pressure_face = density_face * temperature_face
    file_td = write_transport_solution(
        tmp_path / "td.h5", rho=rho, temperature=temperature, density=density,
        rho_face=rho_face, temperature_face=temperature_face, density_face=density_face,
    )
    file_p = write_transport_solution(
        tmp_path / "p.h5", rho=rho, pressure=density * temperature,
        rho_face=rho_face, pressure_face=pressure_face,
    )

    rho_td, total_td, idx_td = fit_mod._load_total_pressure(file_td, time_index=-1, final_time=False)
    rho_p, total_p, idx_p = fit_mod._load_total_pressure(file_p, time_index=-1, final_time=False)

    assert_allclose(total_td, total_p, rtol=1e-12)
    assert_allclose(total_p, np.sum(pressure_face, axis=0), rtol=1e-12)
    # The returned radius is the face grid, which spans the full minor radius.
    assert_allclose(rho_td, rho_face, atol=1e-12)
    assert_allclose(rho_p, rho_face, atol=1e-12)
    assert idx_td is None and idx_p is None  # static profile has no resolved time index


# For a time-resolved file, `_load_total_pressure` should be able to pick a specific time slice. This writes a
# 2-time-step face pressure, then asserts that `final_time=True` selects the last slice (time index 1) and
# `time_index=0` selects the first, with the returned totals matching each slice's sum.
def test_load_total_pressure_final_time_selects_last_slice(tmp_path: Path) -> None:
    rho, rho_face = _center_grid(), _face_grid()
    density, temperature, density_face, temperature_face = _profiles()
    slice0 = density_face * temperature_face
    slice1 = 2.0 * slice0
    pressure_face_3d = np.stack([slice0, slice1])  # (n_time=2, n_species, n_face)
    centered = density * temperature
    f3d = write_transport_solution(
        tmp_path / "p3d.h5", rho=rho, pressure=np.stack([centered, 2.0 * centered]),
        rho_face=rho_face, pressure_face=pressure_face_3d,
    )

    _, total_final, idx_final = fit_mod._load_total_pressure(f3d, time_index=-1, final_time=True)
    _, total_initial, idx_initial = fit_mod._load_total_pressure(f3d, time_index=0, final_time=False)

    assert idx_final == 1 and idx_initial == 0
    assert_allclose(total_final, np.sum(slice1, axis=0), rtol=1e-12)
    assert_allclose(total_initial, np.sum(slice0, axis=0), rtol=1e-12)


# A transport solution predating NEOPAX's staggered cell/face grid carries a complete cell-centered state and no faces.
# Falling back to it would silently fit over radii that stop short of the plasma edge, so the loader must refuse it.
def test_load_total_pressure_missing_face_grid_raises(tmp_path: Path) -> None:
    density, temperature, _, _ = _profiles()
    legacy = write_transport_solution(
        tmp_path / "centers_only.h5", rho=_center_grid(), temperature=temperature, density=density
    )
    with pytest.raises(KeyError, match="rho_face"):
        fit_mod._load_total_pressure(legacy, time_index=-1, final_time=False)


def test_load_total_pressure_missing_face_profiles_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_face_profiles.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("rho", data=_center_grid())
        f.create_dataset("rho_face", data=_face_grid())
    with pytest.raises(KeyError, match="pressure_faces"):
        fit_mod._load_total_pressure(bad, time_index=-1, final_time=False)


# `_saved_time_count` reads the length of the strictly increasing prefix of `ts`; the comparison and plotting tools
# use it to pick the last distinctly timed record of a solution. This pins the prefix rule on the shapes a save
# buffer can take, including an all-zero axis, a partly filled one, a fully distinct one, NaN fill, and a repeated
# final value, since neither of the last two compares greater and so both end the prefix.
@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        ([0.0] * 10, 1),
        ([0.0, 5.0, 10.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4),
        (list(np.linspace(0.0, 20.0, 10)), 10),
        ([3.0, 0.0, 0.0], 1),
        ([0.0, 1.0, 2.0, np.nan, np.nan], 3),
        ([0.0, 1.0, 2.0, 2.0, 2.0], 3),
        ([0.0], 1),
    ],
)
def test_saved_time_count_reads_the_increasing_prefix(ts: list[float], expected: int) -> None:
    assert fit_mod._saved_time_count(np.asarray(ts)) == expected


def test_saved_time_count_rejects_a_non_1d_axis() -> None:
    with pytest.raises(ValueError, match="'ts' must be a 1-D time axis"):
        fit_mod._saved_time_count(np.zeros((2, 3)))


# A NaN in the selected slice would be polynomial-fitted straight into the equilibrium's AM coefficients and only
# surface much later inside VMEC, so it is rejected at read time in both datasets that reach the fit. `rho_face` is
# checked as well because it becomes the abscissa s = rho**2 of that same fit.
@pytest.mark.parametrize("broken", ["pressure", "rho"])
def test_non_finite_slice_raises(tmp_path: Path, broken: str) -> None:
    rho_face = _face_grid()
    total_face = 1.0 - 0.5 * rho_face**2
    if broken == "pressure":
        total_face[2] = np.nan
        expected = "total pressure holds non-finite values"
    else:
        rho_face[3] = np.inf
        expected = "rho holds non-finite values"
    f = _write_total_pressure(
        tmp_path / "nonfinite.h5", rho_face=rho_face, total_face=total_face,
        total_center=1.0 - 0.5 * _center_grid() ** 2,
    )
    with pytest.raises(ValueError, match=expected):
        fit_mod._load_total_pressure(f, time_index=-1, final_time=True)


# VMEC evaluates the fitted power series over the whole of s = rho**2 in [0, 1], so the fit's data must reach
# rho = 1. NEOPAX's cell centers stop at 1 - 1/(2n), while the faces include rho = 1. Asserts the fitted series
# reproduces the pedestal at both s = 1 and s = 0 to 1%, at n_radial 5, 10 and 20.
@pytest.mark.parametrize("n_radial", [5, 10, 20])
def test_fit_recovers_the_edge_of_a_pedestal_profile(tmp_path: Path, n_radial: int) -> None:
    rho_face = _face_grid(n_radial)
    solution = _write_total_pressure(
        tmp_path / f"pedestal_{n_radial}.h5",
        rho_face=rho_face,
        total_face=_pedestal(rho_face),
        total_center=_pedestal(_center_grid(n_radial)),
    )

    coeffs, _, _ = fit_mod._fit_from_args(_args(solution))

    assert_allclose(polyval(1.0, coeffs), _pedestal(1.0), rtol=1e-2)  # plasma edge, s = 1
    assert_allclose(polyval(0.0, coeffs), _pedestal(0.0), rtol=1e-2)  # magnetic axis, s = 0


# `--drop-axis` keeps the magnetic-axis sample out of the fit, and the face grid's first point is rho = 0. A
# degree-1 fit to a profile linear in s apart from a deliberate axis outlier recovers the clean line
# [1.0, -0.5] only once that outlier is dropped.
def test_drop_axis_excludes_the_axis_sample(tmp_path: Path) -> None:
    rho_face = _face_grid()
    total = 1.0 - 0.5 * rho_face**2
    total[0] = 10.0  # an axis outlier that only --drop-axis can remove
    solution = _write_total_pressure(
        tmp_path / "axis_outlier.h5", rho_face=rho_face, total_face=total,
        total_center=1.0 - 0.5 * _center_grid() ** 2,
    )

    kept, _, _ = fit_mod._fit_from_args(_args(solution, degree=1, drop_axis=False))
    dropped, _, _ = fit_mod._fit_from_args(_args(solution, degree=1, drop_axis=True))

    assert_allclose(dropped, [1.0, -0.5], atol=1e-12)  # the clean line, axis sample gone
    assert not np.allclose(kept, [1.0, -0.5], atol=1e-3)  # the outlier still drags the fit


# On a grid whose first radius is not the axis, dropping it would discard a real innermost data point. Asserts
# the flag is skipped, the fit is unchanged, and a warning naming --drop-axis is logged.
def test_drop_axis_skipped_with_a_log_when_first_radius_is_not_the_axis(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rho_face = np.linspace(0.2, 1.0, 6)  # a radial grid whose first point is not rho = 0
    total = 1.0 - 0.5 * rho_face**2
    total[0] = 10.0
    solution = _write_total_pressure(
        tmp_path / "off_axis.h5", rho_face=rho_face, total_face=total,
        total_center=1.0 - 0.5 * (0.5 * (rho_face[:-1] + rho_face[1:])) ** 2,
    )

    kept, _, _ = fit_mod._fit_from_args(_args(solution, degree=1, drop_axis=False))
    with caplog.at_level(logging.WARNING, logger=fit_mod.__name__):
        requested, _, _ = fit_mod._fit_from_args(_args(solution, degree=1, drop_axis=True))

    assert_allclose(requested, kept, rtol=1e-12)  # nothing was dropped
    assert "--drop-axis" in caplog.text


# This is the feedback step that turns the fitted pressure back into a VMEC input file for the next loop iteration.
# Starting from a committed template fixture, it writes a new file with the fitted coefficients and asserts the pressure
# keys are set correctly (`PMASS_TYPE`, `PRES_SCALE`, and the `AM` coefficient line). The expected `AM` line is
# hard-coded (not built by the writer's own formatter) so a bug in that formatter can't hide itself. It also asserts the
# template's `AM` line was replaced in place (not duplicated) and that the committed fixture file itself is left
# unmodified.
def test_write_vmec_input_rewrites_pressure_keys(tmp_path: Path) -> None:
    coeffs = np.array([3.0, -2.0, 0.5])
    out = tmp_path / "vmec_out.txt"
    fixture = DATA_DIR / "vmec_indata_min.txt"

    result = fit_mod._write_vmec_input_with_pressure_fit(fixture, coeffs, output_path=out)

    assert result == out
    text = out.read_text()
    assert "PMASS_TYPE = 'power_series'" in text
    assert "PRES_SCALE = 1.0000000000000000E+00" in text
    # Literal AM line for coeffs [3.0, -2.0, 0.5], independent of the writer's own formatter,
    # so a bug in _format_am_line cannot mask itself.
    expected_am_line = "AM = 3.0000000000000000E+00, -2.0000000000000000E+00, 5.0000000000000000E-01"
    assert expected_am_line in text
    assert "AM = 0.0" not in text  # the template AM line is replaced in place, not left behind or duplicated
    assert "AM = 0.0" in fixture.read_text()  # committed fixture left unmodified
