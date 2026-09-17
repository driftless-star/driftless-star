"""Tests for the Stage 5 post-processing convergence signal.

These reuse the real pressure-convergence functions and ``build_signal`` from
``stage5_post_processing.py``, loaded by path. That script imports its sibling
``fit_vmec_pressure_from_transport_h5`` at module top, so loading it also exercises
the sibling-import support in ``load_stage_module``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution

post = load_stage_module("stages/stage5-post-processing/stage5_post_processing.py")


def _static(value: float, n_species: int = 3, n_rho: int = 5) -> np.ndarray:
    return np.full((n_species, n_rho), value)


def _write(path: Path, pressure: np.ndarray, pressure_face: np.ndarray, n_rho: int = 5) -> Path:
    """Write a staggered solution: ``n_rho`` cell centers bounded by ``n_rho + 1`` faces.

    ``_load_total_pressure`` reads the face grid, so ``pressure_face`` is what the convergence
    check sees; the centered array is written only to keep the file shaped like a real one.
    """
    faces = np.linspace(0.0, 1.0, n_rho + 1)
    return write_transport_solution(
        path, rho=0.5 * (faces[:-1] + faces[1:]), pressure=pressure,
        rho_face=faces, pressure_face=pressure_face,
    )


# Both criteria compare the first and final pressure slices. A static profile has no
# change to measure, so neither criterion may report convergence.
@pytest.mark.parametrize(
    "criterion",
    [post.rms_pressure_converged, post.pointwise_pressure_converged],
    ids=["rms", "pointwise"],
)
def test_pressure_converged_false_for_static_profile(tmp_path: Path, criterion) -> None:
    f = _write(tmp_path / "static.h5", _static(2.0), _static(2.0, n_rho=6))
    assert not criterion(f, rel_tol=1e-2, common_config=_clock_template(tmp_path))


@pytest.mark.parametrize(
    "criterion",
    [post.rms_pressure_converged, post.pointwise_pressure_converged],
    ids=["rms", "pointwise"],
)
def test_pressure_converged_true_for_small_change(tmp_path: Path, criterion) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 1.0001])  # 0.01% change between slices
    f = _write(tmp_path / "small.h5", np.stack([_static(2.0)] * 2), pressure_3d)
    assert criterion(f, rel_tol=1e-2, common_config=_clock_template(tmp_path))


@pytest.mark.parametrize(
    "criterion",
    [post.rms_pressure_converged, post.pointwise_pressure_converged],
    ids=["rms", "pointwise"],
)
def test_pressure_converged_false_for_large_change(tmp_path: Path, criterion) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 2.0])  # 100% change between slices
    f = _write(tmp_path / "large.h5", np.stack([_static(2.0)] * 2), pressure_3d)
    assert not criterion(f, rel_tol=1e-2, common_config=_clock_template(tmp_path))


# A 2% change at one of six faces has an RMS of 0.02/sqrt(6), below the 1%
# tolerance, but its maximum pointwise change remains 2%.
def test_pressure_convergence_methods_distinguish_a_single_point_change(tmp_path: Path) -> None:
    slice0 = _static(2.0, n_rho=6)
    slice1 = slice0.copy()
    slice1[:, 3] *= 1.02
    f = _write(tmp_path / "spike.h5", np.stack([_static(2.0)] * 2), np.stack([slice0, slice1]))

    template = _clock_template(tmp_path)
    assert post.rms_pressure_converged(f, rel_tol=1e-2, common_config=template)
    assert not post.pointwise_pressure_converged(f, rel_tol=1e-2, common_config=template)


@pytest.mark.parametrize(
    ("method", "criterion"),
    [
        ("rms", post.rms_pressure_converged),
        ("pointwise", post.pointwise_pressure_converged),
        ("t_final", post.t_final_converged),
    ],
)
def test_resolve_pressure_convergence_returns_registered_function(method: str, criterion) -> None:
    assert post.resolve_pressure_convergence(method) is criterion


def test_resolve_pressure_convergence_rejects_unknown_method() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported pressure convergence method 'RMS'; expected one of: rms, pointwise, t_final",
    ):
        post.resolve_pressure_convergence("RMS")


# Transport horizon

# Each pass resumes where the last pass stopped. Therefore, ``t_final`` is the absolute end time.
# Reaching it leaves no time for another pass.
CLOCK_TEMPLATE = "[transport_solver]\nt0 = 0.0\nt_final = 2.0\ndt = 0.5\n"


def _clock_template(tmp_path: Path, name: str = "common_input.toml") -> Path:
    """Write a minimal template carrying only the horizon the signal is compared against."""
    path = tmp_path / name
    path.write_text(CLOCK_TEMPLATE)
    return path


def _with_clock(path: Path, final_time: float) -> Path:
    """Add the ``final_time`` scalar that NEOPAX writes beside the profiles."""
    import h5py

    with h5py.File(path, "a") as f:
        f.create_dataset("final_time", data=np.asarray(final_time))
    return path


@pytest.mark.parametrize(
    ("final_time", "reached"), [(1.0, False), (2.0, True), (3.0, True)],
    ids=["before", "at", "past"],
)
def test_transport_horizon_reached_compares_the_clock_against_t_final(
    tmp_path: Path, final_time: float, reached: bool
) -> None:
    f = _with_clock(_write(tmp_path / "clock.h5", _static(2.0), _static(2.0, n_rho=6)), final_time)
    assert post.transport_horizon_reached(f, _clock_template(tmp_path)) is reached


# A solution without an exported clock cannot be compared with the horizon. The loop must not guess.
def test_transport_horizon_needs_the_solutions_clock(tmp_path: Path) -> None:
    f = _write(tmp_path / "no_clock.h5", _static(2.0), _static(2.0, n_rho=6))
    with pytest.raises(KeyError, match="final_time"):
        post.transport_horizon_reached(f, _clock_template(tmp_path))


def test_transport_horizon_needs_a_configured_end_time(tmp_path: Path) -> None:
    f = _with_clock(_write(tmp_path / "clock.h5", _static(2.0), _static(2.0, n_rho=6)), 1.0)
    template = tmp_path / "no_horizon.toml"
    template.write_text("[transport_solver]\nt0 = 0.0\ndt = 0.5\n")
    with pytest.raises(KeyError, match=r"\[transport_solver\]\.t_final"):
        post.transport_horizon_reached(f, template)


# The ``t_final`` criterion reads only the clock. The pressure doubles between the two slices, so
# a pressure criterion would never report convergence.
@pytest.mark.parametrize(
    ("final_time", "converged"), [(1.0, False), (2.0, True), (3.0, True)],
    ids=["before", "at", "past"],
)
def test_t_final_converged_tracks_the_clock(tmp_path: Path, final_time: float, converged: bool) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 2.0])
    f = _with_clock(
        _write(tmp_path / f"t_final-{final_time}.h5", np.stack([_static(2.0)] * 2), pressure_3d),
        final_time,
    )
    assert post.t_final_converged(f, rel_tol=1e-2, common_config=_clock_template(tmp_path)) is converged


# Signal construction

# ``build_signal`` returns the status dictionary that the loop reads. Non-positive total pressure
# stops the run with ``halted``. This test makes one radius negative. The pressure check runs before
# the clock check, so this fixture needs no ``final_time``.
def test_build_signal_halts_on_nonpositive_pressure(tmp_path: Path) -> None:
    pressure_face = _static(2.0, n_rho=6)
    pressure_face[:, 2] = -1.0  # summed total pressure is non-positive at one radius
    f = _write(tmp_path / "halt.h5", _static(2.0), pressure_face)
    assert post.build_signal(f, rel_tol=1e-2, common_config=_clock_template(tmp_path)) == {"status": "halted"}


# Convergence takes priority when a settled pass also reaches the horizon. This case must return
# ``converged``.
def test_build_signal_reports_convergence_ahead_of_the_horizon(tmp_path: Path) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 1.0001])
    f = _with_clock(_write(tmp_path / "ok.h5", np.stack([_static(2.0)] * 2), pressure_3d), 2.0)
    assert post.build_signal(f, rel_tol=1e-2, common_config=_clock_template(tmp_path)) == {"status": "converged"}


@pytest.mark.parametrize(
    ("method", "expected_status"),
    [("rms", "converged"), ("pointwise", "continue")],
)
def test_build_signal_uses_the_selected_pressure_convergence_method(
    tmp_path: Path, method: str, expected_status: str
) -> None:
    slice0 = _static(2.0, n_rho=6)
    slice1 = slice0.copy()
    slice1[:, 3] *= 1.02
    f = _with_clock(
        _write(tmp_path / f"spike-{method}.h5", np.stack([_static(2.0)] * 2), np.stack([slice0, slice1])),
        1.0,
    )
    assert post.build_signal(
        f,
        rel_tol=1e-2,
        common_config=_clock_template(tmp_path),
        convergence_method=method,
    ) == {"status": expected_status}


# Under ``t_final`` only the clock decides. A pass with no transport time left is converged even
# though its pressure doubled.
def test_build_signal_t_final_converges_at_the_horizon(tmp_path: Path) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 2.0])
    f = _with_clock(_write(tmp_path / "t_final_done.h5", np.stack([_static(2.0)] * 2), pressure_3d), 2.0)
    assert post.build_signal(
        f, rel_tol=1e-2, common_config=_clock_template(tmp_path), convergence_method="t_final"
    ) == {"status": "converged"}


# Under ``t_final`` a settled profile with transport time left runs another pass. A change of 0.01%
# would meet the pointwise criterion.
def test_build_signal_t_final_continues_before_the_horizon(tmp_path: Path) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 1.0001])
    f = _with_clock(_write(tmp_path / "t_final_more.h5", np.stack([_static(2.0)] * 2), pressure_3d), 1.0)
    assert post.build_signal(
        f, rel_tol=1e-2, common_config=_clock_template(tmp_path), convergence_method="t_final"
    ) == {"status": "continue"}


# The non-positive pressure check runs before every criterion. Under ``t_final`` a collapsed profile
# at the horizon still halts.
def test_build_signal_t_final_still_halts_on_nonpositive_pressure(tmp_path: Path) -> None:
    pressure_face = _static(2.0, n_rho=6)
    pressure_face[:, 2] = -1.0
    f = _with_clock(_write(tmp_path / "t_final_halt.h5", _static(2.0), pressure_face), 2.0)
    assert post.build_signal(
        f, rel_tol=1e-2, common_config=_clock_template(tmp_path), convergence_method="t_final"
    ) == {"status": "halted"}


# The CLI must reach every registered criterion. The same unsettled profile continues under
# ``pointwise`` with time left. It converges under ``t_final`` when the clock reaches 2.0.
@pytest.mark.parametrize(
    ("method", "final_time", "expected_status"),
    [("pointwise", 1.0, "continue"), ("t_final", 2.0, "converged")],
)
def test_main_accepts_the_convergence_method_from_the_cli(
    tmp_path: Path, monkeypatch, method: str, final_time: float, expected_status: str
) -> None:
    slice0 = _static(2.0, n_rho=6)
    slice1 = slice0.copy()
    slice1[:, 3] *= 1.02
    transport = _with_clock(
        _write(tmp_path / f"cli-spike-{method}.h5", np.stack([_static(2.0)] * 2), np.stack([slice0, slice1])),
        final_time,
    )
    common_config = _clock_template(tmp_path)
    signal = tmp_path / "signal.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage5_post_processing.py",
            "--transport", str(transport),
            "--common-config", str(common_config),
            "--signal", str(signal),
            "--pressure-rel-tol", "0.01",
            "--pressure-convergence-method", method,
        ],
    )

    post.main()

    assert json.loads(signal.read_text()) == {"status": expected_status}


# This profile is not settled and has no transport time left. Another pass would repeat the same
# window and discard the adapted step size. The loop must stop.
def test_build_signal_reports_the_horizon_when_not_converged(tmp_path: Path) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 2.0])  # 100% change, so not converged
    f = _with_clock(_write(tmp_path / "out_of_time.h5", np.stack([_static(2.0)] * 2), pressure_3d), 2.0)
    assert post.build_signal(f, rel_tol=1e-2, common_config=_clock_template(tmp_path)) == {"status": "horizon"}


# Not settled, with transport time left. This is the only status that runs another pass.
def test_build_signal_continues_with_time_left(tmp_path: Path) -> None:
    slice0 = _static(2.0, n_rho=6)
    pressure_3d = np.stack([slice0, slice0 * 2.0])
    f = _with_clock(_write(tmp_path / "more.h5", np.stack([_static(2.0)] * 2), pressure_3d), 1.0)
    assert post.build_signal(f, rel_tol=1e-2, common_config=_clock_template(tmp_path)) == {"status": "continue"}
