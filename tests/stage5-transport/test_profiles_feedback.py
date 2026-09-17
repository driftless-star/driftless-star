"""Test the Stage 5 prescribed-profile feedback writer.

The tests call its pure helpers and command-line entry point. The writer reads one NEOPAX solution
slice and replaces [profiles] in a ``common_input.toml`` template.

The tests check all unit conversions. Density becomes m^-3, temperature becomes eV and ``Er`` stays
in kV/m. Face gradients change from units per metre to units per rho. Text outside [profiles] must
remain unchanged, except for ``t0`` and ``dt`` in [transport_solver]. These tests do not require
NEOPAX.

One test feeds an emitted file back through the Snakefile's parse-time [profiles] validation, which
holds the writer and that validator to the same key set and shapes.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from src.stage5_helper import prepare_neopax_config
from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution

writer = load_stage_module("stages/stage5-post-processing/write_prescribed_profiles_from_transport_h5.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
# The committed quick_run template: 3 species (e, D, T), [geometry].n_radial = 5 and no rho_edge, so
# a slice feeds back only on the grid linspace(0, 1, 5). It is CRLF-terminated and its [profiles]
# section is followed by further sections, which is the layout every tracked-template test relies on.
TRACKED_TEMPLATE = REPO_ROOT / "inputs/quick_run/common_input.toml"

# NEOPAX's minor radius in these fixtures, the metres per unit rho its face gradients are per. Kept
# well away from 1 so a conversion that drops the factor cannot pass by coincidence.
A_NEO = 0.4

# The fixture clock contains the reached time and the next step. Both values precede the template's
# ``t_final``. They also differ from its ``t0`` and ``dt``, so a missed update cannot pass.
FINAL_TIME = 4.0e-7
NEXT_DT = 3.0e-8

# The last saved time equals ``FINAL_TIME``. The writer pairs this final slice with the run clock.
# A shorter save grid would pair state and clock values from different times.
SAVE_TIMES = [0.0, 1.0e-7, FINAL_TIME]


def _face_grid(n_radial: int = 5, rho_edge: float = 1.0) -> np.ndarray:
    """NEOPAX's face grid: the ``n_radial + 1`` cell boundaries spanning ``[0, rho_edge]``."""
    return np.linspace(0.0, rho_edge, n_radial + 1)


def _center_grid(n_radial: int = 5, rho_edge: float = 1.0) -> np.ndarray:
    """NEOPAX's cell centers: face-grid midpoints, so neither 0 nor ``rho_edge`` is a member."""
    faces = _face_grid(n_radial, rho_edge)
    return 0.5 * (faces[:-1] + faces[1:])


def _make_slice(offset: float = 0.0, n_species: int = 3, n_rho: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one transport slice in file units: density in 1e20 m^-3, temperature in keV, Er in kV/m.

    Every entry differs from every other so a transposed, reordered, or wrongly sliced array cannot
    pass an equality assertion by coincidence. ``offset`` separates consecutive time slices.
    """
    ramp = np.arange(n_species * n_rho, dtype=float).reshape(n_species, n_rho)
    return 1.0 + 0.1 * ramp + offset, 2.0 + 0.2 * ramp + offset, np.linspace(-1.0, 3.0, n_rho) + offset


def _make_face_slice(
    offset: float = 0.0, n_species: int = 3, n_rho: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face-grid counterpart of :func:`_make_slice`, on ``n_rho + 1`` radii.

    The base values are deliberately disjoint from ``_make_slice``'s, so a reader that takes the
    cell-centered arrays where it should take the face ones cannot pass by coincidence.
    """
    ramp = np.arange(n_species * (n_rho + 1), dtype=float).reshape(n_species, n_rho + 1)
    return 5.0 + 0.1 * ramp + offset, 7.0 + 0.2 * ramp + offset, np.linspace(-2.0, 4.0, n_rho + 1) + offset


def _make_grad_slice(offset: float = 0.0, n_species: int = 3, n_rho: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Face gradients in the file's units per metre, on ``n_rho + 1`` radii.

    These values differ from both profile fixtures. A swapped profile or gradient cannot pass an
    equality check.
    """
    ramp = np.arange(n_species * (n_rho + 1), dtype=float).reshape(n_species, n_rho + 1)
    return 11.0 + 0.1 * ramp + offset, 13.0 + 0.2 * ramp + offset


def _auto_faces(density: np.ndarray, n_rho: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a face triple whose rank matches ``density``, so only the centered arrays are on trial.

    The rank-consistency tests deliberately break one centered array; they should not also have to
    restate a matching set of face arrays just to reach the check they are about.
    """
    arr = np.asarray(density)
    if arr.ndim != 3:
        return _make_face_slice(n_rho=n_rho)
    slices = [_make_face_slice(offset=float(i), n_rho=n_rho) for i in range(arr.shape[0])]
    return tuple(np.stack([s[k] for s in slices]) for k in range(3))  # type: ignore[return-value]


def _auto_grads(density: np.ndarray, n_rho: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a gradient pair whose rank matches ``density``."""
    arr = np.asarray(density)
    if arr.ndim != 3:
        return _make_grad_slice(n_rho=n_rho)
    slices = [_make_grad_slice(offset=float(i), n_rho=n_rho) for i in range(arr.shape[0])]
    return tuple(np.stack([s[k] for s in slices]) for k in range(2))  # type: ignore[return-value]


def _write_h5(
    path: Path,
    *,
    rho: np.ndarray,
    density: np.ndarray,
    temperature: np.ndarray,
    er: np.ndarray,
    ts: Sequence[float] | None = None,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray] | None | str = "auto",
    grads: tuple[np.ndarray, np.ndarray] | None | str = "auto",
    rho_face: np.ndarray | None = None,
    r_grid_half: np.ndarray | None = None,
    final_time: float | None = FINAL_TIME,
    next_dt: float | None = NEXT_DT,
) -> Path:
    """Write a transport solution and optionally add its ``ts`` axis.

    ``faces`` contains density, temperature and ``Er`` on the face grid. ``grads`` contains density
    and temperature gradients. The ``auto`` value creates arrays with the density rank.

    ``None`` omits those datasets and simulates an older NEOPAX file. By default,
    ``r_grid_half = A_NEO * rho_face``. ``final_time`` and ``next_dt`` are omitted when set to
    ``None``.
    """
    n_rho = int(np.asarray(rho).size)
    if faces == "auto":
        faces = _auto_faces(density, n_rho=n_rho)
    if grads == "auto":
        grads = None if faces is None else _auto_grads(density, n_rho=n_rho)
    face_density, face_temperature, face_er = (None, None, None) if faces is None else faces
    grad_density, grad_temperature = (None, None) if grads is None else grads
    if rho_face is None and faces is not None:
        rho_face = _face_grid(n_rho)
    if r_grid_half is None and grads is not None and rho_face is not None:
        r_grid_half = A_NEO * rho_face
    write_transport_solution(
        path,
        rho=rho,
        density=density,
        temperature=temperature,
        er=er,
        rho_face=rho_face,
        density_face=face_density,
        temperature_face=face_temperature,
        er_face=face_er,
        r_grid_half=r_grid_half,
        density_grad_face=grad_density,
        temperature_grad_face=grad_temperature,
        final_time=final_time,
        next_dt=next_dt,
    )
    if ts is not None:
        with h5py.File(path, "a") as f:
            f.create_dataset("ts", data=np.asarray(ts, dtype=float))
    return path


def _make_transport_slice(
    n_radial: int = 5, rho_edge: float = 1.0, n_species: int = 3, **overrides: Any
) -> Any:
    """Build a ``TransportSolution`` consistent with a template of ``n_radial`` cells.

    Any field can be overridden to put exactly one of ``_validate_against_template``'s checks on
    trial while every other check still passes.
    """
    density, temperature, er = _make_slice(n_species=n_species, n_rho=n_radial)
    density_face, temperature_face, er_face = _make_face_slice(n_species=n_species, n_rho=n_radial)
    density_grad_face, temperature_grad_face = _make_grad_slice(n_species=n_species, n_rho=n_radial)
    fields: dict[str, Any] = {
        "rho": _center_grid(n_radial, rho_edge),
        "density": density,
        "temperature": temperature,
        "er": er,
        "rho_face": _face_grid(n_radial, rho_edge),
        "density_face": density_face,
        "temperature_face": temperature_face,
        "er_face": er_face,
        "density_grad_face": density_grad_face,
        "temperature_grad_face": temperature_grad_face,
        "minor_radius": A_NEO,
        "time_value": None,
        "final_time": FINAL_TIME,
        "next_dt": NEXT_DT,
    }
    return writer.TransportSolution(**{**fields, **overrides})


def _write_time_resolved(
    path: Path, *, n_times: int = 3, ts: Sequence[float] | None = None, final_time: float = FINAL_TIME
) -> Path:
    """Write a 3-D (n_time, n_species, n_rho) solution whose slice ``i`` is ``_make_slice(offset=i)``."""
    slices = [_make_slice(offset=float(i)) for i in range(n_times)]
    face_slices = [_make_face_slice(offset=float(i)) for i in range(n_times)]
    return _write_h5(
        path,
        rho=_center_grid(),
        density=np.stack([s[0] for s in slices]),
        temperature=np.stack([s[1] for s in slices]),
        er=np.stack([s[2] for s in slices]),
        ts=ts,
        final_time=final_time,
        rho_face=_face_grid(),
        faces=(
            np.stack([s[0] for s in face_slices]),
            np.stack([s[1] for s in face_slices]),
            np.stack([s[2] for s in face_slices]),
        ),
    )


def _write_static(path: Path, *, next_dt: float | None = NEXT_DT) -> Path:
    """Write a 2-D (n_species, n_rho) solution holding a single static slice with no time axis."""
    density, temperature, er = _make_slice()
    return _write_h5(
        path,
        rho=_center_grid(),
        density=density,
        temperature=temperature,
        er=er,
        rho_face=_face_grid(),
        faces=_make_face_slice(),
        next_dt=next_dt,
    )


def _run_writer(
    monkeypatch: pytest.MonkeyPatch,
    h5_path: Path,
    template: Path,
    output: Path,
    *extra_args: str,
) -> str:
    """Run the CLI end to end through ``main()`` with a patched argv and return the emitted text."""
    monkeypatch.setattr(
        "sys.argv",
        ["write_prescribed_profiles_from_transport_h5", str(h5_path), str(template),
         "--output-toml", str(output), *extra_args],
    )
    writer.main()
    return output.read_bytes().decode("utf-8")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split TOML at column-zero headers into header and section-text pairs.

    The first pair contains text before the first header and uses an empty name. Each section keeps
    its terminators for byte comparisons.
    """
    starts = [m.start() for m in re.finditer(r"^\[", text, flags=re.MULTILINE)]
    sections: list[tuple[str, str]] = []
    for start, end in zip([0, *starts], [*starts, len(text)], strict=True):
        chunk = text[start:end]
        sections.append((chunk.splitlines()[0] if chunk.startswith("[") else "", chunk))
    return sections


def _template_cfg(n_radial: int = 5, names: Sequence[str] = ("e", "D", "T"), **geometry: float) -> dict:
    """The minimal parsed-template shape ``_validate_against_template`` consumes."""
    return {"species": {"names": list(names)}, "geometry": {"n_radial": n_radial, **geometry}}


# Tracked-template result

# This test covers the complete command. The writer uses the last of three slices and produces valid
# TOML. It converts density to m^-3 and temperature to eV. ``Er`` stays in kV/m. Exact comparisons
# also check the shortest round-trip float format. Rows remain in [species].names order.
def test_main_prescribes_the_final_slice_of_the_tracked_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=SAVE_TIMES)
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    cfg = tomllib.loads(text)
    assert cfg["profiles"]["model"] == "prescribed"

    density, temperature, er = _make_slice(offset=2.0)  # Slice 2 is the final slice.
    assert_array_equal(np.asarray(cfg["profiles"]["density"]), density * 1.0e20)
    assert_array_equal(np.asarray(cfg["profiles"]["temperature"]), temperature * 1.0e3)
    assert_array_equal(np.asarray(cfg["profiles"]["Er"]), er)
    assert f"Rows follow [species].names: {', '.join(cfg['species']['names'])}." in text


# Face arrays for Stages 3 and 4

# NEOPAX evolves centered state, but Stages 3 and 4 sample faces. The writer copies ``*_faces``
# because NEOPAX uses the run's [boundary] settings to build them. Extrapolation can differ at a
# non-Dirichlet edge. The face arrays must have ``n_radial + 1`` SI values. Their fixture values
# differ from the centered values, so copying the wrong grid cannot pass.
def test_emitted_block_carries_the_face_arrays_from_the_solution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=SAVE_TIMES)
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    cfg = tomllib.loads(text)
    assert set(cfg["profiles"]) == {
        "model", "density", "temperature", "Er", "density_face", "temperature_face", "Er_face",
        "density_grad_face", "temperature_grad_face",
    }

    density_face, temperature_face, er_face = _make_face_slice(offset=2.0)
    assert_array_equal(np.asarray(cfg["profiles"]["density_face"]), density_face * 1.0e20)
    assert_array_equal(np.asarray(cfg["profiles"]["temperature_face"]), temperature_face * 1.0e3)
    assert_array_equal(np.asarray(cfg["profiles"]["Er_face"]), er_face)
    # [geometry].n_radial = 5 cells, so 6 faces against 5 centers.
    assert np.asarray(cfg["profiles"]["density_face"]).shape[1] == len(cfg["profiles"]["density"][0]) + 1


# NEOPAX calculates gradients from centered state and [boundary] settings. Face values alone do not
# determine them. The file stores gradients per metre on ``r_grid_half``. Readers use rho, so the
# writer applies one minor-radius factor and the normal unit conversion. Both sides use the same
# multiplication order, so the comparison is exact.
def test_emitted_block_carries_the_face_gradients_rescaled_to_per_rho(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=SAVE_TIMES)
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    profiles = tomllib.loads(text)["profiles"]
    density_grad, temperature_grad = _make_grad_slice(offset=2.0)
    assert_array_equal(np.asarray(profiles["density_grad_face"]), density_grad * A_NEO * 1.0e20)
    assert_array_equal(np.asarray(profiles["temperature_grad_face"]), temperature_grad * A_NEO * 1.0e3)
    assert np.asarray(profiles["density_grad_face"]).shape == np.asarray(profiles["density_face"]).shape


# The solution defines its minor radius through its two face grids. The writer must not assume this
# value. A different machine radius must change the gradient scale.
def test_gradient_scale_is_read_from_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    other_a_neo = 1.7
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "transport_solution.h5",
        rho=_center_grid(), density=density, temperature=temperature, er=er,
        faces=_make_face_slice(), r_grid_half=other_a_neo * _face_grid(),
    )
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    profiles = tomllib.loads(text)["profiles"]
    density_grad, temperature_grad = _make_grad_slice()
    assert_array_equal(np.asarray(profiles["density_grad_face"]), density_grad * other_a_neo * 1.0e20)
    assert_array_equal(np.asarray(profiles["temperature_grad_face"]), temperature_grad * other_a_neo * 1.0e3)


# Shape and finiteness checks cannot detect a radius-dependent scale error.
# The metre grid must be one uniform scaling of ``rho_face``.
# The error names the face with the largest mismatch.
def test_non_linear_face_grid_is_rejected(tmp_path: Path) -> None:
    density, temperature, er = _make_slice()
    bent = A_NEO * _face_grid()
    bent[2] *= 1.5
    h5_path = _write_h5(
        tmp_path / "transport_solution.h5",
        rho=_center_grid(), density=density, temperature=temperature, er=er,
        faces=_make_face_slice(), r_grid_half=bent,
    )
    with pytest.raises(ValueError, match=r"r_grid_half is not rho_face scaled by one minor radius\. At face 2"):
        writer._load_final_profiles(h5_path)


# A solution from before the staggered grid has no ``*_faces`` datasets. The writer rejects it and
# names the missing data before the next iteration reaches Stage 3 or 4.
def test_solution_without_face_datasets_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "transport_solution.h5",
        rho=_center_grid(), density=density, temperature=temperature, er=er,
        faces=None,  # a solution predating the staggered grid
    )
    monkeypatch.setattr(
        "sys.argv",
        ["write_prescribed_profiles_from_transport_h5", str(h5_path), str(TRACKED_TEMPLATE),
         "--output-toml", str(tmp_path / "out.toml")],
    )
    with pytest.raises(KeyError, match=r"density_faces"):
        writer.main()


# A later format has face state but no exported gradients. Reject it before Stage 3 or 4 reads the
# incomplete [profiles] block.
def test_solution_without_gradient_datasets_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "transport_solution.h5",
        rho=_center_grid(), density=density, temperature=temperature, er=er,
        faces=_make_face_slice(), grads=None,  # a solution predating the exported face gradients
    )
    monkeypatch.setattr(
        "sys.argv",
        ["write_prescribed_profiles_from_transport_h5", str(h5_path), str(TRACKED_TEMPLATE),
         "--output-toml", str(tmp_path / "out.toml")],
    )
    with pytest.raises(KeyError, match=r"density_grad_faces"):
        writer.main()


# Text outside [profiles]

# The writer replaces [profiles] and two clock values in [transport_solver]. All other sections
# must keep their bytes and order. The [sources] block also has a ``temperature`` key that must
# survive. This detects a key-only replacement with the wrong scope. The output must also use one
# line-ending style throughout.
def test_sections_outside_profiles_are_byte_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template_bytes = TRACKED_TEMPLATE.read_bytes()
    template_text = template_bytes.decode("utf-8")
    out_path = tmp_path / "out.toml"
    text = _run_writer(monkeypatch, _write_static(tmp_path / "transport_solution.h5"), TRACKED_TEMPLATE, out_path)

    template_sections = _split_sections(template_text)
    output_sections = _split_sections(text)
    assert [name for name, _ in output_sections] == [name for name, _ in template_sections]
    newline = "\r\n" if "\r\n" in template_text else "\n"
    for (name, expected), (_, actual) in zip(template_sections, output_sections, strict=True):
        if name == "[profiles]":
            continue
        if name != "[transport_solver]":
            assert actual == expected, f"section {name!r} was rewritten"
            continue
        expected_lines = expected.splitlines(keepends=True)
        actual_lines = actual.splitlines(keepends=True)
        changed = [pair for pair in zip(expected_lines, actual_lines, strict=True) if pair[0] != pair[1]]
        assert changed == [
            (f"t0 = 0.0{newline}", f"t0 = {FINAL_TIME!r}{newline}"),
            (f"dt = 1.0e-8{newline}", f"dt = {NEXT_DT!r}{newline}"),
        ]

    sources_lines = [line for line in template_text.splitlines() if line.startswith("temperature = [")]
    assert len(sources_lines) == 1  # the [sources] source-term list; the template's [profiles] has no such key
    assert sources_lines[0] in text

    newline_bytes = b"\r\n" if b"\r\n" in template_bytes else b"\n"
    out_bytes = out_path.read_bytes()
    assert out_bytes.count(b"\n") == out_bytes.count(newline_bytes)  # every terminator matches the template's


# Rewriting generated output

# Generated output is the next iteration's template. A second write must change only its clock and
# dated provenance. This detects ambiguous section boundaries and accumulated comments. The second
# solution uses the same profiles at a later time. Reusing the first solution would not advance
# ``t0``, so the writer correctly rejects that case.
def test_rewriting_its_own_output_moves_only_the_clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    later = 2.0 * FINAL_TIME
    first_h5 = _write_time_resolved(tmp_path / "first.h5", n_times=3, ts=SAVE_TIMES)
    second_h5 = _write_time_resolved(
        tmp_path / "second.h5", n_times=3, ts=[*SAVE_TIMES[:-1], later], final_time=later
    )
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    _run_writer(monkeypatch, first_h5, TRACKED_TEMPLATE, first)
    _run_writer(monkeypatch, second_h5, first, second)

    first_text, second_text = first.read_text(), second.read_text()
    profiles_span = writer._section_span(first_text, "profiles")
    assert profiles_span == writer._section_span(second_text, "profiles")
    # Only the dated provenance line can differ in [profiles].
    # All array lines and the line count must stay unchanged.
    first_lines = first_text[slice(*profiles_span)].splitlines()
    second_lines = second_text[slice(*profiles_span)].splitlines()
    differing = [(a, b) for a, b in zip(first_lines, second_lines, strict=True) if a != b]
    provenance = "# Prescribed profiles from transport time slice t = {} s."
    assert differing == [(provenance.format(FINAL_TIME), provenance.format(later))]
    assert tomllib.loads(second_text)["transport_solver"]["t0"] == later

    def _outside_the_rewritten_sections(text: str) -> str:
        """Return the text outside [profiles] and [transport_solver]."""
        profiles = writer._section_span(text, "profiles")
        solver = writer._section_span(text, "transport_solver")
        return text[: profiles[0]] + text[profiles[1] : solver[0]] + text[solver[1] :]

    assert _outside_the_rewritten_sections(first_text) == _outside_the_rewritten_sections(second_text)


# Transport clock

# These values come from the W7-X validation. ``t_final`` is ``10 * t_ref`` and bounds the full run.
# The other values contain the reached time and next step from one real solution.
# All three magnitudes differ, so an incorrect key cannot pass.
W7X_T_FINAL = 0.3740558
W7X_FINAL_TIME = 0.00374056
W7X_NEXT_DT = 0.004822043008189152


def _write_w7x_template(path: Path) -> Path:
    """Write an LF template with the W7-X clock in [transport_solver].

    A comment separates ``t0`` and ``t_final``. An ``rtol`` line follows ``dt``. These lines detect
    replacement of a complete range instead of two values.
    """
    lines = [
        "[geometry]", "n_radial = 5", "",
        "[species]", 'names = ["e", "D", "T"]', "",
        "[transport_solver]", "t0 = 0.0",
        "# Trinity3D's t_max = 10 normalised time units, 10 * t_ref = 10 * 0.037405577 s.",
        f"t_final = {W7X_T_FINAL}", "dt = 0.00374056", "rtol = 1.0e-6", "",
        "[profiles]", 'model = "standard_analytical"', "",
    ]
    path.write_bytes("\n".join(lines).encode("utf-8"))
    return path


# One W7-X call covers about one percent of ``t_final``. Each iteration must resume from the reached
# time. The writer copies ``final_time`` to ``t0`` and ``next_dt`` to ``dt`` without losing bits.
def test_w7x_clock_advances_onto_the_solutions_own(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template = _write_w7x_template(tmp_path / "w7x.toml")
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "w7x.h5", rho=_center_grid(), density=density, temperature=temperature, er=er,
        final_time=W7X_FINAL_TIME, next_dt=W7X_NEXT_DT,
    )
    text = _run_writer(monkeypatch, h5_path, template, tmp_path / "out.toml")

    solver = tomllib.loads(text)["transport_solver"]
    assert solver["t0"] == W7X_FINAL_TIME
    assert solver["dt"] == W7X_NEXT_DT
    assert f"\nt0 = {W7X_FINAL_TIME!r}\n" in text
    assert f"\ndt = {W7X_NEXT_DT!r}\n" in text
    assert "\nrtol = 1.0e-6\n" in text  # the line after dt is not part of the rewrite


# ``t_final`` is the absolute end time and must not move. Advancing it with ``t0`` would extend the
# target after every iteration and prevent termination.
def test_t_final_is_never_rewritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template = _write_w7x_template(tmp_path / "w7x.toml")
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "w7x.h5", rho=_center_grid(), density=density, temperature=temperature, er=er,
        final_time=W7X_FINAL_TIME, next_dt=W7X_NEXT_DT,
    )
    text = _run_writer(monkeypatch, h5_path, template, tmp_path / "out.toml")

    assert tomllib.loads(text)["transport_solver"]["t_final"] == W7X_T_FINAL
    assert f"\nt_final = {W7X_T_FINAL}\n" in text
    assert tomllib.loads(text)["transport_solver"]["t0"] == W7X_FINAL_TIME  # the clock did advance


# NEOPAX accepts ``t0 >= t_final`` without entering its step loop. It can return a zero-filled save
# buffer with ``failed`` set to false. The writer must not seed another iteration from this result.
# ``quick_run`` reaches this case in one call.
# Its [transport_solver] block must remain byte-identical.
@pytest.mark.parametrize("final_time", [1.4e-6, 2.0e-6])
def test_clock_at_or_past_t_final_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, final_time: float
) -> None:
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "done.h5", rho=_center_grid(), density=density, temperature=temperature, er=er,
        final_time=final_time, next_dt=NEXT_DT,
    )
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    solver = tomllib.loads(text)["transport_solver"]
    assert (solver["t0"], solver["dt"], solver["t_final"]) == (0.0, 1.0e-8, 1.4e-6)
    template_sections = dict(_split_sections(TRACKED_TEMPLATE.read_bytes().decode("utf-8")))
    assert dict(_split_sections(text))["[transport_solver]"] == template_sections["[transport_solver]"]


# The writer cannot prevent an overrun without [transport_solver].t_final. A missing block or key
# must raise the same named-key error. The writer must not choose its own horizon.
@pytest.mark.parametrize("cfg", [{}, {"transport_solver": {"t0": 0.0}}])
def test_template_without_a_configured_end_time_raises(cfg: dict) -> None:
    with pytest.raises(KeyError, match=r"\[transport_solver\]\.t_final"):
        writer._advance_transport_clock("[transport_solver]\nt0 = 0.0\n", cfg, final_time=1.0, next_dt=0.5)


# A run with ``final_time <= t0`` integrated nothing. Writing that time back would leave the clock
# unchanged while profiles change. The writer must report this stalled loop.
@pytest.mark.parametrize("final_time", [1.0, 0.5], ids=["equal_to_t0", "before_t0"])
def test_a_run_that_did_not_advance_past_t0_raises(final_time: float) -> None:
    text = "[transport_solver]\nt0 = 1.0\nt_final = 2.0\ndt = 0.5\n"
    with pytest.raises(ValueError, match="integrated nothing"):
        writer._advance_transport_clock(text, tomllib.loads(text), final_time=final_time, next_dt=0.25)


# The final slice and ``final_time`` must describe one instant. If the save grid stops early, the
# next iteration would pair earlier state with a later clock. Shape and finiteness checks cannot
# detect it.
def test_a_save_grid_stopping_short_of_final_time_raises(tmp_path: Path) -> None:
    h5_path = _write_time_resolved(tmp_path / "short.h5", n_times=3, ts=[0.0, 1.0e-7, 0.5 * FINAL_TIME])
    with pytest.raises(ValueError, match="dates its last saved slice"):
        writer._load_final_profiles(h5_path)


# Missing ``t0`` or ``dt`` would leave the clock unchanged. The next iteration would repeat the
# completed window without an error.
@pytest.mark.parametrize("key", ["t0", "dt"])
def test_transport_solver_missing_a_clock_key_raises(key: str) -> None:
    lines = ["[transport_solver]", "t0 = 0.0", "t_final = 2.0", "dt = 0.5", ""]
    text = "\n".join(line for line in lines if not line.startswith(f"{key} ="))
    with pytest.raises(KeyError, match=rf"assigns '{key}' 0 times"):
        writer._advance_transport_clock(text, tomllib.loads(text), final_time=1.0, next_dt=0.25)


# Replacing a value must preserve its spacing, trailing comment and line ending.
def test_replace_section_key_keeps_trailing_comments_and_spacing() -> None:
    section = "[transport_solver]\r\nt0 = 0.0   # start of this window\r\ndt = 1.0e-8\r\n"
    assert writer._replace_section_key(section, section="transport_solver", key="t0", value="0.5") == (
        "[transport_solver]\r\nt0 = 0.5   # start of this window\r\ndt = 1.0e-8\r\n"
    )


# Final [profiles] block in an LF template

def _write_lf_template(path: Path) -> Path:
    """Write a minimal LF template with [profiles] as its final block."""
    lines = [
        "# scratch template", "", "[geometry]", "n_radial = 5", "",
        "[species]", 'names = ["e", "D", "T"]', "",
        "[sources]", 'temperature = ["dt_reaction"]', "",
        "[transport_solver]", "t0 = 0.0", "t_final = 1.4e-6", "dt = 1.0e-8", "",
        "[profiles]", 'model = "standard_analytical"', "n0 = 4.21", "",
    ]
    path.write_bytes("\n".join(lines).encode("utf-8"))
    return path


# A final [profiles] block has no following header. Its replacement must continue to the end of the
# file. This case also checks LF output. No carriage returns can appear. Absence of ``n0`` confirms
# removal of the old analytical keys.
def test_profiles_as_last_section_of_an_lf_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template = _write_lf_template(tmp_path / "template.toml")
    out_path = tmp_path / "out.toml"
    text = _run_writer(monkeypatch, _write_static(tmp_path / "transport_solution.h5"), template, out_path)

    cfg = tomllib.loads(text)
    assert cfg["profiles"]["model"] == "prescribed"
    assert cfg["sources"]["temperature"] == ["dt_reaction"]  # the preceding section is untouched
    assert b"\r" not in out_path.read_bytes()
    assert "n0 = 4.21" not in text


# Slice selection

# The writer copies the final run clock, so it must also copy the final profile slice. An earlier
# slice would describe state that the solver already passed.
def test_the_last_slice_is_prescribed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=SAVE_TIMES)
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    profiles = tomllib.loads(text)["profiles"]
    density, temperature, er = _make_slice(offset=2.0)
    assert_array_equal(np.asarray(profiles["density"]), density * 1.0e20)
    assert_array_equal(np.asarray(profiles["temperature"]), temperature * 1.0e3)
    assert_array_equal(np.asarray(profiles["Er"]), er)
    assert "transport time slice t = 4e-07 s" in text


# A 2-D file holds one static slice and no time axis, so every array is read whole.
def test_static_file_is_read_whole(tmp_path: Path) -> None:
    h5_path = _write_static(tmp_path / "static.h5")
    density, temperature, er = _make_slice()
    density_face, temperature_face, er_face = _make_face_slice()
    density_grad, temperature_grad = _make_grad_slice()

    loaded = writer._load_final_profiles(h5_path)
    assert_array_equal(loaded.rho, _center_grid())
    assert_array_equal(loaded.density, density)
    assert_array_equal(loaded.temperature, temperature)
    assert_array_equal(loaded.er, er)
    assert_array_equal(loaded.rho_face, _face_grid())
    assert_array_equal(loaded.density_face, density_face)
    assert_array_equal(loaded.temperature_face, temperature_face)
    assert_array_equal(loaded.er_face, er_face)
    # The loader is where per metre becomes per unit rho, so the slice already carries the factor.
    assert_array_equal(loaded.density_grad_face, density_grad * A_NEO)
    assert_array_equal(loaded.temperature_grad_face, temperature_grad * A_NEO)
    assert loaded.time_value is None


# ``next_dt`` becomes the next iteration's ``dt``. A non-positive value would stall the solver.
# Finiteness does not detect this error.
@pytest.mark.parametrize("next_dt", [0.0, -1.0e-8], ids=["zero", "negative"])
def test_non_positive_next_dt_is_rejected(tmp_path: Path, next_dt: float) -> None:
    h5_path = _write_static(tmp_path / "static.h5", next_dt=next_dt)
    with pytest.raises(ValueError, match="next_dt"):
        writer._load_final_profiles(h5_path)


# Provenance includes a time only when the file has ``ts``. A time-resolved file without ``ts`` uses
# the undated line. This confirms that ``ts`` presence, not array rank, controls the comment.
def test_time_resolved_file_without_ts_omits_slice_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    no_ts = _write_time_resolved(tmp_path / "no_ts.h5", n_times=2)
    text = _run_writer(monkeypatch, no_ts, TRACKED_TEMPLATE, tmp_path / "no_ts.toml")
    assert "time slice t =" not in text
    assert "# Prescribed profiles from the previous closed-loop transport solution." in text


# File-level loader errors

# A transport solution written by an older or different solver may simply lack a dataset. That is a
# missing-key condition, so it must surface as KeyError naming the dataset, not as an index error
# deeper in the slicing.
def test_missing_dataset_raises_key_error(tmp_path: Path) -> None:
    _, temperature, er = _make_slice()
    density_face, temperature_face, er_face = _make_face_slice()
    density_grad, temperature_grad = _make_grad_slice()
    path = write_transport_solution(tmp_path / "no_density.h5", rho=_center_grid(),
                                    temperature=temperature, er=er, rho_face=_face_grid(),
                                    density_face=density_face, temperature_face=temperature_face,
                                    er_face=er_face, r_grid_half=A_NEO * _face_grid(),
                                    density_grad_face=density_grad,
                                    temperature_grad_face=temperature_grad,
                                    final_time=FINAL_TIME, next_dt=NEXT_DT)
    with pytest.raises(KeyError, match=r"missing required NEOPAX transport datasets: density\."):
        writer._load_final_profiles(path)


# A solution without a clock cannot advance the next iteration. Reusing the template's ``t0`` would
# repeat the completed window. Report it like any other missing dataset.
@pytest.mark.parametrize("dropped", ["final_time", "next_dt"])
def test_solution_without_the_clock_is_rejected(tmp_path: Path, dropped: str) -> None:
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "no_clock.h5",
        rho=_center_grid(), density=density, temperature=temperature, er=er,
        **{dropped: None},
    )
    with pytest.raises(KeyError, match=rf"missing the transport clock datasets: {dropped}\."):
        writer._load_final_profiles(h5_path)


# A NaN or infinity anywhere in the selected slice would be written straight into the next
# iteration's template and only fail much later inside a solver, so it is rejected at read time with
# the offending dataset named.
def test_non_finite_values_in_the_slice_raise(tmp_path: Path) -> None:
    density, temperature, er = _make_slice()
    density[1, 2] = np.nan
    path = _write_h5(tmp_path / "nan.h5", rho=_center_grid(), density=density,
                     temperature=temperature, er=er)
    with pytest.raises(ValueError, match="dataset 'density' holds non-finite values"):
        writer._load_final_profiles(path)


# Density must be 2-D or 3-D. Report another rank against density before writing a malformed block.
def test_density_of_unsupported_rank_raises(tmp_path: Path) -> None:
    rho = _center_grid()
    path = _write_h5(tmp_path / "flat.h5", rho=rho, density=rho, temperature=rho, er=rho)
    with pytest.raises(ValueError, match="dataset 'density' must be 2D"):
        writer._load_final_profiles(path)


# ``Er`` shares the density time and radius axes but has no species axis. Its rank is one lower.
# Reject inconsistent temperature or ``Er`` ranks before selecting a slice.
@pytest.mark.parametrize("broken", ["temperature", "Er"])
def test_inconsistent_dataset_ranks_raise(tmp_path: Path, broken: str) -> None:
    rho = _center_grid()
    density, temperature, er = _make_slice()
    if broken == "temperature":
        # 3-D density against a 2-D temperature.
        arrays = dict(density=np.stack([density, density]), temperature=temperature, er=np.stack([er, er]))
    else:
        # 2-D density against an Er that wrongly carries the species axis.
        arrays = dict(density=density, temperature=temperature, er=np.stack([er, er, er]))
    path = _write_h5(tmp_path / f"{broken}.h5", rho=rho, **arrays)
    with pytest.raises(ValueError, match="inconsistent center dataset ranks"):
        writer._load_final_profiles(path)


# Both states are internally valid. The centered state has three slices, but the face state has two.
# One index cannot select the same instant on both grids. Report the mismatch before indexing.
def test_disagreeing_time_axes_raise(tmp_path: Path) -> None:
    rho = _center_grid()
    density, temperature, er = _make_slice()
    two_slice = np.stack([density, density])
    path = _write_h5(tmp_path / "axes.h5", rho=rho, density=np.stack([density] * 3),
                     temperature=np.stack([temperature] * 3), er=np.stack([er] * 3),
                     faces=_auto_faces(two_slice, n_rho=rho.size), grads=_auto_grads(two_slice, n_rho=rho.size))
    with pytest.raises(ValueError, match="time axes disagree"):
        writer._load_final_profiles(path)


# Template validation

# [profiles] stores no radial coordinates. Readers rebuild both grids from [geometry]. A slice with
# ``rho_edge = 0.7`` must fail against a template that defaults to 1.0, even with equal cell counts.
def test_rho_grid_must_match_the_template_geometry() -> None:
    assert writer._validate_against_template(
        _template_cfg(rho_edge=0.7), slice_=_make_transport_slice(rho_edge=0.7)
    ) == ["e", "D", "T"]
    with pytest.raises(ValueError, match="does not match the template's cell centers"):
        writer._validate_against_template(_template_cfg(), slice_=_make_transport_slice(rho_edge=0.7))


# Check centers and faces separately. Matching centers cannot hide incorrect faces. Otherwise, the
# Stage 3 and 4 flux grid could differ from the NEOPAX interpolation grid.
def test_face_grid_must_match_the_template_geometry() -> None:
    with pytest.raises(ValueError, match="rho_face .* does not match the template's faces"):
        writer._validate_against_template(
            _template_cfg(), slice_=_make_transport_slice(rho_face=np.linspace(0.0, 0.9, 6))
        )


# Face arrays have one more radial entry than centered arrays. A shorter array omits the outer face.
# Gradients must span the same faces as their profiles.
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("density_face", r"density_faces has shape .* expected \(3, 6\)"),
        ("temperature_face", r"temperature_faces has shape .* expected \(3, 6\)"),
        ("density_grad_face", r"density_grad_faces has shape .* expected \(3, 6\)"),
        ("temperature_grad_face", r"temperature_grad_faces has shape .* expected \(3, 6\)"),
    ],
)
def test_face_arrays_must_carry_one_more_radius_than_the_centers(field: str, message: str) -> None:
    trimmed = getattr(_make_transport_slice(), field)[:, :-1]
    with pytest.raises(ValueError, match=message):
        writer._validate_against_template(_template_cfg(), slice_=_make_transport_slice(**{field: trimmed}))


# Er carries no species axis, so its face array is the one whose width is pinned as a bare length.
def test_er_face_must_carry_one_more_radius_than_the_centers() -> None:
    slice_ = _make_transport_slice()
    with pytest.raises(ValueError, match=r"Er_faces has shape .* expected \(6,\)"):
        writer._validate_against_template(
            _template_cfg(), slice_=_make_transport_slice(er_face=slice_.er_face[:-1])
        )


# [profiles] cannot name its rows or grid. The template must supply [species].names and
# [geometry].n_radial. Missing metadata must raise a named ``KeyError`` without a default.
@pytest.mark.parametrize(
    ("missing", "message"),
    [("species", r"\[species\]\.names"), ("geometry", r"\[geometry\]\.n_radial")],
)
def test_template_missing_required_tables_raises_key_error(missing: str, message: str) -> None:
    cfg = _template_cfg()
    del cfg[missing]
    with pytest.raises(KeyError, match=message):
        writer._validate_against_template(cfg, slice_=_make_transport_slice())


# Every profile row needs a name. The species count must match [species].names exactly.
def test_species_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"3 species but the template \[species\]\.names lists 2"):
        writer._validate_against_template(_template_cfg(names=("e", "D")), slice_=_make_transport_slice())


# Readers size arrays from [geometry].n_radial. Reject another radius count before truncation or
# broadcasting can occur.
def test_n_radial_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"5 radii but the template \[geometry\]\.n_radial is 7"):
        writer._validate_against_template(_template_cfg(n_radial=7), slice_=_make_transport_slice())


# Density must include species, temperature must match density and ``Er`` must match ``rho``. Each
# error names the mismatched array.
def test_slice_shape_mismatches_raise() -> None:
    slice_ = _make_transport_slice()
    with pytest.raises(ValueError, match="species-resolved density slice"):
        writer._validate_against_template(_template_cfg(), slice_=_make_transport_slice(density=slice_.density[0]))
    with pytest.raises(ValueError, match="temperature shape"):
        writer._validate_against_template(
            _template_cfg(), slice_=_make_transport_slice(temperature=slice_.temperature[:, :4])
        )
    with pytest.raises(ValueError, match="Er shape"):
        writer._validate_against_template(_template_cfg(), slice_=_make_transport_slice(er=slice_.er[:4]))


# Radial differencing needs at least three faces. A one-cell template passes the other checks but
# cannot produce usable feedback.
def test_fewer_than_three_radii_raises() -> None:
    with pytest.raises(ValueError, match="at least 3 faces"):
        writer._validate_against_template(_template_cfg(n_radial=1), slice_=_make_transport_slice(n_radial=1))


# Two cells give three faces, which is enough to difference.
def test_two_cells_pass() -> None:
    assert writer._validate_against_template(
        _template_cfg(n_radial=2), slice_=_make_transport_slice(n_radial=2)
    ) == ["e", "D", "T"]


# [profiles] replacement

# A missing [profiles] block is a configuration error. Appending one could leave the reader's
# default profile model active.
def test_template_without_a_profiles_section_raises() -> None:
    with pytest.raises(ValueError, match=r"no \[profiles\] section"):
        writer._replace_profiles_section('[general]\r\nmode = "transport"\r\n', "[profiles]\n")


# A trailing header comment must not prevent recognition. Replace that comment with the block, and
# keep the following section in place.
def test_profiles_header_with_a_trailing_comment_is_replaced() -> None:
    text = '[profiles] # analytical\nmodel = "standard_analytical"\n\n[sources]\nx = 1\n'
    out = writer._replace_profiles_section(text, '[profiles]\nmodel = "prescribed"\n\n')
    assert out == '[profiles]\nmodel = "prescribed"\n\n[sources]\nx = 1\n'


# A static solution can have one timestamp. Stages 3 and 4 also read ``ts[0]``. The contract permits
# one entry, so every stage reports the same time.
def test_static_solution_with_a_timestamp_records_its_slice_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_static(tmp_path / "static.h5")
    with h5py.File(h5_path, "a") as f:
        f.create_dataset("ts", data=np.array([FINAL_TIME]))
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")
    assert "transport time slice t = 4e-07 s" in text


# Parse-time validation of the emitted file

# ``prepare_neopax_config`` validates [profiles] while Snakemake parses the workflow, before any
# stage runs. The emitted file is the next iteration's template. This is the only test holding the
# emitted key set, ranks and radial lengths to what that validator demands.
def test_emitted_common_input_passes_the_parse_time_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=SAVE_TIMES)
    emitted = tmp_path / "emitted.toml"
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, emitted)
    assert tomllib.loads(text)["profiles"]["model"] == "prescribed"  # the validated branch is the one reached

    out = tmp_path / "out"
    prepare_neopax_config(
        s5_config_template=str(emitted),
        s5_resolved_config=str(out / "stage5_transport" / "common_input_updated.toml"),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "sfincs_flux.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )
