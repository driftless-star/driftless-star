"""Tests for the ``src.stage5_helper`` readers of the NEOPAX template.

NEOPAX is configured by a TOML file, not CLI flags. ``prepare_neopax_config``
writes a path-resolved copy of the shared template under the run's Stage 5 output
dir, rewriting its five path fields *relative to that copy's own directory* (NEOPAX
runs there) and never touching the committed template. These tests pin the rewrite
targets, the trailing slash on ``transport_output_dir``, and template immutability.
``read_rho_edge`` reads the radial grid's outer edge back out of the same template,
which is what keeps the Stage 4 relabelling step on the grid NEOPAX will build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stage5_helper import (
    prepare_neopax_config,
    read_rho_edge,
    resolve_pressure_convergence_method,
)

_TEMPLATE = (
    'vmec_file = "PLACEHOLDER"\n'
    'boozer_file = "PLACEHOLDER"\n'
    'neoclassical_file = "PLACEHOLDER"\n'
    'turbulence_file = "PLACEHOLDER"\n'
    'transport_output_dir = "PLACEHOLDER"\n'
)


# --- convergence method ---


@pytest.mark.parametrize(
    "config",
    [{}, {"convergence": {}}, {"convergence": {"pressure_rel_tol": 1.0e-2}}],
)
def test_pressure_convergence_method_defaults_to_pointwise(config: dict) -> None:
    assert resolve_pressure_convergence_method(config) == "pointwise"


@pytest.mark.parametrize("method", ["rms", "pointwise", "t_final"])
def test_pressure_convergence_method_accepts_supported_values(method: str) -> None:
    assert resolve_pressure_convergence_method({"convergence": {"method": method}}) == method


@pytest.mark.parametrize("method", ["RMS", "maximum", True, None, 1])
def test_pressure_convergence_method_rejects_unsupported_values(method: object) -> None:
    with pytest.raises(ValueError, match=r"config\['convergence'\]\['method'\]"):
        resolve_pressure_convergence_method({"convergence": {"method": method}})


@pytest.mark.parametrize("convergence", [None, "rms", ["rms"]])
def test_pressure_convergence_method_requires_a_mapping(convergence: object) -> None:
    with pytest.raises(ValueError, match=r"config\['convergence'\] must be a mapping"):
        resolve_pressure_convergence_method({"convergence": convergence})


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a tmp run and resolve the NEOPAX config; return (template, resolved)."""
    template = tmp_path / "inputs" / "common_input.toml"
    template.parent.mkdir(parents=True)
    template.write_text(_TEMPLATE)

    out = tmp_path / "out"
    resolved = out / "stage5_transport" / "common_input_updated.toml"  # parent not pre-created
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(resolved),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "dkx_flux_profiles.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )
    return template, resolved


# NEOPAX (Stage 5) is configured by a TOML file, not CLI flags. `prepare_neopax_config` writes a copy of the template
# into the Stage 5 output dir and rewrites its five input-path fields so each points at the right upstream artifact,
# relative to that copy's own location. This reads the copy and asserts all five fields now hold the expected
# `../stageN/...` relative path, and that the output dir field resolves to `./` (the copy's own directory).
def test_rewrites_five_paths_relative_and_quoted(tmp_path: Path) -> None:
    _, resolved = _prepare(tmp_path)
    text = resolved.read_text()  # also asserts the parent dir was created
    assert 'vmec_file = "../stage1_equilibrium/wout.nc"' in text
    assert 'boozer_file = "../stage2_boozer/boozmn.nc"' in text
    assert 'neoclassical_file = "../stage3_neoclassical/dkx_flux_profiles.h5"' in text
    assert 'turbulence_file = "../stage4_turbulence/neopax_fluxes.h5"' in text
    # The output dir is the copy's own dir, so it resolves to "./" with a trailing slash.
    assert 'transport_output_dir = "./"' in text


# The rewrite must happen on the copy, never the shared template. After running the same preparation, this reads the
# original template file back and asserts its contents are byte-for-byte unchanged, proving the committed template is
# never mutated by a run.
def test_template_left_unmodified(tmp_path: Path) -> None:
    template, _ = _prepare(tmp_path)
    assert template.read_text() == _TEMPLATE


# --- read_rho_edge ---

def _write_template(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "common_input.toml"
    path.write_text(body)
    return path


def test_read_rho_edge_returns_the_configured_value(tmp_path: Path) -> None:
    assert read_rho_edge(str(_write_template(tmp_path, "[geometry]\nrho_edge = 0.7\n"))) == 0.7


# NEOPAX's own default is 1.0, so a template that never mentions rho_edge grids out to the boundary. Both an absent key
# and an absent [geometry] table have to resolve to it, or the relabelling step would reject a perfectly good grid.
@pytest.mark.parametrize("body", ["[geometry]\nn_radial = 5\n", "[species]\nn_species = 1\n"])
def test_read_rho_edge_defaults_to_one(tmp_path: Path, body: str) -> None:
    assert read_rho_edge(str(_write_template(tmp_path, body))) == 1.0


# rho_edge scales every stage's radial grid, so a value outside (0, 1] is never recoverable downstream. It is caught at
# Snakefile parse time, before any stage runs. A bool passes isinstance(x, int) in Python, hence its own case here.
@pytest.mark.parametrize("value", ['"0.7"', "true", "1.5", "0.0", "-0.5", "nan"])
def test_read_rho_edge_rejects_a_value_outside_the_unit_interval(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match=r"\[geometry\].rho_edge"):
        read_rho_edge(str(_write_template(tmp_path, f"[geometry]\nrho_edge = {value}\n")))
