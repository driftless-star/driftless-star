"""Tests for the Stage 4 GKX radial-scan helpers.

These reuse the pure functions from ``gkx_radial_scan.py``, loaded by path.
``_gkx_flux_to_neopax_units`` converts a gyro-Bohm flux to NEOPAX physical
units; ``_expand_axis_zero_if_needed`` prepends the magnetic-axis radius (rho=0,
zero flux) when the scan skipped it. GKX runs via subprocess, so loading
the module and calling these helpers needs no solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import elementary_charge, proton_mass

from tests.helpers.stage_import import load_stage_module

scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")


# Flux unit conversion

# `_gkx_flux_to_neopax_units` converts a dimensionless (gyro-Bohm) flux into physical units. The heat flux Q
# carries one extra factor of temperature (in eV) compared to the particle flux Gamma. Converting the same raw value
# both ways, this asserts their ratio Q/Gamma equals the temperature expressed in eV (2 keV = 2000 eV), a cheap way to
# check the extra-temperature factor without pinning the full absolute scale.
def test_flux_units_q_over_gamma_is_temperature_in_ev() -> None:
    kwargs = dict(density_ref_state=0.5, temperature_ref_keV=2.0, mass_ref_mp=2.0, rho_star_physical=0.01)
    gamma = scan._gkx_flux_to_neopax_units(3.0, kind="Gamma", **kwargs)
    q = scan._gkx_flux_to_neopax_units(3.0, kind="Q", **kwargs)
    assert_allclose(q / gamma, 2.0 * 1.0e3, rtol=1e-12)  # Q carries an extra factor of T expressed in eV


# The absolute-value counterpart to the ratio test above. It recomputes the full expected Gamma from first principles
# (reference density x thermal speed x rho_star squared, with the thermal speed built from physical constants) and
# asserts the helper returns exactly that. Because the ratio tests only compare fluxes to each other, only this one
# would catch a silent error in the overall unit convention.
def test_flux_units_absolute_scale_matches_reference_convention() -> None:
    # Absolute pin for the reference-species convention: Gamma = gamma_hat * n_ref * vth_ref * rho_star^2
    # with vth_ref = sqrt(2 T e / m). Here gamma_hat=3, n=0.5e20 m^-3, T=2000 eV, mass=2 m_p,
    # rho_star=0.01, so the trailing rho_star^2 = 1e-4 factor is exercised end to end. The ratio-only
    # tests below cannot see a silent drift in this overall unit convention.
    vth_ref = np.sqrt(2.0 * 2000.0 * elementary_charge / (2.0 * proton_mass))
    expected_gamma = 3.0 * 0.5e20 * vth_ref * 1.0e-4
    gamma = scan._gkx_flux_to_neopax_units(
        3.0, density_ref_state=0.5, temperature_ref_keV=2.0, mass_ref_mp=2.0, rho_star_physical=0.01, kind="Gamma"
    )
    assert_allclose(gamma, expected_gamma, rtol=1e-12)


# Gyro-Bohm fluxes scale with the square of `rho_star`. Converting the same raw flux at two rho_star values differing by
# 2x (0.01 vs 0.02), this asserts the results differ by 4x (2 squared), confirming the rho_star-squared dependence.
def test_flux_units_scale_as_rho_star_squared() -> None:
    base = dict(density_ref_state=1.0, temperature_ref_keV=1.0, mass_ref_mp=1.0, kind="Gamma")
    small = scan._gkx_flux_to_neopax_units(1.0, rho_star_physical=0.01, **base)
    large = scan._gkx_flux_to_neopax_units(1.0, rho_star_physical=0.02, **base)
    assert_allclose(large / small, 4.0, rtol=1e-12)  # gyro-Bohm scaling goes as rho_star^2


def test_flux_units_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        scan._gkx_flux_to_neopax_units(
            1.0, density_ref_state=1.0, temperature_ref_keV=1.0, mass_ref_mp=1.0, rho_star_physical=0.01, kind="bogus"
        )


# Magnetic-axis expansion

def _expand_inputs(n_species: int = 3, n: int = 2) -> dict:
    return {
        "rho": np.linspace(0.5, 1.0, n),
        "r_physical": np.linspace(0.5, 1.0, n),
        "rho_index": np.arange(1, n + 1),
        "torflux": np.linspace(0.25, 1.0, n),
        "er": np.linspace(1.0, 2.0, n),
        "heat_flux": np.linspace(1.0, 2.0, n),
        "particle_flux": np.linspace(1.0, 2.0, n),
        "heat_flux_species": np.ones((n, n_species)),
        "particle_flux_species": np.ones((n, n_species)),
        "gamma_neopax": np.arange(1.0, n_species * n + 1.0).reshape(n_species, n),
        "q_neopax": np.arange(1.0, n_species * n + 1.0).reshape(n_species, n) + 10.0,
    }


# `_expand_axis_zero_if_needed` re-inserts the magnetic-axis face (rho = 0) that the scan skipped, but only when the
# run manifest describes the full face grid. With an empty manifest (no `source_rho`), it should do nothing: this
# asserts the arrays come back unchanged, still length 2 on the radius axis.
def test_expand_passthrough_without_source_rho() -> None:
    arrays = _expand_inputs()
    out = scan._expand_axis_zero_if_needed({}, **arrays)
    # no source_rho -> arrays returned unchanged (radius axis stays length 2)
    assert out[0].shape == (2,)
    assert_allclose(out[0], arrays["rho"])
    assert out[9].shape == arrays["gamma_neopax"].shape


# The happy path where all guards pass. Given a manifest whose `source_rho` is the full face grid and whose executed
# runs covered every non-axis face, it prepends the missing axis point everywhere: the radial coordinate becomes
# [0, 0.5, 1.0], the physical radius is `a_minor * rho`, toroidal flux is `rho^2`, `Er` is taken from the manifest,
# and the flux arrays gain a new zero-filled axis column while their existing columns are preserved. Each assertion
# checks one of those reconstructed quantities.
def test_expand_prepends_axis_zero() -> None:
    manifest = {"source_rho": [0.0, 0.5, 1.0], "source_er": [0.0, 1.5, 3.0], "geometry": {"a_minor": 2.0}}
    arrays = _expand_inputs()  # rho size 2, rho_index [1, 2]
    (full_rho, full_r, full_rho_index, full_torflux, full_er, _heat, _particle,
     _heat_s, _particle_s, full_gamma, full_q) = scan._expand_axis_zero_if_needed(manifest, **arrays)

    assert_allclose(full_rho, [0.0, 0.5, 1.0], rtol=1e-12)
    assert_allclose(full_r, [0.0, 1.0, 2.0], rtol=1e-12)            # a_minor * full_rho
    assert full_rho_index.tolist() == [0, 1, 2]
    assert_allclose(full_torflux, [0.0, 0.25, 1.0], rtol=1e-12)     # full_rho ** 2
    assert_allclose(full_er, [0.0, 1.5, 3.0], rtol=1e-12)  # Er from manifest source_er, not the per-run er
    assert full_gamma.shape == (3, 3)
    assert_allclose(full_gamma[:, 0], 0.0)              # prepended axis column is zero-filled
    assert_allclose(full_gamma[:, 1:], arrays["gamma_neopax"], rtol=1e-12)
    assert_allclose(full_q[:, 1:], arrays["q_neopax"], rtol=1e-12)


# The radius axis still expands, but here the manifest's `source_er` is the wrong length for the expanded axis. Rather
# than misalign the electric field with the wrong radii, the helper fills `Er` with zeros. This asserts the radius axis
# expanded correctly while `Er` came back all zeros.
def test_expand_er_zero_filled_on_source_er_size_mismatch() -> None:
    manifest = {"source_rho": [0.0, 0.5, 1.0], "source_er": [0.0, 1.5], "geometry": {"a_minor": 2.0}}
    arrays = _expand_inputs()  # rho size 2, rho_index [1, 2]
    out = scan._expand_axis_zero_if_needed(manifest, **arrays)
    # source_er length (2) does not match the expanded radius axis (3): the axis still expands,
    # but Er drops to zeros rather than misaligning.
    assert_allclose(out[0], [0.0, 0.5, 1.0], rtol=1e-12)
    assert_allclose(out[4], 0.0)


# Expansion is only safe if the scanned faces were exactly indices [1, 2] (i.e. the axis face at index 0 really was
# the only one skipped). Here `rho_index` is [0, 1] instead, so a guard trips and the helper refuses to expand. This
# asserts the arrays are returned unchanged (radius axis stays 2), protecting against inserting an axis point when the
# data doesn't line up.
def test_expand_passthrough_on_index_mismatch() -> None:
    manifest = {"source_rho": [0.0, 0.5, 1.0], "geometry": {"a_minor": 2.0}}
    arrays = _expand_inputs()
    arrays["rho_index"] = np.array([0, 1])  # not [1, 2] -> guard trips, no expansion
    out = scan._expand_axis_zero_if_needed(manifest, **arrays)
    assert out[0].shape == (2,)


# Perturbed-species warnings

# A name matching no runtime species must produce a stderr warning (not a silent skip) that names the channel, quotes
# the bad name, lists the available runtime species, and states only the unperturbed base runs execute for it.
def test_warn_unmatched_perturb_species_warns_on_bad_name(capsys: pytest.CaptureFixture[str]) -> None:
    scan._warn_unmatched_perturb_species(["He", "D"], ["e", "D", "T"], flag_label="perturb_density_species")
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1  # only the unmatched name warns; the valid "D" stays quiet
    assert "perturb_density_species 'He' matches no runtime species" in lines[0]
    assert "available: e, D, T" in lines[0]
    assert "only unperturbed base runs will execute" in lines[0]


# The match is case-insensitive and whitespace-trimmed, mirroring the runtime lookup, so a differently cased or padded
# name that resolves to a real runtime species must not warn. Empty request lists must also stay silent.
def test_warn_unmatched_perturb_species_quiet_on_valid_and_empty(capsys: pytest.CaptureFixture[str]) -> None:
    scan._warn_unmatched_perturb_species([" d ", "T"], ["e", "D", "T"], flag_label="perturb_temperature_species")
    scan._warn_unmatched_perturb_species([], ["e", "D", "T"], flag_label="perturb_density_species")
    assert capsys.readouterr().err == ""


# Run identity in flux_summary.h5

def _collect_run_spec(tmp_path: Path, index: int, rho_index: int, rho: float, **response_fields: object) -> dict:
    # Pointing output_prefix at a directory with no diagnostics CSV drives collect down its zero-fill branch, which
    # skips the per-species flux conversion entirely; the run spec can then omit runtime_species and the reference
    # normalization data, keeping the fixture at the minimum the identity datasets need.
    return {
        "index": index,
        "rho_index": rho_index,
        "rho": rho,
        "torflux": rho**2,
        "Er": 0.0,
        "output_prefix": str(tmp_path / f"runs/run_{index:03d}/run"),
        **response_fields,
    }


def _run_collect(tmp_path: Path, runs: list[dict]) -> Path:
    manifest = {
        "runs": runs,
        "runtime_species_names": ["D"],
        "electron_model": "adiabatic",
        "neopax_result": "",
        "common_config": "",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = argparse.Namespace(
        manifest=str(manifest_path),
        out=str(tmp_path / "flux_summary.h5"),
        neopax_flux_out=str(tmp_path / "neopax_fluxes.h5"),
        average_window=5.0,
        t_final=None,
        plot=False,
        plot_run_heat_traces=False,
    )
    assert scan.cmd_collect(args) == 0
    return Path(args.out)


# flux_summary.h5 holds one row per executed run, so in fd_gradients mode base and perturbed rows sit at duplicate rho
# values; the file must carry the run identity as datasets so those rows are distinguishable without joining against
# runs.csv by row order.
def test_collect_flux_summary_carries_run_identity_datasets(tmp_path: Path) -> None:
    runs = [
        _collect_run_spec(tmp_path, 0, 1, 0.25, response_label="base", perturb_species="none", perturb_delta=0.0),
        _collect_run_spec(tmp_path, 1, 1, 0.25, response_label="density_gradient", perturb_species="D", perturb_delta=-0.5),
    ]
    out = _run_collect(tmp_path, runs)
    with h5py.File(out, "r") as f:
        assert f["response_label"].asstr()[...].tolist() == ["base", "density_gradient"]
        assert f["perturb_species"].asstr()[...].tolist() == ["none", "D"]
        assert_allclose(f["perturb_delta"][...], [0.0, -0.5])
        # rows stay per-run in manifest order: both rho entries present, no synthetic axis point inserted
        assert_allclose(f["rho"][...], [0.25, 0.25])


# Manifests written before response_mode existed have run specs without any response/perturb fields; collect must still
# write the identity datasets, defaulting every row to a base run so the schema is uniform across old and new outputs.
def test_collect_flux_summary_identity_defaults_for_legacy_manifest(tmp_path: Path) -> None:
    out = _run_collect(tmp_path, [_collect_run_spec(tmp_path, 0, 1, 0.5)])
    with h5py.File(out, "r") as f:
        assert f["response_label"].asstr()[...].tolist() == ["base"]
        assert f["perturb_species"].asstr()[...].tolist() == ["none"]
        assert_allclose(f["perturb_delta"][...], [0.0])


# Perturbed-species validation

# Species names are pasted into perturbed run-directory names, which Snakemake schedules only when the name contains
# nothing but letters, digits, and underscores; the validator must therefore reject a name like "He-3" early, with an
# error that names both the channel and the offending species.
def test_validate_perturb_species_names_rejects_invalid_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        scan._validate_perturb_species_names(["He-3"], flag_label="perturb_density_species")
    message = str(excinfo.value)
    assert "perturb_density_species" in message
    assert "'He-3'" in message
    with pytest.raises(ValueError):
        scan._validate_perturb_species_names(["D.T"], flag_label="perturb_temperature_species")


# Names made of letters, digits, and underscores (including single-letter ones) and an empty request list are all
# schedulable, so the validator must accept them without raising.
def test_validate_perturb_species_names_accepts_valid_names() -> None:
    scan._validate_perturb_species_names(["D", "He3", "t_D", "e"], flag_label="perturb_density_species")
    scan._validate_perturb_species_names([], flag_label="perturb_temperature_species")


# "none" marks unperturbed runs in the perturb_species identity field and "base" marks them in response_label, so a
# species carrying either name would make run records ambiguous; the validator must reject both, case-insensitively
# (matching the runtime species lookup), with an error naming the channel and the offending species.
def test_validate_perturb_species_names_rejects_reserved_names() -> None:
    for name in ("none", "Base"):
        with pytest.raises(ValueError) as excinfo:
            scan._validate_perturb_species_names([name], flag_label="perturb_density_species")
        message = str(excinfo.value)
        assert "perturb_density_species" in message
        assert f"'{name}'" in message


# Perturbation axis in neopax_fluxes.h5

# In fd_gradients mode collect keys the perturbation axis of neopax_fluxes.h5 on distinct (response_label,
# perturb_species) pairs in first-seen manifest order (deliberately not alphabetical here), joining each perturbed run
# to its base surface through the shared rho_index. Two species perturbed on the same channel must land on separate
# axis rows, surfaces missing a given pair hold zeros and False, and the axis never contains "base" / "none".
def test_collect_neopax_perturbed_axis_keyed_by_channel_and_species(tmp_path: Path) -> None:
    runs = [
        _collect_run_spec(tmp_path, 0, 1, 0.25, response_label="base", perturb_species="none", perturb_delta=0.0),
        _collect_run_spec(tmp_path, 1, 2, 0.5, response_label="base", perturb_species="none", perturb_delta=0.0),
        _collect_run_spec(tmp_path, 2, 1, 0.25, response_label="temperature_gradient", perturb_species="D", perturb_delta=0.5),
        _collect_run_spec(tmp_path, 3, 1, 0.25, response_label="density_gradient", perturb_species="D", perturb_delta=-0.5),
        _collect_run_spec(tmp_path, 4, 2, 0.5, response_label="density_gradient", perturb_species="D", perturb_delta=-0.4),
        _collect_run_spec(tmp_path, 5, 1, 0.25, response_label="density_gradient", perturb_species="T", perturb_delta=-0.3),
    ]
    _run_collect(tmp_path, runs)
    with h5py.File(tmp_path / "neopax_fluxes.h5", "r") as f:
        assert f["response_label"].asstr()[...].tolist() == ["temperature_gradient", "density_gradient", "density_gradient"]
        assert f["perturb_species"].asstr()[...].tolist() == ["D", "D", "T"]
        assert f["perturb_present"][...].tolist() == [[True, False], [True, True], [True, False]]
        assert_allclose(f["perturb_delta"][...], [[0.5, 0.0], [-0.5, -0.4], [-0.3, 0.0]])
        assert f["Gamma_perturbed"].shape == (3, 1, 2)


# A base-only (legacy) manifest has no perturbed runs, so collect must write the required base datasets but none of the
# optional perturbation datasets, keeping the axis absent entirely rather than empty.
def test_collect_neopax_omits_perturbed_datasets_for_base_only_manifest(tmp_path: Path) -> None:
    _run_collect(tmp_path, [_collect_run_spec(tmp_path, 0, 1, 0.5)])
    with h5py.File(tmp_path / "neopax_fluxes.h5", "r") as f:
        for required in ("rho", "Gamma", "Q", "Upar"):
            assert required in f
        for absent in ("Gamma_perturbed", "Q_perturbed", "perturb_delta", "perturb_present", "response_label", "perturb_species"):
            assert absent not in f


# Identity columns in runs.csv

# runs.csv is the flat per-run planning summary; its full header order (basic run fields then the trailing identity
# fields) is a stable contract.
def test_runs_csv_identity_columns(tmp_path: Path) -> None:
    def _run(index: int, label: str, species: str, delta: float) -> dict:
        stem = f"/x/run_{index:03d}"
        return {
            "index": index,
            "rho_index": 1,
            "rho": 0.25,
            "torflux": 0.0625,
            "Er": 0.0,
            "run_dir": stem,
            "config_path": f"{stem}/input.toml",
            "output_prefix": f"{stem}/run",
            "geometry_file": f"{stem}/geom.eik.nc",
            "response_label": label,
            "perturb_species": species,
            "perturb_delta": delta,
        }

    csv_path = tmp_path / "runs.csv"
    scan._write_runs_csv(csv_path, {"runs": [_run(0, "base", "none", 0.0), _run(1, "density_gradient", "D", -0.5)]})
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == [
            "index", "rho_index", "rho", "torflux", "Er", "run_dir", "config_path",
            "output_prefix", "geometry_file", "response_label", "perturb_species", "perturb_delta",
        ]
        rows = list(reader)
    assert [row["response_label"] for row in rows] == ["base", "density_gradient"]
    assert [row["perturb_species"] for row in rows] == ["none", "D"]
    assert [row["perturb_delta"] for row in rows] == ["0.0", "-0.5"]


# Prepare dispatch

# Complete prescribed input for prepare tests. The [species] block supplies charge and mass. The
# [profiles] arrays use SI values on five cells. Face arrays are linear in rho, with constant
# gradients per unit rho.
PRESCRIBED_TOML = """\
[species]
names = ["e", "D", "T"]
charge_qp = [-1.0, 1.0, 1.0]
mass_mp = [0.000544617, 2.0, 3.0]

[geometry]
n_radial = 5

[profiles]
model = "prescribed"
density = [[1.0e19, 1.1e19, 1.2e19, 1.3e19, 1.4e19], [5.0e18, 5.5e18, 6.0e18, 6.5e18, 7.0e18], [5.0e18, 5.5e18, 6.0e18, 6.5e18, 7.0e18]]
temperature = [[1000.0, 900.0, 800.0, 700.0, 600.0], [950.0, 850.0, 750.0, 650.0, 550.0], [940.0, 840.0, 740.0, 640.0, 540.0]]
Er = [0.0, 1.0, 2.0, 3.0, 4.0]
density_face = [[0.95e19, 1.05e19, 1.15e19, 1.25e19, 1.35e19, 1.45e19], [4.75e18, 5.25e18, 5.75e18, 6.25e18, 6.75e18, 7.25e18], [4.75e18, 5.25e18, 5.75e18, 6.25e18, 6.75e18, 7.25e18]]
temperature_face = [[1050.0, 950.0, 850.0, 750.0, 650.0, 550.0], [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0], [990.0, 890.0, 790.0, 690.0, 590.0, 490.0]]
Er_face = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
density_grad_face = [[5.0e18, 5.0e18, 5.0e18, 5.0e18, 5.0e18, 5.0e18], [2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18], [2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18]]
temperature_grad_face = [[-500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0]]
"""


# Other snapshot tests call the prescribed builder directly. This test covers command dispatch from
# the prescribed profiles-source option. It uses no transport result. The command records the source
# and scans the face grid without the magnetic axis. Here, [geometry].n_radial gives five cells and
# six faces. Removing the axis leaves five scan surfaces.
def test_cmd_prepare_dispatches_prescribed_source(tmp_path: Path) -> None:
    from tests.helpers.synthetic import write_wout

    config = tmp_path / "common_input.toml"
    config.write_text(PRESCRIBED_TOML)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = scan.build_parser().parse_args([
        "prepare",
        "--common-config", str(config),
        "--output-dir", str(out_dir),
        "--profiles-source", "prescribed",
        "--vmec-file-override", str(write_wout(tmp_path / "wout_synth.nc")),
    ])
    assert scan.cmd_prepare(args) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert "spectrax_root" not in manifest
    assert manifest["profiles_source"] == "prescribed"
    assert not manifest.get("neopax_result")
    assert [run["rho"] for run in manifest["runs"]] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    # The analytical fallback also lands on the 6-point face grid and profiles_source echoes the CLI, so only a
    # snapshot-derived value proves the prescribed builder ran; the analytical Er is quadratic in rho, never this ramp.
    # It is the Er_face ramp rather than the cell-centered Er, which pins that the face arrays are what got read.
    assert manifest["source_er"] == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]


# --- _t3d_median_estimator ---

# The estimator takes the median of the medians of the trailing windows of widths 1..N-1, matching Trinity3D's
# GX.median_estimator, so a hand-worked vector pins the whole construction. Reversed, [10, 2, 4, 6, 8] is
# [8, 6, 4, 2, 10]; the window medians are 8, 7 ((8+6)/2), 6 and 5 ((6+4)/2), whose median is (6+7)/2 = 6.5. Reducing
# the forward trace instead would give 5.5, and widening the last window to the full trace would give 6.0.
def test_t3d_median_estimator_matches_hand_computed_trailing_medians() -> None:
    assert_allclose(scan._t3d_median_estimator(np.array([10.0, 2.0, 4.0, 6.0, 8.0])), 6.5, rtol=1e-12)


# Trinity3D does not screen samples for finiteness, so a NaN inside the windows must reach the returned value rather
# than be filtered away and report a clean flux for a failed run. The oldest sample enters no window because the widths
# stop at N-1, so a NaN there is invisible upstream and stays invisible here. Reversed, [nan, 2, 4, 6] is
# [6, 4, 2, nan], whose window medians are 6, 5 and 4, with median 5.
def test_t3d_median_estimator_propagates_nan_inside_the_windows() -> None:
    assert np.isnan(scan._t3d_median_estimator(np.array([1.0, np.nan, 3.0, 4.0])))
    assert_allclose(scan._t3d_median_estimator(np.array([np.nan, 2.0, 4.0, 6.0])), 5.0, rtol=1e-12)


# A single-sample trace leaves the trailing-window list empty, where Trinity3D is undefined. That sample is the only
# available reduction, so a trace carrying a real value must not reduce to NaN.
def test_t3d_median_estimator_single_sample_returns_that_sample() -> None:
    assert_allclose(scan._t3d_median_estimator(np.array([2.5])), 2.5, rtol=1e-12)


# Runtime profile gradients

# ``tprim`` and ``fprim`` equal the negative face gradient divided by the value at that face.
# They use no minor-radius factor. NEOPAX uses ``r = a * rho``, so ``a`` cancels against a gradient
# per rho.
# The reference factors also cancel in each ratio. This test detects extra scaling or recalculation.
def test_runtime_species_gradients_are_the_face_gradient_over_the_face_value(tmp_path: Path) -> None:
    from tests.helpers.synthetic import write_wout

    config = tmp_path / "common_input.toml"
    config.write_text(PRESCRIBED_TOML)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = scan.build_parser().parse_args([
        "prepare",
        "--common-config", str(config),
        "--output-dir", str(out_dir),
        "--profiles-source", "prescribed",
        "--electron-model", "kinetic",
        "--vmec-file-override", str(write_wout(tmp_path / "wout_synth.nc")),
    ])
    assert scan.cmd_prepare(args) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    block = tomllib.loads(PRESCRIBED_TOML)["profiles"]
    density_face = np.asarray(block["density_face"])
    temperature_face = np.asarray(block["temperature_face"])
    density_grad_face = np.asarray(block["density_grad_face"])
    temperature_grad_face = np.asarray(block["temperature_grad_face"])

    assert [run["rho_index"] for run in manifest["runs"]] == [1, 2, 3, 4, 5]
    for run in manifest["runs"]:
        face = int(run["rho_index"])
        assert [sp["name"] for sp in run["runtime_species"]] == ["e", "D", "T"]
        for sp_idx, sp in enumerate(run["runtime_species"]):
            expected_tprim = -temperature_grad_face[sp_idx, face] / temperature_face[sp_idx, face]
            expected_fprim = -density_grad_face[sp_idx, face] / density_face[sp_idx, face]
            assert_allclose(sp["tprim"], expected_tprim, rtol=1e-12)
            assert_allclose(sp["fprim"], expected_fprim, rtol=1e-12)


# Runtime TOML text

def _write_template(tmp_path: Path, body: str) -> str:
    # A template with no [output] table leaves resolved_diagnostics on its hard default, and an empty body leaves
    # every other generated value there too. Naming a template also keeps these cases off the repo's own quick-run
    # template, which --gkx-template otherwise defaults to and whose values are free to change.
    path = tmp_path / "gkx_template.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _prepared_runtime_toml(tmp_path: Path, *extra_argv: str) -> dict:
    from tests.helpers.synthetic import write_wout

    config = tmp_path / "common_input.toml"
    config.write_text(PRESCRIBED_TOML)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = scan.build_parser().parse_args([
        "prepare",
        "--common-config", str(config),
        "--output-dir", str(out_dir),
        "--profiles-source", "prescribed",
        "--vmec-file-override", str(write_wout(tmp_path / "wout_synth.nc")),
        *extra_argv,
    ])
    assert scan.cmd_prepare(args) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return tomllib.loads(Path(manifest["runs"][0]["config_path"]).read_text())


# A template that names no [output] table leaves resolved_diagnostics unreachable, so every generated runtime TOML must
# carry the table. The hard default matches GKX's own, making a switched-off run always an explicit act.
def test_runtime_toml_defaults_resolved_diagnostics_on(tmp_path: Path) -> None:
    cfg = _prepared_runtime_toml(tmp_path, "--gkx-template", _write_template(tmp_path, ""))
    assert cfg["output"]["resolved_diagnostics"] is True


# The template tier is reachable only while the CLI flag stays None when the flag is absent. Giving
# --resolved-diagnostics an argparse default of True or False would pin every runtime TOML to that value and the
# template's resolved_diagnostics would silently stop taking effect, which is what this asserts against.
def test_runtime_toml_takes_resolved_diagnostics_from_the_template(tmp_path: Path) -> None:
    template = _write_template(tmp_path, "[output]\nresolved_diagnostics = false\n")
    cfg = _prepared_runtime_toml(tmp_path, "--gkx-template", template)
    assert cfg["output"]["resolved_diagnostics"] is False


# --resolved-diagnostics passed on the command line overrides the same key in a template that switched the spectra off.
def test_runtime_toml_cli_resolved_diagnostics_beats_the_template(tmp_path: Path) -> None:
    template = _write_template(tmp_path, "[output]\nresolved_diagnostics = false\n")
    cfg = _prepared_runtime_toml(tmp_path, "--gkx-template", template, "--resolved-diagnostics")
    assert cfg["output"]["resolved_diagnostics"] is True


# The off-flag has to reach the runtime TOML on its own, without a template that already switched the spectra off.
def test_runtime_toml_cli_switches_resolved_diagnostics_off(tmp_path: Path) -> None:
    cfg = _prepared_runtime_toml(tmp_path, "--no-resolved-diagnostics")
    assert cfg["output"]["resolved_diagnostics"] is False


# Shaping values distinct from every hard fallback, so a resolved value matching one of these came from the template.
_SHAPING_TEMPLATE = """\
[grid]
Nx = 24
Ny = 20
ntheta = 18

[time]
t_max = 33.0
sample_stride = 7
diagnostics_stride = 5
"""


# The chain resolves each shaping setting as CLI flag, then GKX template, then hard fallback. An empty template with no
# flags reaches the last tier, which holds GKX's own default for every key GKX defines one for. GKX leaves ntheta unset,
# so the 48 there is the bridge's own.
def test_runtime_toml_shaping_falls_back_to_gkx_defaults(tmp_path: Path) -> None:
    cfg = _prepared_runtime_toml(tmp_path, "--gkx-template", _write_template(tmp_path, ""))
    assert (cfg["grid"]["Nx"], cfg["grid"]["Ny"], cfg["grid"]["ntheta"]) == (48, 48, 48)
    assert cfg["time"]["t_max"] == 100.0
    assert (cfg["time"]["sample_stride"], cfg["time"]["diagnostics_stride"]) == (1, 1)


# Every shaping flag has to stay None when absent, otherwise argparse fills its dest, the CLI tier always wins and the
# template tier is unreachable. The template values here differ from the hard fallbacks, so these assertions pass only
# while the template tier really decides.
def test_runtime_toml_takes_shaping_from_the_template(tmp_path: Path) -> None:
    cfg = _prepared_runtime_toml(tmp_path, "--gkx-template", _write_template(tmp_path, _SHAPING_TEMPLATE))
    assert (cfg["grid"]["Nx"], cfg["grid"]["Ny"], cfg["grid"]["ntheta"]) == (24, 20, 18)
    assert cfg["time"]["t_max"] == 33.0
    assert (cfg["time"]["sample_stride"], cfg["time"]["diagnostics_stride"]) == (7, 5)


# A shaping flag passed on the command line overrides the same key in the template, so all six reach the runtime TOML.
def test_runtime_toml_cli_shaping_beats_the_template(tmp_path: Path) -> None:
    cfg = _prepared_runtime_toml(
        tmp_path,
        "--gkx-template", _write_template(tmp_path, _SHAPING_TEMPLATE),
        "--nx", "64",
        "--ny", "56",
        "--ntheta", "22",
        "--t-final", "77.0",
        "--sample-stride", "9",
        "--diagnostics-stride", "3",
    )
    assert (cfg["grid"]["Nx"], cfg["grid"]["Ny"], cfg["grid"]["ntheta"]) == (64, 56, 22)
    assert cfg["time"]["t_max"] == 77.0
    assert (cfg["time"]["sample_stride"], cfg["time"]["diagnostics_stride"]) == (9, 3)


# The parser-level counterpart to the template test above, catching a reintroduced argparse default at the parser
# rather than through the resolved runtime TOML. t_max is shared by --t-max and its --t-final alias.
def test_prepare_shaping_flags_default_to_none() -> None:
    args = scan.build_parser().parse_args(["prepare"])
    assert (args.nx, args.ny, args.ntheta) == (None, None, None)
    assert (args.t_max, args.sample_stride, args.diagnostics_stride) == (None, None, None)


# Transport layout recognition

# ``load_neopax_snapshot`` reads only NEOPAX transport solutions. It identifies the layout before
# parsing. A recognized NEOPAX file therefore keeps the loader's specific error.


def _write_ntss_like(path: Path, n_r: int = 6) -> Path:
    """Write a flat NTSS-like file: a radius, an Er, and one density/temperature pair per species."""
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=np.linspace(0.0, 1.0, n_r))
        f.create_dataset("Er", data=np.linspace(-1.0, 3.0, n_r))
        for name in ("ne", "nD", "nT"):
            f.create_dataset(name, data=np.linspace(1.0, 0.2, n_r))
        for name in ("Te", "TD", "TT"):
            f.create_dataset(name, data=np.linspace(4.0, 0.5, n_r))
    return path


# A solution from before the staggered grid is still a NEOPAX file. The caller must receive the
# specific ``missing the transport face datasets`` error instead of a generic layout error.
def test_load_neopax_snapshot_surfaces_the_missing_face_diagnostic(tmp_path: Path) -> None:
    from tests.helpers.synthetic import write_transport_solution

    path = write_transport_solution(
        tmp_path / "old_pin.h5",
        rho=np.linspace(0.1, 0.9, 5),
        density=np.ones((3, 5)), temperature=np.ones((3, 5)), er=np.zeros(5),
    )
    with pytest.raises(KeyError, match="transport face datasets"):
        scan.load_neopax_snapshot(path, time_index=-1)


# A flat NTSS-like file has no face grid or NEOPAX [boundary] models. A reader cannot recover the
# face gradients. Reject the file instead of creating gradients locally.
def test_load_neopax_snapshot_rejects_the_flat_ntss_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a NEOPAX transport_solution.h5"):
        scan.load_neopax_snapshot(_write_ntss_like(tmp_path / "ntss.h5"), time_index=-1)
