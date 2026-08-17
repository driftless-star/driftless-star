#!/usr/bin/env python3
"""NEOPAX -> GKX radial flux-scan driver (Stage 4 turbulence).

Self-contained NEOPAX-facing entrypoint for the NEOPAX <-> GKX
coupling. It reads a NEOPAX transport TOML (and, optionally, its transport
HDF5 output) and, across a set of radii:

- prepares the per-radius GKX runs (manifest + runtime TOMLs),
- executes them in parallel,
- collects the final nonlinear heat / particle fluxes into a single HDF5
  file in NEOPAX units.

Invoked with no subcommand, it runs the full prepare -> run -> collect
pipeline (``cmd_all``). The hidden ``run-one`` subcommand executes a single
radius and is used internally by the parallel launcher (one worker process
per run); it is not intended for direct use.

This module was formed by merging the former NEOPAX flux-bridge helper
into this driver so the Stage-4 radial scan lives in a single
self-contained file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from common.neopax_geometry import read_neopax_minor_radius
from common.neopax_profiles import (
    NEOPAX_DENSITY_REFERENCE_M3,
    NEOPAX_TEMPERATURE_REFERENCE_EV,
    FaceState,
    SpeciesMeta,
    build_analytical_face_state,
    build_prescribed_face_state,
    read_transport_face_state,
)
from netCDF4 import Dataset
from scipy.constants import elementary_charge, proton_mass

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


SCRIPT_DIR = Path(__file__).resolve().parent
STAGES_DIR = SCRIPT_DIR.parent

DEFAULT_COMMON_CONFIG = STAGES_DIR.parent / "inputs" / "quick_run" / "common_input.toml"
DEFAULT_GKX_TEMPLATE = STAGES_DIR.parent / "inputs" / "quick_run" / "HSX_vacuum_ns201_quickrun.toml"
DEFAULT_OUTPUT_DIR = STAGES_DIR.parent / "outputs" / "quick_run" / "stage4_turbulence"
DEFAULT_FD_PERTURB_REL_STEP = 0.5
DEFAULT_FD_DKAP_DENSITY = 0.5
DEFAULT_FD_DKAP_TEMPERATURE = 0.5

# Maps a run's gradient channel to the short fd_[nt] run-directory suffix stem the Snakemake surf wildcard matches.
FD_CHANNEL_DIR_SUFFIX: dict[str, str] = {
    "density_gradient": "fd_n",
    "temperature_gradient": "fd_t",
}

def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN cannot be written to TOML")
        if math.isinf(value):
            raise ValueError("inf cannot be written to TOML")
        return repr(float(value))
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _runtime_toml_text(manifest: dict[str, Any], run_spec: dict[str, Any]) -> str:
    grid = manifest["grid"]
    time_cfg = manifest["time"]
    geom = manifest["geometry"]
    init = manifest["init"]
    phys = manifest["physics"]
    coll = manifest["collisions"]
    norm = manifest["normalization"]
    terms = manifest["terms"]
    run = manifest["run"]
    output = manifest["output"]

    lines: list[str] = []
    for species in run_spec["runtime_species"]:
        lines.extend(
            [
                "[[species]]",
                f"name = {_toml_scalar(species['name'])}",
                f"charge = {_toml_scalar(species['charge'])}",
                f"mass = {_toml_scalar(species['mass'])}",
                f"density = {_toml_scalar(species['density'])}",
                f"temperature = {_toml_scalar(species['temperature'])}",
                f"tprim = {_toml_scalar(species['tprim'])}",
                f"fprim = {_toml_scalar(species['fprim'])}",
                f"nu = {_toml_scalar(species['nu'])}",
                "kinetic = true",
                "",
            ]
        )

    lines.extend(
        [
            "[grid]",
            f"Nx = {_toml_scalar(grid['Nx'])}",
            f"Ny = {_toml_scalar(grid['Ny'])}",
            f"Nz = {_toml_scalar(grid['Nz'])}",
            f"Lx = {_toml_scalar(grid['Lx'])}",
            f"Ly = {_toml_scalar(grid['Ly'])}",
            f"boundary = {_toml_scalar(grid['boundary'])}",
            f"y0 = {_toml_scalar(grid['y0'])}",
            f"ntheta = {_toml_scalar(grid['ntheta'])}",
            f"nperiod = {_toml_scalar(grid['nperiod'])}",
            "",
            "[time]",
            f"t_max = {_toml_scalar(time_cfg['t_max'])}",
            f"dt = {_toml_scalar(time_cfg['dt'])}",
            f"method = {_toml_scalar(time_cfg['method'])}",
            f"use_diffrax = {_toml_scalar(time_cfg['use_diffrax'])}",
            f"sample_stride = {_toml_scalar(time_cfg['sample_stride'])}",
            f"diagnostics_stride = {_toml_scalar(time_cfg['diagnostics_stride'])}",
            f"chunk_steps = {_toml_scalar(time_cfg['chunk_steps'])}" if time_cfg["chunk_steps"] is not None else "",
            f"fixed_dt = {_toml_scalar(time_cfg['fixed_dt'])}",
            f"cfl = {_toml_scalar(time_cfg['cfl'])}",
            f"state_sharding = {_toml_scalar(time_cfg['state_sharding'])}" if time_cfg["state_sharding"] is not None else "",
            "",
            "[geometry]",
            f"model = {_toml_scalar(geom['model'])}",
            f"vmec_file = {_toml_scalar(manifest['vmec_file'])}",
            f"geometry_file = {_toml_scalar(run_spec['geometry_file_toml'])}",
            f"geometry_backend = {_toml_scalar(geom['geometry_backend'])}",
            f"torflux = {_toml_scalar(run_spec['torflux'])}",
            f"alpha = {_toml_scalar(geom['alpha'])}",
            f"npol = {_toml_scalar(geom['npol'])}",
            "",
            "[init]",
            f"init_field = {_toml_scalar(init['init_field'])}",
            f"init_amp = {_toml_scalar(init['init_amp'])}",
            f"gaussian_init = {_toml_scalar(init['gaussian_init'])}",
            f"init_single = {_toml_scalar(init['init_single'])}",
            "",
            "[physics]",
            "linear = false",
            "nonlinear = true",
            f"electrostatic = {_toml_scalar(phys['electrostatic'])}",
            f"electromagnetic = {_toml_scalar(phys['electromagnetic'])}",
            f"adiabatic_electrons = {_toml_scalar(phys['adiabatic_electrons'])}",
            "adiabatic_ions = false",
            f"tau_e = {_toml_scalar(1.0 if run_spec['tau_e'] is None else run_spec['tau_e'])}",
            f"beta = {_toml_scalar(phys['beta'])}",
            f"collisions = {_toml_scalar(phys['collisions'])}",
            f"hypercollisions = {_toml_scalar(phys['hypercollisions'])}",
            "",
            "[collisions]",
            f"nu_hermite = {_toml_scalar(coll['nu_hermite'])}",
            f"nu_laguerre = {_toml_scalar(coll['nu_laguerre'])}",
            f"nu_hyper = {_toml_scalar(coll['nu_hyper'])}",
            f"p_hyper = {_toml_scalar(coll['p_hyper'])}",
            f"hypercollisions_const = {_toml_scalar(coll['hypercollisions_const'])}",
            f"hypercollisions_kz = {_toml_scalar(coll['hypercollisions_kz'])}",
            f"D_hyper = {_toml_scalar(coll['D_hyper'])}",
            f"damp_ends_amp = {_toml_scalar(coll['damp_ends_amp'])}",
            f"damp_ends_widthfrac = {_toml_scalar(coll['damp_ends_widthfrac'])}",
            "",
            "[normalization]",
            f"contract = {_toml_scalar(norm['contract'])}",
            f"diagnostic_norm = {_toml_scalar(norm['diagnostic_norm'])}",
            "",
            "[terms]",
            f"streaming = {_toml_scalar(terms['streaming'])}",
            f"mirror = {_toml_scalar(terms['mirror'])}",
            f"curvature = {_toml_scalar(terms['curvature'])}",
            f"gradb = {_toml_scalar(terms['gradb'])}",
            f"diamagnetic = {_toml_scalar(terms['diamagnetic'])}",
            f"collisions = {_toml_scalar(terms['collisions'])}",
            f"hypercollisions = {_toml_scalar(terms['hypercollisions'])}",
            f"hyperdiffusion = {_toml_scalar(terms['hyperdiffusion'])}",
            f"end_damping = {_toml_scalar(terms['end_damping'])}",
            f"apar = {_toml_scalar(terms['apar'])}",
            f"bpar = {_toml_scalar(terms['bpar'])}",
            f"nonlinear = {_toml_scalar(terms['nonlinear'])}",
            "",
            "[run]",
            f"ky = {_toml_scalar(run['ky'])}",
            f"Nl = {_toml_scalar(run['Nl'])}",
            f"Nm = {_toml_scalar(run['Nm'])}",
            "",
            "[output]",
            f"resolved_diagnostics = {_toml_scalar(output['resolved_diagnostics'])}",
            "",
        ]
    )
    return "\n".join(line for line in lines if line != "")


def _write_runtime_toml(path: Path, manifest: dict[str, Any], run_spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_runtime_toml_text(manifest, run_spec), encoding="utf-8")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _maybe_load_toml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return _load_toml(path)


def _resolve_relative(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    if re.match(r"^[A-Za-z]:[\\/]", expanded):
        return expanded
    path = Path(expanded)
    if path.is_absolute():
        return str(path.resolve())
    return str((base / path).resolve())


def _resolve_cli_path(value: str | None) -> str | None:
    """Resolve a CLI path argument.

    CLI overrides use standard path semantics (absolute or relative to CWD),
    not relative to the NEOPAX config's directory.
    """
    if not value:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    if re.match(r"^[A-Za-z]:[\\/]", expanded):
        return expanded
    return str(Path(expanded).resolve())


def _infer_neopax_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.parent.name == "examples":
        return parent.parent.parent
    return parent


def _resolve_equilibrium_paths(common_config: Path, args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve the VMEC and Boozer file paths.

    Command-line overrides have priority. The NEOPAX config supplies each fallback.

    Parameters
    ----------
    common_config : Path
        NEOPAX common input whose [geometry] block names both files.
    args : argparse.Namespace
        Parsed arguments, read for ``vmec_file_override`` and ``boozer_file_override``.

    Returns
    -------
    tuple of (str or None, str or None)
        ``(vmec_path, boozer_path)`` as absolute paths. An entry is None if neither source names it.
    """
    neopax_root = _infer_neopax_root(common_config)
    geometry_cfg = _load_toml(common_config).get("geometry", {})
    vmec_path = _resolve_cli_path(args.vmec_file_override)
    if vmec_path is None:
        vmec_path = _resolve_relative(neopax_root, geometry_cfg.get("vmec_file"))
    booz_path = _resolve_cli_path(args.boozer_file_override)
    if booz_path is None:
        booz_path = _resolve_relative(neopax_root, geometry_cfg.get("boozer_file"))
    return vmec_path, booz_path


def _coalesce(value: Any, template_value: Any, fallback: Any) -> Any:
    if value is not None:
        return value
    if template_value is not None:
        return template_value
    return fallback


def _template_section(template_cfg: dict[str, Any], name: str) -> dict[str, Any]:
    section = template_cfg.get(name, {})
    return section if isinstance(section, dict) else {}


def _resolve_template_path(template_arg: str | None, base: Path) -> Path | None:
    if template_arg is None or not str(template_arg).strip():
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(template_arg)))
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _parse_species_from_common_config(cfg: dict[str, Any]) -> list[SpeciesMeta]:
    species_cfg = cfg.get("species", {})
    names = list(species_cfg.get("names", []))
    masses = list(species_cfg.get("mass_mp", []))
    charges = list(species_cfg.get("charge_qp", []))
    if not names or len(names) != len(masses) or len(names) != len(charges):
        raise ValueError("NEOPAX config [species] must define matching names, mass_mp, and charge_qp arrays")
    return [
        SpeciesMeta(name=str(name), charge=float(charge), mass_mp=float(mass))
        for name, mass, charge in zip(names, masses, charges)
    ]


def _canonical_species_key(name: str) -> str:
    return str(name).strip().lower()


def _validate_perturb_species_names(perturb_species: list[str], *, flag_label: str) -> None:
    """Reject perturbation species names that cannot work inside run-directory names or run identity fields.

    Snakemake schedules perturbed run-directory names only when the species name contains nothing but letters,
    digits, and underscores. This function prevents names like ``He-3``, which would otherwise fail as a "No rule
    to produce" scheduling error. It also rejects the reserved names ``none`` and ``base`` (case-insensitively),
    which mark unperturbed runs in the ``perturb_species`` and ``response_label`` identity fields respectively.
    """
    for name in perturb_species:
        if re.fullmatch(r"\w+", name) is None:
            raise ValueError(
                f"{flag_label} '{name}' cannot name a schedulable run directory: "
                "species names may contain only letters, digits, and underscores"
            )
        if _canonical_species_key(name) in {"none", "base"}:
            raise ValueError(
                f"{flag_label} '{name}' is a reserved run-identity word: 'none' marks unperturbed runs in "
                "perturb_species and 'base' marks them in response_label, so neither can name a species"
            )


def _warn_unmatched_perturb_species(
    perturb_species: list[str], runtime_species_names: list[str], *, flag_label: str
) -> None:
    """Warn once per perturbation species name that matches no runtime species.

    Parameters
    ----------
    perturb_species : list[str]
        Names requested for a finite-difference gradient channel.
    runtime_species_names : list[str]
        Species that enter the runtime set for every flux surface.
    flag_label : str
        Channel identifier shown in the warning (e.g. ``perturb_density_species``).
    """
    available_keys = {_canonical_species_key(name) for name in runtime_species_names}
    available_display = ", ".join(runtime_species_names)
    for name in perturb_species:
        if _canonical_species_key(name) in available_keys:
            continue
        print(
            f"warning: {flag_label} '{name}' matches no runtime species "
            f"(available: {available_display}); no perturbed runs generated for it, "
            f"only unperturbed base runs will execute",
            file=sys.stderr,
        )


def load_neopax_snapshot(h5_path: Path, *, time_index: int) -> FaceState:
    """Read a face state from NEOPAX transport_solution.h5.

    The dataset names identify the layout before parsing. A parse failure does not identify the file
    format. A file without face datasets can still have the NEOPAX solution layout.
    In that case, the shared reader identifies the missing datasets.

    Parameters
    ----------
    h5_path : Path
        Transport solution to read.
    time_index : int
        Time slice to take, counting back from the last when negative.

    Returns
    -------
    FaceState
        Face state and face gradients on the transport face grid.

    Raises
    ------
    ValueError
        If the file is not a transport solution. A flat radius list carries no face grid and no
        boundary models, so NEOPAX's face gradients cannot be derived from one.
    """
    with h5py.File(h5_path, "r") as f:
        is_transport_solution = {"rho", "density", "Er"}.issubset(f.keys())
    if not is_transport_solution:
        raise ValueError(
            f"{h5_path} is not a NEOPAX transport_solution.h5. Every run needs the face gradients "
            "NEOPAX builds from its cell-centered state under the run's boundary models, and only "
            "that layout carries them."
        )
    return read_transport_face_state(h5_path, time_index=time_index)


def _infer_vmec_minor_radius(vmec_path: str) -> float:
    with Dataset(vmec_path, mode="r") as vfile:
        if "Aminor_p" in vfile.variables:
            return float(vfile.variables["Aminor_p"][:])
        if "volume_p" in vfile.variables and "Rmajor_p" in vfile.variables:
            volume_p = float(vfile.variables["volume_p"][:])
            r_major = float(vfile.variables["Rmajor_p"][:])
            return float(np.sqrt(volume_p / (2.0 * np.pi**2 * r_major)))
    raise KeyError(f"Could not infer minor radius from VMEC file {vmec_path}")


def _infer_local_rho_star(
    *,
    vmec_path: str,
    booz_path: str | None,
    a_minor: float | None,
    rho_value: float,
    temperature_keV: float,
    mass_mp: float,
    charge_qp: float,
) -> float:
    with Dataset(vmec_path, mode="r") as vfile:
        if a_minor is None:
            a_minor = _infer_vmec_minor_radius(vmec_path)
        b0_scalar = float(vfile.variables["b0"][:]) if "b0" in vfile.variables else float("nan")
        ns = int(np.asarray(vfile.variables["ns"][:]).reshape(())) if "ns" in vfile.variables else None

    b_ref = b0_scalar
    if booz_path is not None and ns is not None:
        try:
            with Dataset(booz_path, mode="r") as bfile:
                bmnc_b = np.asarray(bfile.variables["bmnc_b"][:], dtype=float)
                ixm_b = np.asarray(bfile.variables["ixm_b"][:], dtype=int)
                ixn_b = np.asarray(bfile.variables["ixn_b"][:], dtype=int)
            mode00 = None
            for idx in range(ixm_b.size):
                if int(ixm_b[idx]) == 0 and int(ixn_b[idx]) == 0:
                    mode00 = idx
                    break
            if mode00 is not None:
                s_half = (np.arange(ns, dtype=float) - 0.5) / float(ns - 1)
                rho_half = np.sqrt(np.clip(s_half[1:], 0.0, None))
                b00 = np.asarray(bmnc_b[:, mode00], dtype=float)
                b_ref = float(np.interp(float(rho_value), rho_half, b00, left=b00[0], right=b00[-1]))
        except Exception:
            b_ref = b0_scalar

    temperature_eV = float(temperature_keV) * NEOPAX_TEMPERATURE_REFERENCE_EV
    mass_kg = float(mass_mp) * proton_mass
    vth = float(np.sqrt(2.0 * temperature_eV * elementary_charge / mass_kg))
    omega_c = float(abs(charge_qp) * elementary_charge * b_ref / mass_kg)
    rho_i = vth / omega_c
    return float(rho_i / a_minor)


def _select_reference_ion_index(species: list[SpeciesMeta], preferred_name: str | None = None) -> int:
    if preferred_name is not None:
        key = _canonical_species_key(preferred_name)
        for idx, sp in enumerate(species):
            if _canonical_species_key(sp.name) == key:
                return idx
        raise ValueError(f"reference ion {preferred_name!r} not found in NEOPAX species list")
    for idx, sp in enumerate(species):
        if sp.charge > 0.0:
            return idx
    raise ValueError("No positively charged species found; cannot choose a reference ion")


def _parse_index_list(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def _choose_radius_indices(
    rho: np.ndarray,
    *,
    explicit: list[int] | None,
    rho_min: float,
    rho_max: float,
    num_radii: int,
) -> list[int]:
    if explicit is not None:
        idxs = sorted(set(int(i) for i in explicit))
        for idx in idxs:
            if idx < 0 or idx >= rho.size:
                raise IndexError(f"rho index {idx} out of range [0, {rho.size - 1}]")
        return idxs
    mask = (rho >= float(rho_min)) & (rho <= float(rho_max))
    mask &= ~np.isclose(rho, 0.0, atol=1.0e-8)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise ValueError("No radii satisfy the requested rho range")
    if int(num_radii) <= 0 or int(num_radii) >= candidates.size:
        return [int(v) for v in candidates]
    picks = np.linspace(0, candidates.size - 1, int(num_radii))
    return sorted(set(int(candidates[int(round(p))]) for p in picks))


def _build_manifest(
    *,
    neopax_result: Path,
    common_config: Path,
    output_dir: Path,
    snapshot: FaceState,
    species: list[SpeciesMeta],
    electron_model: str,
    reference_ion: str | None,
    rho_indices: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rho = np.asarray(snapshot.rho, dtype=float)
    density = np.asarray(snapshot.density, dtype=float)
    temperature = np.asarray(snapshot.temperature, dtype=float)
    er = np.asarray(snapshot.er, dtype=float)
    density_grad = np.asarray(snapshot.density_grad, dtype=float)
    temperature_grad = np.asarray(snapshot.temperature_grad, dtype=float)
    if density.shape[0] != len(species) or temperature.shape[0] != len(species):
        raise ValueError("NEOPAX HDF5 species dimension does not match the species list from the TOML")

    template_path = _resolve_template_path(getattr(args, "gkx_template", None), common_config.resolve().parent)
    template_cfg = _maybe_load_toml(template_path)
    template_grid = _template_section(template_cfg, "grid")
    template_time = _template_section(template_cfg, "time")
    template_geom = _template_section(template_cfg, "geometry")
    template_init = _template_section(template_cfg, "init")
    template_phys = _template_section(template_cfg, "physics")
    template_coll = _template_section(template_cfg, "collisions")
    template_norm = _template_section(template_cfg, "normalization")
    template_terms = _template_section(template_cfg, "terms")
    template_run = _template_section(template_cfg, "run")
    template_output = _template_section(template_cfg, "output")
    parse_species_list = lambda raw: [part.strip() for part in str(raw or "").split(",") if part.strip()]
    response_mode = str(getattr(args, "response_mode", "none")).strip().lower()
    perturb_density_species = parse_species_list(getattr(args, "perturb_density_species", ""))
    perturb_temperature_species = parse_species_list(getattr(args, "perturb_temperature_species", ""))
    perturb_rel_step = getattr(args, "perturb_rel_step", None)
    dkap_density = getattr(args, "dkap_density", None)
    dkap_temperature = getattr(args, "dkap_temperature", None)
    if response_mode == "fd_gradients":
        _validate_perturb_species_names(perturb_density_species, flag_label="perturb_density_species")
        _validate_perturb_species_names(perturb_temperature_species, flag_label="perturb_temperature_species")
        if perturb_rel_step is None:
            perturb_rel_step = DEFAULT_FD_PERTURB_REL_STEP
        if dkap_density is None:
            dkap_density = DEFAULT_FD_DKAP_DENSITY
        if dkap_temperature is None:
            dkap_temperature = DEFAULT_FD_DKAP_TEMPERATURE

    ref_idx = _select_reference_ion_index(species, preferred_name=reference_ion)
    ref_species = species[ref_idx]
    ref_density = np.maximum(density[ref_idx], float(args.density_floor))
    ref_temperature = np.maximum(temperature[ref_idx], float(args.temperature_floor))
    density_denominator = np.maximum(density, float(args.density_floor))
    temperature_denominator = np.maximum(temperature, float(args.temperature_floor))

    vmec_path, booz_path = _resolve_equilibrium_paths(common_config, args)
    if vmec_path is None:
        raise ValueError("Could not resolve a VMEC file path from the NEOPAX config")
    a_minor = _infer_vmec_minor_radius(vmec_path)

    electron_idx = None
    for idx, sp in enumerate(species):
        if sp.charge < 0.0:
            electron_idx = idx
            break

    runs: list[dict[str, Any]] = []
    electron_model_value = electron_model
    if electron_model_value is None:
        if "adiabatic_electrons" in template_phys:
            electron_model_value = "adiabatic" if bool(template_phys["adiabatic_electrons"]) else "kinetic"
        else:
            electron_model_value = "adiabatic"

    runtime_species_names: list[str] = []
    if str(electron_model_value).lower() == "adiabatic":
        runtime_species_names = [sp.name for sp in species if sp.charge > 0.0]
    elif str(electron_model_value).lower() == "kinetic":
        runtime_species_names = [sp.name for sp in species]
    else:
        raise ValueError("electron_model must be either 'adiabatic' or 'kinetic'")

    if response_mode == "fd_gradients":
        _warn_unmatched_perturb_species(
            perturb_density_species, runtime_species_names, flag_label="perturb_density_species"
        )
        _warn_unmatched_perturb_species(
            perturb_temperature_species, runtime_species_names, flag_label="perturb_temperature_species"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_index = 0

    def _append_perturbed_run(
        run_spec: dict[str, Any],
        runtime_species: list[dict[str, Any]],
        *,
        response_label: str,
        perturb_species: str,
        perturb_delta: float,
        local_geom_name: str,
    ) -> None:
        nonlocal run_index
        name_perturbed = f"{base_name}_{FD_CHANNEL_DIR_SUFFIX[response_label]}_{perturb_species}".replace(".", "p")
        run_dir_perturbed = (runs_root / name_perturbed).resolve()
        run_dir_perturbed.mkdir(parents=True, exist_ok=True)
        runs.append(
            {
                **run_spec,
                "index": run_index,
                "run_dir": str(run_dir_perturbed),
                "config_path": str((run_dir_perturbed / "input.toml").resolve()),
                "output_prefix": str((run_dir_perturbed / "run").resolve()),
                "geometry_file": str((run_dir_perturbed / local_geom_name).resolve()),
                "runtime_species": runtime_species,
                "response_label": response_label,
                "perturb_species": perturb_species,
                "perturb_delta": float(perturb_delta),
            }
        )
        run_index += 1

    def _perturb_delta(grad_base: float, floor: float | None, *, signed: float) -> float:
        return signed * max(
            float(floor) if floor is not None else 0.0,
            0.0 if perturb_rel_step is None else float(perturb_rel_step) * abs(grad_base),
        )

    for rho_idx in rho_indices:
        rho_val = float(rho[rho_idx])
        torflux = float(rho_val ** 2)
        ref_n = float(ref_density[rho_idx])
        ref_t = float(ref_temperature[rho_idx])
        if ref_n <= 0.0 or ref_t <= 0.0:
            raise ValueError(f"Reference density/temperature must stay positive at rho index {rho_idx}")

        runtime_species: list[dict[str, Any]] = []
        for sp_idx, sp in enumerate(species):
            include = str(electron_model_value).lower() == "kinetic" or sp.charge > 0.0
            if not include:
                continue
            is_electron = sp.charge < 0.0
            runtime_species.append(
                {
                    "name": sp.name,
                    "charge": float(sp.charge),
                    "mass": float(sp.mass_mp),
                    "density": float(density[sp_idx, rho_idx] / ref_n),
                    "temperature": float(temperature[sp_idx, rho_idx] / ref_t),
                    "tprim": float(
                        args.tprim_scale
                        * (-temperature_grad[sp_idx, rho_idx] / temperature_denominator[sp_idx, rho_idx])
                    ),
                    "fprim": float(
                        args.fprim_scale * (-density_grad[sp_idx, rho_idx] / density_denominator[sp_idx, rho_idx])
                    ),
                    "nu": float(args.nu_electron if is_electron else args.nu_ion),
                    "density_physical": float(density[sp_idx, rho_idx]),
                    "temperature_physical": float(temperature[sp_idx, rho_idx]),
                    "density_reference_physical": ref_n,
                    "temperature_reference_physical": ref_t,
                }
            )

        tau_e = None
        if electron_idx is not None:
            te_val = max(float(temperature[electron_idx, rho_idx]), float(args.temperature_floor))
            tau_e = float(ref_t / te_val)
        elif str(electron_model_value).lower() == "adiabatic":
            tau_e = float(_coalesce(args.tau_e_override, template_phys.get("tau_e"), 1.0))

        rho_star_physical = (
            float(args.rho_star_physical)
            if args.rho_star_physical is not None
            else _infer_local_rho_star(
                vmec_path=vmec_path,
                booz_path=booz_path,
                a_minor=a_minor,
                rho_value=rho_val,
                temperature_keV=ref_t,
                mass_mp=float(ref_species.mass_mp),
                charge_qp=float(ref_species.charge),
            )
        )

        base_name = f"rho_{rho_idx:03d}_r{rho_val:.4f}".replace(".", "p")
        run_dir = (runs_root / base_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        local_geom_name = f"{Path(vmec_path).stem}.eik.nc"
        output_prefix = str((run_dir / "run").resolve())
        geometry_file = str((run_dir / local_geom_name).resolve())
        config_path = str((run_dir / "input.toml").resolve())
        run_spec = {
            "index": run_index,
            "rho_index": int(rho_idx),
            "rho": rho_val,
            "r_physical": float(a_minor * rho_val),
            "torflux": torflux,
            "Er": float(er[rho_idx]),
            "run_dir": str(run_dir),
            "config_path": config_path,
            "output_prefix": output_prefix,
            "geometry_file": geometry_file,
            "geometry_file_toml": f"./{local_geom_name}",
            "runtime_species": runtime_species,
            "tau_e": tau_e,
            "rho_star_physical": rho_star_physical,
            "a_minor": float(a_minor),
            "response_label": "base",
            "perturb_species": "none",
            "perturb_delta": 0.0,
        }
        runs.append(run_spec)
        run_index += 1

        if response_mode != "fd_gradients":
            continue

        runtime_species_map = {str(sp["name"]).strip().lower(): sp for sp in runtime_species}

        for species_name in perturb_density_species:
            species_key = species_name.strip().lower()
            species_base = runtime_species_map.get(species_key)
            if species_base is None:
                continue
            delta = _perturb_delta(float(species_base["fprim"]), dkap_density, signed=-1.0)
            species_perturbed = [dict(sp) for sp in runtime_species]
            for sp in species_perturbed:
                if str(sp["name"]).strip().lower() == species_key:
                    sp["fprim"] = float(sp["fprim"]) + delta
                    sp["tprim"] = float(sp["tprim"]) - delta
                    break
            _append_perturbed_run(
                run_spec,
                species_perturbed,
                response_label="density_gradient",
                perturb_species=species_name,
                perturb_delta=delta,
                local_geom_name=local_geom_name,
            )

        for species_name in perturb_temperature_species:
            species_key = species_name.strip().lower()
            species_base = runtime_species_map.get(species_key)
            if species_base is None:
                continue
            delta = _perturb_delta(float(species_base["tprim"]), dkap_temperature, signed=1.0)
            species_perturbed = [dict(sp) for sp in runtime_species]
            for sp in species_perturbed:
                if str(sp["name"]).strip().lower() == species_key:
                    sp["tprim"] = float(sp["tprim"]) + delta
                    break
            _append_perturbed_run(
                run_spec,
                species_perturbed,
                response_label="temperature_gradient",
                perturb_species=species_name,
                perturb_delta=delta,
                local_geom_name=local_geom_name,
            )

    manifest = {
        "schema_version": 2,
        "profiles_source": str(args.profiles_source).lower(),
        "neopax_result": "" if neopax_result is None else str(neopax_result.resolve()),
        "common_config": str(common_config.resolve()),
        "output_dir": str(output_dir.resolve()),
        "snapshot_time": snapshot.time_value,
        "electron_model": str(electron_model_value).lower(),
        "gkx_template": None if template_path is None else str(template_path),
        "response_mode": response_mode,
        "perturb_density_species": perturb_density_species,
        "perturb_temperature_species": perturb_temperature_species,
        "dkap_density": None if dkap_density is None else float(dkap_density),
        "dkap_temperature": None if dkap_temperature is None else float(dkap_temperature),
        "perturb_rel_step": None if perturb_rel_step is None else float(perturb_rel_step),
        "source_rho": [float(v) for v in rho],
        "source_er": [float(v) for v in er],
        "runtime_species_names": runtime_species_names,
        "booz_file": booz_path,
        "vmec_file": vmec_path,
        "grid": {
            "Nx": int(_coalesce(args.nx, template_grid.get("Nx"), 48)),
            "Ny": int(_coalesce(args.ny, template_grid.get("Ny"), 48)),
            "Nz": int(_coalesce(args.nz, template_grid.get("Nz"), 48)),
            "Lx": float(_coalesce(args.lx, template_grid.get("Lx"), 62.8)),
            "Ly": float(_coalesce(args.ly, template_grid.get("Ly"), 62.8)),
            "boundary": str(_coalesce(args.boundary, template_grid.get("boundary"), "fix aspect")),
            "y0": float(_coalesce(args.y0, template_grid.get("y0"), 21.0)),
            "ntheta": int(_coalesce(args.ntheta, template_grid.get("ntheta"), 48)),
            "nperiod": int(_coalesce(args.nperiod, template_grid.get("nperiod"), 1)),
        },
        "time": {
            "t_max": float(_coalesce(args.t_max, template_time.get("t_max"), 100.0)),
            "dt": float(_coalesce(args.dt, template_time.get("dt"), 0.1)),
            "method": str(_coalesce(args.method, template_time.get("method"), "rk3")),
            "use_diffrax": bool(_coalesce(args.use_diffrax, template_time.get("use_diffrax"), False)),
            "sample_stride": int(_coalesce(args.sample_stride, template_time.get("sample_stride"), 1)),
            "diagnostics_stride": int(_coalesce(args.diagnostics_stride, template_time.get("diagnostics_stride"), 1)),
            "chunk_steps": None
            if _coalesce(args.chunk_steps, template_time.get("chunk_steps"), None) is None
            else int(_coalesce(args.chunk_steps, template_time.get("chunk_steps"), None)),
            "fixed_dt": bool(_coalesce(args.fixed_dt, template_time.get("fixed_dt"), False)),
            "cfl": float(_coalesce(args.cfl, template_time.get("cfl"), 1.0)),
            "state_sharding": None
            if str(_coalesce(args.state_sharding, template_time.get("state_sharding"), "none")).lower() == "none"
            else str(_coalesce(args.state_sharding, template_time.get("state_sharding"), "none")),
        },
        "physics": {
            "electrostatic": bool(_coalesce(None, template_phys.get("electrostatic"), True)),
            "electromagnetic": bool(_coalesce(None, template_phys.get("electromagnetic"), False)),
            "adiabatic_electrons": str(electron_model_value).lower() == "adiabatic",
            "collisions": bool(_coalesce(None, template_phys.get("collisions"), True)),
            "hypercollisions": bool(_coalesce(None, template_phys.get("hypercollisions"), True)),
            "beta": float(_coalesce(args.beta, template_phys.get("beta"), 0.0)),
        },
        "collisions": {
            "nu_hermite": float(_coalesce(args.nu_hermite, template_coll.get("nu_hermite"), 1.0)),
            "nu_laguerre": float(_coalesce(args.nu_laguerre, template_coll.get("nu_laguerre"), 2.0)),
            "nu_hyper": float(_coalesce(args.nu_hyper, template_coll.get("nu_hyper"), 0.0)),
            "p_hyper": float(_coalesce(args.p_hyper, template_coll.get("p_hyper"), 4.0)),
            "hypercollisions_const": float(_coalesce(args.hypercollisions_const, template_coll.get("hypercollisions_const"), 0.0)),
            "hypercollisions_kz": float(_coalesce(args.hypercollisions_kz, template_coll.get("hypercollisions_kz"), 1.0)),
            "D_hyper": float(_coalesce(args.d_hyper, template_coll.get("D_hyper"), 0.05)),
            "damp_ends_amp": float(_coalesce(args.damp_ends_amp, template_coll.get("damp_ends_amp"), 0.1)),
            "damp_ends_widthfrac": float(_coalesce(args.damp_ends_widthfrac, template_coll.get("damp_ends_widthfrac"), 0.125)),
        },
        "normalization": {
            "contract": str(_coalesce(args.normalization_contract, template_norm.get("contract"), "kinetic")),
            "diagnostic_norm": str(_coalesce(args.diagnostic_norm, template_norm.get("diagnostic_norm"), "gx")),
            "rho_star_physical": None if args.rho_star_physical is None else float(args.rho_star_physical),
            "reference_species_name": str(ref_species.name),
            "reference_species_index": int(ref_idx),
            "density_normalization": "n_s / n_ref_ion(rho)",
            "temperature_normalization": "T_s / T_ref_ion(rho)",
            "gradient_definition": (
                "tprim/fprim come from NEOPAX's own face gradients d(T)/d(rho) and d(n)/d(rho), per unit rho; "
                "tprim = -(dT/drho) / T_face and fprim = -(dn/drho) / n_face at the same face."
            ),
        },
        "terms": {
            "streaming": float(_coalesce(None, template_terms.get("streaming"), 1.0)),
            "mirror": float(_coalesce(None, template_terms.get("mirror"), 1.0)),
            "curvature": float(_coalesce(None, template_terms.get("curvature"), 1.0)),
            "gradb": float(_coalesce(None, template_terms.get("gradb"), 1.0)),
            "diamagnetic": float(_coalesce(None, template_terms.get("diamagnetic"), 1.0)),
            "collisions": float(_coalesce(None, template_terms.get("collisions"), 1.0)),
            "hypercollisions": float(_coalesce(None, template_terms.get("hypercollisions"), 1.0)),
            "hyperdiffusion": float(_coalesce(args.hyperdiffusion, template_terms.get("hyperdiffusion"), 1.0)),
            "end_damping": float(_coalesce(None, template_terms.get("end_damping"), 1.0)),
            "apar": float(_coalesce(None, template_terms.get("apar"), 0.0)),
            "bpar": float(_coalesce(None, template_terms.get("bpar"), 0.0)),
            "nonlinear": float(_coalesce(None, template_terms.get("nonlinear"), 1.0)),
        },
        "run": {
            "ky": float(_coalesce(args.ky, template_run.get("ky"), 1.0 / 21.0)),
            "Nl": int(_coalesce(args.nl, template_run.get("Nl"), 4)),
            "Nm": int(_coalesce(args.nm, template_run.get("Nm"), 8)),
        },
        "output": {
            "resolved_diagnostics": bool(
                _coalesce(args.resolved_diagnostics, template_output.get("resolved_diagnostics"), True)
            ),
        },
        "init": {
            "init_field": str(_coalesce(args.init_field, template_init.get("init_field"), "density")),
            "init_amp": float(_coalesce(args.init_amp, template_init.get("init_amp"), 1.0e-3)),
            "gaussian_init": bool(_coalesce(None, template_init.get("gaussian_init"), False)),
            "init_single": bool(_coalesce(None, template_init.get("init_single"), False)),
        },
        "geometry": {
            "model": str(_coalesce(None, template_geom.get("model"), "vmec")),
            "alpha": float(_coalesce(args.alpha, template_geom.get("alpha"), 0.0)),
            "npol": float(_coalesce(args.npol, template_geom.get("npol"), 1.0)),
            "geometry_backend": str(_coalesce(None, template_geom.get("geometry_backend"), "internal")),
            "a_minor": float(a_minor),
        },
        "species_meta": [
            {"name": sp.name, "charge": sp.charge, "mass_mp": sp.mass_mp}
            for sp in species
        ],
        "runs": runs,
    }
    for run_spec in runs:
        _write_runtime_toml(Path(run_spec["config_path"]), manifest, run_spec)
    return manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runs_csv(path: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for run in manifest["runs"]:
        row = {
            "index": run["index"],
            "rho_index": run["rho_index"],
            "rho": run["rho"],
            "torflux": run["torflux"],
            "Er": run["Er"],
            "run_dir": run["run_dir"],
            "config_path": run["config_path"],
            "output_prefix": run["output_prefix"],
            "geometry_file": run["geometry_file"],
            "response_label": run.get("response_label", "base"),
            "perturb_species": run.get("perturb_species", "none"),
            "perturb_delta": run.get("perturb_delta", 0.0),
        }
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)


def _build_normalization_audit_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    electron_model = str(manifest["electron_model"])
    normalization = manifest.get("normalization", {})
    ref_species_name = str(normalization.get("reference_species_name", ""))
    ref_species_index = int(normalization.get("reference_species_index", -1))
    for run in manifest["runs"]:
        rho = float(run["rho"])
        torflux = float(run["torflux"])
        tau_e = run["tau_e"]
        for sp in run["runtime_species"]:
            tprim = float(sp["tprim"])
            fprim = float(sp["fprim"])
            rows.append(
                {
                    "run_index": int(run["index"]),
                    "rho_index": int(run["rho_index"]),
                    "rho": rho,
                    "torflux": torflux,
                    "electron_model": electron_model,
                    "species_name": str(sp["name"]),
                    "charge": float(sp["charge"]),
                    "mass_mp": float(sp["mass"]),
                    "density_physical": float(sp["density_physical"]),
                    "temperature_physical": float(sp["temperature_physical"]),
                    "reference_species_name": ref_species_name,
                    "reference_species_index": ref_species_index,
                    "reference_density_physical": float(sp["density_reference_physical"]),
                    "reference_temperature_physical": float(sp["temperature_reference_physical"]),
                    "density_normalized_to_ref_ion": float(sp["density"]),
                    "temperature_normalized_to_ref_ion": float(sp["temperature"]),
                    "fprim_used": fprim,
                    "tprim_used": tprim,
                    "tau_e_used": math.nan if tau_e is None else float(tau_e),
                    "Er_input": float(run["Er"]),
                }
            )
    return rows


def _write_normalization_audit(output_dir: Path, manifest: dict[str, Any]) -> None:
    rows = _build_normalization_audit_rows(manifest)
    if not rows:
        return
    csv_path = output_dir / "normalization_audit.csv"
    json_path = output_dir / "normalization_audit.json"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "assumptions": {
            "reference_species_normalization": "All runtime densities and temperatures are normalized to the chosen reference ion at the same radius.",
            "gradient_definition_used": (
                "tprim/fprim are -X'(rho)/X evaluated on the transport cell faces, from the face gradients "
                "NEOPAX builds out of its cell-centered state under the run's boundary models."
            ),
            "torflux_mapping": "The torflux column is rho^2. It labels each row and takes no part in tprim/fprim.",
        },
        "rows": rows,
    }
    _write_json(json_path, payload)


def cmd_prepare(args: argparse.Namespace) -> int:
    common_config = Path(args.common_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    # Anchor a relative --gkx-template at the invocation CWD like every other CLI path
    # argument; the template resolver downstream would otherwise anchor it at the
    # common-config directory and mangle CWD-relative values.
    if args.gkx_template:
        args.gkx_template = str(Path(args.gkx_template).resolve())

    cfg = _load_toml(common_config)
    species = _parse_species_from_common_config(cfg)
    geometry_cfg = cfg.get("geometry", {})
    profiles_source = str(args.profiles_source).strip().lower()
    if profiles_source == "analytical":
        analytical_n_radii = args.analytical_n_radii
        if analytical_n_radii is None or int(analytical_n_radii) <= 0:
            # [geometry].n_radial counts transport cells. The scan samples their bounding faces.
            # Therefore, the default is the face count. An explicit analytical radius count is also
            # a face count. It can give NEOPAX a finer or coarser interpolation grid.
            analytical_n_radii = int(geometry_cfg.get("n_radial", 51)) + 1
        vmec_path, booz_path = _resolve_equilibrium_paths(common_config, args)
        if vmec_path is None or booz_path is None:
            raise ValueError(
                "--profiles-source=analytical reconstructs the transport faces on NEOPAX's metre grid, so "
                "[geometry].vmec_file and [geometry].boozer_file must both resolve"
            )
        snapshot = build_analytical_face_state(
            cfg,
            species=species,
            n_faces=int(analytical_n_radii),
            minor_radius=read_neopax_minor_radius(Path(vmec_path), Path(booz_path)),
        )
        neopax_result: Path | None = None
    elif profiles_source == "transport_h5":
        if args.neopax_result is None:
            raise ValueError("--neopax-result is required when --profiles-source=transport_h5")
        neopax_result = Path(args.neopax_result).resolve()
        snapshot = load_neopax_snapshot(neopax_result, time_index=int(args.time_index))
    elif profiles_source == "prescribed":
        snapshot = build_prescribed_face_state(cfg, n_species=len(species))
        neopax_result = None
    else:
        raise ValueError("--profiles-source must be 'transport_h5', 'analytical', or 'prescribed'")
    rho_indices = _choose_radius_indices(
        snapshot.rho,
        explicit=_parse_index_list(args.rho_indices),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        num_radii=int(args.num_radii),
    )
    manifest = _build_manifest(
        neopax_result=neopax_result,
        common_config=common_config,
        output_dir=output_dir,
        snapshot=snapshot,
        species=species,
        electron_model=args.electron_model,
        reference_ion=args.reference_ion,
        rho_indices=rho_indices,
        args=args,
    )
    manifest_path = output_dir / "manifest.json"
    csv_path = output_dir / "runs.csv"
    _write_json(manifest_path, manifest)
    _write_runs_csv(csv_path, manifest)
    _write_normalization_audit(output_dir, manifest)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote run table: {csv_path}")
    print(f"Wrote normalization audit: {output_dir / 'normalization_audit.csv'}")
    if neopax_result is not None:
        source_label = neopax_result.name
    elif profiles_source == "prescribed":
        source_label = "prescribed profiles from TOML"
    else:
        source_label = "analytical profiles from TOML"
    print(f"Prepared {len(manifest['runs'])} GKX runs from {source_label}")
    return 0


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostics_csv_path(run_spec: dict[str, Any]) -> Path:
    return Path(f"{run_spec['output_prefix']}.diagnostics.csv")


def _summary_json_path(run_spec: dict[str, Any]) -> Path:
    return Path(f"{run_spec['output_prefix']}.summary.json")


def _has_completed_output(run_spec: dict[str, Any]) -> bool:
    diag_csv = _diagnostics_csv_path(run_spec)
    if not diag_csv.exists():
        return False
    try:
        columns = _read_diagnostics_csv(diag_csv)
    except Exception:
        return False
    times = np.asarray(columns.get("t", []), dtype=float)
    return bool(times.size > 0)


def _apply_backend_env(
    env: dict[str, str],
    backend: str,
    gpu_id: int | str | None,
    threads_per_run: int | None,
) -> dict[str, str]:
    # GKX workers pick their JAX device and thread count from these variables. The parallel
    # launcher and the single-run worker both route through here so a CPU/GPU slot is configured the
    # same way no matter which path started the run. A missing GPU id maps to device 0 and a missing
    # CPU thread count maps to a single thread, matching the launcher's per-slot defaults.
    mode = str(backend).lower()
    if mode == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = str(0 if gpu_id is None else gpu_id)
        env["JAX_PLATFORM_NAME"] = "gpu"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["OMP_NUM_THREADS"] = str(1 if threads_per_run is None else int(threads_per_run))
    return env


def _resolve_run_spec(manifest: dict[str, Any], *, index: int | None, run_name: str | None) -> dict[str, Any]:
    # A per-surface run can be selected either by its manifest ordinal or by its directory name, where
    # the name is the basename of run_dir (the rho_{idx}_r{value} surface folder). Exactly one selector
    # must be supplied so the worker targets a single, unambiguous surface.
    if (index is None) == (run_name is None):
        raise ValueError("run-one requires exactly one of --index or --run-name")
    runs = manifest["runs"]
    if index is not None:
        return runs[int(index)]
    matches = [run for run in runs if os.path.basename(str(run["run_dir"])) == run_name]
    if len(matches) != 1:
        raise ValueError(
            f"--run-name {run_name!r} matched {len(matches)} runs in the manifest; expected exactly one"
        )
    return matches[0]


def cmd_run_one(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest).resolve())
    run_spec = _resolve_run_spec(manifest, index=args.index, run_name=getattr(args, "run_name", None))
    if _has_completed_output(run_spec):
        diag_csv = _diagnostics_csv_path(run_spec)
        row = _read_last_row_csv(diag_csv)
        print(
            json.dumps(
                {
                    "index": int(run_spec["index"]),
                    "rho": float(run_spec["rho"]),
                    "run_dir": str(Path(run_spec["run_dir"]).resolve()),
                    "config_path": str(Path(run_spec["config_path"]).resolve()),
                    "output_prefix": str(run_spec["output_prefix"]),
                    "geometry_file": str(run_spec["geometry_file"]),
                    "heat_flux_last": row.get("heat_flux", math.nan),
                    "particle_flux_last": row.get("particle_flux", math.nan),
                    "status": "already_completed",
                }
            )
        )
        return 0
    run_dir = Path(run_spec["run_dir"]).resolve()
    config_path = Path(run_spec["config_path"]).resolve()
    output_prefix = str(run_spec["output_prefix"])
    env = os.environ.copy()
    # When the parallel launcher starts this worker it exports the backend selectors into the
    # environment already, so --backend is omitted and the inherited variables are left untouched.
    # A direct invocation with --backend configures those selectors here instead. --gpu-ids may
    # carry a comma-separated list; a single surface consumes a single device, so only the first
    # id is pinned and the rest are ignored.
    backend = getattr(args, "backend", None)
    if backend is not None:
        gpu_id = None
        if str(backend).lower() == "gpu":
            gpu_tokens = [token.strip() for token in str(getattr(args, "gpu_ids", None) or "").split(",") if token.strip()]
            gpu_id = gpu_tokens[0] if gpu_tokens else None
        _apply_backend_env(env, backend, gpu_id, getattr(args, "threads_per_run", None))
    cmd = [
        sys.executable,
        "-m",
        "gkx.cli",
        "run",
        "--config",
        str(config_path),
        "--out",
        output_prefix,
    ]
    if bool(getattr(args, "verbose_worker", False)):
        cmd.append("--progress")
    else:
        cmd.append("--no-progress")
    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        env=env,
        text=True,
        capture_output=not bool(getattr(args, "verbose_worker", False)),
    )
    if proc.returncode != 0:
        if not bool(getattr(args, "verbose_worker", False)) and proc.stdout:
            print(proc.stdout.rstrip())
        if not bool(getattr(args, "verbose_worker", False)) and proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        return int(proc.returncode)
    diag_csv = _diagnostics_csv_path(run_spec)
    summary_json = _summary_json_path(run_spec)
    if not diag_csv.exists():
        message = (
            "GKX worker exited successfully but did not write the expected "
            f"diagnostics file: {diag_csv}"
        )
        if summary_json.exists():
            message += f" (summary exists: {summary_json})"
        if not bool(getattr(args, "verbose_worker", False)) and proc.stdout:
            print(proc.stdout.rstrip())
        if not bool(getattr(args, "verbose_worker", False)) and proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        print(message, file=sys.stderr)
        return 2
    heat_last = float("nan")
    pflux_last = float("nan")
    row = _read_last_row_csv(diag_csv)
    heat_last = row.get("heat_flux", math.nan)
    pflux_last = row.get("particle_flux", math.nan)
    print(
        json.dumps(
            {
                "index": int(run_spec["index"]),
                "rho": float(run_spec["rho"]),
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "output_prefix": output_prefix,
                "geometry_file": str(run_spec["geometry_file"]),
                "heat_flux_last": heat_last,
                "particle_flux_last": pflux_last,
            }
        )
    )
    return 0


def _launch_subprocess(
    *,
    manifest_path: Path,
    index: int,
    env_overrides: dict[str, str],
    verbose_workers: bool,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-one",
        "--manifest",
        str(manifest_path),
        "--index",
        str(index),
    ]
    if verbose_workers:
        cmd.append("--verbose-worker")
    return subprocess.Popen(
        cmd,
        stdout=None if verbose_workers else subprocess.PIPE,
        stderr=None if verbose_workers else subprocess.PIPE,
        text=True,
        env=env,
    )


def cmd_run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    runs = manifest["runs"]
    max_parallel = int(args.max_parallel)
    if max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    mode = str(args.backend).lower()
    if mode not in {"cpu", "gpu"}:
        raise ValueError("--backend must be 'cpu' or 'gpu'")

    gpu_ids = []
    if mode == "gpu":
        gpu_ids = _parse_index_list(args.gpu_ids) or [0]
        max_parallel = min(max_parallel, len(gpu_ids))

    pending = list(range(len(runs)))
    already_done = [idx for idx, run in enumerate(runs) if _has_completed_output(run)]
    if already_done:
        pending = [idx for idx in pending if idx not in set(already_done)]
    active: dict[Any, tuple[subprocess.Popen[str], int, dict[str, str]]] = {}
    failures = 0
    completed = len(already_done)
    total = len(runs)

    if already_done:
        print(f"skipping {len(already_done)} already completed runs")

    def _env_for_slot(slot: int) -> dict[str, str]:
        env: dict[str, str] = {}
        gpu_id = gpu_ids[slot % len(gpu_ids)] if mode == "gpu" else None
        _apply_backend_env(env, mode, gpu_id, args.threads_per_run)
        return env

    while pending or active:
        while pending and len(active) < max_parallel:
            slot = len(active)
            run_idx = pending.pop(0)
            env = _env_for_slot(slot)
            proc = _launch_subprocess(
                manifest_path=manifest_path,
                index=run_idx,
                env_overrides=env,
                verbose_workers=bool(getattr(args, "verbose_workers", False)),
            )
            active[proc] = (proc, run_idx, env)
            where = env.get("CUDA_VISIBLE_DEVICES", f"cpu x{env.get('OMP_NUM_THREADS', '1')}")
            print(f"started run {run_idx} on {where} ({completed}/{total} completed)")

        if not active:
            break

        done = []
        for proc, run_idx, env in list(active.values()):
            rc = proc.poll()
            if rc is None:
                continue
            if bool(getattr(args, "verbose_workers", False)):
                stdout = ""
                stderr = ""
            else:
                stdout, stderr = proc.communicate()
            if rc == 0:
                completed += 1
                if bool(getattr(args, "verbose_workers", False)):
                    print(f"completed run {run_idx} ({completed}/{total})")
                else:
                    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
                    print(f"completed run {run_idx} ({completed}/{total}): {line}")
            else:
                failures += 1
                print(f"run {run_idx} failed with code {rc} ({completed}/{total} completed)")
                if stdout.strip():
                    print(stdout.strip())
                if stderr.strip():
                    print(stderr.strip())
            done.append(proc)
        for proc in done:
            active.pop(proc, None)

        if active and not done:
            wait_timeout = float(args.poll_interval)
            import time

            time.sleep(wait_timeout)

    if failures:
        print(f"{failures} runs failed")
        return 1
    print("All runs finished successfully")
    return 0


def _read_last_row_csv(path: Path) -> dict[str, float]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"No rows found in diagnostics CSV {path}")
    row = data[-1] if getattr(data, "shape", ()) else data
    out: dict[str, float] = {}
    for name in data.dtype.names or ():
        out[str(name)] = float(row[name])
    return out


def _read_diagnostics_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"No rows found in diagnostics CSV {path}")
    if getattr(data, "shape", ()) == ():
        data = np.asarray([data], dtype=data.dtype)
    out: dict[str, np.ndarray] = {}
    for name in data.dtype.names or ():
        out[str(name)] = np.asarray(data[name], dtype=float)
    return out


def _time_average_columns(
    columns: dict[str, np.ndarray],
    *,
    average_window: float,
    t_final_override: float | None,
    average_reducer: str,
) -> tuple[dict[str, float], float, float]:
    if "t" not in columns:
        raise KeyError("Diagnostics CSV must contain a 't' column for time averaging")
    times = np.asarray(columns["t"], dtype=float)
    if times.size == 0:
        raise ValueError("Diagnostics CSV has an empty time axis")
    reducer = str(average_reducer).strip().lower()
    if reducer == "t3d_median":
        t_start = float(times[0])
        t_end = float(times[-1])
        return {
            name: _t3d_median_estimator(np.asarray(values, dtype=float))
            for name, values in columns.items()
            if np.asarray(values, dtype=float).ndim == 1
        }, t_start, t_end
    t_end = float(times[-1]) if t_final_override is None else float(t_final_override)
    if float(average_window) <= 0.0:
        raise ValueError("--average-window must be > 0")
    t_start = max(float(times[0]), t_end - float(average_window))
    mask = times >= t_start
    if not np.any(mask):
        mask[-1] = True
    return {
        name: float(np.nanmean(np.asarray(values, dtype=float)[mask]))
        for name, values in columns.items()
        if np.asarray(values, dtype=float).ndim == 1
    }, t_start, t_end


def _t3d_median_estimator(values: np.ndarray) -> float:
    """Reduce a diagnostics time trace to the median of its trailing-window medians.

    The windows cover the last k samples for k = 1 .. N-1, matching Trinity3D's
    ``GX.median_estimator``, so the oldest sample enters no window. Samples are not screened for
    finiteness, so a NaN the windows cover propagates to the returned value rather than being
    dropped, which keeps a failed run visible instead of reporting the median of its surviving
    samples. Trinity3D leaves a single-sample trace undefined because its window list is empty,
    and that sample is returned here.

    Parameters
    ----------
    values : np.ndarray
        One diagnostics column as a 1-D time trace, oldest sample first.

    Returns
    -------
    float
        Median of the trailing-window medians, NaN for an empty trace or a NaN inside a window.
    """
    trace = np.asarray(values, dtype=float)
    if trace.size == 0:
        return math.nan
    if trace.size == 1:
        return float(trace[0])
    trailing_medians = [np.median(trace[::-1][:k]) for k in range(1, trace.size)]
    return float(np.median(trailing_medians))


def _write_summary_plots(
    *,
    output_dir: Path,
    rho: np.ndarray,
    species_names: list[str],
    gamma: np.ndarray,
    q: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Plotting was requested but matplotlib is not installed."
        ) from exc

    traces = (
        ("Gamma", np.asarray(gamma, dtype=np.float64), output_dir / "Gamma_vs_rho.png"),
        ("Q", np.asarray(q, dtype=np.float64), output_dir / "Q_vs_rho.png"),
    )

    for label, values, path in traces:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        for i, name in enumerate(species_names):
            ax.plot(rho, values[i], marker="o", linewidth=1.5, markersize=4.0, label=name)
        ax.set_xlabel("rho")
        ax.set_ylabel(label)
        ax.set_title(f"{label} from GKX radial scan")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _write_run_heat_flux_trace_plots(
    *,
    manifest: dict[str, Any],
    species_names: list[str],
) -> tuple[list[Path], list[str]]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Plotting was requested but matplotlib is not installed."
        ) from exc

    runtime_species_names = list(manifest.get("runtime_species_names", []))
    written: list[Path] = []
    skipped: list[str] = []
    for run in manifest["runs"]:
        diag_csv = Path(f"{run['output_prefix']}.diagnostics.csv")
        if not diag_csv.exists():
            skipped.append(f"rho={float(run['rho']):.4f}: missing diagnostics CSV {diag_csv}")
            continue
        try:
            columns = _read_diagnostics_csv(diag_csv)
        except Exception as exc:
            skipped.append(f"rho={float(run['rho']):.4f}: failed to read {diag_csv} ({exc})")
            continue
        times = np.asarray(columns.get("t", []), dtype=float)
        if times.size == 0:
            skipped.append(f"rho={float(run['rho']):.4f}: diagnostics CSV {diag_csv} has no time samples")
            continue

        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        total_heat = np.asarray(columns.get("heat_flux", []), dtype=float)
        if total_heat.size == times.size:
            ax.plot(times, total_heat, linewidth=2.0, label="total")

        for runtime_idx, runtime_name in enumerate(runtime_species_names):
            key = f"heat_flux_s{runtime_idx}"
            values = np.asarray(columns.get(key, []), dtype=float)
            if values.size != times.size:
                continue
            label = runtime_name if runtime_idx < len(species_names) else f"s{runtime_idx}"
            ax.plot(times, values, linewidth=1.4, alpha=0.9, label=label)

        ax.set_xlabel("t")
        ax.set_ylabel("heat_flux")
        ax.set_title(f"Heat flux trace: rho={float(run['rho']):.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_path = Path(run["run_dir"]).resolve() / "heat_flux_trace.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)
    return written, skipped


def _thermal_speed_ms(temperature_keV: float, mass_mp: float) -> float:
    temp_eV = float(temperature_keV) * NEOPAX_TEMPERATURE_REFERENCE_EV
    mass_kg = float(mass_mp) * proton_mass
    return float(np.sqrt(2.0 * temp_eV * elementary_charge / mass_kg))


def _gkx_flux_to_neopax_units(
    flux_gb: float,
    *,
    density_ref_state: float,
    temperature_ref_keV: float,
    mass_ref_mp: float,
    rho_star_physical: float,
    kind: str,
) -> float:
    n_ref_m3 = float(density_ref_state) * NEOPAX_DENSITY_REFERENCE_M3
    t_ref_eV = float(temperature_ref_keV) * NEOPAX_TEMPERATURE_REFERENCE_EV
    vth_ref = _thermal_speed_ms(float(temperature_ref_keV), float(mass_ref_mp))
    if kind == "Gamma":
        scale = n_ref_m3 * vth_ref * float(rho_star_physical) ** 2
    elif kind == "Q":
        scale = n_ref_m3 * t_ref_eV * vth_ref * float(rho_star_physical) ** 2
    else:
        raise ValueError(f"Unknown flux kind {kind!r}")
    return float(flux_gb) * float(scale)


def _expand_axis_zero_if_needed(
    manifest: dict[str, Any],
    rho: np.ndarray,
    r_physical: np.ndarray,
    rho_index: np.ndarray,
    torflux: np.ndarray,
    er: np.ndarray,
    heat_flux: np.ndarray,
    particle_flux: np.ndarray,
    heat_flux_species: np.ndarray,
    particle_flux_species: np.ndarray,
    gamma_neopax: np.ndarray,
    q_neopax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_rho = np.asarray(manifest.get("source_rho", []), dtype=float)
    source_er = np.asarray(manifest.get("source_er", []), dtype=float)
    if source_rho.size == 0:
        return rho, r_physical, rho_index, torflux, er, heat_flux, particle_flux, heat_flux_species, particle_flux_species, gamma_neopax, q_neopax
    if source_rho.size != rho.size + 1:
        return rho, r_physical, rho_index, torflux, er, heat_flux, particle_flux, heat_flux_species, particle_flux_species, gamma_neopax, q_neopax
    if not np.isclose(source_rho[0], 0.0):
        return rho, r_physical, rho_index, torflux, er, heat_flux, particle_flux, heat_flux_species, particle_flux_species, gamma_neopax, q_neopax
    expected = np.arange(1, source_rho.size, dtype=int)
    if not np.array_equal(np.sort(rho_index), expected):
        return rho, r_physical, rho_index, torflux, er, heat_flux, particle_flux, heat_flux_species, particle_flux_species, gamma_neopax, q_neopax

    a_minor = float(manifest.get("geometry", {}).get("a_minor", 1.0))
    full_n = source_rho.size
    full_rho = np.asarray(source_rho, dtype=float)
    full_r = a_minor * full_rho
    full_rho_index = np.arange(full_n, dtype=int)
    full_torflux = full_rho**2
    full_er = np.zeros(full_n, dtype=float)
    if source_er.size == full_n:
        full_er[:] = source_er

    full_heat = np.zeros(full_n, dtype=float)
    full_particle = np.zeros(full_n, dtype=float)
    full_heat_species = np.zeros((full_n, heat_flux_species.shape[1]), dtype=float)
    full_particle_species = np.zeros((full_n, particle_flux_species.shape[1]), dtype=float)
    full_gamma_neopax = np.zeros((gamma_neopax.shape[0], full_n), dtype=float)
    full_q_neopax = np.zeros((q_neopax.shape[0], full_n), dtype=float)

    full_heat[1:] = np.nan_to_num(heat_flux, nan=0.0)
    full_particle[1:] = np.nan_to_num(particle_flux, nan=0.0)
    full_heat_species[1:, :] = np.nan_to_num(heat_flux_species, nan=0.0)
    full_particle_species[1:, :] = np.nan_to_num(particle_flux_species, nan=0.0)
    full_gamma_neopax[:, 1:] = np.nan_to_num(gamma_neopax, nan=0.0)
    full_q_neopax[:, 1:] = np.nan_to_num(q_neopax, nan=0.0)
    return (
        full_rho,
        full_r,
        full_rho_index,
        full_torflux,
        full_er,
        full_heat,
        full_particle,
        full_heat_species,
        full_particle_species,
        full_gamma_neopax,
        full_q_neopax,
    )


def cmd_collect(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest).resolve())
    runs = manifest["runs"]
    indices_base = [i for i, run in enumerate(runs) if str(run.get("response_label", "base")) == "base"]
    indices_perturbed = [i for i, run in enumerate(runs) if str(run.get("response_label", "base")) != "base"]
    species_meta = list(manifest.get("species_meta", []))
    species_names = [str(sp["name"]) for sp in species_meta] if species_meta else list(manifest["runtime_species_names"])
    runtime_species_names = list(manifest["runtime_species_names"])
    runtime_to_full = {name: idx for idx, name in enumerate(species_names)}
    out_h5 = Path(args.out).resolve()
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    n = len(runs)
    rho = np.full(n, np.nan, dtype=float)
    r_physical = np.full(n, np.nan, dtype=float)
    rho_index = np.full(n, -1, dtype=int)
    torflux = np.full(n, np.nan, dtype=float)
    er = np.full(n, np.nan, dtype=float)
    heat_flux = np.full(n, np.nan, dtype=float)
    particle_flux = np.full(n, np.nan, dtype=float)
    heat_flux_species = np.zeros((n, len(species_names)), dtype=float)
    particle_flux_species = np.zeros((n, len(species_names)), dtype=float)
    average_window_used = np.full(n, np.nan, dtype=float)
    average_t_start = np.full(n, np.nan, dtype=float)
    average_t_end = np.full(n, np.nan, dtype=float)
    gamma_neopax = np.zeros((len(species_names), n), dtype=float)
    q_neopax = np.zeros((len(species_names), n), dtype=float)

    requested_t_final = args.t_final
    if requested_t_final is None:
        requested_t_final = manifest.get("time", {}).get("t_max")
    requested_t_final = None if requested_t_final is None else float(requested_t_final)

    for i, run in enumerate(runs):
        rho[i] = float(run["rho"])
        r_physical[i] = float(run.get("r_physical", run.get("a_minor", 1.0) * run["rho"]))
        rho_index[i] = int(run["rho_index"])
        torflux[i] = float(run["torflux"])
        er[i] = float(run["Er"])
        diag_csv = Path(f"{run['output_prefix']}.diagnostics.csv")
        if not diag_csv.exists():
            print(f"missing diagnostics; zero-filling run fluxes: {diag_csv}")
            heat_flux[i] = 0.0
            particle_flux[i] = 0.0
            average_window_used[i] = float(args.average_window)
            continue
        try:
            columns = _read_diagnostics_csv(diag_csv)
            row, t_start_used, t_end_used = _time_average_columns(
                columns,
                average_window=float(args.average_window),
                t_final_override=requested_t_final,
                average_reducer=str(args.average_reducer),
            )
        except Exception as exc:
            print(f"failed to read diagnostics; zero-filling run fluxes: {diag_csv} ({exc})")
            heat_flux[i] = 0.0
            particle_flux[i] = 0.0
            average_window_used[i] = float(args.average_window)
            continue
        average_window_used[i] = float(args.average_window)
        average_t_start[i] = t_start_used
        average_t_end[i] = t_end_used
        heat_flux[i] = row.get("heat_flux", math.nan)
        particle_flux[i] = row.get("particle_flux", math.nan)
        ref_name = str(manifest.get("normalization", {}).get("reference_species_name", ""))
        ref_runtime_species = next(
            (sp for sp in run["runtime_species"] if str(sp.get("name", "")).strip().lower() == ref_name.strip().lower()),
            run["runtime_species"][0],
        )
        rho_star_physical = float(run.get("rho_star_physical", 1.0))
        a_minor = float(run.get("a_minor", manifest.get("geometry", {}).get("a_minor", 1.0)))
        for runtime_idx, runtime_name in enumerate(runtime_species_names):
            full_idx = runtime_to_full.get(runtime_name)
            if full_idx is None:
                continue
            heat_flux_species[i, full_idx] = row.get(f"heat_flux_s{runtime_idx}", 0.0)
            particle_flux_species[i, full_idx] = row.get(f"particle_flux_s{runtime_idx}", 0.0)
            q_neopax[full_idx, i] = _gkx_flux_to_neopax_units(
                heat_flux_species[i, full_idx],
                density_ref_state=float(ref_runtime_species["density_reference_physical"]),
                temperature_ref_keV=float(ref_runtime_species["temperature_reference_physical"]),
                mass_ref_mp=float(ref_runtime_species["mass"]),
                rho_star_physical=rho_star_physical,
                kind="Q",
            )
            gamma_neopax[full_idx, i] = _gkx_flux_to_neopax_units(
                particle_flux_species[i, full_idx],
                density_ref_state=float(ref_runtime_species["density_reference_physical"]),
                temperature_ref_keV=float(ref_runtime_species["temperature_reference_physical"]),
                mass_ref_mp=float(ref_runtime_species["mass"]),
                rho_star_physical=rho_star_physical,
                kind="Gamma",
            )
        q_neopax[:, i] *= a_minor
        gamma_neopax[:, i] *= a_minor

    idx_base = np.asarray(indices_base, dtype=int)
    rho_base = rho[idx_base]
    r_physical_base = r_physical[idx_base]
    rho_index_base = rho_index[idx_base]
    torflux_base = torflux[idx_base]
    er_base = er[idx_base]
    heat_flux_base = heat_flux[idx_base]
    particle_flux_base = particle_flux[idx_base]
    heat_flux_species_base = heat_flux_species[idx_base, :]
    particle_flux_species_base = particle_flux_species[idx_base, :]
    gamma_neopax_base = gamma_neopax[:, idx_base]
    q_neopax_base = q_neopax[:, idx_base]

    (
        rho_base,
        r_physical_base,
        rho_index_base,
        torflux_base,
        er_base,
        heat_flux_base,
        particle_flux_base,
        heat_flux_species_base,
        particle_flux_species_base,
        gamma_neopax_base,
        q_neopax_base,
    ) = _expand_axis_zero_if_needed(
        manifest,
        rho_base,
        r_physical_base,
        rho_index_base,
        torflux_base,
        er_base,
        heat_flux_base,
        particle_flux_base,
        heat_flux_species_base,
        particle_flux_species_base,
        gamma_neopax_base,
        q_neopax_base,
    )

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("r", data=r_physical)
        f.create_dataset("rho_index", data=rho_index)
        f.create_dataset("torflux", data=torflux)
        f.create_dataset("Er", data=er)
        f.create_dataset("heat_flux_total", data=heat_flux)
        f.create_dataset("particle_flux_total", data=particle_flux)
        f.create_dataset("average_window", data=average_window_used)
        f.create_dataset("average_t_start", data=average_t_start)
        f.create_dataset("average_t_end", data=average_t_end)
        dt = h5py.string_dtype(encoding="utf-8")
        # Per-run identity columns mirroring _write_runs_csv: rows here are one-per-executed-run, so in fd_gradients
        # response mode base and perturbed runs land at duplicate rho values and the (response_label, perturb_species)
        # pair is what distinguishes them.
        f.create_dataset("response_label", data=np.asarray([str(run.get("response_label", "base")) for run in runs], dtype=object), dtype=dt)
        f.create_dataset("perturb_species", data=np.asarray([str(run.get("perturb_species", "none")) for run in runs], dtype=object), dtype=dt)
        f.create_dataset("perturb_delta", data=np.asarray([float(run.get("perturb_delta", 0.0)) for run in runs], dtype=float))
        grp = f.create_group("species")
        grp.create_dataset("names", data=np.asarray(species_names, dtype=object), dtype=dt)
        grp.create_dataset("heat_flux", data=heat_flux_species)
        grp.create_dataset("particle_flux", data=particle_flux_species)
        meta = f.create_group("meta")
        meta.attrs["manifest"] = str(Path(args.manifest).resolve())
        meta.attrs["electron_model"] = str(manifest["electron_model"])
        meta.attrs["neopax_result"] = str(manifest["neopax_result"])
        meta.attrs["common_config"] = str(manifest["common_config"])
        meta.attrs["average_window"] = float(args.average_window)
        if manifest.get("normalization", {}).get("rho_star_physical", None) is not None:
            meta.attrs["rho_star_physical_override"] = float(manifest["normalization"]["rho_star_physical"])
        if requested_t_final is not None:
            meta.attrs["t_final_requested"] = requested_t_final

    neopax_flux_out = Path(args.neopax_flux_out).resolve()
    neopax_flux_out.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(rho_base)
    rho_sorted = rho_base[order]
    r_sorted = r_physical_base[order]
    rho_index_sorted = rho_index_base[order]
    gamma_sorted = gamma_neopax_base[:, order]
    q_sorted = q_neopax_base[:, order]
    perturb_keys: list[tuple[str, str]] = []
    for i in indices_perturbed:
        key = (str(runs[i]["response_label"]), str(runs[i]["perturb_species"]))
        if key not in perturb_keys:
            perturb_keys.append(key)
    with h5py.File(neopax_flux_out, "w") as f:
        f.create_dataset("rho", data=rho_sorted)
        f.create_dataset("r", data=r_sorted)
        f.create_dataset("Gamma", data=gamma_sorted)
        f.create_dataset("Q", data=q_sorted)
        f.create_dataset("Upar", data=np.zeros_like(gamma_sorted))
        meta = f.create_group("meta")
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("species_names", data=np.asarray(species_names, dtype=object), dtype=dt)
        meta.attrs["particle_flux_units"] = "m^-2 s^-1"
        meta.attrs["heat_flux_units"] = "eV m^-2 s^-1"
        meta.attrs["rho_star_source"] = "per-radius VMEC/profile derived" if manifest.get("normalization", {}).get("rho_star_physical", None) is None else "manual override"
        if manifest.get("normalization", {}).get("rho_star_physical", None) is not None:
            meta.attrs["rho_star_physical_override"] = float(manifest["normalization"]["rho_star_physical"])
        meta.attrs["minor_radius_m"] = float(manifest.get("geometry", {}).get("a_minor", 1.0))
        meta.attrs["radial_flux_coordinate"] = "saved Gamma/Q are converted to the physical minor-radius coordinate r = a*rho"
        meta.attrs["conversion"] = "Gamma_r = a * Gamma_rho = a * Gamma_gB * n_ref[m^-3] * vth_ref[m/s] * rho_star^2; Q_r = a * Q_rho = a * Q_gB * T_ref[eV] * n_ref[m^-3] * vth_ref[m/s] * rho_star^2"
        meta.attrs["reference_species_name"] = str(manifest.get("normalization", {}).get("reference_species_name", ""))
        meta.attrs["manifest"] = str(Path(args.manifest).resolve())
        if perturb_keys:
            key_to_index = {key: i for i, key in enumerate(perturb_keys)}
            rho_index_to_axis = {int(rho_idx): axis for axis, rho_idx in enumerate(rho_index_sorted)}
            gamma_perturbed = np.zeros((len(perturb_keys), len(species_names), rho_sorted.size), dtype=float)
            q_perturbed = np.zeros((len(perturb_keys), len(species_names), rho_sorted.size), dtype=float)
            perturb_delta = np.zeros((len(perturb_keys), rho_sorted.size), dtype=float)
            perturb_present = np.zeros((len(perturb_keys), rho_sorted.size), dtype=bool)
            for i in indices_perturbed:
                perturb_axis = key_to_index[(str(runs[i]["response_label"]), str(runs[i]["perturb_species"]))]
                rho_axis = rho_index_to_axis.get(int(runs[i].get("rho_index", -1)))
                if rho_axis is None:
                    continue
                gamma_perturbed[perturb_axis, :, rho_axis] = gamma_neopax[:, i]
                q_perturbed[perturb_axis, :, rho_axis] = q_neopax[:, i]
                perturb_delta[perturb_axis, rho_axis] = float(runs[i].get("perturb_delta", 0.0))
                perturb_present[perturb_axis, rho_axis] = True
            f.create_dataset("Gamma_perturbed", data=gamma_perturbed)
            f.create_dataset("Q_perturbed", data=q_perturbed)
            f.create_dataset("perturb_delta", data=perturb_delta)
            f.create_dataset("perturb_present", data=perturb_present)
            f.create_dataset(
                "response_label",
                data=np.asarray([key[0] for key in perturb_keys], dtype=object),
                dtype=dt,
            )
            f.create_dataset(
                "perturb_species",
                data=np.asarray([key[1] for key in perturb_keys], dtype=object),
                dtype=dt,
            )

    if bool(args.plot):
        _write_summary_plots(
            output_dir=neopax_flux_out.parent,
            rho=rho_sorted,
            species_names=species_names,
            gamma=gamma_sorted,
            q=q_sorted,
        )
    if bool(getattr(args, "plot_run_heat_traces", False)):
        written, skipped = _write_run_heat_flux_trace_plots(
            manifest=manifest,
            species_names=species_names,
        )
        if written:
            print(f"Wrote {len(written)} per-run heat-flux trace plot(s)")
            for path in written[:5]:
                print(f"  {path}")
            if len(written) > 5:
                print(f"  ... and {len(written) - 5} more")
        else:
            print("No per-run heat-flux trace plots were written")
        for message in skipped[:10]:
            print(f"skipped heat-trace plot: {message}")
        if len(skipped) > 10:
            print(f"... and {len(skipped) - 10} more skipped heat-trace plot(s)")

    print(f"Wrote collected flux summary: {out_h5}")
    print(f"Wrote NEOPAX flux profile: {neopax_flux_out}")
    return 0


def _resolve_from_neopax_root(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    root = _infer_neopax_root(config_path)
    resolved = _resolve_relative(root, value)
    return None if resolved is None else Path(resolved)


def _default_transport_solution_path(config_path: Path, cfg: dict) -> Path:
    output_cfg = cfg.get("transport_output", {})
    output_dir = _resolve_from_neopax_root(config_path, output_cfg.get("transport_output_dir"))
    if output_dir is None:
        raise ValueError("Could not resolve [transport_output].transport_output_dir from the NEOPAX config")
    return output_dir / "transport_solution.h5"


def _warn_if_hdf5_disabled(cfg: dict) -> None:
    output_cfg = cfg.get("transport_output", {})
    if not bool(output_cfg.get("transport_write_hdf5", False)):
        print(
            "warning: [transport_output].transport_write_hdf5 is false; "
            "NEOPAX will not write transport_solution.h5 unless you enable it.",
            file=sys.stderr,
        )


def cmd_all(args: argparse.Namespace) -> int:
    config_path = Path(args.common_config).resolve()
    cfg = _load_toml(config_path)
    if str(args.profiles_source).lower() == "transport_h5":
        _warn_if_hdf5_disabled(cfg)

    result_path = None
    if str(args.profiles_source).lower() == "transport_h5":
        result_path = (
            Path(args.neopax_result).resolve()
            if args.neopax_result is not None
            else _default_transport_solution_path(config_path, cfg)
        )
    output_dir = Path(args.output_dir).resolve()
    gkx_template = (
        str(Path(args.gkx_template).resolve()) if args.gkx_template else None
    )

    prepare_args = argparse.Namespace(
        profiles_source=args.profiles_source,
        neopax_result=None if result_path is None else str(result_path),
        common_config=str(config_path),
        gkx_template=gkx_template,
        output_dir=str(output_dir),
        time_index=args.time_index,
        analytical_n_radii=args.analytical_n_radii,
        electron_model=args.electron_model,
        reference_ion=args.reference_ion,
        rho_indices=args.rho_indices,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        num_radii=args.num_radii,
        vmec_file_override=args.vmec_file_override,
        boozer_file_override=args.boozer_file_override,
        density_floor=args.density_floor,
        temperature_floor=args.temperature_floor,
        tprim_scale=args.tprim_scale,
        fprim_scale=args.fprim_scale,
        tau_e_override=args.tau_e_override,
        nu_ion=args.nu_ion,
        nu_electron=args.nu_electron,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        boundary=args.boundary,
        y0=args.y0,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
        t_max=args.t_max,
        dt=args.dt,
        method=args.method,
        use_diffrax=args.use_diffrax,
        fixed_dt=args.fixed_dt,
        sample_stride=args.sample_stride,
        diagnostics_stride=args.diagnostics_stride,
        resolved_diagnostics=args.resolved_diagnostics,
        chunk_steps=args.chunk_steps,
        cfl=args.cfl,
        state_sharding=args.state_sharding,
        ky=args.ky,
        nl=args.nl,
        nm=args.nm,
        init_field=args.init_field,
        init_amp=args.init_amp,
        alpha=args.alpha,
        npol=args.npol,
        beta=args.beta,
        nu_hermite=args.nu_hermite,
        nu_laguerre=args.nu_laguerre,
        nu_hyper=args.nu_hyper,
        p_hyper=args.p_hyper,
        hypercollisions_const=args.hypercollisions_const,
        hypercollisions_kz=args.hypercollisions_kz,
        d_hyper=args.d_hyper,
        damp_ends_amp=args.damp_ends_amp,
        damp_ends_widthfrac=args.damp_ends_widthfrac,
        hyperdiffusion=args.hyperdiffusion,
        normalization_contract=args.normalization_contract,
        diagnostic_norm=args.diagnostic_norm,
        rho_star_physical=args.rho_star_physical,
        response_mode=args.response_mode,
        perturb_density_species=args.perturb_density_species,
        perturb_temperature_species=args.perturb_temperature_species,
        dkap_density=args.dkap_density,
        dkap_temperature=args.dkap_temperature,
        perturb_rel_step=args.perturb_rel_step,
    )
    rc = cmd_prepare(prepare_args)
    if rc != 0:
        return rc

    manifest_path = output_dir / "manifest.json"
    run_args = argparse.Namespace(
        manifest=str(manifest_path),
        backend=args.backend,
        gpu_ids=args.gpu_ids,
        max_parallel=args.max_parallel,
        threads_per_run=args.threads_per_run,
        poll_interval=args.poll_interval,
        verbose_workers=args.verbose_workers,
    )
    rc = cmd_run(run_args)
    run_failed = rc != 0
    if run_failed and not bool(args.collect_even_if_failures):
        return rc
    if run_failed:
        print(
            "warning: one or more GKX runs failed; continuing to collect "
            "available outputs and zero-filling missing runs.",
            file=sys.stderr,
        )

    collect_args = argparse.Namespace(
        manifest=str(manifest_path),
        out=str(output_dir / "flux_summary.h5"),
        neopax_flux_out=str(output_dir / "neopax_fluxes.h5"),
        average_window=args.average_window,
        t_final=args.t_max,
        plot=args.plot,
        plot_run_heat_traces=args.plot_run_heat_traces,
    )
    collect_rc = cmd_collect(collect_args)
    if run_failed and collect_rc == 0:
        return rc
    return collect_rc


def _add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--common-config",
        default=str(DEFAULT_COMMON_CONFIG),
        help="Path to the shared common_input TOML.",
    )
    parser.add_argument("--neopax-result", default=None, help="Optional explicit path to transport_solution.h5")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for manifest, GKX outputs, and collected fluxes",
    )
    parser.add_argument(
        "--gkx-template",
        default=str(DEFAULT_GKX_TEMPLATE),
        help="Base GKX runtime TOML used as the model template",
    )


def _add_prepare_shaping_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profiles-source", choices=("transport_h5", "analytical", "prescribed"), default="analytical")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--analytical-n-radii", type=int, default=None, help="Number of analytical cell faces; defaults to [geometry].n_radial + 1 from the NEOPAX config, the transport face grid")
    parser.add_argument("--electron-model", choices=("adiabatic", "kinetic"), default=None)
    parser.add_argument("--reference-ion", default=None)
    parser.add_argument("--rho-indices", default=None)
    parser.add_argument("--rho-min", type=float, default=0.0)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--num-radii", type=int, default=-1, help="Number of radii to sample; use <=0 for all available nonzero radii")
    parser.add_argument("--vmec-file-override", default=None)
    parser.add_argument("--boozer-file-override", default=None)
    parser.add_argument("--density-floor", type=float, default=1.0e-8)
    parser.add_argument("--temperature-floor", type=float, default=1.0e-8)
    parser.add_argument("--tprim-scale", type=float, default=1.0)
    parser.add_argument("--fprim-scale", type=float, default=1.0)
    parser.add_argument("--tau-e-override", type=float, default=None)
    parser.add_argument("--nu-ion", type=float, default=0.01)
    parser.add_argument("--nu-electron", type=float, default=0.0)
    parser.add_argument("--nx", type=int, default=None, help="Nonlinear spectral resolution in kx / x")
    parser.add_argument("--ny", type=int, default=None, help="Nonlinear spectral resolution in ky / y")
    parser.add_argument("--nz", type=int, default=None, help="Parallel/grid resolution in z")
    parser.add_argument("--lx", type=float, default=None)
    parser.add_argument("--ly", type=float, default=None)
    parser.add_argument("--boundary", default=None)
    parser.add_argument("--y0", type=float, default=None)
    parser.add_argument("--ntheta", type=int, default=None, help="Number of theta points for generated VMEC geometry")
    parser.add_argument("--nperiod", type=int, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--t-final", dest="t_max", type=float, help="Alias for --t-max")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--use-diffrax", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fixed-dt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument(
        "--resolved-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain the per-chunk resolved kx/ky/z diagnostic spectra in host RAM",
    )
    parser.add_argument("--chunk-steps", type=int, default=None, help="Adaptive nonlinear chunk size in steps for each GKX run")
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--state-sharding", default=None)
    parser.add_argument("--ky", type=float, default=None, help="Default nonlinear reference ky retained unless overridden")
    parser.add_argument("--nl", type=int, default=None)
    parser.add_argument("--nm", type=int, default=None)
    parser.add_argument("--init-field", default=None)
    parser.add_argument("--init-amp", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Field-line label alpha for the local geometry")
    parser.add_argument("--npol", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--nu-hermite", type=float, default=None)
    parser.add_argument("--nu-laguerre", type=float, default=None)
    parser.add_argument("--nu-hyper", type=float, default=None)
    parser.add_argument("--p-hyper", type=float, default=None)
    parser.add_argument("--hypercollisions-const", type=float, default=None)
    parser.add_argument("--hypercollisions-kz", type=float, default=None)
    parser.add_argument("--d-hyper", type=float, default=None)
    parser.add_argument("--damp-ends-amp", type=float, default=None)
    parser.add_argument("--damp-ends-widthfrac", type=float, default=None)
    parser.add_argument("--hyperdiffusion", type=float, default=None)
    parser.add_argument("--normalization-contract", default=None)
    parser.add_argument("--diagnostic-norm", default=None)
    parser.add_argument("--rho-star-physical", type=float, default=None, help="Optional manual rho_star override; otherwise derive it per radius from VMEC geometry and the reference-ion profile")
    parser.add_argument(
        "--response-mode",
        choices=("none", "fd_gradients"),
        default="none",
        help="Optional turbulence-response mode. 'fd_gradients' reserves baseline plus perturbed gradient runs for a future finite-difference response path.",
    )
    parser.add_argument(
        "--perturb-density-species",
        default=None,
        help="Comma-separated species names to perturb through the density-gradient channel when response mode is enabled.",
    )
    parser.add_argument(
        "--perturb-temperature-species",
        default=None,
        help="Comma-separated species names to perturb through the temperature-gradient channel when response mode is enabled.",
    )
    parser.add_argument(
        "--dkap-density",
        type=float,
        default=None,
        help="Optional absolute floor for density-gradient perturbations in fd_gradients mode.",
    )
    parser.add_argument(
        "--dkap-temperature",
        type=float,
        default=None,
        help="Optional absolute floor for temperature-gradient perturbations in fd_gradients mode.",
    )
    parser.add_argument(
        "--perturb-rel-step",
        type=float,
        default=None,
        help="Optional relative perturbation factor used later with fd_gradients mode.",
    )


def _add_collect_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--average-window", type=float, default=1.0, help="Average turbulent fluxes over the final time window")
    parser.add_argument(
        "--average-reducer",
        choices=("window_mean", "t3d_median"),
        default="window_mean",
        help="Reduction used on the turbulent time trace: final-window mean or Trinity3D-style median estimator.",
    )
    parser.add_argument("--plot", dest="plot", action="store_true", help="Write PNG plots of Gamma and Q versus rho.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Skip PNG plots.")
    parser.set_defaults(plot=True)
    parser.add_argument(
        "--plot-run-heat-traces",
        dest="plot_run_heat_traces",
        action="store_true",
        help="Write per-run heat-flux time-trace PNGs from existing diagnostics CSV files.",
    )
    parser.add_argument(
        "--no-plot-run-heat-traces",
        dest="plot_run_heat_traces",
        action="store_false",
        help="Skip per-run heat-flux time-trace PNGs.",
    )
    parser.set_defaults(plot_run_heat_traces=True)


def _add_collect_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, help="Path to the prepare-stage manifest.json")
    parser.add_argument("--out", required=True, help="Destination flux_summary.h5 path")
    parser.add_argument("--neopax-flux-out", required=True, help="Destination neopax_fluxes.h5 path")
    parser.add_argument(
        "--t-final",
        dest="t_final",
        type=float,
        default=None,
        help="Final time to average up to; falls back to the manifest t_max",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--threads-per-run", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--verbose-workers",
        dest="verbose_workers",
        action="store_true",
        help="Show the stdout/stderr from each GKX worker run",
    )
    parser.add_argument(
        "--no-verbose-workers",
        dest="verbose_workers",
        action="store_false",
        help="Silence GKX worker stdout/stderr.",
    )
    parser.set_defaults(verbose_workers=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    _add_common_io_args(p)
    _add_prepare_shaping_args(p)
    _add_collect_tuning_args(p)
    _add_run_args(p)
    p.add_argument(
        "--collect-even-if-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue to the collect stage after partial run failures, zero-filling missing runs.",
    )
    p.set_defaults(func=cmd_all)

    sub = p.add_subparsers(dest="cmd", required=False)

    prepare = sub.add_parser("prepare", help="Build the per-radius manifest and GKX runtime TOMLs.")
    _add_common_io_args(prepare)
    _add_prepare_shaping_args(prepare)
    prepare.set_defaults(func=cmd_prepare)

    collect = sub.add_parser("collect", help="Reduce per-radius diagnostics into NEOPAX-unit flux HDF5 files.")
    _add_collect_io_args(collect)
    _add_collect_tuning_args(collect)
    collect.set_defaults(func=cmd_collect)

    run_one = sub.add_parser("run-one", help=argparse.SUPPRESS)
    run_one.add_argument("--manifest", required=True)
    run_one.add_argument("--index", type=int, default=None)
    run_one.add_argument("--run-name", default=None, help="Per-surface run_dir basename to execute instead of --index")
    run_one.add_argument("--verbose-worker", action="store_true")
    run_one.add_argument("--backend", choices=("cpu", "gpu"), default=None)
    run_one.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids; the worker pins the first one.")
    run_one.add_argument("--threads-per-run", type=int, default=None)
    run_one.set_defaults(func=cmd_run_one)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
