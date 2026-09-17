"""Export NEOPAX face pressure to VMEX in pascals.

``write-input`` fits a polynomial in ``s = rho_face**2`` by default.
Select Akima or cubic splines to interpolate through every face sample.
``fit`` prints polynomial coefficients in pascals. The shared pressure loader
keeps NEOPAX units for convergence and comparison functions.

The comparison and plotting siblings also import :func:`_saved_time_count` from here to pick
the last distinctly timed record of a transport solution.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

NEOPAX_PRESSURE_TO_PA = 16021.76634
PROFILE_TYPES = ("akima_spline", "cubic_spline", "power_series")
MAX_SPLINE_KNOTS = 101
ENDPOINT_ATOL = 1e-12

# Face-grid datasets of a NEOPAX transport_solution.h5 this fit reads, named per stage because Phase 1 keeps
# the stage scripts free of cross-stage imports. ``rho_face`` is always required; the pressure comes from
# ``pressure_faces``, or from ``temperature_faces`` times ``density_faces``.
_TRANSPORT_FACE_DATASETS = ("rho_face", "pressure_faces", "temperature_faces", "density_faces")


def _load_dataset_at_time(arr: np.ndarray, time_index: int) -> np.ndarray:
    if arr.ndim == 1:
        return np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=float)
    idx = int(time_index)
    if idx < 0:
        idx = arr.shape[0] + idx
    if idx < 0 or idx >= arr.shape[0]:
        raise IndexError(f"time index {time_index} out of range for shape {arr.shape}")
    return np.asarray(arr[idx], dtype=float)


def _resolve_time_index(n_times: int, *, time_index: int, final_time: bool) -> int:
    if final_time:
        return n_times - 1
    idx = int(time_index)
    if idx < 0:
        idx = n_times + idx
    if idx < 0 or idx >= n_times:
        raise IndexError(f"time index {time_index} out of range for n_times={n_times}")
    return idx


def _saved_time_count(ts: np.ndarray) -> int:
    """Count the leading strictly increasing prefix of a solution's time axis.

    The comparison and plotting siblings use this to pick the last distinctly timed record of a
    transport solution. Repeated values, ``NaN`` fill and preallocated zeros all fail to compare
    greater than their predecessor, so any of them ends the prefix. A completed solve may repeat
    timestamps when several save slots land on one accepted step, so the prefix length is a
    record-selection aid, not a health verdict.

    Parameters
    ----------
    ts : numpy.ndarray
        The file's ``ts`` time axis in seconds, shape ``(n_time,)``.

    Returns
    -------
    int
        Length of the strictly increasing prefix. The first slot always counts.

    Raises
    ------
    ValueError
        If ``ts`` is not a 1-D time axis.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1:
        raise ValueError(f"'ts' must be a 1-D time axis, got shape {ts.shape}")
    increasing = np.diff(ts) > 0.0
    if bool(np.all(increasing)):
        return int(ts.size)
    return int(np.argmin(increasing)) + 1


def _load_total_pressure(h5_path: Path, *, time_index: int, final_time: bool) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Read one time slice of ``transport_solution.h5`` as a total pressure on its **face** grid.

    NEOPAX evolves its state on the ``n_radial`` cell centers. It writes these values as
    ``rho`` / ``pressure`` / ``temperature`` / ``density``. The cell centers include neither
    ``rho = 0`` nor ``rho = rho_edge``. VMEC evaluates pressure over ``s = rho**2`` in
    ``[0, 1]``. Thus, this function reads the ``n_radial + 1`` faces. The export functions
    separately check that these faces cover the full minor radius.

    Parameters
    ----------
    h5_path : Path
        NEOPAX ``transport_solution.h5``.
    time_index : int
        Time slice to read from a time-resolved file; negative indices count from the end.
    final_time : bool
        Select the last saved time slice, overriding ``time_index``.

    Returns
    -------
    rho : numpy.ndarray
        Face radial coordinate, shape ``(n_radial + 1,)``.
    total_pressure : numpy.ndarray
        Species-summed face pressure in NEOPAX units, shape ``(n_radial + 1,)``.
    resolved_index : int or None
        The time index actually read, or ``None`` for a static profile with no time axis.

    Raises
    ------
    KeyError
        If the face datasets are absent, as in a solution from a NEOPAX predating the staggered grid.
    IndexError
        If ``time_index`` falls outside the file's time axis.
    """
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        if "rho_face" not in keys:
            raise KeyError(
                f"{h5_path} is missing the transport face dataset 'rho_face'. The face datasets "
                f"({', '.join(_TRANSPORT_FACE_DATASETS)}) are written only by NEOPAX revisions that solve on "
                "the staggered cell/face grid, and the cell-centered datasets cannot stand in: the centers "
                "span neither rho = 0 nor rho = rho_edge, so a power series fitted on them is extrapolated "
                "across the plasma edge VMEC evaluates it at."
            )
        rho = np.asarray(f["rho_face"][()], dtype=float)
        resolved_index: int | None = None
        if "pressure_faces" in keys:
            pressure_all = np.asarray(f["pressure_faces"][()])
            if pressure_all.ndim >= 3:
                resolved_index = _resolve_time_index(pressure_all.shape[0], time_index=int(time_index), final_time=final_time)
                pressure = np.asarray(pressure_all[resolved_index], dtype=float)
            else:
                pressure = _load_dataset_at_time(pressure_all, time_index)
        elif "temperature_faces" in keys and "density_faces" in keys:
            temperature_all = np.asarray(f["temperature_faces"][()])
            density_all = np.asarray(f["density_faces"][()])
            if temperature_all.ndim >= 3 or density_all.ndim >= 3:
                n_times = temperature_all.shape[0] if temperature_all.ndim >= 3 else density_all.shape[0]
                resolved_index = _resolve_time_index(n_times, time_index=int(time_index), final_time=final_time)
                temperature = np.asarray(temperature_all[resolved_index], dtype=float) if temperature_all.ndim >= 3 else np.asarray(temperature_all, dtype=float)
                density = np.asarray(density_all[resolved_index], dtype=float) if density_all.ndim >= 3 else np.asarray(density_all, dtype=float)
            else:
                temperature = _load_dataset_at_time(temperature_all, time_index)
                density = _load_dataset_at_time(density_all, time_index)
            pressure = density * temperature
        else:
            raise KeyError(
                f"{h5_path} must contain either 'pressure_faces' or both 'temperature_faces' and 'density_faces'"
            )

    if pressure.ndim != 2:
        raise ValueError(f"Expected species-resolved pressure with shape (species, rho_face), got {pressure.shape}")

    total_pressure = np.sum(pressure, axis=0)
    if total_pressure.shape != rho.shape:
        raise ValueError(
            f"Pressure/rho_face shape mismatch: total_pressure.shape={total_pressure.shape}, rho_face.shape={rho.shape}"
        )
    # The loader rejects non-finite values when it reads the profile.
    for name, arr in (("rho", rho), ("total pressure", total_pressure)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{h5_path} {name} holds non-finite values in the selected time slice")
    return rho, total_pressure, resolved_index


def _load_ion_temperature(h5_path: Path, *, time_index: int, final_time: bool) -> tuple[np.ndarray, int | None]:
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        if "temperature" not in keys:
            raise KeyError(f"{h5_path} is missing required dataset 'temperature'")
        temperature_all = np.asarray(f["temperature"][()])
        resolved_index: int | None = None
        if temperature_all.ndim >= 3:
            resolved_index = _resolve_time_index(
                temperature_all.shape[0],
                time_index=int(time_index),
                final_time=final_time,
            )
            temperature = np.asarray(temperature_all[resolved_index], dtype=float)
        else:
            temperature = _load_dataset_at_time(temperature_all, time_index)
        if temperature.ndim != 2:
            raise ValueError(f"Expected species-resolved temperature with shape (species, rho), got {temperature.shape}")
        species_names = None
        if "species_names" in keys:
            species_names = [
                bytes(name).decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in np.asarray(f["species_names"][()]).reshape(-1)
            ]
    ion_index = 0
    if species_names:
        ion_index = next(
            (i for i, name in enumerate(species_names) if name.strip().lower() not in {"e", "electron"}),
            0,
        )
    elif temperature.shape[0] > 1:
        ion_index = 1
    return np.asarray(temperature[ion_index], dtype=float), resolved_index


def _pressure_in_pascals(pressure: np.ndarray) -> np.ndarray:
    """Convert NEOPAX pressure to pascals once. Reject overflow before export."""
    with np.errstate(over="ignore", invalid="ignore"):
        pressure_pa = pressure * NEOPAX_PRESSURE_TO_PA
    if not np.all(np.isfinite(pressure_pa)):
        raise ValueError("Pressure in pascals must be finite")
    return pressure_pa


def _fit_power_series(s: np.ndarray, p: np.ndarray, degree: int) -> np.ndarray:
    coeffs = np.polynomial.polynomial.polyfit(s, p, deg=int(degree))
    return np.asarray(coeffs, dtype=float)


def _format_am_line(coeffs: np.ndarray) -> str:
    return "AM = " + ", ".join(f"{float(c):.16E}" for c in np.asarray(coeffs, dtype=float))


def _validate_export_profile(rho: np.ndarray, pressure: np.ndarray, *, spline: bool) -> np.ndarray:
    """Check coverage of the full radius. Correct only endpoint roundoff."""
    rho = np.asarray(rho, dtype=float).copy()
    pressure = np.asarray(pressure, dtype=float)
    if rho.ndim != 1 or rho.size < 2 or pressure.shape != rho.shape:
        raise ValueError("Pressure export requires matching 1-D arrays with at least two radii")
    if not np.all(np.isfinite(rho)) or not np.all(np.diff(rho) > 0):
        raise ValueError("Pressure radii must be finite and strictly increasing")
    if abs(rho[0]) > ENDPOINT_ATOL or abs(rho[-1] - 1.0) > ENDPOINT_ATOL:
        raise ValueError("Pressure export requires rho_face spanning [0, 1]. Do not extrapolate a truncated grid")
    rho[0], rho[-1] = 0.0, 1.0
    if not np.all(np.diff(rho) > 0):
        raise ValueError("Pressure radii must remain strictly increasing after endpoint roundoff correction")
    if not np.all(np.isfinite(pressure)) or np.any(pressure < 0):
        raise ValueError("Pressure export requires finite, non-negative pressure")
    if spline and rho.size > MAX_SPLINE_KNOTS:
        raise ValueError(f"VMEX accepts at most {MAX_SPLINE_KNOTS} spline knots, got {rho.size}")
    return rho


# Mask quoted strings and comments before reading namelist syntax. A slash or
# assignment in a title or comment must not mark the end of the INDATA block.
_NAMELIST_LITERAL = re.compile(r"'(?:(?:'')|[^'])*'|\"(?:(?:\"\")|[^\"])*\"|![^\n]*")
_ASSIGNMENT = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*=")
_PRESSURE_KEYS = {"PMASS_TYPE", "PRES_SCALE", "AM", "AM_AUX_S", "AM_AUX_F"}


def _rewrite_pressure(text: str, assignments: list[str]) -> str:
    """Replace complete pressure assignments, including indexed and continued arrays."""
    masked = _NAMELIST_LITERAL.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    start = re.search(r"&\s*INDATA\b", masked, re.IGNORECASE)
    if start is None:
        raise ValueError("Input does not contain an &INDATA block")
    end = re.search(r"/|&end\b", masked[start.end():], re.IGNORECASE)
    if end is None:
        raise ValueError("Input does not contain a terminating '/' for &INDATA")
    body_end = start.end() + end.start()
    matches = list(_ASSIGNMENT.finditer(masked, start.end(), body_end))
    body = text[start.end():body_end]
    remove = np.zeros(len(body), dtype=bool)
    for i, match in enumerate(matches):
        if match.group(1).upper() not in _PRESSURE_KEYS:
            continue
        stop = matches[i + 1].start() if i + 1 < len(matches) else body_end
        stop = match.start() + len(text[match.start():stop].rstrip())
        remove[match.start() - start.end():stop - start.end()] = True
    # Keep comments, including those between continuation lines. Leave all
    # unrelated assignments, blank lines, and namelist blocks as supplied.
    for literal in _NAMELIST_LITERAL.finditer(body):
        if literal.group().startswith("!"):
            remove[literal.start():literal.end()] = False
    lines = []
    offset = 0
    for line in body.splitlines(keepends=True):
        deleted = remove[offset:offset + len(line)]
        kept = "".join(char for char, drop in zip(line, deleted) if not drop or char == "\n")
        if kept.strip() or not np.any(deleted):
            lines.append(kept)
        offset += len(line)
    body = "".join(lines)
    if not body.endswith("\n"):
        body += "\n"
    return text[:start.end()] + body + "".join("  " + line + "\n" for line in assignments) + text[body_end:]


def _write_vmec_input_with_pressure_fit(
    vmec_input: Path, coeffs: np.ndarray, *, output_path: Path | None,
) -> Path:
    """Write polynomial coefficients already in pascals."""
    assignments = ["PMASS_TYPE = 'power_series'", "PRES_SCALE = 1.0000000000000000E+00", _format_am_line(coeffs)]
    dst = output_path if output_path is not None else vmec_input
    dst.write_text(_rewrite_pressure(vmec_input.read_text(), assignments))
    return dst


def _write_vmec_input_with_pressure_spline(
    vmec_input: Path, rho: np.ndarray, pressure_pa: np.ndarray, *,
    profile_type: str = "akima_spline", output_path: Path | None = None,
) -> Path:
    """Write all supplied knots without fitting, resampling, or clipping pressure."""
    if profile_type not in PROFILE_TYPES[:2]:
        raise ValueError(f"Unsupported spline profile type {profile_type!r}")
    rho = _validate_export_profile(rho, pressure_pa, spline=True)
    assignments = [f"PMASS_TYPE = '{profile_type}'", "PRES_SCALE = 1.0000000000000000E+00"]
    for key, values in (("AM_AUX_S", rho**2), ("AM_AUX_F", pressure_pa)):
        # Short continuation lines also work with Fortran namelist readers.
        for offset in range(0, len(values), 4):
            prefix = f"{key} = " if offset == 0 else "  "
            assignments.append(prefix + ", ".join(f"{float(v):.16E}" for v in values[offset:offset + 4]) + ",")
    dst = output_path if output_path is not None else vmec_input
    dst.write_text(_rewrite_pressure(vmec_input.read_text(), assignments))
    return dst


def _fit_from_args(args) -> tuple[np.ndarray, int | None, int]:
    rho, total_pressure, resolved_index = _load_total_pressure(
        args.h5_path,
        time_index=int(args.time_index),
        final_time=bool(args.final_time),
    )
    rho = _validate_export_profile(rho, total_pressure, spline=False)
    total_pressure = _pressure_in_pascals(total_pressure)
    s = rho**2

    if args.drop_axis and s.size > 1:
        s = s[1:]
        total_pressure = total_pressure[1:]

    # A degree-d power series has d+1 coefficients and needs at least d+1 sample
    # points to be well-posed. Clamp the effective degree to the number of points
    # minus one so the fit stays well-conditioned regardless of the radial resolution.
    effective_degree = min(int(args.degree), s.size - 1)
    coeffs = _fit_power_series(s, total_pressure, degree=effective_degree)
    return coeffs, resolved_index, effective_degree


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common(subparser):
        subparser.add_argument("h5_path", type=Path, help="Path to transport_solution.h5")
        subparser.add_argument("--degree", type=int, default=None, help="Polynomial degree for AM coefficients")
        subparser.add_argument("--time-index", type=int, default=-1, help="Time slice to read if the file is time-dependent")
        subparser.add_argument(
            "--final-time",
            action="store_true",
            help="Use the final saved time slice explicitly. Equivalent to the last time index.",
        )
        subparser.add_argument(
            "--drop-axis",
            action="store_true",
            help="Exclude the magnetic axis point from the polynomial fit.",
        )

    fit_parser = subparsers.add_parser("fit", help="Print fitted VMEC AM coefficients from transport_solution.h5")
    _add_common(fit_parser)

    write_parser = subparsers.add_parser(
        "write-input",
        help="Export pressure from transport_solution.h5 into a VMEC input file",
    )
    _add_common(write_parser)
    write_parser.add_argument("--profile-type", choices=PROFILE_TYPES, default="power_series")
    write_parser.add_argument("vmec_input", type=Path, help="Path to VMEC input.* file to update")
    write_parser.add_argument(
        "--output-input",
        type=Path,
        default=None,
        help="Optional output path for the updated VMEC input. Defaults to overwriting vmec_input.",
    )

    args = parser.parse_args()
    profile_type = args.profile_type if args.command == "write-input" else "power_series"
    if profile_type != "power_series" and (args.degree is not None or args.drop_axis):
        parser.error("--degree and --drop-axis require --profile-type power_series")
    if profile_type == "power_series":
        args.degree = 8 if args.degree is None else args.degree
        if args.degree < 0:
            parser.error("--degree must be non-negative")
        coeffs, resolved_index, effective_degree = _fit_from_args(args)
        if args.command == "write-input":
            out_path = _write_vmec_input_with_pressure_fit(args.vmec_input, coeffs, output_path=args.output_input)
        print(f"# requested_degree: {args.degree}")
        print(f"# effective_degree: {effective_degree}")
        print(_format_am_line(coeffs))
    else:
        rho, pressure, resolved_index = _load_total_pressure(
            args.h5_path, time_index=args.time_index, final_time=args.final_time,
        )
        out_path = _write_vmec_input_with_pressure_spline(
            args.vmec_input, rho, _pressure_in_pascals(pressure),
            profile_type=profile_type, output_path=args.output_input,
        )
        print(f"# pressure_knots: {rho.size}")
    if args.command == "write-input":
        print(f"# wrote_vmec_input: {out_path}")
    print(f"# input: {args.h5_path}")
    print(f"# resolved_time_index: {resolved_index if resolved_index is not None else 'static_profile'}")
    print(f"# pressure_profile_type: {profile_type}")
    print("# pressure_units: Pa")
    print("PRES_SCALE = 1.0")


if __name__ == "__main__":
    main()
