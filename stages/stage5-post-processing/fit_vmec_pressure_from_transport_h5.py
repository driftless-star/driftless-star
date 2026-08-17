"""Fit or write a VMEC-style pressure power series from NEOPAX transport HDF5 output.

This script reads ``transport_solution.h5``, extracts a time slice, sums the
species pressure profiles on its **face** grid into a total pressure profile
``P(rho)``, converts to ``s = rho**2``, and fits

    P(s) ~= sum_k AM[k] * s**k

which matches the VMEC / vmec_jax ``PMASS_TYPE = "power_series"`` convention
when ``PRES_SCALE = 1``.

Modes
-----
- ``fit``:
  print the fitted ``AM`` coefficients
- ``write-input``:
  update the ``AM`` / ``PRES_SCALE`` / ``PMASS_TYPE`` assignments inside the
  ``&INDATA`` block of a VMEC ``input.*`` file

The comparison and plotting siblings also import :func:`_saved_time_count` from here to pick
the last distinctly timed record of a transport solution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

import h5py
import numpy as np

logger = logging.getLogger(__name__)

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

    NEOPAX evolves its state on the ``n_radial`` cell centers, written as ``rho`` / ``pressure`` /
    ``temperature`` / ``density``, and those centers span neither ``rho = 0`` nor ``rho = rho_edge``. VMEC
    evaluates the fitted power series over the whole of ``s = rho**2`` in ``[0, 1]``, so this reads the
    ``n_radial + 1`` faces, which do span the full minor radius.

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
        Species-summed face pressure, shape ``(n_radial + 1,)``.
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
    # A NaN here would be polynomial-fitted straight into the equilibrium's AM coefficients and only
    # fail much later inside VMEC, so it is rejected at read time.
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


def _fit_power_series(s: np.ndarray, p: np.ndarray, degree: int) -> np.ndarray:
    coeffs = np.polynomial.polynomial.polyfit(s, p, deg=int(degree))
    return np.asarray(coeffs, dtype=float)


def _format_am_line(coeffs: np.ndarray) -> str:
    return "AM = " + ", ".join(f"{float(c):.16E}" for c in np.asarray(coeffs, dtype=float))


def _rewrite_indata_scalar_line(block_text: str, key: str, value_text: str) -> str:
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(key)}\s*=.*$", flags=re.IGNORECASE | re.MULTILINE)
    replacement_done = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal replacement_done
        replacement_done = True
        indent = match.group("indent")
        return f"{indent}{value_text}"

    updated = pattern.sub(_replace, block_text, count=1)
    if replacement_done:
        return updated

    lines = block_text.splitlines()
    insert_at = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == "/":
            insert_at = i
            break
    lines.insert(insert_at, f"  {value_text}")
    return "\n".join(lines)


def _write_vmec_input_with_pressure_fit(
    vmec_input: Path,
    coeffs: np.ndarray,
    *,
    output_path: Path | None,
) -> Path:
    text = vmec_input.read_text()
    m_start = re.search(r"&\s*INDATA\b", text, flags=re.IGNORECASE)
    if not m_start:
        raise ValueError(f"{vmec_input} does not contain an &INDATA block")
    m_end = re.search(r"^\s*/\s*$", text[m_start.end():], flags=re.MULTILINE)
    if not m_end:
        raise ValueError(f"{vmec_input} does not contain a terminating '/' for &INDATA")

    block_start = m_start.start()
    body_start = m_start.end()
    body_end = body_start + m_end.start()
    block_end = body_start + m_end.end()

    prefix = text[:body_start]
    block_body = text[body_start:block_end]
    suffix = text[block_end:]

    block_body = _rewrite_indata_scalar_line(block_body, "PMASS_TYPE", "PMASS_TYPE = 'power_series'")
    block_body = _rewrite_indata_scalar_line(block_body, "PRES_SCALE", "PRES_SCALE = 1.0000000000000000E+00")
    block_body = _rewrite_indata_scalar_line(block_body, "AM", _format_am_line(coeffs))

    dst = output_path if output_path is not None else vmec_input
    dst.write_text(prefix + block_body + suffix)
    return dst


def _fit_from_args(args) -> tuple[np.ndarray, int | None, int]:
    rho, total_pressure, resolved_index = _load_total_pressure(
        args.h5_path,
        time_index=int(args.time_index),
        final_time=bool(args.final_time),
    )
    s = rho**2

    # Drop the first point only when the grid actually starts on the magnetic axis. On a grid that does not
    # start at rho = 0 the first point is a real innermost sample.
    if args.drop_axis and s.size > 1:
        if np.isclose(rho[0], 0.0):
            s = s[1:]
            total_pressure = total_pressure[1:]
        else:
            logger.warning(
                "--drop-axis was requested but the first radius is rho = %.6g, not the magnetic axis; "
                "keeping it, because dropping it would discard a real innermost data point.",
                float(rho[0]),
            )

    # A degree-d power series has d+1 coefficients and needs at least d+1 sample
    # points to be well-posed. Clamp the effective degree to the number of points 
    # minus one so the fit stays well-conditioned regardless of the radial resolution.
    effective_degree = min(int(args.degree), s.size - 1)
    coeffs = _fit_power_series(s, total_pressure, degree=effective_degree)
    return coeffs, resolved_index, effective_degree


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=False)

    def _add_common(subparser):
        subparser.add_argument("h5_path", type=Path, help="Path to transport_solution.h5")
        subparser.add_argument("--degree", type=int, default=8, help="Polynomial degree for AM coefficients")
        subparser.add_argument("--time-index", type=int, default=-1, help="Time slice to read if the file is time-dependent")
        subparser.add_argument(
            "--final-time",
            action="store_true",
            help="Use the final saved time slice explicitly. Equivalent to the last time index.",
        )
        subparser.add_argument(
            "--drop-axis",
            action="store_true",
            help="Exclude the magnetic axis point from the fit if rho_face[0] = 0 is causing trouble. "
                 "Skipped with a warning if the first radius is not the axis.",
        )

    fit_parser = subparsers.add_parser("fit", help="Print fitted VMEC AM coefficients from transport_solution.h5")
    _add_common(fit_parser)

    write_parser = subparsers.add_parser(
        "write-input",
        help="Fit AM coefficients from transport_solution.h5 and write them into a VMEC input file",
    )
    _add_common(write_parser)
    write_parser.add_argument("vmec_input", type=Path, help="Path to VMEC input.* file to update")
    write_parser.add_argument(
        "--output-input",
        type=Path,
        default=None,
        help="Optional output path for the updated VMEC input. Defaults to overwriting vmec_input.",
    )

    args = parser.parse_args()
    if args.command is None:
        args.command = "fit"

    coeffs, resolved_index, effective_degree = _fit_from_args(args)

    if args.command == "write-input":
        out_path = _write_vmec_input_with_pressure_fit(
            args.vmec_input,
            coeffs,
            output_path=args.output_input,
        )
        print(f"# wrote_vmec_input: {out_path}")

    print(f"# input: {args.h5_path}")
    print(f"# requested_degree: {int(args.degree)}")
    print(f"# effective_degree: {effective_degree}")
    if resolved_index is not None:
        print(f"# resolved_time_index: {resolved_index}")
    else:
        print("# resolved_time_index: static_profile")
    print("# VMEC / vmec_jax power-series pressure fit")
    print("# P(s) ~= sum_k AM[k] * s**k")
    print("s = rho**2")
    print(_format_am_line(coeffs))
    print("PRES_SCALE = 1.0")


if __name__ == "__main__":
    main()
