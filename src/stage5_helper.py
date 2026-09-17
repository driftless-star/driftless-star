"""Stage 5 (NEOPAX transport) workflow helpers.

NEOPAX is configured via a TOML file rather than CLI flags. :func:`prepare_neopax_config`
writes a path-resolved copy of the shared ``common_input`` template under the run's output
directory (leaving the committed template untouched); the Snakefile then runs NEOPAX on that
copy. Called at Snakefile parse time.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .utils import apply_assignments

PRESSURE_CONVERGENCE_METHODS = ("rms", "pointwise", "t_final")


def resolve_pressure_convergence_method(config: dict) -> str:
    """Return the configured pressure convergence method, defaulting to pointwise.

    Parameters
    ----------
    config : dict
        Parsed run config. ``convergence.method`` may be ``"rms"``, ``"pointwise"`` or
        ``"t_final"``.

    Returns
    -------
    str
        The validated convergence method.

    Raises
    ------
    ValueError
        If the convergence block is not a mapping or its method is unsupported.
    """
    convergence = config.get("convergence", {})
    if not isinstance(convergence, dict):
        raise ValueError(
            f"config['convergence'] must be a mapping, got {convergence!r}."
        )
    method = convergence.get("method", "pointwise")
    if not isinstance(method, str) or method not in PRESSURE_CONVERGENCE_METHODS:
        choices = ", ".join(repr(choice) for choice in PRESSURE_CONVERGENCE_METHODS)
        raise ValueError(
            f"config['convergence']['method'] must be one of {choices}, got {method!r}."
        )
    return method


def read_rho_edge(s5_config_template: str) -> float:
    """Return ``[geometry].rho_edge`` from the NEOPAX template, or NEOPAX's default of 1.0.

    NEOPAX solves on a staggered grid whose ``n_radial + 1`` faces are
    ``linspace(0, rho_edge, n_radial + 1)``, and the Stage 3 and Stage 4 scans sample those faces,
    so ``rho_edge`` bounds every radial grid in the pipeline. It therefore has to reach the Stage 4
    flux-file relabelling step too, which otherwise assumes the grid runs out to ``rho = 1``. Read
    here rather than duplicated into ``config.yaml`` so the NEOPAX template stays the single source
    of truth for the transport grid.

    Parameters
    ----------
    s5_config_template : str
        Path to the shared NEOPAX template (``inputs/<run>/common_input.toml``).

    Returns
    -------
    float
        ``[geometry].rho_edge``, defaulting to ``1.0`` when the key is absent.

    Raises
    ------
    ValueError
        If the key is present but not a positive, finite number.
    """
    cfg = tomllib.loads(Path(s5_config_template).read_bytes().decode("utf-8"))
    rho_edge = cfg.get("geometry", {}).get("rho_edge", 1.0)
    if not isinstance(rho_edge, (int, float)) or isinstance(rho_edge, bool) or not 0.0 < float(rho_edge) <= 1.0:
        raise ValueError(
            f"{s5_config_template}: [geometry].rho_edge must be a number in (0, 1], got {rho_edge!r}."
        )
    return float(rho_edge)


def prepare_neopax_config(
    *,
    s5_config_template: str,
    s5_resolved_config: str,
    s1_output: str,
    s2_output: str,
    s3_output: str,
    s4_output: str,
    s5_output_dir: str,
) -> None:
    """Write a path-resolved copy of the NEOPAX template for the current run.

    Writes a path-resolved copy of ``s5_config_template`` to ``s5_resolved_config``, with
    its five path fields rewritten relative to the copy's own directory (NEOPAX runs
    there). The committed template is never modified.

    Parameters
    ----------
    s5_config_template : str
        Path to the shared NEOPAX template (``inputs/<run>/common_input.toml``).
    s5_resolved_config : str
        Path of the resolved copy to write (under ``outputs/<run>/stage5_transport/``).
    s1_output, s2_output, s3_output, s4_output : str
        Paths to Stage 1-4 output artifacts referenced by NEOPAX.
    s5_output_dir : str
        Stage 5 output directory (where NEOPAX writes ``transport_solution.h5``).
    """
    resolved = Path(s5_resolved_config)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    base = resolved.parent.resolve()

    def _rel(p: str) -> str:
        return str(Path(p).resolve().relative_to(base, walk_up=True))

    assignments = {
        "vmec_file": f'"{_rel(s1_output)}"',
        "boozer_file": f'"{_rel(s2_output)}"',
        "neoclassical_file": f'"{_rel(s3_output)}"',
        "turbulence_file": f'"{_rel(s4_output)}"',
        "transport_output_dir": f'"{_rel(s5_output_dir)}/"',
    }
    template_text = Path(s5_config_template).read_bytes().decode("utf-8")
    resolved.write_bytes(apply_assignments(template_text, assignments).encode("utf-8"))
