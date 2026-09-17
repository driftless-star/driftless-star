"""Write the closed-loop status signal after Stage 5.

Runs in Stage 5's container immediately after the pressure fit has written the
new Stage 1 input. It decides whether the loop has converged (the transport
pressure profile has reached steady state between the initial and final time
slices) and writes that verdict to a small JSON signal file. The external loop
driver reads this file to decide whether to run another forward pass.

Non-positive pressure, steady pressure or the [transport_solver] horizon stops
the loop. The ``t_final`` criterion is a debugging criterion that stops the loop
when the transport clock reaches [transport_solver].t_final.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

# Reuse the stage's pressure loader (sibling script in the same Stage 5 container).
from fit_vmec_pressure_from_transport_h5 import _load_total_pressure

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

logger = logging.getLogger(__name__)


def _pressure_converged(transport: Path, *, rel_tol: float, method: str) -> bool:
    """Return whether Stage 5's transport evolution meets ``method``'s criterion.

    Compares the total pressure profile `P(rho)` at the final time slice of `transport`
    against the initial time slice. A small final-vs-initial change means the transport
    evolution barely moved the input profile, so the boundary fed back to Stage 1 is
    self-consistent.

    Parameters
    ----------
    transport : Path
        This pass's ``transport_solution.h5`` (species-resolved ``pressure_faces`` or
        ``temperature_faces`` and ``density_faces``, plus ``rho_face``).
    rel_tol : float
        Relative tolerance; convergence requires the selected measure to be below it.
    method : str
        Relative-change measure: ``"rms"`` or ``"pointwise"``.

    Returns
    -------
    bool
        ``True`` if the selected relative-change measure is below ``rel_tol``, else
        ``False``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"rms"`` or ``"pointwise"``.

    Notes
    -----
    If the solution has fewer than two distinct time slices (a static profile, or a
    single saved step) the initial and final slices coincide, so convergence cannot
    be assessed; this returns ``False`` with a warning so the loop never stops on a
    non-evolving profile.
    """
    if method not in {"rms", "pointwise"}:
        raise ValueError(
            f"Unsupported pressure convergence method {method!r}; expected one of: rms, pointwise"
        )

    _, p_initial, idx_initial = _load_total_pressure(transport, time_index=0, final_time=False)
    _, p_final, idx_final = _load_total_pressure(transport, time_index=-1, final_time=True)

    if idx_initial == idx_final:
        logger.warning(
            "%s has fewer than two distinct time slices (initial index %s == final index %s); "
            "cannot assess convergence, reporting not converged.",
            transport, idx_initial, idx_final,
        )
        return False

    rel = (p_final - p_initial) / p_initial
    if method == "rms":
        # Normalising by the number of rho points keeps the RMS grid-resolution independent.
        rel_change = float(np.sqrt(np.mean(rel**2)))
        measure = "relative RMS"
    else:
        # Every rho point must settle within rel_tol; a local change cannot be averaged away.
        rel_change = float(np.max(np.abs(rel)))
        measure = "maximum pointwise relative"

    logger.info("pressure %s change = %.3e (rel_tol = %.3e)", measure, rel_change, rel_tol)
    return rel_change < rel_tol


def rms_pressure_converged(transport: Path, *, rel_tol: float, common_config: Path) -> bool:
    """Return whether the relative RMS pressure change is below ``rel_tol``.

    ``common_config`` is unused. Every registered criterion takes the same arguments.
    """
    return _pressure_converged(transport, rel_tol=rel_tol, method="rms")


def pointwise_pressure_converged(transport: Path, *, rel_tol: float, common_config: Path) -> bool:
    """Return whether every point's relative pressure change is below ``rel_tol``.

    ``common_config`` is unused. Every registered criterion takes the same arguments.
    """
    return _pressure_converged(transport, rel_tol=rel_tol, method="pointwise")


def transport_horizon_reached(transport: Path, common_config: Path) -> bool:
    """Return whether the transport clock has reached the configured end time.

    Each pass resumes when the previous pass stopped. Therefore, [transport_solver].t_final is an
    absolute horizon. No transport time remains after ``final_time`` reaches this value.

    NEOPAX accepts ``t0 >= t_final`` and can return an empty solution. This check stops the loop
    before another pass starts.

    Parameters
    ----------
    transport : Path
        This pass's ``transport_solution.h5``, read for its ``final_time``.
    common_config : Path
        The ``common_input.toml`` used for this pass. It supplies [transport_solver].t_final.

    Returns
    -------
    bool
        ``True`` once ``final_time`` has reached ``t_final``.

    Raises
    ------
    KeyError
        If the solution lacks ``final_time`` or the config lacks [transport_solver].t_final.
    """
    with h5py.File(transport, "r") as f:
        if "final_time" not in f:
            raise KeyError(
                f"{transport} carries no 'final_time' dataset, so the transport clock cannot be compared "
                "against the configured horizon. It is written only by NEOPAX revisions that export the "
                "clock the next iteration resumes from."
            )
        final_time = float(np.asarray(f["final_time"][()]))
    solver_cfg = tomllib.loads(common_config.read_bytes().decode("utf-8")).get("transport_solver")
    if not isinstance(solver_cfg, dict) or "t_final" not in solver_cfg:
        raise KeyError(
            f"{common_config} must define [transport_solver].t_final, which is the horizon the loop stops at"
        )
    t_final = float(solver_cfg["t_final"])
    logger.info("transport clock at %r against [transport_solver].t_final %r", final_time, t_final)
    return final_time >= t_final


def t_final_converged(transport: Path, *, rel_tol: float, common_config: Path) -> bool:
    """Return whether the transport clock has reached [transport_solver].t_final.

    This is a debugging criterion. It reports convergence when the transport clock reaches
    ``t_final``. It does not read the pressure change.

    Parameters
    ----------
    transport : Path
        This pass's ``transport_solution.h5``, read for its ``final_time``.
    rel_tol : float
        Unused. Every registered criterion takes the same arguments.
    common_config : Path
        The ``common_input.toml`` used for this pass. It supplies [transport_solver].t_final.

    Returns
    -------
    bool
        ``True`` once ``final_time`` has reached ``t_final``.

    Raises
    ------
    KeyError
        If the solution lacks ``final_time`` or the config lacks [transport_solver].t_final.
    """
    return transport_horizon_reached(transport, common_config)


_PRESSURE_CONVERGENCE_METHODS: dict[str, Callable[..., bool]] = {
    "rms": rms_pressure_converged,
    "pointwise": pointwise_pressure_converged,
    "t_final": t_final_converged,
}


def resolve_pressure_convergence(method: str) -> Callable[..., bool]:
    """Return the pressure convergence function registered for ``method``."""
    try:
        return _PRESSURE_CONVERGENCE_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unsupported pressure convergence method {method!r}; expected one of: rms, pointwise, t_final"
        ) from None


def build_signal(
    transport: Path,
    *,
    rel_tol: float,
    common_config: Path,
    convergence_method: str = "pointwise",
) -> dict[str, str]:
    """Build the closed-loop status signal for the driver.

    The status is ``continue``, ``converged``, ``horizon`` or ``halted``. The driver runs another
    pass only for ``continue``. Convergence takes priority when a pass also reaches the horizon.
    ``t_final`` is a debugging criterion. It reports a clock at ``t_final`` as ``converged``, so the
    ``horizon`` status cannot occur under it.

    Parameters
    ----------
    transport : Path
        This pass's ``transport_solution.h5``.
    rel_tol : float
        Relative tolerance forwarded to the selected convergence criterion.
    common_config : Path
        The ``common_input.toml`` used for this pass. It supplies the configured horizon.
    convergence_method : str
        Convergence criterion, ``"rms"``, ``"pointwise"`` (the default) or ``"t_final"``.

    Returns
    -------
    dict
        Mapping with one ``status`` string.
    """
    pressure_converged = resolve_pressure_convergence(convergence_method)

    for label, time_index, final_time in (("initial", 0, False), ("final", -1, True)):
        _, pressure, _ = _load_total_pressure(transport, time_index=time_index, final_time=final_time)
        if np.any(pressure <= 0.0):
            logger.warning(
                "%s total pressure is non-positive at profile index/indices %s; the equilibrium "
                "was not sustained. Halting the loop; restart from different initial conditions.",
                label, np.flatnonzero(pressure <= 0.0).tolist(),
            )
            return {"status": "halted"}

    if pressure_converged(transport, rel_tol=rel_tol, common_config=common_config):
        return {"status": "converged"}
    if transport_horizon_reached(transport, common_config):
        logger.warning(
            "the transport clock has reached [transport_solver].t_final, so the next pass would have no "
            "time to advance into and would re-integrate this window under the times it already used. "
            "Stopping the loop; raise t_final to keep evolving."
        )
        return {"status": "horizon"}
    return {"status": "continue"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Stage 5 Post-Processing: write the closed-loop status signal.")
    parser.add_argument("--transport", type=Path, required=True,
                        help="This pass's transport_solution.h5.")
    parser.add_argument("--common-config", type=Path, required=True,
                        help="The common_input.toml this pass ran with, read for [transport_solver].t_final.")
    parser.add_argument("--signal", type=Path, required=True,
                        help="Output path for the status JSON the driver reads.")
    parser.add_argument("--pressure-rel-tol", type=float, required=True,
                        help="Relative tolerance on the final-vs-initial total pressure profile.")
    parser.add_argument(
        "--pressure-convergence-method",
        choices=_PRESSURE_CONVERGENCE_METHODS,
        default="pointwise",
        help="Convergence criterion: rms, pointwise (default) or t_final.",
    )
    args = parser.parse_args()
    if args.pressure_rel_tol <= 0.0:
        parser.error(f"--pressure-rel-tol must be positive, got {args.pressure_rel_tol}.")

    status = build_signal(
        args.transport,
        rel_tol=args.pressure_rel_tol,
        common_config=args.common_config,
        convergence_method=args.pressure_convergence_method,
    )
    args.signal.parent.mkdir(parents=True, exist_ok=True)
    args.signal.write_text(json.dumps(status) + "\n")
    print(f"# converge_status: {status}")
    print(f"# wrote_signal: {args.signal}")


if __name__ == "__main__":
    main()
