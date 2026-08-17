"""Stage 4 (turbulence) shell-command composition for the Snakemake workflow.

The Stage 4 GKX radial-scan script accepts many optional flags; this
module turns the user-facing ``config.yaml`` ``stage4.gkx`` block into
the per-phase shell commands run by the Snakefile's ``stage4_prepare``
checkpoint and its ``stage4_run_one``/``stage4_collect`` rules.
"""

from __future__ import annotations

_SCRIPT = "stages/stage4-turbulence/gkx_radial_scan.py"
_RELABEL_SCRIPT = "stages/stage4-turbulence/relabel_neopax_flux_radius.py"

# Minor-radius conventions stage4.neopax_radius_relabel accepts, matching the script's own choices.
RELABEL_CONVENTIONS: tuple[str, ...] = ("boozer_volume",)

# (config_key, cli_flag) accepted by the `prepare` subcommand; emitted as `<flag> <value>` when set.
# The config key t_max is spelled --t-final, an accepted alias whose argparse dest is t_max.
_PREPARE_OPTIONAL_FLAGS: list[tuple[str, str]] = [
    ("profiles_source",             "--profiles-source"),
    ("neopax_result",               "--neopax-result"),
    ("nx",                          "--nx"),
    ("ny",                          "--ny"),
    ("ntheta",                      "--ntheta"),
    ("t_max",                       "--t-final"),
    ("sample_stride",               "--sample-stride"),
    ("diagnostics_stride",          "--diagnostics-stride"),
    ("analytical_n_radii",          "--analytical-n-radii"),
    ("rho_indices",                 "--rho-indices"),
    ("rho_min",                     "--rho-min"),
    ("rho_max",                     "--rho-max"),
    ("num_radii",                   "--num-radii"),
    ("response_mode",               "--response-mode"),
    ("perturb_density_species",     "--perturb-density-species"),
    ("perturb_temperature_species", "--perturb-temperature-species"),
    ("dkap_density",                "--dkap-density"),
    ("dkap_temperature",            "--dkap-temperature"),
    ("perturb_rel_step",            "--perturb-rel-step"),
]

# (config_key, on_flag, off_flag) tri-state toggles accepted by the `prepare` subcommand.
_PREPARE_BOOL_FLAGS: list[tuple[str, str, str]] = [
    ("resolved_diagnostics", "--resolved-diagnostics", "--no-resolved-diagnostics"),
]

# Flux averaging and plotting happen in the collect step. Config t_max is deliberately not re-emitted
# at collect: it reaches the manifest via prepare, and collect's --t-final falls back to the manifest
# value, keeping a single source of truth for the time window.
_COLLECT_OPTIONAL_FLAGS: list[tuple[str, str]] = [
    ("average_window",  "--average-window"),
    ("average_reducer", "--average-reducer"),
]

_COLLECT_BOOL_FLAGS: list[tuple[str, str, str]] = [
    ("plot",                 "--plot",                 "--no-plot"),
    ("plot_run_heat_traces", "--plot-run-heat-traces", "--no-plot-run-heat-traces"),
]


def _append_optional_flags(parts: list[str], stage_cfg: dict, table: list[tuple[str, str]]) -> None:
    """Append ``<flag> <value>`` to ``parts`` for each table entry whose config value is not None."""
    for key, flag in table:
        value = stage_cfg.get(key)
        if value is not None:
            parts.append(f"{flag} {value}")


def _append_bool_flags(parts: list[str], stage_cfg: dict, table: list[tuple[str, str, str]]) -> None:
    """Append the on-flag for True and the off-flag for False; a missing/None key appends nothing."""
    for key, on, off in table:
        value = stage_cfg.get(key)
        if value is True:
            parts.append(on)
        elif value is False:
            parts.append(off)


def prepare_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
) -> str:
    """Compose the Stage 4 GKX ``prepare`` shell command.

    Concretely: build the CLI arguments for the scan script's ``prepare``
    subcommand from ``stage_cfg`` and wrap them in a ``docker run`` invocation
    of the stage image. Nothing executes here; the returned string is the
    command Snakemake runs when the checkpoint's job fires.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-gkx-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage4.gkx`` block.
    output_dir : str
        Stage 4 output directory (already ``{run_name}``-substituted).

    Returns
    -------
    str
        A single-line ``prepare`` subcommand that writes the per-radius manifest
        and GKX runtime TOMLs. ``{input.*}`` placeholders remain literal so
        Snakemake substitutes them at rule-execution time.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} prepare",
        "--common-config {input.common_config}",
        "--gkx-template {input.config_file}",
        "--vmec-file-override {input.wout}",
        "--boozer-file-override {input.boozer}",
        f"--output-dir {output_dir}",
    ]
    _append_optional_flags(parts, stage_cfg, _PREPARE_OPTIONAL_FLAGS)
    _append_bool_flags(parts, stage_cfg, _PREPARE_BOOL_FLAGS)
    return " ".join(parts)


def run_one_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
    device: str,
) -> str:
    """Compose the Stage 4 GKX ``run-one`` shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-gkx-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage4.gkx`` block.
    output_dir : str
        Stage 4 output directory (already ``{run_name}``-substituted).
    device : str
        ``"cpu"`` or ``"gpu"``; selects the worker's JAX backend. Device
        assignment lives in ``docker_prefix``, so no GPU flag is emitted here
        and the worker pins the device its container exposes.

    Returns
    -------
    str
        A single-line ``run-one`` subcommand that evolves one surface. The
        ``{wildcards.surf}`` placeholder names the manifest run to execute and
        stays literal so Snakemake substitutes it at rule-execution time.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} run-one",
        f"--manifest {output_dir}/manifest.json",
        "--run-name {wildcards.surf}",
        f"--backend {device}",
    ]
    # --verbose-worker is a plain store_true with no negative form, so only an explicit config True
    # emits it; False and a missing key both leave the worker at its quiet default.
    if stage_cfg.get("verbose_workers") is True:
        parts.append("--verbose-worker")
    return " ".join(parts)


def collect_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
) -> str:
    """Compose the Stage 4 GKX ``collect`` shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-gkx-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage4.gkx`` block.
    output_dir : str
        Stage 4 output directory (already ``{run_name}``-substituted).

    Returns
    -------
    str
        A single-line ``collect`` subcommand that reduces per-radius
        diagnostics into ``flux_summary.h5`` and ``neopax_fluxes.h5``.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} collect",
        f"--manifest {output_dir}/manifest.json",
        f"--out {output_dir}/flux_summary.h5",
        f"--neopax-flux-out {output_dir}/neopax_fluxes.h5",
    ]
    _append_optional_flags(parts, stage_cfg, _COLLECT_OPTIONAL_FLAGS)
    _append_bool_flags(parts, stage_cfg, _COLLECT_BOOL_FLAGS)
    return " ".join(parts)


def resolve_radius_relabel(config: dict) -> str | None:
    """Turn a run config's ``stage4.neopax_radius_relabel`` key into a minor-radius convention.

    Parameters
    ----------
    config : dict
        Parsed run config. ``stage4.neopax_radius_relabel`` is ``false`` (the default) to write the
        flux file's radial grid through unchanged, or ``"boozer_volume"`` for the convention NEOPAX
        builds its grid on today.

    Returns
    -------
    str or None
        The convention name, or ``None`` when the step is off.

    Raises
    ------
    ValueError
        If the key is neither ``false`` nor one of :data:`RELABEL_CONVENTIONS`.
    """
    value = config.get("stage4", {}).get("neopax_radius_relabel", False)
    if value is False:
        return None
    if not isinstance(value, str) or value not in RELABEL_CONVENTIONS:
        raise ValueError(
            f"config['stage4']['neopax_radius_relabel'] is {value!r}; write false to skip the step, or one of "
            f"{', '.join(repr(c) for c in RELABEL_CONVENTIONS)} to name the minor-radius convention NEOPAX builds "
            "its radial grid on."
        )
    return value


def relabel_cmd(
    *,
    docker_prefix: str,
    image: str,
    flux_file: str,
    wout: str,
    boozer: str,
    convention: str,
    rho_edge: float,
) -> str:
    """Compose the shell command that puts the collected flux file on NEOPAX's radial grid.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile. Rewriting one HDF5 dataset needs no
        GPU, so the caller passes the prefix that claims neither a GPU nor a scheduling slot.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-gkx-cpu``).
    flux_file : str
        ``neopax_fluxes.h5`` to rewrite in place.
    wout : str
        Stage 1 VMEC output supplying the plasma volume.
    boozer : str
        Stage 2 Boozer output supplying ``rmnc_b``.
    convention : str
        Minor-radius convention from :func:`resolve_radius_relabel`.
    rho_edge : float
        NEOPAX's ``[geometry].rho_edge``, from ``stage5_helper.read_rho_edge``.

    Returns
    -------
    str
        A single-line command that rewrites the flux file's ``r`` dataset in place.
    """
    return " ".join([
        f"{docker_prefix} {image}",
        f"python {_RELABEL_SCRIPT}",
        f"--flux-file {flux_file}",
        f"--wout {wout}",
        f"--boozer {boozer}",
        f"--convention {convention}",
        f"--rho-edge {rho_edge}",
    ])
