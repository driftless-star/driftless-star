"""Tests for the ``src.stage4_helper`` per-phase command composers.

``prepare_cmd``, ``run_one_cmd``, and ``collect_cmd`` turn the ``stage4.gkx``
config block into the three shell commands that the Snakefile's per-phase Stage 4
rules run. The exact strings are the contract with those rules, so this pins them:
the static base flags (the ``--vmec-file-override`` and ``--boozer-file-override``
geometry inputs plus the literal ``{input.*}`` and ``{wildcards.surf}`` placeholders
Snakemake substitutes at run time), the not-None-only optional flags on ``prepare``
(where ``t_max`` is emitted as ``--t-final``) and ``collect``, the tri-state and
on-only booleans, and the absence of scan-level parallelism and GPU flags now
that surface concurrency belongs to ``snakemake --cores`` and device assignment
to the docker prefix.
"""

from __future__ import annotations

import argparse
import inspect
import re
from collections.abc import Callable

import pytest

from src.stage4_helper import (
    RELABEL_CONVENTIONS,
    collect_cmd,
    prepare_cmd,
    relabel_cmd,
    resolve_radius_relabel,
    run_one_cmd,
)
from tests.helpers.stage_import import load_stage_module

_STAGE4_SCRIPT = "stages/stage4-turbulence/gkx_radial_scan.py"
# Snakemake substitutes {input.*}/{output.*} at run time; swap them for literal path
# tokens so the emitted command parses as a plain argument vector here.
_PLACEHOLDER = re.compile(r"\{(?:input|output)\.[A-Za-z0-9_]+\}")
_scan = load_stage_module(_STAGE4_SCRIPT)

# Every prepare-phase optional key with a quick-run-like value, used both to exercise
# each flag individually and to prove that a fully-populated config never leaks
# scan-level flags. The t_max key is deliberately spelled --t-final, the prepare
# parser's alias whose argparse dest is t_max.
_PREPARE_OPTIONALS: list[tuple[str, str, object]] = [
    ("profiles_source",               "--profiles-source",               "prescribed"),  # the loop's iteration-2+ value, not the default
    ("neopax_result",                 "--neopax-result",                 "stage5_transport/transport_solution.h5"),
    ("nx",                            "--nx",                            12),
    ("ny",                            "--ny",                            12),
    ("ntheta",                        "--ntheta",                        30),
    ("t_max",                         "--t-final",                       10.0),
    ("sample_stride",                 "--sample-stride",                 50),
    ("diagnostics_stride",            "--diagnostics-stride",            1),
    ("analytical_n_radii",            "--analytical-n-radii",            5),
    ("rho_indices",                   "--rho-indices",                   "1,5,10"),
    ("rho_min",                       "--rho-min",                       0.1),
    ("rho_max",                       "--rho-max",                       0.9),
    ("num_radii",                     "--num-radii",                     4),
    ("response_mode",                 "--response-mode",                 "fd_gradients"),
    ("perturb_density_species",       "--perturb-density-species",       "D"),
    ("perturb_temperature_species",   "--perturb-temperature-species",   "D,T"),
    ("dkap_density",                  "--dkap-density",                  0.5),
    ("dkap_temperature",              "--dkap-temperature",              0.5),
    ("perturb_rel_step",              "--perturb-rel-step",              0.5),
]

# max_parallel, collect_even_if_failures, and gpu_ids are retired keys that older
# configs may still carry; every composer must ignore them.
_FULL_CFG: dict = {key: value for key, _, value in _PREPARE_OPTIONALS} | {
    "resolved_diagnostics": False,
    "average_window": 1.0,
    "average_reducer": "t3d_median",
    "gpu_ids": "0,1",
    "plot": False,
    "plot_run_heat_traces": True,
    "verbose_workers": True,
    "max_parallel": 1,
    "collect_even_if_failures": True,
}


def compose(composer: Callable[..., str], **overrides) -> str:
    """Build a command with quick-run-like defaults, overriding only what a test varies.

    Each composer takes exactly the arguments its phase uses, so the shared defaults are narrowed to the ones the
    called composer accepts.
    """
    base = dict(
        docker_prefix="docker run --rm",
        image="ghcr.io/driftless-star/driftless-star:stage-4-gkx-cpu",
        stage_cfg={},
        output_dir="outputs/quick_run/stage4_turbulence",
        device="cpu",
    )
    base.update(overrides)
    accepted = inspect.signature(composer).parameters
    return composer(**{name: value for name, value in base.items() if name in accepted})


def parse_with_stage_script(command: str) -> argparse.Namespace:
    """Feed a composed command's argv into the real stage script parser and return the namespace.

    Fails the test if argparse rejects any flag (argparse exits via ``sys.exit(2)``).
    """
    concrete = _PLACEHOLDER.sub("placeholder_path", command).replace("{wildcards.surf}", "rho_001_r0p1000")
    tokens = concrete.split()
    argv = tokens[tokens.index(_STAGE4_SCRIPT) + 1:]
    try:
        return _scan.build_parser().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"stage script parser rejected a composer-emitted flag: argv={argv} (exit {exc.code})")


# `prepare_cmd` builds the shell command that writes the per-radius manifest and GKX runtime TOMLs. With an empty
# config it should emit just the fixed base command: the two geometry-input overrides plus the literal `{input.*}`
# placeholders Snakemake fills in at run time. The prepare parser takes no backend flag, so none may appear.
def test_prepare_base_command_with_empty_config() -> None:
    assert compose(prepare_cmd) == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-4-gkx-cpu "
        "python stages/stage4-turbulence/gkx_radial_scan.py prepare "
        "--common-config {input.common_config} "
        "--gkx-template {input.config_file} "
        "--vmec-file-override {input.wout} "
        "--boozer-file-override {input.boozer} "
        "--output-dir outputs/quick_run/stage4_turbulence"
    )


# `run_one_cmd` builds the per-surface worker command. This pins the exact string: the manifest path is derived from the
# output dir, the surface is selected by run-directory basename through the literal `{wildcards.surf}` placeholder that
# Snakemake substitutes at rule-execution time, and only the backend is added with an empty config on cpu.
def test_run_one_base_command_with_empty_config() -> None:
    assert compose(run_one_cmd) == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-4-gkx-cpu "
        "python stages/stage4-turbulence/gkx_radial_scan.py run-one "
        "--manifest outputs/quick_run/stage4_turbulence/manifest.json "
        "--run-name {wildcards.surf} "
        "--backend cpu"
    )


# `collect_cmd` builds the reduction command that folds per-radius diagnostics into the two flux HDF5 files. This pins
# the exact string, including the flux_summary.h5 and neopax_fluxes.h5 destination filenames the pipeline consumes.
def test_collect_base_command_with_empty_config() -> None:
    assert compose(collect_cmd) == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-4-gkx-cpu "
        "python stages/stage4-turbulence/gkx_radial_scan.py collect "
        "--manifest outputs/quick_run/stage4_turbulence/manifest.json "
        "--out outputs/quick_run/stage4_turbulence/flux_summary.h5 "
        "--neopax-flux-out outputs/quick_run/stage4_turbulence/neopax_fluxes.h5"
    )


# Each prepare-phase optional key set on its own must emit its flag-value pair exactly once; an absent key must emit
# nothing. Parametrizing over the full optional table also locks the config-key to CLI-flag spelling, in particular the
# t_max to --t-final rename and analytical_n_radii to --analytical-n-radii.
@pytest.mark.parametrize("key, flag, value", _PREPARE_OPTIONALS)
def test_prepare_optional_flag_emitted_exactly_once(key: str, flag: str, value: object) -> None:
    out = compose(prepare_cmd, stage_cfg={key: value})
    assert f"{flag} {value}" in out
    assert out.split().count(flag) == 1
    assert flag not in compose(prepare_cmd, stage_cfg={})


# resolved_diagnostics is the prepare-phase tri-state. True emits the on-flag, False the off-flag, and an absent key
# emits neither, leaving the value to the GKX template. Token membership, because --resolved-diagnostics is a
# substring of --no-resolved-diagnostics and a plain `in` check would pass on the off-flag alone.
def test_prepare_resolved_diagnostics_tristate() -> None:
    assert "--resolved-diagnostics" in compose(prepare_cmd, stage_cfg={"resolved_diagnostics": True}).split()
    assert "--no-resolved-diagnostics" in compose(prepare_cmd, stage_cfg={"resolved_diagnostics": False}).split()
    absent = compose(prepare_cmd, stage_cfg={}).split()
    assert "--resolved-diagnostics" not in absent
    assert "--no-resolved-diagnostics" not in absent


# Flux averaging happens in the reduction step, so average_window is a collect-phase optional: present exactly once
# when set, absent otherwise.
def test_collect_average_window_emitted_exactly_once() -> None:
    out = compose(collect_cmd, stage_cfg={"average_window": 1.0})
    assert "--average-window 1.0" in out
    assert out.split().count("--average-window") == 1
    assert "--average-window" not in compose(collect_cmd, stage_cfg={})


# The reducer picks which estimator folds each time trace into one number, so leaving it unemitted silently pins every
# run to window_mean and makes t3d_median unreachable from a config. Absent must stay absent, since the stage script's
# own default is what an omitted key is meant to select.
def test_collect_average_reducer_emitted_exactly_once() -> None:
    out = compose(collect_cmd, stage_cfg={"average_reducer": "t3d_median"})
    assert "--average-reducer t3d_median" in out
    assert out.split().count("--average-reducer") == 1
    assert "--average-reducer" not in compose(collect_cmd, stage_cfg={})


# Plot toggles live on `collect` and are tri-state: True emits the on-flag, False the off-flag, absent emits neither.
# Token membership so a short flag like --plot is not falsely found inside --no-plot or --plot-run-heat-traces.
@pytest.mark.parametrize(
    "key, on, off",
    [
        ("plot", "--plot", "--no-plot"),
        ("plot_run_heat_traces", "--plot-run-heat-traces", "--no-plot-run-heat-traces"),
    ],
)
def test_collect_plot_tristates(key: str, on: str, off: str) -> None:
    assert on in compose(collect_cmd, stage_cfg={key: True}).split()
    assert off in compose(collect_cmd, stage_cfg={key: False}).split()
    absent = compose(collect_cmd, stage_cfg={}).split()
    assert on not in absent and off not in absent


# The run-one worker's --verbose-worker is a plain store_true with no negative form, unlike the tri-state toggles: only
# an explicit config True emits it, while False and an absent key both emit nothing and leave the worker at its quiet
# default. This asymmetry is asserted explicitly.
def test_run_one_verbose_worker_on_only() -> None:
    assert "--verbose-worker" in compose(run_one_cmd, stage_cfg={"verbose_workers": True}).split()
    assert "--verbose-worker" not in compose(run_one_cmd, stage_cfg={"verbose_workers": False}).split()
    assert "--verbose-worker" not in compose(run_one_cmd, stage_cfg={}).split()
    assert "--no-verbose-worker" not in compose(run_one_cmd, stage_cfg={"verbose_workers": False})


# Device assignment lives in the docker run prefix, so the worker command itself carries no GPU flag. Even a config
# still holding the retired gpu_ids key must never make run-one emit --gpu-ids; the worker then pins the one device
# its container exposes.
def test_run_one_never_emits_gpu_ids() -> None:
    cfg = {"gpu_ids": "0,1"}
    assert "--gpu-ids" not in compose(run_one_cmd, stage_cfg=cfg, device="gpu")
    assert "--gpu-ids" not in compose(run_one_cmd, stage_cfg=cfg, device="cpu")
    assert "--gpu-ids" not in compose(run_one_cmd, stage_cfg={}, device="gpu")


# A drift guard. The exact-string tests only check the composer against itself; this checks it against the real stage
# script. It configures every optional (including the t_max to --t-final rename), substitutes the Snakemake
# placeholders, and feeds the argv into the stage script's own `build_parser`. Parsing must succeed and dispatch to
# cmd_prepare, so a renamed or dropped flag on the prepare subparser fails here.
def test_prepare_flags_parse_and_dispatch_to_cmd_prepare() -> None:
    args = parse_with_stage_script(compose(prepare_cmd, stage_cfg=_FULL_CFG))
    assert args.func.__name__ == "cmd_prepare"
    assert args.t_max == 10.0


# A drift guard for the worker phase: the composed run-one command (gpu variant with verbose_workers on, so
# --verbose-worker is included) must parse with the real stage script and dispatch to cmd_run_one, with the
# substituted surface name landing on the run-one --run-name argument. gpu_ids staying at the parser's None default
# proves the worker self-pins the device its container exposes instead of receiving an id list.
def test_run_one_flags_parse_and_dispatch_to_cmd_run_one() -> None:
    args = parse_with_stage_script(compose(run_one_cmd, stage_cfg=_FULL_CFG, device="gpu"))
    assert args.func.__name__ == "cmd_run_one"
    assert args.run_name == "rho_001_r0p1000"
    assert args.gpu_ids is None


# A drift guard for the reduction phase: the composed collect command must parse with the real stage script, dispatch
# to cmd_collect, and route the two destination paths onto --out and --neopax-flux-out. t_final must stay None so it
# falls back to the manifest's t_max, which prepare recorded from the same config key.
def test_collect_flags_parse_and_dispatch_to_cmd_collect() -> None:
    args = parse_with_stage_script(compose(collect_cmd, stage_cfg=_FULL_CFG))
    assert args.func.__name__ == "cmd_collect"
    assert args.out == "outputs/quick_run/stage4_turbulence/flux_summary.h5"
    assert args.neopax_flux_out == "outputs/quick_run/stage4_turbulence/neopax_fluxes.h5"
    assert args.t_final is None
    assert args.plot is False
    assert args.average_reducer == "t3d_median"


# The time window has a single source of truth: config t_max reaches the manifest via prepare, and collect's --t-final
# falls back to that manifest value. Even with t_max configured, collect must therefore never emit --t-final.
def test_collect_never_emits_t_final() -> None:
    assert "--t-final" not in compose(collect_cmd, stage_cfg=_FULL_CFG)


# Surface-level concurrency now belongs to `snakemake --cores`, so even a fully-populated config (including the retired
# max_parallel and collect_even_if_failures keys) must never make any composer emit a scan-level flag.
def test_no_composer_emits_scan_level_flags() -> None:
    for composer in (prepare_cmd, run_one_cmd, collect_cmd):
        for device in ("cpu", "gpu"):
            out = compose(composer, stage_cfg=_FULL_CFG, device=device)
            assert "--max-parallel" not in out
            assert "--collect-even-if-failures" not in out


# --- the NEOPAX radial-grid relabelling step ---

_RELABEL_SCRIPT = "stages/stage4-turbulence/relabel_neopax_flux_radius.py"
_relabel = load_stage_module(_RELABEL_SCRIPT)

_RELABEL_ARGS = dict(
    flux_file="outputs/quick_run/stage4_turbulence/neopax_fluxes.h5",
    wout="outputs/quick_run/stage1_equilibrium/wout_run.nc",
    boozer="outputs/quick_run/stage2_boozer/boozmn_run.nc",
    convention="boozer_volume",
    rho_edge=0.7,
)


# An absent key and an explicit false both leave the flux file's grid alone. There is no boolean spelling of "on",
# since the convention has to be named, so a config can never enable the step without saying which grid it targets.
@pytest.mark.parametrize("cfg", [{}, {"stage4": {}}, {"stage4": {"neopax_radius_relabel": False}}])
def test_radius_relabel_off_by_default(cfg: dict) -> None:
    assert resolve_radius_relabel(cfg) is None


@pytest.mark.parametrize("convention", RELABEL_CONVENTIONS)
def test_radius_relabel_accepts_each_convention(convention: str) -> None:
    assert resolve_radius_relabel({"stage4": {"neopax_radius_relabel": convention}}) == convention


# The convention names live in two places, since a stage script runs inside its container and never imports `src`.
# A name added on one side only would be accepted by config and then rejected by argparse inside the container.
def test_the_config_conventions_match_the_script_conventions() -> None:
    assert RELABEL_CONVENTIONS == _relabel.CONVENTIONS


# True is rejected rather than mapped onto a default convention, because the key is a name precisely so that
# NEOPAX's choice of minor radius, the thing that might change, has to be stated by the config.
@pytest.mark.parametrize("bad", [True, "true", "yes", "boozer", "vmec_aminor", 1, None])
def test_radius_relabel_rejects_anything_that_is_not_a_convention(bad: object) -> None:
    with pytest.raises(ValueError, match="neopax_radius_relabel"):
        resolve_radius_relabel({"stage4": {"neopax_radius_relabel": bad}})


def test_relabel_base_command() -> None:
    assert compose(relabel_cmd, **_RELABEL_ARGS) == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-4-gkx-cpu "
        "python stages/stage4-turbulence/relabel_neopax_flux_radius.py "
        "--flux-file outputs/quick_run/stage4_turbulence/neopax_fluxes.h5 "
        "--wout outputs/quick_run/stage1_equilibrium/wout_run.nc "
        "--boozer outputs/quick_run/stage2_boozer/boozmn_run.nc "
        "--convention boozer_volume "
        "--rho-edge 0.7"
    )


# The same drift guard the other phases get: the composed command is fed to the relabel script's own parser, so a
# renamed or dropped flag fails here rather than inside the container. rho_edge in particular has to survive as a
# number, since it is read from the NEOPAX template rather than from config.yaml.
def test_relabel_flags_parse_with_the_stage_script() -> None:
    command = compose(relabel_cmd, **_RELABEL_ARGS)
    argv = command.split()[command.split().index(_RELABEL_SCRIPT) + 1:]
    try:
        args = _relabel.build_parser().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"relabel script parser rejected a composer-emitted flag: argv={argv} (exit {exc.code})")
    assert args.convention == "boozer_volume"
    assert args.rho_edge == 0.7
    assert args.dry_run is False


# Rewriting one HDF5 dataset needs no GPU, so the Snakefile hands this composer the slot-free prefix. Nothing in the
# composer itself may add a device flag back.
def test_relabel_command_carries_no_gpu_flag() -> None:
    out = compose(relabel_cmd, **_RELABEL_ARGS)
    assert "--gpus" not in out
    assert "gpu_slots" not in out
