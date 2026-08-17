"""Dry-run tests of the Snakemake forward-pass DAG.

``snakemake -n`` parses the Snakefile and plans the job graph without running any
container, so it catches wiring and parse-time errors with no Docker.

Stages 3 and 4 fan out one job per flux surface behind ``prepare`` checkpoints, so
a dry-run plans only up to each checkpoint plus the deferred ``collect`` gathers;
the per-surface ``run_one`` layer enters the DAG only after a real ``prepare``
writes its manifest.

Overriding ``output_dir`` to a tmp dir keeps the Snakefile's parse-time
``prepare_neopax_config`` write, and every planned artifact path, out of the repo,
and ``--runtime-source-cache-path`` keeps Snakemake's own runtime source cache under
tmp. This is not full filesystem hermeticity: Snakemake still writes ``.snakemake/``
metadata into the repo root during a dry run, but that directory is gitignored.
``--workflow-profile none`` stops a user-level profile from injecting flags, so the
planned DAG is the same everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from src.ouroboros import _write_loop_overrides
from src.utils import resolve_pipeline_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_RULES = (
    "stage1_vmec",
    "stage2_boozer",
    "stage3_prepare",
    "stage3_collect",
    "stage4_prepare",
    "stage4_collect",
    "stage5_neopax",
)
# The per-surface run_one rules are gated behind the prepare checkpoints, so a plan built
# before any manifest exists must not schedule them.
DEFERRED_RULES = ("stage3_run_one", "stage4_run_one")


def _dry_run(
    tmp_path: Path,
    targets: list[str],
    config_overrides: list[str],
    extra_configfiles: list[str] | None = None,
    printshellcmds: bool = False,
    configfile: str = "inputs/quick_run/config.yaml",
) -> subprocess.CompletedProcess:
    """Plan a run with ``snakemake -n`` and redirect writes to ``tmp_path``.

    ``extra_configfiles`` are appended after the base config file under the same single
    ``--configfile`` flag, exactly as the loop driver passes its overrides file, and
    ``printshellcmds`` adds ``-p`` so planned shell commands can be asserted on.
    """
    return subprocess.run(
        ["snakemake", "-n", *(["-p"] if printshellcmds else []), *targets,
         "--configfile", configfile, *(extra_configfiles or []),
         "--workflow-profile", "none",
         "--runtime-source-cache-path", f"{tmp_path}/srccache",
         "--config", f"output_dir={tmp_path}/out", *config_overrides],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _write_convergence_override(tmp_path: Path, method: str) -> str:
    """Write a layered run config that changes only the convergence method."""
    path = tmp_path / f"convergence-{method}.yaml"
    path.write_text(yaml.safe_dump({"convergence": {"method": method}}))
    return str(path)


# This runs the default forward pass and asserts it plans successfully (exit code 0), that every stage rule visible
# before the checkpoints run is scheduled (including both deferred collect gathers), and that the post-processing rule
# is NOT scheduled, because the default target is a pure forward pass with no loop-closing step. This catches Snakefile
# wiring/parse errors without Docker.
def test_forward_pass_dag_dry_run(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for rule in FORWARD_RULES:
        assert rule in output, f"rule {rule} not scheduled:\n{output}"
    assert "stage5_post_processing" not in output  # rule all is a pure forward pass
    # Dry-run visibility ends at the unexecuted prepare checkpoints: the per-surface jobs
    # must be absent and Snakemake must announce that the DAG grows after the checkpoints.
    for rule in DEFERRED_RULES:
        assert rule not in output, f"per-surface rule {rule} planned before its checkpoint ran:\n{output}"
    assert "checkpoint jobs" in output, output


# When you explicitly ask Snakemake to build the convergence-signal file (the loop-closing target), the post-processing
# rule SHOULD be pulled in. This resolves that target path, dry-runs with it as the goal, and asserts
# `stage5_post_processing` now appears in the plan, confirming the loop-closing rule wires up on demand.
def test_s5_signal_target_schedules_post_processing(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    s5_signal = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["s5_signal"]
    result = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    # Requesting the convergence signal pulls in the loop-closing rule and formats its
    # shell string, which rule all (a pure forward pass) never reaches.
    assert "stage5_post_processing" in output, output


def test_convergence_method_reaches_the_post_processing_command(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    s5_signal = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["s5_signal"]

    baseline = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[], printshellcmds=True)
    baseline_output = baseline.stdout + baseline.stderr
    assert baseline.returncode == 0, baseline_output
    assert "--pressure-convergence-method pointwise" in baseline_output, baseline_output

    rms = _dry_run(
        tmp_path,
        targets=[s5_signal],
        config_overrides=[],
        extra_configfiles=[_write_convergence_override(tmp_path, "rms")],
        printshellcmds=True,
    )
    rms_output = rms.stdout + rms.stderr
    assert rms.returncode == 0, rms_output
    assert "--pressure-convergence-method rms" in rms_output, rms_output


def test_unknown_convergence_method_fails_at_parse_time(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        targets=[],
        config_overrides=[],
        extra_configfiles=[_write_convergence_override(tmp_path, "maximum")],
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "config['convergence']['method']" in output, output
    assert "'rms', 'pointwise'" in output, output


# From iteration 2 the loop driver appends its loop_overrides.yaml after the base config file, switching both stage
# prepares to the prescribed profiles carried by the seeded common_input.toml. This plans the loop-closing target with
# and without that overrides file (written by the real driver helper) and asserts the printed shell commands flip from
# analytical to prescribed, and that the post-processing rule always invokes the prescribed-profiles writer.
def test_loop_overrides_switch_stage_prepares_to_prescribed(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    s5_signal = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["s5_signal"]

    baseline = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[], printshellcmds=True)
    output = baseline.stdout + baseline.stderr
    assert baseline.returncode == 0, output
    assert output.count("--profiles-source analytical") == 2, output
    assert "write_prescribed_profiles_from_transport_h5.py" in output, output
    assert "--output-toml" in output, output

    overrides = _write_loop_overrides(tmp_path)
    result = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[],
                      extra_configfiles=[str(overrides)], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("--profiles-source prescribed") == 2, output
    assert "--profiles-source analytical" not in output, output


# The Snakefile validates the GPU pool config at parse time, before any job runs. The retired `device` key must be
# rejected (non-zero exit) with a migration hint naming the gpu_ids replacement, confirming an out-of-date config is
# caught early rather than silently planning a CPU run.
def test_device_key_fails_at_parse_with_migration_hint(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["device=gpu"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "gpu_ids" in output, output


def test_apptainer_runtime_plans_native_gpu_image_and_absolute_binds(tmp_path: Path) -> None:
    """The CHTC runtime must not leave a nested ``docker run`` in planned jobs."""
    result = _dry_run(
        tmp_path,
        targets=[],
        config_overrides=["container_runtime=apptainer", "gpu_ids=all"],
        printshellcmds=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "apptainer run --unsquash --nv" in output, output
    assert 'oras://ghcr.io/driftless-star/driftless-star:apptainer-stage-1-vmec-gpu' in output, output
    assert f'--bind "{tmp_path}/out:{tmp_path}/out"' in output, output
    assert "docker run" not in output, output


def test_unknown_container_runtime_fails_at_parse(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["container_runtime=singularity"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "container_runtime" in output, output


# A perturbed fd_gradients sibling run-directory must be schedulable by stage4_run_one exactly like a baseline
# surface: its basename satisfies the shared SURF_PATTERN wildcard constraint, so asking Snakemake to build that
# run.diagnostics.csv plans the per-surface rule. A sibling whose channel letter falls outside n/t violates the
# constraint, so no rule can produce it and the DAG build aborts at parse time.
def test_perturbed_surface_target_matches_run_one(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    stage4_dir = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["stage4_dir"]
    good = f"{stage4_dir}/runs/rho_003_r0p2500_fd_n_D/run.diagnostics.csv"
    result = _dry_run(tmp_path, targets=[good], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "stage4_run_one" in output, output
    # A malformed sibling (fd_x is not a valid density/temperature channel) matches no rule.
    bad = f"{stage4_dir}/runs/rho_003_r0p2500_fd_x_D/run.diagnostics.csv"
    result = _dry_run(tmp_path, targets=[bad], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "No rule to produce" in output, output


# With a real manifest on disk and its upstream equilibrium and Boozer outputs already present, the stage4_prepare
# checkpoint is up to date, so a dry run expands its gather and schedules one stage4_run_one per manifest entry,
# including the perturbed sibling. Every committed file the DAG consumes lives under inputs/quick_run/ and carries its
# checkout mtime, so the dummy upstream outputs and the manifest are stamped strictly newer than all of them, in DAG
# order, keeping every rule up to date however recently the repo was cloned. Only the manifest basenames matter, so
# container-absolute run_dir paths work.
def test_perturbed_manifest_expands_fan_out(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    paths = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")
    newest_input = max(p.stat().st_mtime for p in (REPO_ROOT / paths["input_dir"]).rglob("*") if p.is_file())
    for key in ("s1_output", "s2_output"):
        artifact = Path(paths[key])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("")
        os.utime(artifact, (newest_input + 100, newest_input + 100))
    manifest = Path(paths["stage4_manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_dir": "/container/abs/runs/rho_001_r0p2500"},
                    {"run_dir": "/container/abs/runs/rho_001_r0p2500_fd_n_D"},
                ]
            }
        )
    )
    os.utime(manifest, (newest_input + 200, newest_input + 200))
    result = _dry_run(tmp_path, targets=[paths["s4_output"]], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "stage4_run_one" in output, output
    assert "rho_001_r0p2500_fd_n_D" in output, output


# From iteration 2 with frozen stages, the driver's overrides file names the iteration 1 tree and restates the validated
# rerun map. This freezes Stages 1 and 2 exactly as the driver would (via the real overrides writer), fakes their
# artifacts in the reuse tree, and asserts the plan omits both frozen rules while still scheduling Stages 3/4/5 and
# post-processing, with the composed prepare commands reading the wout and boozmn from the reuse tree rather than the
# current output tree. Omitted rules mean Snakemake treats the reuse-tree files as plain inputs it can never rebuild,
# so iteration 1 records cannot be overwritten by a later pass.
def test_frozen_stages_absent_from_plan(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    reuse = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/reuse")
    for key in ("s1_output", "s2_output"):
        artifact = Path(reuse[key])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("")
    overrides = _write_loop_overrides(
        tmp_path,
        rerun={"stage1": False, "stage2": False, "stage3": True, "stage4": True, "stage5": True},
        reuse_output_dir=reuse["output_dir"],
    )
    s5_signal = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["s5_signal"]
    result = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[],
                      extra_configfiles=[str(overrides)], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for rule in ("stage3_prepare", "stage4_prepare", "stage5_neopax", "stage5_post_processing"):
        assert rule in output, f"rule {rule} not scheduled:\n{output}"
    # The Stage 2 output directory is also named stage2_boozer, and it legitimately appears in the reuse-tree paths
    # of the planned commands, so frozen rules are matched against the job header form instead of the bare name.
    for rule in ("rule stage1_vmec:", "rule stage2_boozer:"):
        assert rule not in output, f"frozen {rule.rstrip(':')} scheduled:\n{output}"
    # The composed commands must consume the frozen artifacts from the reuse tree, not the current output tree.
    assert reuse["s1_output"] in output, output
    assert reuse["s2_output"] in output, output


# A frozen Stage 3 keeps its iteration 1 provenance. Its override omits the Stage 3 profile-source
# switch, and no Stage 3 rule runs. The NEOPAX config must point ``neoclassical_file`` to the reuse
# tree with a relative path. NEOPAX resolves it from the output directory. Omitted rerun keys
# default to true. This case freezes Stages 1, 2 and 3. Stage 3 reads the Boozer transform.
# Therefore, Stage 2 must also remain frozen.
def test_frozen_stage3_keeps_iter1_provenance(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    reuse = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/reuse")
    for key in ("s1_output", "s2_output", "s3_output"):
        artifact = Path(reuse[key])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("")
    overrides = tmp_path / "loop_overrides.yaml"
    overrides.write_text(
        "stage4:\n"
        "  gkx:\n"
        "    profiles_source: prescribed\n"
        "loop:\n"
        "  rerun:\n"
        "    stage1: false\n"
        "    stage2: false\n"
        "    stage3: false\n"
        f"  reuse_output_dir: {reuse['output_dir']}\n"
    )
    paths_out = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")
    result = _dry_run(tmp_path, targets=[paths_out["s5_signal"]], config_overrides=[],
                      extra_configfiles=[str(overrides)], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "stage3_prepare" not in output, output
    # Stage 4 still reruns and takes the frozen boozmn from the reuse tree.
    assert "stage4_prepare" in output, output
    assert reuse["s2_output"] in output, output
    assert output.count("--profiles-source prescribed") == 1, output
    assert "--profiles-source analytical" not in output, output
    resolved = Path(paths_out["s5_resolved_config"]).read_text()
    assert 'neoclassical_file = "../../reuse/stage3_neoclassical/sfincs_jax_flux_profiles.h5"' in resolved, resolved


# The rerun flags are validated at Snakefile parse time even without a reuse tree, so a hand-edited config freezing a
# stage whose input stages still rerun is rejected before any job runs, mirroring the device guard.
def test_invalid_rerun_combo_fails_at_parse(tmp_path: Path) -> None:
    overrides = tmp_path / "loop_overrides.yaml"
    overrides.write_text("loop:\n  rerun:\n    stage2: false\n")
    result = _dry_run(tmp_path, targets=[], config_overrides=[], extra_configfiles=[str(overrides)])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "config['loop']['rerun']['stage2']" in output, output


# Rerun flags without a reuse tree must not change the plan. Only the loop driver provides loop.reuse_output_dir, so a
# base config carrying frozen flags still plans the full forward pass on a plain invocation and on iteration 1.
def test_rerun_flags_without_a_reuse_tree_leave_plan_unchanged(tmp_path: Path) -> None:
    overrides = tmp_path / "loop_overrides.yaml"
    overrides.write_text(
        "loop:\n  rerun:\n    stage1: false\n    stage2: false\n    stage3: false\n    stage4: false\n"
    )
    result = _dry_run(tmp_path, targets=[], config_overrides=[], extra_configfiles=[str(overrides)])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for rule in FORWARD_RULES:
        assert rule in output, f"rule {rule} not scheduled:\n{output}"


# Freezing every stage while naming a reuse tree leaves nothing to build, so the Snakefile rejects it at parse time.
# The driver never emits this shape because an all-frozen config stops after iteration 1 without naming a reuse tree.
def test_reuse_tree_with_all_frozen_fails_at_parse(tmp_path: Path) -> None:
    overrides = tmp_path / "loop_overrides.yaml"
    overrides.write_text(
        "loop:\n"
        "  rerun:\n"
        "    stage1: false\n    stage2: false\n    stage3: false\n    stage4: false\n    stage5: false\n"
        f"  reuse_output_dir: {tmp_path}/reuse\n"
    )
    result = _dry_run(tmp_path, targets=[], config_overrides=[], extra_configfiles=[str(overrides)])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "freezes every stage" in output, output
