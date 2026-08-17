# Running on an HTCondor cluster

This directory contains the files used to run the pipeline on an HTCondor cluster, via [`snakemake-executor-plugin-htcondor`](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/htcondor.html). It was developed and verified on CHTC, so the profile and the example paths below are CHTC-shaped; adapt them for another pool.

A staging directory for inputs and outputs is required so the execute nodes can read/write inputs/outputs without going through the submit node. The commands below reach it through the shell variable `$RUN_ROOT`, which is set once at the start of the Run section.

The demo below runs the `quick_run` example. For real runs, substitute the relevant config and inputs.

## Install

The executor plugin is declared in the root `pixi.toml`, so installing the orchestration environment is all that is needed:

```
pixi install --environment pipeline
```

## Build the parent runtime image

From the **repository root** (the definition file copies `./pixi.toml` and `./pixi.lock`, which is where they live):

```
apptainer build htcondor-runtime.sif executors/htcondor/apptainer.def
```

The `.sif` must sit in the repository root, which is what the profile's `container_image` resolves against. Rebuild it whenever the root `pixi.lock` changes the `htcondor-runtime` environment.

## Run

A staging run needs **absolute** `input_dir` and `output_dir`. A relative value, which is what the committed `inputs/quick_run/config.yaml` ships, resolves against the checkout on the submit host, so pointing a run at a config file that itself sits on the staging filesystem (e.g. `/staging` on CHTC) still reads its inputs from, and writes its outputs to, the local repository. The closed-loop driver has no flag for either, so its config file has to carry them; the forward pass can override both on the command line.

First point `RUN_ROOT` at your own staging run directory, replacing `my_run` with the name you gave it. Every command below expands it, so set it once per shell (the group path is an example, from CHTC):

```
export RUN_ROOT=/staging/groups/driftless_star/my_run
```

Then, from the repository root, run the closed-loop driver to iterate toward transport-consistent profiles. It forwards the profile to every iteration's forward pass. Its `--config` names the run config file rather than `key=value` overrides, so edit that file's `input_dir` and `output_dir` to the absolute staging paths before running:

```
pixi run driftless-star \
    --config $RUN_ROOT/inputs/quick_run/config.yaml \
    --profile executors/htcondor/profiles/htcondor-gpu \
    --container-runtime apptainer --gpu-ids all \
    --max-iters 3 --cores 8
```

To run a single forward pass with the same profile. It takes `input_dir` and `output_dir` as `--config` overrides, so the committed config file can be left as it is:

```
pixi run driftless-star-fwd \
    --profile executors/htcondor/profiles/htcondor-gpu \
    --configfile $RUN_ROOT/inputs/quick_run/config.yaml \
    --config \
        gpu_ids=all container_runtime=apptainer \
        input_dir=$RUN_ROOT/inputs/quick_run \
        output_dir=$RUN_ROOT/outputs/quick_run \
    --cores 8
```

`--container-runtime apptainer` is not optional on a cluster: the execute nodes have no Docker daemon, and the committed configs default to `docker`.

### The W7-X runs

`inputs/w7-x_quick_run/` and `inputs/w7-x_t3d_validation/` use the same HTCondor profile as `quick_run`. Their Stage 3 SFINCS namelist is not committed. Put `sfincs_input.w7x_t3d_reconstruction` in the selected staging input directory before you submit the run. Otherwise, `stage3_prepare` fails.

Edit the selected staging `config.yaml` so that `input_dir` and `output_dir` are absolute paths. Then start the closed-loop validation run from the repository root:

```
pixi run driftless-star \
    --config $RUN_ROOT/inputs/w7-x_t3d_validation/config.yaml \
    --profile executors/htcondor/profiles/htcondor-gpu \
    --container-runtime apptainer --gpu-ids all \
    --max-iters 10 --cores 8
```

Use `--gpu-ids all` for both W7-X configurations. The quick configuration already uses `all`, but the validation configuration lists local device IDs. Those IDs do not identify HTCondor execute nodes. In Apptainer mode, this option selects the GPU images and records the effective cluster configuration.

The W7-X runs queue more work than `quick_run`. The W7-X quick run has 9 transport cells and 10 faces. Stages 3 and 4 scan its 9 non-axis faces. This produces 9 SFINCS jobs and 18 GKX jobs in iteration 1. The validation run has 8 cells and 9 faces. It scans 8 non-axis faces and produces 8 SFINCS jobs and 16 GKX jobs. Its 8 SFINCS jobs run in iteration 1 only, because its run config freezes Stages 1 through 3 after the first iteration. Later iterations queue Stages 4 and 5 alone. The quick run instead reruns every stage each iteration. Each GKX count includes one base run and one temperature-gradient perturbation for each face. The profile permits 8 concurrent jobs, so it submits these jobs in multiple waves.

### Recovering a stuck run

If a `LockException` is raised, it is usually because a previous run did not finish. To unlock:

```
pixi run driftless-star-fwd --unlock \
    --profile executors/htcondor/profiles/htcondor-gpu \
    --configfile $RUN_ROOT/inputs/quick_run/config.yaml \
    --config \
        gpu_ids=all container_runtime=apptainer \
        input_dir=$RUN_ROOT/inputs/quick_run \
        output_dir=$RUN_ROOT/outputs/quick_run \
    --cores 8
```

Then rerun the previous command. To recover the incomplete run rather than redo it, also pass `--rerun-incomplete`.

### Where the logs go

The executor writes HTCondor's own log, stdout, and stderr under the profile's `htcondor-jobdir`, which the GPU profile sets to `jobs/`:

- `jobs/snakemake-rules.log`: the unified HTCondor event log for every job
- `jobs/<rule>/<rule>-<jobid>_<ClusterId>.out` / `.err`: per-job output

Those paths are chosen by the plugin and cannot be redirected through `default-resources`. The closed-loop driver's `--htcondor-jobdir` replaces `jobs/` for one run, and every iteration of that run is given the same directory. Two controllers sharing a checkout need distinct values: the event log is one file per directory, so otherwise both append to the same `jobs/snakemake-rules.log` and each reads the other's job events. Point it at a path on the submit host, which is where HTCondor writes the event log and where it returns each job's `.out` and `.err`.

Each job's `.err` opens with `[job_wrapper]` diagnostics naming the Snakemake it resolved and that Snakemake's version, which is the first thing to read when jobs fail immediately.

## Further Explanation

The parent image built from `apptainer.def` exists only to give remote HTCondor jobs a stable runtime. Inside it, the workflow launches the stage-specific images for VMEC, BOOZ_XFORM, SFINCS, GKX, and NEOPAX.

So the layering is:

1. HTCondor launches the parent image `htcondor-runtime.sif`.
2. Inside that parent image, Snakemake executes one rule.
3. That rule launches the stage container command.

Because the executor does not ship Snakemake to the node (it is baked into the parent image), the submit host and the image must run the **same** Snakemake version. The executor formats each remote command line using the submit host's own flag vocabulary, and a mismatch surfaces on the node as an unhelpful `unrecognized arguments` error. `pixi.toml` pins the two together and `tests/orchestration/test_pixi_envs.py` fails the suite if they drift apart.

### What each file does

- `../../htcondor-runtime.sif`
  The built parent runtime image, consumed by HTCondor `universe=container` jobs. Must be in the repository root. Gitignored.

- `apptainer.def`
  Builds `htcondor-runtime.sif` from the root workspace's `htcondor-runtime` environment. That environment carries Snakemake and nothing else: the executor rewrites every remote job into a plain `snakemake --mode remote ...` with no `--executor` flag, so the plugin itself is only ever needed on the submit host. The image additionally installs `apptainer` and `git` globally, since it launches the nested stage containers.

- `profiles/htcondor-gpu/`
  An example profile, as used on CHTC: `universe=container` with the parent image, one GPU per job, and `/staging` treated as shared rather than transferred. This is the one the commands above use; another pool needs its own `requirements` ClassAd and shared-FS prefixes.

- `job_wrapper.sh`
  Used by the GPU profile as the job executable. The executor hands it the Snakemake arguments with the `python -m snakemake` prefix stripped, and the wrapper runs them with the parent image's one Snakemake, `/app/.pixi/envs/htcondor-runtime/bin/snakemake`. There is no `PATH` fallback: if that path is missing, it dumps diagnostics to stderr and exits non-zero.
