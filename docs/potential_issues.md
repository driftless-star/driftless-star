# Potential Issues

## Cross-Stage Infrastructure

- [ ] Use the standard `outputs/` tree as a cache to quickly retrieve results that have already been processed, kind of like a database.

## Documentation

- [ ] Add more details for each software, e.g. for GKX's `omega_t` what are units, what is it normalized to (gyro freq, etc), scale lengths.

## Code Quality / Tooling

- [ ] No type checker is configured, so the type hints being added across the codebase (e.g. `stages/stage4-turbulence/gkx_radial_scan.py`) are unverified. Adding one (e.g. `ty`, mypy, or pyright) to CI and/or pre-commit would catch incorrect annotations. Raised on PR #78; deferred to a future PR.
- [ ] No workflow runs the test suite. `.github/workflows/` holds only `containers.yml`, which builds and pushes the stage images, so nothing runs `pytest`, a linter, or a type checker on a pull request. The only wired entry point is the local `[feature.test.tasks.test]` task in the root `pixi.toml`.
- [ ] Tests that need a stage environment can never run. `tests/common/test_profile_gradients.py::test_matches_neopax_on_random_inputs` compares the ported cell-variable operator against live NEOPAX, but the `stage-5-neopax` environment has NEOPAX and no `pytest` while the root `test` environment has `pytest` and no NEOPAX, and the two pixi workspaces cannot see each other. The test always skips, which reads as a pass. Running it takes a throwaway venv: `stages/.pixi/envs/stage-5-neopax/bin/python -m venv --system-site-packages <dir>`, then `pip install pytest` in it. A `stage-test` feature carrying `pytest` plus a `stage-5-neopax-test` environment would make it a plain command without putting `pytest` in the published image, since `stages/Dockerfile` copies only the selected environment. The ten golden cases in the same module are pinned from NEOPAX and do run everywhere, so the operator is still guarded. This has now cost something concrete: `tests/common/test_neopax_profiles.py::test_short_form_matches_live_neopax` skipped through a green suite while the adapter read the electron charge from the wrong place, because the only thing that could have caught it was running the pinned solver.

- [ ] Four stage scripts fail at `--help` when run from the repository root. `python stages/stage4-turbulence/gkx_radial_scan.py --help` raises `ModuleNotFoundError: No module named 'common'`, and the Stage 3 scan, the Stage 4 relabel step and the Stage 5 feedback writer do the same. The shared modules under `stages/common/` resolve only through an injected `PYTHONPATH`, which `Snakefile`, both pixi scan tasks, the container launchers and `pytest.ini` all set. That is path manipulation by another name and `CLAUDE.md` rules it out under Project Organization. The fix is to make the shared code an installable package with real entry points and drop the `PYTHONPATH` lines, which touches `stages/pixi.toml`, `stages/Dockerfile` and the container CI, so it belongs in its own change.

## Stage 1 -- Equilibrium

- [ ] vmec/vmec_jax and DESC do not have directly compatible inputs; an adapter or input translation layer will be needed to support both implementations behind the same pipeline entry point
- [ ] vmec_jax only consumes a subset of the full VMEC INDATA file; need to document which fields are supported/ignored, or validate inputs to warn when unsupported fields are present
- [ ] DESC can output Boozer coordinates directly, so with the right flag/argument it can handle both Stage 1 and Stage 2; the pipeline should support this shortcut path

## Stage 2 -- Boozer Transform

- [ ] Future boundary condition optimization can be added as additional functions in Stage 2

## Stage 3 -- Neoclassical

- [ ] sfincs/sfincs_jax and NEO_JAX do not have directly compatible inputs; same adapter/translation issue as Stage 1.
- [ ] NEO_JAX is fast, but its output can't be used for future stages. sfincs is slower, but more accurate.
- [ ] NEO_JAX is excluded from the MVP, but should be included in the final pipeline as an optional stage; its effective ripple output is valuable as a figure of merit even though it does not feed later stages

## Stage 4 -- Turbulence

- [ ] GKX/GX, and GENE likely do not have directly compatible inputs; same adapter/translation issue as Stages 1 and 3

## Stage 5 -- Transport

- [ ] A run whose transport window completes in one NEOPAX call stops the loop at iteration 1 with `horizon`. That is the correct reading of `[transport_solver].t_final` as an absolute end time, but such a config exercises none of the feedback path beyond the first pass. `inputs/quick_run` is such a config. Its archived solutions reach `t_final = 1.4e-6` in one call, so the documented `pixi run driftless-star --max-iters 3` example is a one-iteration run. Accepted for now, because the quick run is a smoke config. To exercise the feedback path again, the config must stop each call short of `t_final`, for example with a `stop_after_accepted_steps` cap as the W7-X validation uses. A larger `t_final` does not help, because one call covers it regardless. The W7-X validation is unaffected; it covers about one percent of its `t_final` per call.

- [ ] The `q`/`gamma` flux-versus-gradient scatter panels have not yet drawn from a real artifact. `stages/stage5-post-processing/plot_transport_panels.py`, which sits in this tree as an untracked file, reads `density_grad_faces` / `temperature_grad_faces` and pairs them with `rho_face`. The scatters therefore use the same per-rho `a/L_X` that drove GKX; one multiplication by the minor radius converts the stored per-metre gradients. The pairing guard compares the flux grid against the face grid. But every `transport_solution.h5` currently under `outputs/` predates the NEOPAX revision that exports the gradient datasets, so on the archived artifacts the panels stay omitted with a reason naming the absent face data. A copy of a real artifact, augmented with contract-shaped gradients, demonstrates the wiring. Confirm the `q` panel draws on the first run at the current pin, then drop this entry.

## W&B / Output Tracking

- [ ] Decide whether W&B dashboards are internal (maintainers only) or public-facing.
- [ ] Eventually it'll be a public challenge SDK. For the official submission: we don't need to worry about API keys. 

## Workflow Engine

- [ ] Container registry for external collaborators: where do external contributors host their alternative stage implementations? Maybe their own GHCR or Docker Hub, with the workflow engine pulling from there.
- [ ] How to expose the workflow engine to external users.
- [ ] How to validate that externally submitted containers are not a security threat.
- [ ] A Snakemake profile with `rerun-triggers: mtime`, opted into by exporting `SNAKEMAKE_PROFILE` around a resume command, for restarting a long run against an existing output tree. Snakemake's default rerun triggers count any change to a rule's rendered shell command as staleness, which requeues completed Stage 4 GPU work.
