"""Test profile beta and collision calculations and their use during preparation."""
from __future__ import annotations

import csv
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset
from scipy.constants import elementary_charge, proton_mass

from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution, write_wout

scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")
parameters = load_stage_module("stages/stage4-turbulence/profile_parameters.py")
fixtures = load_stage_module("tests/stage4-turbulence/test_gkx_scan_helpers.py")


@pytest.mark.parametrize("density,temperature,mass,charge,logarithm,rate", [
    (0.7, 2.3, 2.0, 1.0, 18.324664833606935, 741.0921074592969),
    (0.9, 1.7, 0.000544617, -1.0, 15.372964459356517, 76230.01649587744),
    (0.02, 3.2, 12.0, 6.0, 15.222422986973342, 5670.848594636906),
])
def test_self_collision_reference_values(density, temperature, mass, charge, logarithm, rate):
    # Reference cases evaluated separately from the helpers using the pinned T3D expressions.
    actual_rate, actual_log = parameters.self_collision(density, temperature, mass, charge, context="case")
    assert actual_log == pytest.approx(logarithm, rel=1e-13)
    assert actual_rate == pytest.approx(rate, rel=1e-13)


def test_ion_collision_scalings_include_coulomb_logarithm():
    rate, log = parameters.self_collision(0.7, 2.3, 2, 1, context="D")
    for n, t, mass, charge, expected_log, prefactor in [
        (2.8, 2.3, 2, 1, log - math.log(2), 4),
        (0.7, 9.2, 2, 1, log + 1.5 * math.log(4), 1 / 8),
        (0.7, 2.3, 8, 1, log, 1 / 2),
        (0.7, 2.3, 2, 2, log - 3 * math.log(2), 16),
    ]:
        actual, actual_log = parameters.self_collision(n, t, mass, charge, context="ion")
        assert actual_log == pytest.approx(expected_log)
        assert actual / rate == pytest.approx(prefactor * expected_log / log)


def test_beta_and_speed_scales():
    beta = parameters.reference_beta(0.7, 2.3, 2.5, "D")
    assert beta == pytest.approx(0.01038128)
    assert parameters.reference_beta(1.4, 2.3, 2.5, "D") == pytest.approx(2 * beta)
    assert parameters.reference_beta(0.7, 4.6, 2.5, "D") == pytest.approx(2 * beta)
    assert parameters.reference_beta(0.7, 2.3, 5, "D") == pytest.approx(beta / 4)
    speed = parameters.reference_speed(2.3, "D")
    assert speed == pytest.approx(math.sqrt(2300 * elementary_charge / proton_mass))
    assert parameters.reference_speed(9.2, "D") == pytest.approx(2 * speed)


def test_geometry_flux_sign_and_minimal_collision_geometry(tmp_path):
    path = write_wout(tmp_path / "wout.nc")
    positive_scales = parameters.geometry_scales(path, need_field=True)
    with Dataset(path, "a") as data:
        data["phi"][:] *= -1
    assert parameters.geometry_scales(path, need_field=True) == positive_scales
    assert positive_scales == pytest.approx((0.3, 1 / (math.pi * 0.3**2)))
    no_field = write_wout(tmp_path / "no_field.nc", omit=["phi"])
    assert parameters.geometry_scales(no_field, need_field=False) == (0.3, None)
    with pytest.raises(ValueError, match="require phi"):
        parameters.geometry_scales(no_field, need_field=True)


@pytest.mark.parametrize("name,value", [
    ("density", -1), ("density", float("nan")), ("density", float("inf")),
    ("temperature", 0), ("temperature", -1), ("temperature", float("nan")),
    ("mass", 0), ("mass", float("inf")), ("charge", 0), ("charge", float("nan")),
])
def test_invalid_collision_inputs_identify_species_and_radius(name, value):
    inputs = dict(density=1, temperature=1, mass=2, charge=1)
    inputs[name] = value
    with pytest.raises(ValueError, match="species D at rho 0.4"):
        parameters.self_collision(**inputs, context="species D at rho 0.4")


def test_zero_density_trace_and_invalid_coulomb_log():
    assert parameters.self_collision(0, 1, 12, 6, context="trace") == (0, None)
    with pytest.raises(ValueError, match="Coulomb logarithm"):
        parameters.self_collision(1, 1e-12, 12, 6, context="trace")
    with pytest.raises(ValueError, match="reference density"):
        parameters.reference_beta(0, 1, 1, "species D at rho 0.2")


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_scaling_factor(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        parameters.scaling_factor(value)


def prepare(tmp_path, *extra, config_text=None, template_text="", source="prescribed", infer_rho_star=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "common.toml"
    config.write_text(config_text or fixtures.PRESCRIBED_TOML)
    template = tmp_path / "template.toml"
    template.write_text(template_text)
    out = tmp_path / "out"
    vmec = write_wout(tmp_path / "wout.nc")
    args = scan.build_parser().parse_args([
        "prepare", "--common-config", str(config), "--output-dir", str(out),
        "--profiles-source", source, "--vmec-file-override", str(vmec),
        "--gkx-template", str(template), "--electron-model", "kinetic",
        *([] if infer_rho_star else ["--rho-star-physical", "0.01"]), *extra,
    ])
    assert scan.cmd_prepare(args) == 0
    return json.loads((out / "manifest.json").read_text())


@pytest.mark.parametrize("beta_source", ["fixed", "profiles"])
@pytest.mark.parametrize("nu_source", ["fixed", "profiles"])
def test_sources_toml_audit_and_face_values(tmp_path, beta_source, nu_source):
    manifest = prepare(tmp_path, "--beta-source", beta_source,
        "--collisionality-source", nu_source, "--beta", "0.123",
        "--nu-ion", "0.456", "--nu-electron", "0.789")
    audit = json.loads((tmp_path / "out/normalization_audit.json").read_text())["rows"]
    with (tmp_path / "out/normalization_audit.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(audit) == len(csv_rows) == len(manifest["runs"]) * 3
    source = tomllib.loads(fixtures.PRESCRIBED_TOML)["profiles"]
    for run in manifest["runs"]:
        face = run["rho_index"]
        nr = source["density_face"][1][face] / 1e20
        tr = source["temperature_face"][1][face] / 1000
        field = 1 / (math.pi * 0.3**2)
        expected_beta = 0.0403 * nr * tr / field**2 if beta_source == "profiles" else 0.123
        assert run["beta"] == pytest.approx(expected_beta)
        runtime = tomllib.loads(Path(run["config_path"]).read_text())
        assert runtime["physics"]["beta"] == pytest.approx(expected_beta)
        assert runtime["physics"]["use_apar"] is False
        assert runtime["physics"]["use_bpar"] is False
        for index, sp in enumerate(run["runtime_species"]):
            row = next(row for row in audit if row["run_index"] == run["index"] and row["species_name"] == sp["name"])
            assert sp["density_physical"] == pytest.approx(source["density_face"][index][face] / 1e20)
            assert sp["temperature_physical"] == pytest.approx(source["temperature_face"][index][face] / 1000)
            if nu_source == "profiles":
                rate, log = parameters.self_collision(sp["density_physical"], sp["temperature_physical"], sp["mass"], sp["charge"], context="expected")
                expected_nu = rate * 0.3 / math.sqrt(1000 * elementary_charge * tr / proton_mass)
                assert row["self_collision_rate_s"] == pytest.approx(rate)
                assert row["coulomb_logarithm"] == pytest.approx(log)
                assert row["reference_speed_ms"] == pytest.approx(math.sqrt(1000 * elementary_charge * tr / proton_mass))
            else:
                expected_nu = 0.789 if index == 0 else 0.456
                assert row["self_collision_rate_s"] is None
            assert sp["nu"] == pytest.approx(expected_nu)
            assert runtime["species"][index]["nu"] == pytest.approx(expected_nu)
            assert row["beta_source"] == beta_source
            assert row["collisionality_source"] == nu_source
            assert row["beta_configured"] == pytest.approx(expected_beta)
            assert row["beta_effective"] == 0.0
            assert row["nu_used"] == pytest.approx(expected_nu)
            assert row["fixed_beta"] == 0.123
            assert row["fixed_nu"] == (0.789 if index == 0 else 0.456)
            csv_row = next(r for r in csv_rows if int(r["run_index"]) == run["index"] and r["species_name"] == sp["name"])
            assert float(csv_row["nu_used"]) == pytest.approx(expected_nu)


def test_scaling_and_gradient_perturbations(tmp_path):
    manifest = prepare(tmp_path, "--beta-source", "profiles", "--collisionality-source", "profiles",
        "--collisionality-scaling-factor", "0", "--response-mode", "fd_gradients",
        "--perturb-density-species", "D", "--perturb-temperature-species", "e", "--perturb-rel-step", "0.1")
    assert any(run["response_label"] != "base" for run in manifest["runs"])
    for run in manifest["runs"]:
        base = next(r for r in manifest["runs"] if r["rho_index"] == run["rho_index"] and r["response_label"] == "base")
        assert run["beta"] == base["beta"]
        for sp, base_sp in zip(run["runtime_species"], base["runtime_species"]):
            assert sp["nu"] == base_sp["nu"] == 0
            assert sp["self_collision_rate_s"] == base_sp["self_collision_rate_s"] > 0


@pytest.mark.parametrize("extra,expected", [([], 0.08), (["--beta", "0.12"], 0.12)])
def test_fixed_beta_precedence_and_explicit_fields(tmp_path, extra, expected):
    manifest = prepare(tmp_path, *extra, template_text="""[physics]
beta = 0.08
electromagnetic = true
electrostatic = false
use_apar = true
use_bpar = true
collisions = false
hypercollisions = true
[terms]
apar = 0.5
bpar = 0.7
""")
    runtime = tomllib.loads(Path(manifest["runs"][0]["config_path"]).read_text())
    assert runtime["physics"]["beta"] == expected
    for flag in ("electromagnetic", "use_apar", "use_bpar", "hypercollisions"):
        assert runtime["physics"][flag] is True
    assert runtime["physics"]["collisions"] is False
    assert runtime["terms"]["apar"] == 0.5
    assert runtime["terms"]["bpar"] == 0.7


def test_default_sources_and_collision_values(tmp_path):
    manifest = prepare(tmp_path)
    for run in manifest["runs"]:
        assert run["beta"] == 0
        assert run["parameter_audit"]["beta_source"] == "fixed"
        assert run["parameter_audit"]["collisionality_source"] == "fixed"
        assert [sp["nu"] for sp in run["runtime_species"]] == [0, 0.01, 0.01]


def transport_history(path):
    profiles = tomllib.loads(fixtures.PRESCRIBED_TOML)["profiles"]
    def history(key, scale):
        values = np.asarray(profiles[key]) / scale
        return np.stack([values, 2 * values])
    faces = np.linspace(0, 1, 6)
    return write_transport_solution(path,
        rho=(faces[:-1] + faces[1:]) / 2, rho_face=faces, r_grid_half=faces * 0.3,
        density=history("density", 1e20), temperature=history("temperature", 1000),
        er=history("Er", 1000), density_face=history("density_face", 1e20),
        temperature_face=history("temperature_face", 1000), er_face=history("Er_face", 1000),
        density_grad_face=history("density_grad_face", 1e20 * 0.3),
        temperature_grad_face=history("temperature_grad_face", 1000 * 0.3),
        final_time=1, next_dt=0.1)


def assert_parameters_equal(left, right):
    for a, b in zip(left["runs"], right["runs"], strict=True):
        assert a["rho"] == b["rho"]
        assert a["beta"] == pytest.approx(b["beta"])
        for spa, spb in zip(a["runtime_species"], b["runtime_species"], strict=True):
            for key in ("nu", "density_physical", "temperature_physical", "fprim", "tprim"):
                assert spa[key] == pytest.approx(spb[key])


def test_transport_time_selection_and_prescribed_equivalence(tmp_path):
    history = transport_history(tmp_path / "transport.h5")
    options = ["--beta-source", "profiles", "--collisionality-source", "profiles"]
    prescribed = prepare(tmp_path / "prescribed", *options)
    first = prepare(tmp_path / "first", *options, "--neopax-result", str(history), "--time-index", "0", source="transport_h5")
    last = prepare(tmp_path / "last", *options, "--neopax-result", str(history), "--time-index", "-1", source="transport_h5")
    assert_parameters_equal(first, prescribed)
    for a, b in zip(first["runs"], last["runs"]):
        assert b["beta"] == pytest.approx(4 * a["beta"])
        assert b["runtime_species"][1]["nu"] != pytest.approx(a["runtime_species"][1]["nu"])


def test_stage5_feedback_changes_parameters_and_matches_final_transport(tmp_path, monkeypatch):
    writer = load_stage_module("stages/stage5-post-processing/write_prescribed_profiles_from_transport_h5.py")
    history = transport_history(tmp_path / "transport.h5")
    template = tmp_path / "template.toml"
    template.write_text(fixtures.PRESCRIBED_TOML + "\n[transport_solver]\nt0 = 0.0\ndt = 0.01\nt_final = 10.0\n")
    feedback = tmp_path / "feedback.toml"
    monkeypatch.setattr("sys.argv", ["feedback", str(history), str(template), "--output-toml", str(feedback)])
    writer.main()
    options = ["--beta-source", "profiles", "--collisionality-source", "profiles"]
    initial = prepare(tmp_path / "initial", *options)
    fed = prepare(tmp_path / "fed", *options, config_text=feedback.read_text())
    final_transport = prepare(tmp_path / "final", *options, "--neopax-result", str(history), source="transport_h5")
    assert_parameters_equal(fed, final_transport)
    assert fed["runs"][0]["beta"] != initial["runs"][0]["beta"]
    assert fed["runs"][0]["runtime_species"][1]["nu"] != initial["runs"][0]["runtime_species"][1]["nu"]


@pytest.mark.parametrize("reducer", ["window_mean", "t3d_median"])
def test_full_scan_forwards_sources_scaling_and_reducer(tmp_path, monkeypatch, reducer):
    config = tmp_path / "common.toml"
    config.write_text(fixtures.PRESCRIBED_TOML)
    template = tmp_path / "gkx.toml"
    template.write_text("")
    vmec = write_wout(tmp_path / "wout.nc")
    monkeypatch.setattr(scan, "cmd_run", lambda args: 0)

    def collect(args):
        assert args.average_reducer == reducer
        return 0

    monkeypatch.setattr(scan, "cmd_collect", collect)
    args = scan.build_parser().parse_args([
        "--common-config", str(config), "--output-dir", str(tmp_path / "out"),
        "--profiles-source", "prescribed", "--vmec-file-override", str(vmec),
        "--gkx-template", str(template), "--rho-star-physical", "0.01",
        "--beta-source", "profiles", "--collisionality-source", "profiles",
        "--collisionality-scaling-factor", "2.5",
        "--average-reducer", reducer,
    ])
    assert scan.cmd_all(args) == 0
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    for run in manifest["runs"]:
        audit = run["parameter_audit"]
        assert audit["beta_source"] == audit["collisionality_source"] == "profiles"
        assert audit["collisionality_scaling_factor"] == 2.5
        for sp in run["runtime_species"]:
            assert sp["nu"] == pytest.approx(sp["self_collision_rate_s"] * audit["reference_length_m"] / audit["reference_speed_ms"] * 2.5)


def test_adiabatic_electrons_remain_filtered(tmp_path):
    manifest = prepare(tmp_path, "--electron-model", "adiabatic", "--collisionality-source", "profiles")
    assert all([sp["name"] for sp in run["runtime_species"]] == ["D", "T"] for run in manifest["runs"])


@pytest.mark.parametrize("option", ["--beta-source", "--collisionality-source"])
def test_profile_parameters_reject_non_vmec_geometry(tmp_path, option):
    with pytest.raises(ValueError, match="require VMEC geometry"):
        prepare(tmp_path, option, "profiles", template_text='[geometry]\nmodel = "miller"\n')


@pytest.mark.parametrize("field,bad_value", [("density_face", -1), ("temperature_face", 0)])
def test_raw_reference_values_are_not_replaced_with_floors(tmp_path, field, bad_value):
    cfg = tomllib.loads(fixtures.PRESCRIBED_TOML)
    # Only this reference-ion face differs from the positive prescribed fixture.
    cfg["profiles"][field][1][1] = bad_value
    lines = fixtures.PRESCRIBED_TOML.splitlines()
    text = "\n".join(f"{field} = {json.dumps(cfg['profiles'][field])}" if line.startswith(field + " =") else line for line in lines)
    with pytest.raises(ValueError, match="species D at rho 0.2"):
        prepare(tmp_path, "--beta-source", "profiles", "--collisionality-source", "profiles",
            "--density-floor", "1", "--temperature-floor", "1", config_text=text)


def test_analytical_first_run_matches_equivalent_prescribed_and_transport_faces(tmp_path):
    from tests.helpers.synthetic import write_boozmn

    config_text = (Path(__file__).resolve().parents[2] / "inputs/quick_run/common_input.toml").read_text()
    vmec = write_wout(tmp_path / "analytical_wout.nc")
    booz = write_boozmn(tmp_path / "booz.nc")
    with Dataset(vmec, "a") as data:
        data.createVariable("volume_p", "f8")[...] = 2 * math.pi**2 * 3 * 0.4**2
    with Dataset(booz, "a") as data:
        data.createVariable("rmnc_b", "f8", ("pack_rad", "mn_mode"))[:] = 3
    options = ["--beta-source", "profiles", "--collisionality-source", "profiles"]
    analytical = prepare(tmp_path / "analytical", *options,
        "--vmec-file-override", str(vmec), "--boozer-file-override", str(booz),
        config_text=config_text, source="analytical")
    cfg = tomllib.loads(config_text)
    state = scan.build_analytical_face_state(cfg, species=scan._parse_species_from_common_config(cfg), n_faces=6, minor_radius=0.4)
    # Prescribed cell values can differ because preparation uses the supplied face values.
    prescribed = tomllib.loads(fixtures.PRESCRIBED_TOML)["profiles"]
    for key, value, scale in [
        ("density_face", state.density, 1e20), ("temperature_face", state.temperature, 1000),
        ("density_grad_face", state.density_grad, 1e20), ("temperature_grad_face", state.temperature_grad, 1000),
        ("Er_face", state.er, 1000),
    ]:
        prescribed[key] = (value * scale).tolist()
    prefix = fixtures.PRESCRIBED_TOML.split("[profiles]")[0]
    text = prefix + "[profiles]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key, value in prescribed.items())
    from_prescribed = prepare(tmp_path / "prescribed_equivalent", *options, config_text=text)
    faces = np.linspace(0, 1, 6)
    path = write_transport_solution(tmp_path / "analytical_state.h5", rho=(faces[:-1] + faces[1:]) / 2,
        rho_face=faces, r_grid_half=faces * 0.4, density=np.ones((3, 5)), temperature=np.ones((3, 5)), er=np.zeros(5),
        density_face=state.density, temperature_face=state.temperature, er_face=state.er,
        density_grad_face=state.density_grad / 0.4, temperature_grad_face=state.temperature_grad / 0.4,
        final_time=1, next_dt=0.1)
    from_transport = prepare(tmp_path / "transport_equivalent", *options, "--neopax-result", str(path), source="transport_h5")
    assert_parameters_equal(analytical, from_prescribed)
    assert_parameters_equal(analytical, from_transport)


def test_zero_density_trace_is_audited_without_a_coulomb_logarithm(tmp_path):
    profiles = tomllib.loads(fixtures.PRESCRIBED_TOML)["profiles"]
    profiles["density_face"][2] = [0.0] * 6
    profiles["density_grad_face"][2] = [0.0] * 6
    text = "\n".join(f"{key} = {json.dumps(profiles[key])}" if (key := line.split(" =")[0]) in ("density_face", "density_grad_face") else line for line in fixtures.PRESCRIBED_TOML.splitlines())
    manifest = prepare(tmp_path, "--collisionality-source", "profiles", config_text=text)
    audit = json.loads((tmp_path / "out/normalization_audit.json").read_text())["rows"]
    for run in manifest["runs"]:
        trace = next(sp for sp in run["runtime_species"] if sp["name"] == "T")
        assert trace["density_physical"] == trace["nu"] == trace["self_collision_rate_s"] == 0
        assert trace["coulomb_logarithm"] is None
        row = next(row for row in audit if row["run_index"] == run["index"] and row["species_name"] == "T")
        assert row["coulomb_logarithm"] is None
        assert row["nu_used"] == 0


@pytest.mark.parametrize("beta_source,nu_source", [("profiles", "fixed"), ("fixed", "profiles"), ("profiles", "profiles")])
def test_active_floors_do_not_change_normalization_when_sources_switch(tmp_path, beta_source, nu_source):
    options = ["--density-floor", "1", "--temperature-floor", "2", "--collisionality-scaling-factor", "0"]
    fixed = prepare(tmp_path / "fixed", *options, infer_rho_star=True)
    enabled = prepare(tmp_path / "enabled", *options, "--beta-source", beta_source, "--collisionality-source", nu_source, infer_rho_star=True)
    for old, new in zip(fixed["runs"], enabled["runs"], strict=True):
        for key in ("tau_e", "rho_star_physical", "a_minor"):
            assert new[key] == old[key]
        for a, b in zip(old["runtime_species"], new["runtime_species"], strict=True):
            for key in ("density", "temperature", "tprim", "fprim", "density_reference_physical", "temperature_reference_physical"):
                assert b[key] == a[key]
            for kind in ("Gamma", "Q"):
                def converted(sp):
                    return scan._gkx_flux_to_neopax_units(1.0, density_ref_state=sp["density_reference_physical"],
                        temperature_ref_keV=sp["temperature_reference_physical"], mass_ref_mp=sp["mass"],
                        rho_star_physical=new["rho_star_physical"], kind=kind)
                assert converted(a) == converted(b)
        audit = new["parameter_audit"]
        assert audit["raw_reference_density"] < new["runtime_species"][1]["density_reference_physical"]
        assert audit["raw_reference_temperature_keV"] < new["runtime_species"][1]["temperature_reference_physical"]
        if beta_source == "profiles":
            assert new["beta"] == pytest.approx(0.0403 * audit["raw_reference_density"] * audit["raw_reference_temperature_keV"] / audit["reference_field_T"]**2)
        if nu_source == "profiles":
            assert audit["reference_speed_ms"] == pytest.approx(math.sqrt(1000 * elementary_charge * audit["raw_reference_temperature_keV"] / proton_mass))
            assert all(sp["nu"] == 0 for sp in new["runtime_species"])


@pytest.mark.parametrize("electromagnetic", [False, True])
def test_audit_distinguishes_configured_and_effective_parameters(tmp_path, electromagnetic):
    template = f"""[physics]
electromagnetic = {str(electromagnetic).lower()}
use_apar = true
use_bpar = true
collisions = true
[terms]
apar = 0.5
bpar = 0.25
collisions = 0.75
"""
    manifest = prepare(tmp_path, "--beta-source", "profiles", "--collisionality-source", "profiles",
                       "--collisionality-scaling-factor", "0", template_text=template)
    rows = scan._build_normalization_audit_rows(manifest)
    assert all(row["beta_configured"] > 0 for row in rows)
    for row in rows:
        assert row["beta_effective"] == (row["beta_configured"] if electromagnetic else 0)
        assert row["apar_term_effective"] == (0.5 if electromagnetic else 0)
        assert row["bpar_term_effective"] == (0.25 if electromagnetic else 0)
        assert row["collisions_term_effective"] == 0


def test_invalid_geometry_model_and_missing_required_geometry_are_labelled(tmp_path):
    with pytest.raises(ValueError, match="require VMEC geometry"):
        prepare(tmp_path / "bad_model", "--beta-source", "profiles", template_text="[geometry]\nmodel = 1\n")
    path = write_wout(tmp_path / "missing_radius.nc", omit=["Aminor_p"])
    with pytest.raises(ValueError, match="Aminor_p.*missing_radius.nc"):
        parameters.geometry_scales(path, need_field=False)


def test_scaling_cli_error_explains_allowed_values(capsys):
    with pytest.raises(SystemExit):
        scan.build_parser().parse_args(["prepare", "--collisionality-scaling-factor", "-1"])
    assert "must be finite and nonnegative" in capsys.readouterr().err


@pytest.mark.parametrize("profiles,convention_only,reused", [(False, False, True), (True, False, False), (False, True, True), (True, True, True)])
def test_reprepare_fingerprints_numerical_changes_not_audit_choices(tmp_path, monkeypatch, profiles, convention_only, reused):
    validity = load_stage_module("stages/stage4-turbulence/run_validity.py")
    options = ["--collisionality-source", "profiles" if profiles else "fixed"]
    original = prepare(tmp_path, *options)
    runtime_bytes = [Path(run["config_path"]).read_bytes() for run in original["runs"]]
    for run in original["runs"]:
        token = validity.begin_attempt(original, run)
        validity.diagnostics_path(run).write_text("t,heat_flux,particle_flux\n0,1,2\n1,2,3\n")
        validity.certify_completion(original, run, token)
    if convention_only:
        monkeypatch.setattr(scan, "CONVENTION", "new documentation or pin label")
    else:
        options += ["--collisionality-scaling-factor", "2"]
    updated = prepare(tmp_path, *options)
    for index, (old, new) in enumerate(zip(original["runs"], updated["runs"], strict=True)):
        assert (Path(new["config_path"]).read_bytes() == runtime_bytes[index]) is reused
        assert (old["input_fingerprint"] == new["input_fingerprint"]) is reused
        assert validity.completion_status(updated, new)[0] is reused


@pytest.mark.parametrize("flags,enabled", [([], False), (["--allow-incomplete"], True), (["--collect-even-if-failures"], True),
    (["--allow-incomplete", "--no-collect-even-if-failures"], False)])
def test_incomplete_alias_has_one_parser_destination(flags, enabled):
    args = scan.build_parser().parse_args(flags)
    assert args.allow_incomplete is enabled
    assert not hasattr(args, "collect_even_if_failures")
