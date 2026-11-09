import json
from dataclasses import replace

import numpy as np
import pytest

from crashdamage.obstacles import PRESET_NAMES, default_obstacle, obstacle_from_dict
from crashdamage.physics import Scenario, result_to_dict, scenario_from_dict, simulate
from crashdamage.vehicle import EXTENT, N_PARTS, PART_INDEX, default_vehicle, sample_vehicle
from crashdamage import dataset


def scenario(speed_kmh=50.0, obstacle="concrete_wall", face="front", offset=0.0, angle=0.0, vehicle=None, **ob):
    veh = vehicle or default_vehicle()
    o = default_obstacle(obstacle)
    if ob:
        o = replace(o, **ob)
    return Scenario(veh, o, face=face, offset=offset, angle_deg=angle, speed=speed_kmh / 3.6)


def test_energy_balance_holds_for_random_crashes(rng):
    for _ in range(150):
        res = simulate(dataset.random_scenario(rng))
        total = res.e_car_kj + res.e_obstacle_kj + res.unabsorbed_kj
        assert total == pytest.approx(res.e_diss_kj, rel=1e-6, abs=1e-6)


def test_outputs_are_in_range(rng):
    for _ in range(100):
        sc = dataset.random_scenario(rng)
        res = simulate(sc)
        assert res.damage.shape == (N_PARTS,)
        assert np.all((res.damage >= 0) & (res.damage <= 1))
        assert np.all((res.level >= 0) & (res.level <= 4))
        assert 0.0 <= res.obstacle_damage <= 1.0
        assert 0.0 <= res.crush_depth <= EXTENT[sc.face] + 1e-9
        assert np.isfinite(res.peak_force_kn)


def test_more_speed_means_more_damage_and_deeper_crush():
    speeds = [10, 20, 30, 40, 50, 70, 90]
    results = [simulate(scenario(v)) for v in speeds]
    depth = [r.crush_depth for r in results]
    total = [r.damage.sum() for r in results]
    assert all(b >= a - 1e-9 for a, b in zip(depth, depth[1:]))
    assert all(b >= a - 1e-9 for a, b in zip(total, total[1:]))
    assert depth[-1] > depth[0]


def test_damage_of_every_part_grows_with_speed(rng):
    speeds = np.linspace(4, 36, 9)
    for _ in range(25):
        sc = dataset.random_scenario(rng)
        damage = []
        for v in speeds:
            sc.speed = float(v)
            damage.append(simulate(sc).damage)
        assert np.all(np.diff(np.stack(damage), axis=0) >= -1e-9)


def test_stronger_steel_crushes_less():
    base = default_vehicle()
    parts = []
    for p in base.parts:
        if p.name in ("beam_front", "rail_front_l", "rail_front_r"):
            p = replace(p, material="ultra_high_strength_steel")
        parts.append(p)
    strong = replace(base, parts=parts)
    soft_parts = [replace(p, material="mild_steel") if p.name in ("beam_front", "rail_front_l", "rail_front_r") else p
                  for p in base.parts]
    soft = replace(base, parts=soft_parts)
    assert simulate(scenario(56, vehicle=strong)).crush_depth < simulate(scenario(56, vehicle=soft)).crush_depth


def test_thicker_sheet_crushes_less():
    base = default_vehicle()
    thick = replace(base, parts=[replace(p, thickness_mm=p.thickness_mm * 1.5) for p in base.parts])
    assert simulate(scenario(50, vehicle=thick)).crush_depth < simulate(scenario(50, vehicle=base)).crush_depth


def test_sharp_objects_cut_deeper_than_blunt_ones():
    kw = dict(face="left", width=0.3, perimeter=0.6)
    blunt = simulate(scenario(35, "steel_pole", shape="flat", **kw))
    sharp = simulate(scenario(35, "steel_pole", shape="sharp", **kw))
    assert sharp.crush_depth > blunt.crush_depth
    assert sharp.intrusion > blunt.intrusion


def test_heavier_car_has_more_impact_energy():
    light = simulate(scenario(50, vehicle=default_vehicle(900)))
    heavy = simulate(scenario(50, vehicle=default_vehicle(1800)))
    assert heavy.e_diss_kj > light.e_diss_kj


def test_free_obstacle_takes_less_energy_than_a_fixed_one():
    fixed = simulate(scenario(40, "concrete_block", mass_kg=None))
    free = simulate(scenario(40, "concrete_block", mass_kg=1500.0))
    assert free.e_diss_kj < fixed.e_diss_kj
    assert free.obstacle_speed_after > 0.0
    assert fixed.obstacle_speed_after == 0.0


def test_left_and_right_are_mirror_images():
    left = simulate(scenario(40, "steel_pole", face="left", offset=0.3))
    right = simulate(scenario(40, "steel_pole", face="right", offset=0.3))
    assert left.crush_depth == pytest.approx(right.crush_depth)
    assert left.damage[PART_INDEX["door_l"]] == pytest.approx(right.damage[PART_INDEX["door_r"]])
    assert left.damage[PART_INDEX["door_r"]] == pytest.approx(right.damage[PART_INDEX["door_l"]])


def test_wooden_fence_breaks_and_the_car_hardly_notices():
    res = simulate(scenario(50, "wood_fence"))
    assert res.breakthrough
    assert res.obstacle_damage == 1.0
    assert res.e_car_kj < 0.1 * res.e_diss_kj


def test_foam_barrier_is_crushed_and_spares_the_car():
    foam = simulate(scenario(56, "foam_barrier"))
    wall = simulate(scenario(56, "concrete_wall"))
    assert foam.e_obstacle_kj > 0 and foam.obstacle_damage > 0.3
    assert foam.crush_depth < wall.crush_depth
    assert foam.e_car_kj < wall.e_car_kj


def test_narrow_hard_object_can_go_through_and_energy_stays_accounted_for():
    res = simulate(scenario(120, "steel_edge", face="left", offset=0.0))
    assert res.through
    assert res.unabsorbed_kj > 0
    assert res.crush_depth <= EXTENT["left"] + 1e-9


def test_standing_still_causes_no_damage():
    res = simulate(scenario(0.0))
    assert res.damage.sum() == 0.0
    assert not res.airbag_front and not res.airbag_side


def test_airbags_follow_the_side_that_was_hit():
    front = simulate(scenario(56, face="front"))
    side = simulate(scenario(40, "steel_pole", face="left"))
    assert front.airbag_front and not front.airbag_side
    assert side.airbag_side and not side.airbag_front


def test_invalid_scenarios_are_rejected():
    with pytest.raises(ValueError):
        Scenario(default_vehicle(), default_obstacle("steel_pole"), face="top")
    with pytest.raises(ValueError):
        Scenario(default_vehicle(), default_obstacle("steel_pole"), offset=1.5)
    with pytest.raises(ValueError):
        Scenario(default_vehicle(), default_obstacle("steel_pole"), speed=-1.0)


def test_every_preset_runs_on_every_side():
    for name in PRESET_NAMES:
        for face in ("front", "rear", "left", "right"):
            res = simulate(scenario(45, name, face=face, offset=-0.4, angle=15))
            assert np.isfinite(res.damage).all()


def test_report_is_json_and_has_all_parts():
    sc = scenario(50, "steel_pole", face="left")
    report = result_to_dict(sc, simulate(sc))
    text = json.dumps(report)
    back = json.loads(text)
    assert len(back["parts"]) == N_PARTS
    assert {"summary", "effects", "impact", "obstacle"} <= set(back)
    assert back["impact"]["speed_kmh"] == pytest.approx(50.0, abs=0.1)


def test_scenario_json_accepts_unity_style_requests():
    sc = scenario_from_dict({
        "speed_kmh": 50, "speed_ms": 0, "face": "front", "offset": 0, "angle_deg": 0,
        "vehicle": {"mass_kg": 1200, "thickness_scale": 0,
                    "materials": [{"part": "door_l", "material": "aluminium"}],
                    "thickness_mm": [{"part": "roof", "mm": 1.5}]},
        "obstacle": {"name": "concrete_wall", "width": 0, "material": "", "mass_kg": 0, "fixed_in_place": True}})
    assert sc.speed == pytest.approx(50 / 3.6)
    assert sc.vehicle.mass_kg == 1200
    assert sc.vehicle.parts[PART_INDEX["door_l"]].material == "aluminium"
    assert sc.vehicle.parts[PART_INDEX["roof"]].thickness_mm == 1.5
    assert sc.obstacle.width == default_obstacle("concrete_wall").width


def test_unknown_names_give_clear_errors():
    with pytest.raises(ValueError):
        scenario_from_dict({"speed_kmh": 30, "obstacle": {"name": "unobtainium"}})
    with pytest.raises(ValueError):
        scenario_from_dict({"speed_kmh": 30, "vehicle": {"materials": {"door_l": "cheese"}}})
    with pytest.raises(ValueError):
        obstacle_from_dict({"name": "steel_pole", "material": "cheese"})


def test_random_vehicles_stay_valid(rng):
    for _ in range(20):
        v = sample_vehicle(rng)
        assert len(v.parts) == N_PARTS
        assert 900 <= v.mass_kg <= 1900
