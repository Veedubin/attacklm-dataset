"""Hermetic tests for scripts/extract_mitre_atlas.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_extractor():
    spec = importlib.util.spec_from_file_location(
        "extract_mitre_atlas", _REPO_ROOT / "scripts" / "extract_mitre_atlas.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_mitre_atlas"] = module
    spec.loader.exec_module(module)
    return module


ex = _load_extractor()

FIXTURE = """
format-version: 6.0.0
collection:
  id: ATLAS-collection
  version: "2026.09"
  name: ATLAS
  description: test fixture
matrix:
  id: ATLAS-matrix
  name: ATLAS
tactics:
  AML.TA0002:
    id: AML.TA0002
    object-type: tactic
    name: Reconnaissance
techniques:
  AML.T0001:
    id: AML.T0001
    object-type: technique
    name: Test Technique
    description: Does test things.
    references: []
    created-date: "2021-05-13"
    modified-date: "2026-05-27"
    platforms: [Generative AI]
    maturity: Demonstrated
    attack-reference:
      id: T1596
      url: https://attack.mitre.org/techniques/T1596/
  AML.T0001.001:
    id: AML.T0001.001
    object-type: technique
    name: Test Sub Technique
    description: Does sub test things.
    references: []
    created-date: "2021-05-13"
    modified-date: "2026-05-27"
    platforms: [Generative AI]
    maturity: Feasible
mitigations:
  AML.M0000:
    id: AML.M0000
    object-type: mitigation
    name: Test Mitigation
    description: Mitigates the test technique.
    references: []
    created-date: "2023-04-12"
    modified-date: "2026-07-31"
    lifecycle-phases: [Deployment]
    categories: [Policy]
case-studies:
  AML.CS0000:
    id: AML.CS0000
    object-type: case-study
    name: Test Case Study
    description: A test case study.
    references: []
    created-date: "2020-12-15"
    modified-date: "2025-03-14"
    type: Exercise
    actor: Test Actor
    target: Test Target
relationships:
  ATLAS-matrix:
    sequences:
      - {source: ATLAS-matrix, target: AML.TA0002, relationship-type: sequences}
  AML.T0001:
    achieves:
      - {source: AML.T0001, target: AML.TA0002, relationship-type: achieves}
  AML.T0001.001:
    achieves:
      - {source: AML.T0001.001, target: AML.TA0002, relationship-type: achieves}
    specializes:
      - {source: AML.T0001.001, target: AML.T0001, relationship-type: specializes}
  AML.M0000:
    mitigates:
      - {source: AML.M0000, target: AML.T0001, relationship-type: mitigates}
  AML.CS0000:
    employs:
      - source: AML.CS0000
        target: AML.T0001.001
        relationship-type: employs
        description: Step two did Y.
        tactic: AML.TA0002
        step-id: S01
        leads-to: []
      - source: AML.CS0000
        target: AML.T0001
        relationship-type: employs
        description: Step one did X.
        tactic: AML.TA0002
        step-id: S00
        leads-to: [S01]
"""


@pytest.fixture()
def atlas():
    return yaml.safe_load(FIXTURE)


@pytest.fixture()
def idx(atlas):
    return ex.build_indexes(atlas)


def test_tactic_slug_all_16():
    names = [
        ("AI Model Access", "ai_model_access"),
        ("AI Attack Adaptation", "ai_attack_adaptation"),
        ("Reconnaissance", "reconnaissance"),
        ("Resource Development", "resource_development"),
        ("Initial Access", "initial_access"),
        ("Execution", "execution"),
        ("Persistence", "persistence"),
        ("Defense Evasion", "defense_evasion"),
        ("Discovery", "discovery"),
        ("Collection", "collection"),
        ("Exfiltration", "exfiltration"),
        ("Impact", "impact"),
        ("Privilege Escalation", "privilege_escalation"),
        ("Credential Access", "credential_access"),
        ("Command and Control", "command_and_control"),
        ("Lateral Movement", "lateral_movement"),
    ]
    for name, slug in names:
        assert ex.tactic_slug(name) == slug


def test_indexes(idx):
    assert idx["tech_to_tactics"]["AML.T0001"] == ["AML.TA0002"]
    assert idx["sub_to_parent"]["AML.T0001.001"] == "AML.T0001"
    assert idx["tech_to_mitigations"]["AML.T0001"] == ["AML.M0000"]
    assert len(idx["case_steps"]["AML.CS0000"]) == 2


def test_technique_pairs_shape(atlas, idx):
    tech = atlas["techniques"]["AML.T0001"]
    pairs = ex.technique_pairs(tech, atlas, idx)
    assert len(pairs) >= 3  # desc + variant + platforms/maturity (+cross-ref)
    assert len(pairs) == 4  # fixture has attack-reference
    for p in pairs:
        assert p["source"] == "mitre-atlas"
        assert p["license"] == "Apache-2.0"
        assert p["license_uri"] == "https://www.apache.org/licenses/LICENSE-2.0"
        assert p["source_uri"] == "https://github.com/mitre-atlas/atlas-data"
        assert p["rights_contact"] == "see data/REMOVAL.md"
        assert p["tactic"] == "reconnaissance"
        assert "atlas-2026.09" in p["tags"]
        assert "AML.T0001" in p["mitre_ids"]
        roles = [m["role"] for m in p["messages"]]
        assert roles == ["system", "user", "assistant"]
    # ATT&CK cross-ref pair
    cross = [p for p in pairs if "T1596" in p["mitre_ids"]]
    assert len(cross) == 1


def test_sub_technique_names_parent(atlas, idx):
    sub = atlas["techniques"]["AML.T0001.001"]
    pairs = ex.technique_pairs(sub, atlas, idx)
    desc_pair = pairs[0]
    assert "Test Technique" in desc_pair["messages"][2]["content"]
