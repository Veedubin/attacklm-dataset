"""Inversion-attack audit harness for AttackLM.

Defensive audit tooling that probes the user's OWN models for memorized
training data. Two attack strategies are implemented:

1. **Carlini prefix-completion extraction** — feed the first N tokens of
   a training record as a prompt, generate K completions, and score
   how closely the best completion matches the original assistant turn.

2. **Membership-inference attack (MIA)** — compute NLL + zlib entropy
   for each record and calibrate a threshold on held-out data to
   distinguish members from non-members.

Threat model: identify what an adversary could extract so the model
owner can harden and produce evidence for regulators.
"""

__version__ = "0.4.0"
