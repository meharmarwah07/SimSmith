"""MOCK reach script: pure Python, no simulator. Exists so the pipeline can be
tested end to end without a GPU. It is NOT an Isaac Sim script.

Replace/extend with a real hand-written Isaac reach script for your pinned
version (examples/reach_isaac.py) and use that in scripts/gate_day1.py.
"""
from simsmith_contract import get_rng, get_perturb, write_result

rng = get_rng()
perturb = get_perturb()
target = [round(rng.uniform(0.3, 0.6), 3), round(rng.uniform(-0.2, 0.2), 3), round(rng.uniform(0.2, 0.5), 3)]
noise = float(perturb.get("pose_noise_std", 0.0))
ee = [t + rng.gauss(0, noise) for t in target]
write_result({"ee_pos": ee, "target_pos": target})
