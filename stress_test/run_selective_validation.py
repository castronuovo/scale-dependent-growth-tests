#!/usr/bin/env python3
"""Calibrate selective mass intervals and validate partial entry into ST2."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import scipy
from scipy.stats import beta as beta_distribution
from scipy.stats import binom

from run_stress_test import HERE, KEYS, mock_stats, rate, summarize

ALPHA = 0.05
N_REFERENCE = 1_999
N_COVERAGE = 4_000
N_PILOT = 512
N_BOUNDARY = 20_000
CHUNK = 2_048
PROPOSAL_CAP = 2_000_000
PILOT_MASSES = (.040, .0675, .1225, .1575, .220)
ENTRY_TARGETS = (.20, .50, .80)


def stage_masks(stats, cuts):
    nonnull, reject1, reject2 = (stats > cuts).T
    entry = nonnull & ~reject1
    return nonnull, entry, entry & ~reject2


def projected_rows(benchmark, signal, noise):
    y = signal @ benchmark.whitener.T + noise
    return y - (y @ benchmark.q_nuisance) @ benchmark.q_nuisance.T


def mass_scores(benchmark, y, masses):
    costs = benchmark.const_quad - 2 * (y @ benchmark.const_bank.T)
    n_amplitudes = len(benchmark.const_bank) // len(masses)
    by_mass = costs.reshape(len(y), n_amplitudes, len(masses)).min(axis=1)
    return by_mass - by_mass.min(axis=1, keepdims=True)


def mass_hulls(scores, masses, critical):
    retained = scores <= critical
    lower = masses[retained.argmax(axis=1)]
    upper = masses[len(masses) - 1 - retained[:, ::-1].argmax(axis=1)]
    return retained, lower, upper


def selective_mass_interval(benchmark, vector, thresholds, mass_calibration):
    """Return the closed mass interval for one raw pass, or None otherwise.

    The raw vector and benchmark must use the archived nominal contract.
    Calibration supports exactly the recorded finite mass/amplitude grid.
    """
    if benchmark.raw_class(benchmark.stat(vector), thresholds) != "ST1/ST2 pass":
        return None
    masses = np.asarray(mass_calibration["mass_nodes_h_mpc"])
    score = mass_scores(benchmark, benchmark.project(vector)[None, :], masses)
    _, lower, upper = mass_hulls(score, masses, mass_calibration["critical_by_mass"])
    return float(lower[0]), float(upper[0])


def noise_rng(mod, phase, beta, mass):
    return mod.seeded(f"adp-4004254-selective-{phase}-{beta:.4f}-{mass:.4f}")


def selected_indices(benchmark, signal, noise, cuts):
    # The same non-null expression as mock_stats cheaply screens proposals.
    y = projected_rows(benchmark, signal, noise)
    flexible_residual = y - (y @ benchmark.q_flexible) @ benchmark.q_flexible.T
    nonnull = np.sum(y**2, axis=1) - np.sum(flexible_residual**2, axis=1)
    candidates = np.flatnonzero(nonnull > cuts[0])
    if not len(candidates):
        return candidates, y[:0]
    stats = mock_stats(benchmark, signal, noise[candidates])
    _, _, report = stage_masks(stats, cuts)
    keep = candidates[report]
    return keep, y[keep]


def draw_selected(mod, benchmark, signal, beta, mass, phase, count, cuts, masses):
    rng = noise_rng(mod, phase, beta, mass)
    proposed, remaining = 0, count
    scores = []
    while remaining and proposed < PROPOSAL_CAP:
        n = min(CHUNK, PROPOSAL_CAP - proposed)
        noise = rng.standard_normal((n, benchmark.chol.shape[0]))
        indices, y = selected_indices(benchmark, signal, noise, cuts)
        take = min(remaining, len(indices))
        remaining -= take
        proposed += n if remaining else int(indices[take - 1]) + 1
        if take:
            scores.append(mass_scores(benchmark, y[:take], masses))
    if remaining:
        raise RuntimeError(f"Insufficient {phase} selections at beta={beta}, mass={mass}")
    return np.vstack(scores), proposed


def exact_interval(hits, total, alpha):
    if not total:
        return None
    lo = beta_distribution.ppf(alpha / 2, hits, total - hits + 1) if hits else 0.0
    hi = beta_distribution.isf(alpha / 2, hits + 1, total - hits) if hits < total else 1.0
    return [float(lo), float(hi)]


def coverage(scores, mass, masses, critical, cells):
    if not len(scores):
        return None
    retained, lower, upper = mass_hulls(scores, masses, critical)
    hits = int(np.sum((lower <= mass) & (mass <= upper)))
    truth_node = np.flatnonzero(masses == mass)
    node_hits = int(retained[:, truth_node[0]].sum()) if len(truth_node) else None
    n = len(scores)
    return {
        "covered": hits,
        "reported": n,
        "fraction": hits / n,
        "clopper_pearson_95": exact_interval(hits, n, ALPHA),
        "simultaneous_clopper_pearson_95": exact_interval(hits, n, ALPHA / cells),
        "retained_truth_node": node_hits,
        "median_width_h_mpc": float(np.median(upper - lower)),
        "full_range": int(np.sum((lower == masses[0]) & (upper == masses[-1]))),
        "touching_lower": int(np.sum(lower == masses[0])),
        "touching_upper": int(np.sum(upper == masses[-1])),
        "disconnected_node_sets": int(np.sum((retained[:, 1:] & ~retained[:, :-1]).sum(axis=1)
                                             + retained[:, 0] > 1)),
    }


def check_contract(mod, benchmark, betas, masses, cuts):
    rng = mod.seeded("adp-4004254-selective-scalar-check")
    cuts_dict = dict(zip(KEYS, cuts))
    for beta, mass in [(.004, .04), (.012, .105), (.064, .140), (.18, .220)]:
        signal = benchmark.one(beta, mass)
        noise = rng.standard_normal((48, benchmark.chol.shape[0]))
        values = signal + noise @ benchmark.chol.T
        scalar = [benchmark.stat(row) for row in values]
        stats = mock_stats(benchmark, signal, noise)
        np.testing.assert_allclose(stats, [[row[k] for k in ("nonnull", "st1", "st2")]
                                           for row in scalar], rtol=0, atol=1e-10)
        keep, _ = selected_indices(benchmark, signal, noise, cuts)
        np.testing.assert_array_equal(keep, np.flatnonzero(stage_masks(stats, cuts)[2]))
        np.testing.assert_array_equal(
            keep,
            np.flatnonzero([benchmark.raw_class(row, cuts_dict) == "ST1/ST2 pass" for row in scalar]),
        )
        y = benchmark.project(values[0])
        reference = []
        for node in masses:
            bank = np.array([benchmark.project(benchmark.one(b, node)) for b in betas])
            reference.append(benchmark.best(y, bank, np.sum(bank**2, axis=1)))
        reference = np.array(reference) - min(reference)
        np.testing.assert_allclose(mass_scores(benchmark, y[None, :], masses)[0], reference,
                                   rtol=0, atol=1e-8)


def validate_mass(mod, benchmark, betas, masses, cuts):
    cells = len(betas) * len(masses)
    tail_budget = ALPHA / cells
    rank = int(binom.isf(tail_budget, N_REFERENCE, 1 - ALPHA)) + 1
    assert binom.sf(rank - 1, N_REFERENCE, 1 - ALPHA) <= tail_budget
    assert binom.sf(rank - 2, N_REFERENCE, 1 - ALPHA) > tail_budget
    truths = [(beta, mass, i, benchmark.one(beta, mass))
              for beta in betas for i, mass in enumerate(masses)]
    rows = []
    critical = np.zeros(len(masses))
    for beta, mass, i, signal in truths:
        scores, n = draw_selected(mod, benchmark, signal, beta, mass, "reference",
                                  N_REFERENCE, cuts, masses)
        cut = float(np.partition(scores[:, i], rank - 1)[rank - 1])
        critical[i] = max(critical[i], cut)
        rows.append({"beta": beta, "mass_h_mpc": mass, "reference_proposals": n,
                     "reference_critical": cut})
    print("Mass reference calibrated.", file=sys.stderr, flush=True)
    for row, (beta, mass, _, signal) in zip(rows, truths):
        scores, n = draw_selected(mod, benchmark, signal, beta, mass, "coverage",
                                  N_COVERAGE, cuts, masses)
        row["validation_proposals"] = n
        row["validation"] = coverage(scores, mass, masses, critical, cells)
    return {
        "beta_nodes": betas.tolist(),
        "mass_nodes_h_mpc": masses.tolist(),
        "reference_selected_per_cell": N_REFERENCE,
        "validation_selected_per_cell": N_COVERAGE,
        "reference_rank_one_based": rank,
        "reference_binomial_tail": float(binom.sf(rank - 1, N_REFERENCE, 1 - ALPHA)),
        "critical_by_mass": critical.tolist(),
        "cells": rows,
    }


def pilot_pairs(mod, benchmark, betas, cuts):
    pairs, pilot = [], []
    weak_betas = np.arange(4, 65) / 1000
    midpoint_betas = np.round((betas[1:] + betas[:-1]) / 2, 4)
    strong_betas = midpoint_betas[midpoint_betas >= .056]
    for mass in PILOT_MASSES:
        at_mass = []
        for beta in np.unique(np.r_[weak_betas, strong_betas]):
            rng = noise_rng(mod, "pilot", beta, mass)
            noise = rng.standard_normal((N_PILOT, benchmark.chol.shape[0]))
            stats = mock_stats(benchmark, benchmark.one(beta, mass), noise)
            nonnull, entry, _ = stage_masks(stats, cuts)
            row = {"beta": float(beta), "mass_h_mpc": mass, "nonnull": int(nonnull.sum()),
                   "entry": int(entry.sum())}
            at_mass.append(row)
        pilot.extend(at_mass)
        weak = [row for row in at_mass if row["beta"] in weak_betas
                and .10 <= row["entry"] / N_PILOT <= .90
                and row["entry"] >= .95 * row["nonnull"]]
        for target in ENTRY_TARGETS:
            if weak:
                row = min(weak, key=lambda row: abs(row["entry"] / N_PILOT - target))
                pairs.append({**row, "pilot_gate": "nonnull", "pilot_target": target})
                weak.remove(row)
        strong = [row for row in at_mass if row["beta"] in strong_betas
                  and row["nonnull"] >= .95 * N_PILOT
                  and .10 <= row["entry"] / row["nonnull"] <= .90]
        if strong:
            row = min(strong, key=lambda row: abs(row["entry"] / row["nonnull"] - .50))
            pairs.append({**row, "pilot_gate": "st1", "pilot_target": .50})
    return pairs, pilot


def validate_boundary(mod, benchmark, betas, masses, cuts, critical):
    pairs, pilot = pilot_pairs(mod, benchmark, betas, cuts)
    off_grid_cells = sum(row["beta"] not in betas or row["mass_h_mpc"] not in masses for row in pairs)
    results = []
    for pair in pairs:
        beta, mass = pair["beta"], pair["mass_h_mpc"]
        signal = benchmark.one(beta, mass)
        rng = noise_rng(mod, "boundary", beta, mass)
        noise = rng.standard_normal((N_BOUNDARY, benchmark.chol.shape[0]))
        stats = mock_stats(benchmark, signal, noise)
        operating_rates = summarize(mod, stats, cuts)
        rejected = operating_rates["st2_reject_given_entry"]
        rejected["simultaneous_clopper_pearson_95"] = exact_interval(
            rejected["n"], rejected["denominator"], ALPHA / len(pairs)
        )
        _, _, report = stage_masks(stats, cuts)
        hull_coverage = None
        if beta not in betas or mass not in masses:
            y = projected_rows(benchmark, signal, noise[report])
            hull_coverage = coverage(mass_scores(benchmark, y, masses), mass, masses,
                                     critical, off_grid_cells)
        results.append({"pilot_pair": pair, "operating_rates": operating_rates,
                        "off_grid_hull_coverage_given_report": hull_coverage})
    return {"pilot_draws_per_pair": N_PILOT, "validation_draws_per_pair": N_BOUNDARY,
            "pilot_cells": pilot, "validated_pairs": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "selective_validation_results.json")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="adp-frozen-") as tmp, ZipFile(HERE.parent / "reproducibility.zip") as archive:
        source = archive.read("benchmark_st1_st2.py")
        baseline = archive.read("benchmark_st1_st2_results.json")
        (Path(tmp) / "benchmark_st1_st2.py").write_bytes(source)
        sys.path.insert(0, tmp)
        import benchmark_st1_st2 as mod

        frozen = json.loads(baseline)
        contract = frozen["contract"]["one_turnover_bank"]
        betas = np.asarray(contract["beta_nodes"])
        masses = np.asarray(contract["mass_nodes_h_mpc"])
        cuts = np.array([frozen["thresholds"][key] for key in KEYS])
        benchmark = mod.Benchmark(mod.Contract())
        check_contract(mod, benchmark, betas, masses, cuts)
        mass = validate_mass(mod, benchmark, betas, masses, cuts)
        print("Mass coverage assessed. Running independent boundary validation.", file=sys.stderr, flush=True)
        boundary = validate_boundary(mod, benchmark, betas, masses, cuts, mass["critical_by_mass"])
    result = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__},
        "frozen_benchmark_sha256": hashlib.sha256(source).hexdigest(),
        "frozen_results_sha256": hashlib.sha256(baseline).hexdigest(),
        "thresholds": dict(zip(KEYS, cuts.tolist())),
        "proposal_cap_per_selected_sample": PROPOSAL_CAP,
        "alpha": ALPHA,
        "mass": mass,
        "boundary": boundary,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
