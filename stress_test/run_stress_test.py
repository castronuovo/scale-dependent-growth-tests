#!/usr/bin/env python3
"""Evaluate the revision stress tests using the frozen benchmark in the sibling ZIP."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
N = 4_000
KEYS = ("nonnull_99", "st1_95", "st2_95")
CLASSES = ("null", "ST1 fail", "ST2 fail", "ST1/ST2 pass")


def mock_stats(b, signal, white_noise):
    y = signal @ b.whitener.T + white_noise
    y -= (y @ b.q_nuisance) @ b.q_nuisance.T
    d0 = np.sum(y**2, axis=1)
    df = np.sum((y - (y @ b.q_flexible) @ b.q_flexible.T)**2, axis=1)
    c = np.min(b.const_quad - 2 * (y @ b.const_bank.T), axis=1)
    h = np.min(b.one_quad - 2 * (y @ b.one_bank.T), axis=1)
    return np.column_stack((d0 - df, d0 + h - df, c - h))


def expanded_bank(b, frozen):
    out = copy(b)
    const, one = [], []
    for mass in (.030, .260):
        for beta in frozen["contract"]["one_turnover_bank"]["beta_nodes"]:
            for power in frozen["contract"]["one_turnover_bank"]["power_nodes"]:
                signal = b.project(b.one(beta, mass, power))
                one.append(signal)
                if power == 1:
                    const.append(signal)
    out.const_bank = np.vstack((b.const_bank, const))
    out.one_bank = np.vstack((b.one_bank, one))
    out.const_quad = np.sum(out.const_bank**2, axis=1)
    out.one_quad = np.sum(out.one_bank**2, axis=1)
    return out


def calibration(mod, b, expanded):
    size = b.chol.shape[0]

    def draw(signal, n, label):
        noise = mod.seeded(label).standard_normal((n, size))
        return [mock_stats(bank, signal, noise) for bank in (b, expanded)]

    null = draw(np.zeros(size), 1200, "cal-null")
    st1, st2 = [], []
    for i, (beta, mass) in enumerate(
        (beta, mass) for beta in (.012, .024, .040, .064) for mass in (.055, .105, .175)
    ):
        st2.append(draw(b.one(beta, mass), 240, f"cal-c-{i}"))
        for p in (.75, 1., 1.25):
            st1.append(draw(b.one(beta, mass, p), 240, f"cal-h1-{i}-{p}"))
    for i, signal in enumerate(b.causal_signals(mod.CAUSAL_CALIBRATION_SEED, 16)):
        st1.append(draw(signal, 240, f"cal-causal-{i}"))
    quantiles = ((.98, .99, .995), (.90, .95, .975), (.90, .95, .975))
    cells = [np.asarray([null]), np.asarray(st1), np.asarray(st2)]
    return [
        [
            np.quantile(cell[:, bank, :, stage], quantiles[stage], axis=1).max(axis=1)
            for stage, cell in enumerate(cells)
        ]
        for bank in range(2)
    ]


def smeared_signal(mod, b, d, f, sigma_v, order=32):
    mu, w = leggauss(order)
    a = mod.A_OBS
    e = np.sqrt(mod.OMEGA_M0 / a**3 + 1 - mod.OMEGA_M0)
    sigma_parallel = sigma_v / (100 * a * e)
    damping = np.exp(-(sigma_parallel[:, None, None] * b.k[None, :, None] * mu)**2)
    pm = mod.primordial_shape(b.k)[None, :] * d**2
    power = pm[:, :, None] * (mod.B_FID[:, None, None] + f[:, :, None] * mu**2)**2
    p0 = (power * damping) @ (w / 2)
    p2 = (power * damping) @ (5 * w * (3 * mu**2 - 1) / 4)
    windowed = np.stack((p0 @ b.window.T, p2 @ b.window.T), axis=1)
    return (windowed / b.gr_multipoles - 1).reshape(-1)


def rate(mod, numerator, denominator):
    n, d = int(numerator), int(denominator)
    if not d:
        return {"n": n, "denominator": d, "fraction": None, "wilson_95": None}
    fraction = n / d
    lo, hi = mod.wilson_interval(fraction, d, 1.959963984540054)
    return {
        "n": n, "denominator": d, "fraction": fraction,
        "wilson_95": [max(0., lo), min(1., hi)],
    }


def summarize(mod, stats, cuts):
    nonnull, reject1, reject2 = (stats > cuts).T
    entry = nonnull & ~reject1
    masks = (~nonnull, nonnull & reject1, entry & reject2, entry & ~reject2)
    assert np.all(np.sum(masks, axis=0) == 1)
    n = len(stats)
    return {
        "raw": {label: rate(mod, mask.sum(), n) for label, mask in zip(CLASSES, masks)},
        "nonnull": rate(mod, nonnull.sum(), n),
        "st1_marginal_crossing": rate(mod, reject1.sum(), n),
        "st1_reject_given_nonnull": rate(mod, masks[1].sum(), nonnull.sum()),
        "st2_marginal_crossing": rate(mod, reject2.sum(), n),
        "st2_entry": rate(mod, entry.sum(), n),
        "st2_reject_given_entry": rate(mod, masks[2].sum(), entry.sum()),
        "st2_nonrejection_given_entry": rate(mod, masks[3].sum(), entry.sum()),
    }


def evaluate(mod, frozen):
    b = mod.Benchmark(mod.Contract())
    ext = expanded_bank(b, frozen)
    archived_cuts = np.array([frozen["thresholds"][key] for key in KEYS])
    base, wide = calibration(mod, b, ext)
    np.testing.assert_allclose([v[1] for v in base], archived_cuts, rtol=0, atol=1e-10)
    cuts = {"nominal": archived_cuts}
    for stage, label in enumerate(("null", "st1", "st2")):
        for idx, side in ((0, "lower"), (2, "upper")):
            new = archived_cuts.copy()
            new[stage] = base[stage][idx]
            cuts[f"{label}_{side}"] = new
    cuts["expanded_mass_support"] = np.array([v[1] for v in wide])
    cases = {
        "gr": (0., .105, 1.),
        "constant_a": (.033, .090, 1.),
        "constant_b": (.036, .150, 1.),
        "constant_c": (.053, .120, 1.),
        "drift_0.5": (.080, .105, 1.5),
        "drift_1": (.080, .105, 2.),
    }
    signals = {key: b.one(*value) for key, value in cases.items()}
    signals["resolved_two_scale"] = b.two(.100, .045, .100, .195)
    size = b.chol.shape[0]
    np.testing.assert_allclose(signals["gr"], np.zeros(size), rtol=0, atol=1e-14)
    # Check the batched equivalent against the released scalar statistic.
    sample = mod.seeded("adp-statistic-equivalence").standard_normal((4, size))
    for bank in (b, ext):
        direct = [bank.stat(signals["drift_1"] + bank.chol @ draw) for draw in sample]
        np.testing.assert_allclose(
            mock_stats(bank, signals["drift_1"], sample),
            [[row[key] for key in ("nonnull", "st1", "st2")] for row in direct],
            rtol=0, atol=1e-10,
        )

    validation, smearing, geometry, drift_decisions = {}, {}, {}, {}
    bank_nodes = frozen["contract"]["one_turnover_bank"]
    const_params = [(beta, mass) for beta in bank_nodes["beta_nodes"] for mass in bank_nodes["mass_nodes_h_mpc"]]
    extended_params = const_params + [
        (beta, mass) for mass in (.030, .260) for beta in bank_nodes["beta_nodes"]
    ]
    for key, signal in signals.items():
        noise_group = "paired_drift" if key.startswith("drift_") else key
        noise = mod.seeded("adp-major-revision-evaluation-" + noise_group).standard_normal((N, size))
        base_stats = mock_stats(b, signal, noise)
        extended_stats = mock_stats(ext, signal, noise)
        validation[key] = {
            rule: summarize(mod, extended_stats if rule == "expanded_mass_support" else base_stats, cut)
            for rule, cut in cuts.items()
        }
        if key in ("gr", "constant_a", "drift_1"):
            beta, mass, p = cases[key]
            d, f = b.integrate(lambda k, a: 1 + mod.activation(a) * beta * k**2 / (k**2 + (mass * a**p)**2))
            np.testing.assert_allclose(smeared_signal(mod, b, d, f, 0), signal, rtol=0, atol=1e-13)
            smearing[key] = {"0": validation[key]["nominal"]}
            for sigma_v in (85.7, 200.):
                damped = smeared_signal(mod, b, d, f, sigma_v)
                np.testing.assert_allclose(damped, smeared_signal(mod, b, d, f, sigma_v, 64), rtol=0, atol=1e-13)
                stats = mock_stats(b, damped, noise)
                smearing[key][f"{sigma_v:g}"] = {
                    **summarize(mod, stats, archived_cuts),
                    "squared_projected_mean_shift": float(np.sum(b.project(damped - signal)**2)),
                }
        if key.startswith("drift_"):
            _, mass, p = cases[key]
            y = b.project(signal)
            geometry[key] = {"q_h_mpc": (mass * mod.A_OBS**p).tolist()}
            for name, bank, params in (("nominal", b, const_params), ("expanded", ext, extended_params)):
                distances = np.sum((bank.const_bank - y)**2, axis=1)
                idx = np.argmin(distances)
                one_distance = np.min(np.sum((bank.one_bank - y)**2, axis=1))
                assert one_distance < 1e-20
                geometry[key][name] = {
                    "constant_distance_squared": float(distances[idx]),
                    "one_turnover_distance_squared": float(one_distance),
                    "nearest_constant_beta": params[idx][0],
                    "nearest_constant_mass_h_mpc": params[idx][1],
                }
            nonnull, reject1, reject2 = (base_stats > archived_cuts).T
            drift_decisions[key] = nonnull & ~reject1 & reject2
    p05, p1 = drift_decisions["drift_0.5"], drift_decisions["drift_1"]
    delta = p1.astype(float) - p05.astype(float)
    mean, se = np.mean(delta), np.std(delta, ddof=1) / np.sqrt(N)
    return {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "evaluation_draws_per_truth": N,
        "truth_beta_mass_power": cases,
        "resolved_two_scale_beta_mass_pairs": [[.100, .045], [.100, .195]],
        "thresholds": {rule: dict(zip(KEYS, cut.tolist())) for rule, cut in cuts.items()},
        "clean_validation": validation,
        "redshift_smearing_sigma_v_km_s": smearing,
        "noiseless_drift_geometry": geometry,
        "paired_drift_change_in_raw_ST2_exit_1_minus_0.5": {
            "entry_at_0.5": validation["drift_0.5"]["nominal"]["st2_entry"]["n"],
            "entry_at_1": validation["drift_1"]["nominal"]["st2_entry"]["n"],
            "reject_0.5_only": int(np.sum(p05 & ~p1)),
            "reject_1_only": int(np.sum(p1 & ~p05)),
            "fraction_difference": float(mean),
            "normal_95": [float(mean - 1.96 * se), float(mean + 1.96 * se)],
        },
    }


def main():
    with TemporaryDirectory(prefix="adp-frozen-") as tmp, ZipFile(HERE.parent / "reproducibility.zip") as archive:
        source = archive.read("benchmark_st1_st2.py")
        (Path(tmp) / "benchmark_st1_st2.py").write_bytes(source)
        baseline = archive.read("benchmark_st1_st2_results.json")
        sys.path.insert(0, tmp)
        import benchmark_st1_st2 as mod
        results = evaluate(mod, json.loads(baseline))
    results["frozen_benchmark_sha256"] = hashlib.sha256(source).hexdigest()
    results["frozen_results_sha256"] = hashlib.sha256(baseline).hexdigest()
    (HERE / "stress_test_results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
