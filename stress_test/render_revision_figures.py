#!/usr/bin/env python3
"""Render the revision schematics and diagnostics from the equations and frozen results."""
from pathlib import Path
from zipfile import ZipFile
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def kernel_schematics(main_figs, supplement_figs):
    """Plot the analytic curves and parameter values shown in the manuscript."""
    k = np.geomspace(.001, 12., 300)
    r = np.geomspace(.1, 110., 300)
    masses = (.20, .35, .60)
    amplitudes = (.005, .010, .020, .030)
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")

    def save(fig, ax, xlabel, ylabel, title, destination):
        ax.set_xscale("log")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(alpha=.20)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(destination, bbox_inches="tight", metadata={"CreationDate": None})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for mass, color in zip(masses, colors):
        label = fr"$m_\ast={mass:g}\,h\,\mathrm{{Mpc}}^{{-1}}$"
        ax.plot(k, 1 + .02 * k**2 / (k**2 + mass**2), label=label, color=color)
        ax.axvline(mass, color=color, ls="--", lw=.8, alpha=.7)
    save(fig, ax, r"$k\ [h\,\mathrm{Mpc}^{-1}]$", r"$\mu(k,a)$",
         r"$a=1,\ \beta_0=0.02$", main_figs / "si_fig1_mu_k.pdf")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for beta, color in zip(amplitudes, colors):
        ax.plot(k, beta * k**2 / (k**2 + .35**2), label=fr"$\beta_0={beta:g}$", color=color)
    save(fig, ax, r"$k\ [h\,\mathrm{Mpc}^{-1}]$", r"$\Delta\mu(k,a)$",
         r"$a=1,\ m_\ast=0.35\,h\,\mathrm{Mpc}^{-1}$",
         supplement_figs / "si_fig2_deltamu_k_supplement.pdf")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for mass, color in zip(masses, colors):
        label = fr"$m_\ast={mass:g}\,h\,\mathrm{{Mpc}}^{{-1}}$"
        ax.plot(r, 1 + .02 * np.exp(-mass * r), label=label, color=color)
    save(fig, ax, r"$r\ [h^{-1}\,\mathrm{Mpc}]$", r"$G_\mu/G_{\rm GR}$",
         r"$a=1,\ \beta_0=0.02$", supplement_figs / "si_fig3_green_ratio_supplement.pdf")


def schematic(output, it):
    z = np.linspace(.5, 2., 251)
    a = 1 / (1 + z)
    ratio = a / .5
    # The neutrino guide reuses the comoving, non-relativistic matter-era
    # limit k_fs proportional to a**0.5 (Lesgourgues & Pastor 2006).
    labels = (
        ("Scala fisica costante", "Guida di free streaming", "Scala fisica variabile")
        if it else ("Constant physical scale", "Free-streaming guide", "Evolving physical scale")
    )
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for guide, style, color, label, formula in (
        (ratio, "-", "#0072B2", labels[0], r"$q \propto a$"),
        (np.sqrt(ratio), "--", "#D55E00", labels[1], r"$k_{\rm fs} \propto a^{1/2}$"),
        (ratio**2, "-.", "#009E73", labels[2], r"$q \propto a^2$"),
    ):
        ax.plot(z, guide, style, color=color, lw=1.65, label=f"{label}, {formula}")
    ax.set(
        xlim=(.5, 2.), ylim=(.4, 1.82), xlabel="Redshift $z$",
        ylabel=(r"Numero d'onda normalizzato" if it else "Normalized wavenumber"),
        xticks=[.5, 1., 1.5, 2.],
    )
    ax.axvline(1., lw=.7, color=".6", zorder=0)
    ax.grid(alpha=.20)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def diagnostics(results, output, it):
    noise = results["noise_operating_characteristics"]
    drift = results["epsilon_ST2_power"]
    banks = results["finite_bank_convergence"]
    fig, axs = plt.subplots(1, 3, figsize=(10.25, 3.25), layout="constrained")
    axs[0].plot([r["noise_multiplier"] for r in noise], [r["detection_power"] for r in noise],
                "o-", color="#0f8b7e")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks([1, 2, 4], ["1", "2", "4"])
    axs[0].set(
        xlabel="Moltiplicatore del rumore" if it else "Noise multiplier",
        ylabel="Potenza di rivelazione" if it else "Detection power",
    )
    axs[1].plot([r["epsilon"] for r in drift], [r["ST2_conditional_power"] for r in drift],
                "o-", color="#c46a00")
    at_null = next(row for row in drift if row["epsilon"] == 0)
    axs[1].scatter([0], [at_null["ST2_conditional_power"]], marker="D", zorder=3,
                   facecolors="white", edgecolors="#c46a00")
    axs[1].set(
        xlabel=r"$\epsilon$ in $q\propto a^{1+\epsilon}$",
        ylabel="Rigetto ST2 condizionale" if it else "Conditional ST2 rejection",
    )
    axs[2].plot(range(len(banks)), [r["positive_two_scale_ST1_fail"] for r in banks],
                "o-", color="#bd1e24")
    axs[2].set_xticks(range(len(banks)), [f"{r['n_causal']}/{r['n_k_basis']}" for r in banks])
    axs[2].set(
        xlabel=r"Storie causali / base $\ln k$" if it else r"Causal histories / $\ln k$ basis",
        ylabel="Frazione di rigetto ST1" if it else "ST1 rejection fraction",
    )
    for ax in axs[:2]:
        ax.axhline(results["power_target_for_classified_status"], color="#756bb1", ls="--", lw=1)
    for ax in axs:
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=.22)
    fig.savefig(output, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    with ZipFile(ROOT / "reproducibility.zip") as archive:
        frozen = json.loads(archive.read("benchmark_st1_st2_results.json"))
    for name, it in (("main", False), ("ita", True)):
        folder = ROOT / name / "figs"
        supplement = ROOT / ("ita" if it else "supplement") / "figs"
        kernel_schematics(folder, supplement)
        schematic(folder / "kturn_drift_test.pdf", it)
        diagnostics(frozen, folder / "st1_st2_operating_diagnostics.pdf", it)
