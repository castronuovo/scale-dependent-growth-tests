#!/usr/bin/env python3
"""Reproducible semi-numerical operating-characteristic benchmark for ST1--ST2.

This is deliberately a benchmark-level calculation, not a survey forecast or a
Boltzmann-code validation.  It integrates the linear growth equation, forms
windowed Kaiser multipoles, employs a non-diagonal covariance, projects the
bias and k^2 nuisance directions, and calibrates all decisions on fixed mocks.
The finite bank is an operational approximation to the stated hypotheses.
"""

from __future__ import annotations

import json
import zlib
import warnings
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Some macOS Accelerate builds emit spurious floating-point warnings for finite
# matrix products.  The script stores only finite checked numerical outputs.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmark_st1_st2_results.json"
FIG = ROOT / "figs" / "st1_st2_end_to_end_benchmark.png"
FIG_ITA = ROOT / "figs" / "st1_st2_end_to_end_benchmark_ita.png"
DIAG = ROOT / "figs" / "st1_st2_operating_diagnostics.png"
DIAG_ITA = ROOT / "figs" / "st1_st2_operating_diagnostics_ita.png"

A_OBS = np.array([1.0 / 3.0, 0.4, 0.5, 2.0 / 3.0])
B_FID = np.array([1.85, 1.75, 1.65, 1.55])
OMEGA_M0, NS, A_INI, A_ON, DELTA_ON = 0.315, 0.965, 0.05, 0.25, 0.035
BASE_SEED = 70230
CAUSAL_CALIBRATION_SEED = 70233
CAUSAL_VALIDATION_SEED = 70234
N_CAUSAL_COVERAGE_CELLS = 16
POWER_TARGET = 0.80


@dataclass(frozen=True)
class Contract:
    """All tunable parts of the predeclared semi-numerical experiment."""

    n_k: int = 18
    kmin: float = 0.012
    kmax: float = 0.14
    lnk_window_width: float = 0.065
    noise_scale: float = 1.0
    n_causal: int = 128
    causal_seed: int = 70231
    n_k_basis: int = 6
    n_time_steps: int = 160


def seeded(label: str) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + zlib.crc32(label.encode("utf-8")))


def omega_m(a: float | np.ndarray) -> float | np.ndarray:
    e2 = OMEGA_M0 * np.asarray(a) ** -3 + (1.0 - OMEGA_M0)
    return OMEGA_M0 * np.asarray(a) ** -3 / e2


def activation(a: float | np.ndarray) -> float | np.ndarray:
    return 0.5 * (1.0 + np.tanh((np.asarray(a) - A_ON) / DELTA_ON))


def primordial_shape(k: np.ndarray) -> np.ndarray:
    return (k / 0.05) ** NS / (1.0 + (k / 0.075) ** 2) ** 2


def hat_basis(x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    out = np.zeros((nodes.size, x.size))
    for i, node in enumerate(nodes):
        if i == 0:
            out[i] = np.clip((nodes[1] - x) / (nodes[1] - node), 0.0, 1.0)
        elif i == nodes.size - 1:
            out[i] = np.clip((x - nodes[-2]) / (node - nodes[-2]), 0.0, 1.0)
        else:
            left = np.clip((x - nodes[i - 1]) / (node - nodes[i - 1]), 0.0, 1.0)
            right = np.clip((nodes[i + 1] - x) / (nodes[i + 1] - node), 0.0, 1.0)
            out[i] = np.minimum(left, right)
    return out / np.where(out.sum(axis=0) > 0.0, out.sum(axis=0), 1.0)


class Benchmark:
    def __init__(self, contract: Contract):
        self.c = contract
        self.k = np.geomspace(contract.kmin, contract.kmax, contract.n_k)
        x = np.log(self.k)
        w = np.exp(-0.5 * ((x[:, None] - x[None, :]) / contract.lnk_window_width) ** 2)
        self.window = w / w.sum(axis=1, keepdims=True)
        self.k_basis = hat_basis(x, np.linspace(x.min(), x.max(), contract.n_k_basis))
        self.time_edges = np.sqrt(A_OBS[:-1] * A_OBS[1:])
        self.gr_d, self.gr_f = self.integrate(lambda k, a: np.ones_like(k))
        self.gr_multipoles = self.multipoles(self.gr_d, self.gr_f)
        self.covariance, nuisance = self.covariance_and_nuisance()
        self.chol = np.linalg.cholesky(self.covariance)
        self.whitener = np.linalg.inv(self.chol)
        uw, sw, _ = np.linalg.svd(self.whitener @ nuisance, full_matrices=False)
        self.nuisance_rank = int(np.sum(sw > sw[0] * 1.0e-10))
        self.q_nuisance = uw[:, : self.nuisance_rank]
        self.prepare_flexible()
        self.prepare_banks()

    def integrate(self, mu_function, n_time_steps=None):
        # The base grid is supplemented by every output epoch and causal edge.
        # This prevents a Runge--Kutta step from straddling a discontinuity in a
        # piecewise-constant causal history.  Endpoint evaluations are displaced
        # infinitesimally into the current step, so each side of an edge uses the
        # appropriate half-open interval without imposing artificial smoothing.
        steps = self.c.n_time_steps if n_time_steps is None else int(n_time_steps)
        base = np.linspace(np.log(A_INI), np.log(A_OBS.max()), steps + 1)
        breaks = np.log(np.concatenate((A_OBS, self.time_edges)))
        n_grid = np.unique(np.concatenate((base, breaks)))
        d = np.full(self.k.size, A_INI)
        v = np.full(self.k.size, A_INI)
        d_out = np.zeros((A_OBS.size, self.k.size))
        f_out = np.zeros_like(d_out)
        targets, itarget = np.log(A_OBS), 0

        def rhs(n, d0, v0):
            a = float(np.exp(n)); om = float(omega_m(a)); friction = 2.0 - 1.5 * om
            return v0, 1.5 * om * mu_function(self.k, a) * d0 - friction * v0

        for n0, n1 in zip(n_grid[:-1], n_grid[1:]):
            h = n1 - n0
            n0_inner = np.nextafter(n0, n1)
            n1_inner = np.nextafter(n1, n0)
            k1d, k1v = rhs(n0_inner, d, v)
            k2d, k2v = rhs(n0 + h / 2, d + h * k1d / 2, v + h * k1v / 2)
            k3d, k3v = rhs(n0 + h / 2, d + h * k2d / 2, v + h * k2v / 2)
            k4d, k4v = rhs(n1_inner, d + h * k3d, v + h * k3v)
            dn = d + h * (k1d + 2*k2d + 2*k3d + k4d) / 6
            vn = v + h * (k1v + 2*k2v + 2*k3v + k4v) / 6
            while itarget < targets.size and targets[itarget] <= n1 + 1e-12:
                t = (targets[itarget] - n0) / h
                dt, vt = d + t*(dn-d), v + t*(vn-v)
                d_out[itarget], f_out[itarget] = dt, vt / dt
                itarget += 1
            d, v = dn, vn
        return d_out, f_out

    def multipoles(self, d, f, bias=B_FID):
        pm = primordial_shape(self.k)[None, :] * d**2
        p0 = (bias[:,None]**2 + 2*bias[:,None]*f/3 + f**2/5) * pm
        p2 = (4*bias[:,None]*f/3 + 4*f**2/7) * pm
        return np.stack([p0 @ self.window.T, p2 @ self.window.T], axis=1)

    def vector(self, mu):
        d, f = self.integrate(mu)
        return (self.multipoles(d, f) / self.gr_multipoles - 1.0).reshape(-1)

    def one(self, beta, mass, power=1.0):
        return self.vector(lambda k, a: 1 + activation(a)*beta*k**2/(k**2+(mass*a**power)**2))

    def two(self, beta1, mass1, beta2, mass2):
        return self.vector(lambda k, a: 1 + activation(a) * (
            beta1*k**2/(k**2+(mass1*a)**2) + beta2*k**2/(k**2+(mass2*a)**2)))

    def history_mu(self, beta_nodes, q_nodes):
        def mu(k, a):
            j = int(np.searchsorted(self.time_edges, a, side="right"))
            return 1 + activation(a)*beta_nodes[j]*k**2/(k**2+q_nodes[j]**2)
        return mu

    def history(self, beta_nodes, q_nodes):
        return self.vector(self.history_mu(beta_nodes, q_nodes))

    def flexible(self, coeff):
        def mu(k, a):
            time = np.array([1.0, np.log(a/0.5)])
            delta = np.einsum("ik,ij,j->k", self.k_basis, coeff, time)
            return 1 + activation(a)*delta
        return self.vector(mu)

    def covariance_and_nuisance(self):
        x = np.log(self.k)
        ck = np.exp(-np.abs(x[:,None]-x[None,:]) / 0.23)
        cz = 0.16 ** np.abs(np.arange(A_OBS.size)[:,None]-np.arange(A_OBS.size)[None,:])
        cl = np.array([[1., .28],[.28, 1.]])
        corr = .90*np.kron(cz, np.kron(cl, ck)) + .10*np.eye(A_OBS.size*2*self.k.size)
        sig = self.c.noise_scale*np.tile(np.repeat([.002,.004], self.k.size), A_OBS.size)
        cov = corr * sig[:,None] * sig[None,:]
        cols = []
        for iz in range(A_OBS.size):
            bp, bm = B_FID.copy(), B_FID.copy(); bp[iz] += 1e-3; bm[iz] -= 1e-3
            cols.append(((self.multipoles(self.gr_d,self.gr_f,bp)/self.gr_multipoles -
                          self.multipoles(self.gr_d,self.gr_f,bm)/self.gr_multipoles)/2e-3).reshape(-1))
            for ell in range(2):
                col = np.zeros((A_OBS.size,2,self.k.size)); col[iz,ell] = (self.k/.1)**2
                cols.append(col.reshape(-1))
        return cov, np.column_stack(cols)

    def project(self, vector):
        white = self.whitener @ vector
        return white - self.q_nuisance @ (self.q_nuisance.T @ white)

    def prepare_flexible(self):
        cols = []
        eps = 2e-3
        for ik in range(self.c.n_k_basis):
            for it in range(2):
                coeff = np.zeros((self.c.n_k_basis,2)); coeff[ik,it] = eps
                cols.append(self.vector(lambda k,a,c=coeff: 1 + activation(a)*np.einsum(
                    "ik,ij,j->k", self.k_basis, c, np.array([1.,np.log(a/.5)]))) / eps)
        flexible = np.column_stack(cols)
        projected = np.column_stack([self.project(flexible[:,i]) for i in range(flexible.shape[1])])
        u, s, _ = np.linalg.svd(projected, full_matrices=False)
        self.flexible_rank = int(np.sum(s > s[0]*1e-8))
        self.q_flexible = u[:,:self.flexible_rank]

    def prepare_banks(self):
        # Fixed finite bank; the upper amplitude accommodates the positive resolved injection.
        # Every composite-calibration cell is represented exactly in the bank.
        betas = np.array([.004,.012,.024,.040,.048,.064,.080,.100,.140,.180,.220])
        masses = np.array([.040,.045,.055,.080,.105,.140,.175,.195,.220])
        powers = np.array([.50,.75,1.,1.25,1.50,2.])
        const, one = [], []
        for beta in betas:
            for mass in masses:
                const.append(self.one(beta,mass,1.))
                for power in powers: one.append(self.one(beta,mass,power))
        rng = np.random.default_rng(self.c.causal_seed)
        for _ in range(self.c.n_causal):
            bn = rng.uniform(.004,.10,A_OBS.size)
            qn = np.exp(rng.uniform(np.log(.012),np.log(.145),A_OBS.size))
            one.append(self.history(bn,qn))
        self.const_bank = np.asarray([self.project(v) for v in const])
        self.one_bank = np.asarray([self.project(v) for v in one])
        self.const_quad = np.einsum("ij,ij->i",self.const_bank,self.const_bank)
        self.one_quad = np.einsum("ij,ij->i",self.one_bank,self.one_bank)

    def causal_signals(self, seed, n_signals):
        """Draw fixed causal histories independently of the finite fitting bank."""
        rng = np.random.default_rng(seed)
        signals = []
        for _ in range(n_signals):
            beta_nodes = rng.uniform(.004,.10,A_OBS.size)
            q_nodes = np.exp(rng.uniform(np.log(.012),np.log(.145),A_OBS.size))
            signals.append(self.history(beta_nodes,q_nodes))
        return signals

    @staticmethod
    def best(y, bank, quad):
        values = y@y - 2*(bank@y) + quad
        return float(values.min())

    def stat(self, vector):
        y = self.project(vector)
        chi0 = float(y@y)
        yf = y-self.q_flexible@(self.q_flexible.T@y)
        chif = float(yf@yf)
        chiconst = self.best(y,self.const_bank,self.const_quad)
        chione = self.best(y,self.one_bank,self.one_quad)
        return {"nonnull":chi0-chif, "st1":chione-chif, "st2":chiconst-chione}

    def sample_stats(self, signal, n, label):
        rng = seeded(label)
        return [self.stat(signal + self.chol@rng.standard_normal(self.chol.shape[0])) for _ in range(n)]

    def calibrate(self, n_null=1200, n_cell=240,
                  n_causal_cells=N_CAUSAL_COVERAGE_CELLS):
        # Composite calibration: each quoted threshold is the worst cell quantile.
        # Causal calibration histories are independent of the finite fitting bank;
        # a second seed supplies an out-of-calibration validation set.
        grid = [(b,m) for b in (.012,.024,.040,.064) for m in (.055,.105,.175)]
        null = self.sample_stats(np.zeros(self.chol.shape[0]),n_null,"cal-null")
        powerlaw_st1_cells, st2_cells = [], []
        for i,(b,m) in enumerate(grid):
            st2_cells.append(self.sample_stats(self.one(b,m,1.),n_cell,f"cal-c-{i}"))
            for p in (.75,1.,1.25):
                powerlaw_st1_cells.append(self.sample_stats(self.one(b,m,p),n_cell,f"cal-h1-{i}-{p}"))
        causal_calibration_cells = [
            self.sample_stats(signal,n_cell,f"cal-causal-{i}")
            for i,signal in enumerate(
                self.causal_signals(CAUSAL_CALIBRATION_SEED,n_causal_cells))
        ]
        st1_cells = powerlaw_st1_cells + causal_calibration_cells
        threshold = {
            "nonnull_99":float(np.quantile([r["nonnull"] for r in null],.99)),
            "st1_95":float(max(np.quantile([r["st1"] for r in cell],.95) for cell in st1_cells)),
            "st2_95":float(max(np.quantile([r["st2"] for r in cell],.95) for cell in st2_cells)),
        }
        causal_validation_cells = [
            self.sample_stats(signal,n_cell,f"validation-causal-{i}")
            for i,signal in enumerate(
                self.causal_signals(CAUSAL_VALIDATION_SEED,n_causal_cells))
        ]

        def st1_rejection_range(cells):
            rates = [np.mean([r["st1"]>threshold["st1_95"] for r in cell])
                     for cell in cells]
            return [float(min(rates)),float(max(rates))]

        coverage = {
            "constant_ST2_false_rejection_range": [float(min(np.mean([r["st2"]>threshold["st2_95"] for r in c]) for c in st2_cells)),
                                                   float(max(np.mean([r["st2"]>threshold["st2_95"] for r in c]) for c in st2_cells))],
            "powerlaw_H1_ST1_false_rejection_range": st1_rejection_range(powerlaw_st1_cells),
            "causal_H1_ST1_calibration_false_rejection_range": st1_rejection_range(causal_calibration_cells),
            "causal_H1_ST1_validation_false_rejection_range": st1_rejection_range(causal_validation_cells),
            "grid_beta": [.012,.024,.040,.064], "grid_mass": [.055,.105,.175],
            "power_nodes_used_for_ST1_coverage": [.75,1.,1.25],
            "n_causal_histories_per_coverage_set": n_causal_cells,
            "causal_calibration_seed": CAUSAL_CALIBRATION_SEED,
            "causal_validation_seed": CAUSAL_VALIDATION_SEED,
            "coverage_scope":"finite sampled power-law and piecewise-causal families; not uniform coverage of a continuous function space",
        }
        return threshold, coverage

    @staticmethod
    def raw_class(stat, threshold):
        if stat["nonnull"] <= threshold["nonnull_99"]: return "null"
        if stat["st1"] > threshold["st1_95"]: return "ST1 fail"
        if stat["st2"] > threshold["st2_95"]: return "ST2 fail"
        return "ST1/ST2 pass"

    def evaluate(self, signal, threshold, n, label):
        stats = self.sample_stats(signal,n,label)
        return {key:float(np.mean([self.raw_class(s,threshold)==key for s in stats]))
                for key in ("null","ST1/ST2 pass","ST2 fail","ST1 fail")}

    def response_cosines(self):
        base = self.one(.040,.105,1.)
        dm=.01
        mass = (self.one(.040,.105*np.exp(dm),1.)-self.one(.040,.105*np.exp(-dm),1.))/(2*dm)
        def nu(k,a):
            kfs=.075*a**.5
            return 1+activation(a)*(-.018)*k**2/(k**2+kfs**2)
        nuv=self.vector(nu)/-.018
        def cosine(x,y):
            x,y=self.project(x),self.project(y)
            return float(x@y/np.sqrt((x@x)*(y@y)))
        return {"mass_vs_neutrino_like":cosine(mass,nuv), "amplitude_vs_neutrino_like":cosine(base/.04,nuv),
                "note":"neutrino-like semi-numerical kernel; not a Boltzmann derivative"}

    def with_noise_scale(self, scale):
        """Return the same signal bank evaluated at a uniformly scaled noise level.

        A global covariance rescaling leaves the projected nuisance and flexible
        subspaces unchanged; only all whitened response vectors rescale.  This
        avoids silently changing the signal model when diagnosing noise power.
        """
        out = copy(self)
        out.covariance = self.covariance * scale**2
        out.chol = self.chol * scale
        out.whitener = self.whitener / scale
        out.const_bank, out.one_bank = self.const_bank / scale, self.one_bank / scale
        out.const_quad, out.one_quad = self.const_quad / scale**2, self.one_quad / scale**2
        return out

    def time_step_convergence(self):
        """Check deterministic response convergence under time-grid refinement."""
        histories = {
            "constant_scale": lambda k, a: 1 + activation(a)*.040*k**2/(k**2+(.105*a)**2),
            "evolving_scale": lambda k, a: 1 + activation(a)*.080*k**2/(k**2+(.105*a**2)**2),
            "piecewise_causal": self.history_mu(
                np.array([.020, .060, .030, .080]),
                np.array([.025, .090, .045, .130])),
        }
        step_counts = (80, 160, 320)
        rows = {}
        for name, mu in histories.items():
            vectors = {}
            for steps in step_counts:
                gd, gf = self.integrate(lambda k, a: np.ones_like(k), steps)
                md, mf = self.integrate(mu, steps)
                gr = self.multipoles(gd, gf)
                vectors[steps] = (self.multipoles(md, mf) / gr - 1.0).reshape(-1)
            reference = vectors[320]
            peak = max(float(np.max(np.abs(reference))), 1.0e-15)
            rows[name] = {
                str(steps): {
                    "max_absolute_response_difference_vs_320": float(
                        np.max(np.abs(vectors[steps] - reference))),
                    "max_difference_over_reference_peak_vs_320": float(
                        np.max(np.abs(vectors[steps] - reference)) / peak),
                }
                for steps in (80, 160)
            }
        return {
            "base_steps_tested": list(step_counts),
            "reference_base_steps": 320,
            "mandatory_breakpoints": "all output epochs and causal interval edges",
            "models": rows,
        }


def short_contract(contract):
    return {"n_k":contract.n_k,"k_h_mpc":[contract.kmin,contract.kmax],
            "lnk_window_width":contract.lnk_window_width,"fractional_errors_P0_P2":[.002*contract.noise_scale,.004*contract.noise_scale],
            "growth_integrator":{"method":"RK4","base_steps_in_ln_a":contract.n_time_steps,
                                 "a_initial":A_INI,
                                 "mandatory_breakpoints":"all output epochs and causal interval edges",
                                 "edge_evaluation":"one-sided within each integration step"},
            "activation":{"form":"0.5*(1+tanh((a-a_on)/delta_a))","a_on":A_ON,"delta_a":DELTA_ON},
            "one_turnover_bank":{"beta_nodes":[.004,.012,.024,.040,.048,.064,.080,.100,.140,.180,.220],
                                   "mass_nodes_h_mpc":[.040,.045,.055,.080,.105,.140,.175,.195,.220],
                                   "power_nodes":[.50,.75,1.,1.25,1.50,2.]},
            "n_causal_node_histories":contract.n_causal,"causal_history_seed":contract.causal_seed,
            "causal_history_sampling":{"beta":"independent uniform [0.004,0.10] per interval",
                                         "q_h_mpc":"independent log-uniform [0.012,0.145] per interval",
                                         "negative_amplitudes":False,
                                         "interval_convention":"left-closed/right-open; final interval closed at a_max",
                                         "smoothing_or_regularization":"none; piecewise constant on four causal intervals"},
            "n_k_basis":contract.n_k_basis}


def wilson_interval(fraction, n, z=1.0):
    """Return the two-sided Wilson interval (default: nominal 68 percent)."""
    denominator = 1.0 + z*z/n
    centre = (fraction + z*z/(2.0*n)) / denominator
    half = z*np.sqrt(fraction*(1.0-fraction)/n + z*z/(4.0*n*n)) / denominator
    return [float(centre-half), float(centre+half)]


def classify_with_deferred(benchmark, signal, threshold, n, label, criterion):
    """Operational status: deferred if the predeclared power is below 80 percent."""
    raw=benchmark.evaluate(signal,threshold,n,label)
    if criterion < POWER_TARGET:
        return {"deferred":1.0, **{k:0.0 for k in raw}}, raw
    return {"deferred":0.0, **raw}, raw


def main():
    b = Benchmark(Contract())
    threshold, coverage = b.calibrate()
    injections = {
        "GR/null": np.zeros(b.chol.shape[0]),
        "constant physical scale": b.one(.040,.105,1.),
        "evolving scale ($q\\propto a^2$)": b.one(.080,.105,2.),
        # Positive spectral weights, both turnover scales inside the declared k window.
        "resolved positive two-scale": b.two(.100,.045,.100,.195),
        "compressed positive two-scale": b.two(.020,.095,.020,.120),
    }
    raw = {name:b.evaluate(signal,threshold,400,"inj-"+name) for name,signal in injections.items()}
    classifications = {
        name:{"fractions": {"deferred":0., **frac},
              "wilson_68_intervals": {key:wilson_interval(value,400) for key,value in frac.items()}}
        for name,frac in raw.items()
    }

    # Noise sensitivity is evaluated for a weak, otherwise constant-scale signal.
    noise_rows=[]
    for scale in (1.,2.,4.):
        bn=b.with_noise_scale(scale)
        th,_=bn.calibrate(n_null=600,n_cell=120)
        # This fiducial constant-scale signal is safely classifiable at the
        # nominal contract and becomes information-limited as errors increase.
        weak=b.one(.040,.105,1.)
        frac=bn.evaluate(weak,th,240,f"noise-{scale}")
        detectable=1.-frac["null"]
        stated, _ = classify_with_deferred(bn,weak,th,240,f"noise-status-{scale}",detectable)
        noise_rows.append({"noise_multiplier":scale,"raw_fractions":frac,"detection_power":detectable,
                           "operational_fractions":stated,"status":"classified" if detectable>=POWER_TARGET else "deferred"})

    epsilon_rows=[]
    for eps in (-.25,-.10,0.,.10,.25,.50,1.):
        frac=b.evaluate(b.one(.080,.105,1.+eps),threshold,360,f"epsilon-{eps}")
        # Conditional power is the relevant ST2 capability after non-null, ST1-compatible entry.
        denominator=frac["ST1/ST2 pass"]+frac["ST2 fail"]
        power=frac["ST2 fail"]/denominator if denominator else 0.
        epsilon_rows.append({"epsilon":eps,"raw_fractions":frac,"ST2_conditional_power":power,
                             "status":"classified" if power>=POWER_TARGET else "deferred"})

    convergence=[]
    for ncausal,seed,nbasis in ((64,70231,5),(128,70231,6),(192,70231,7),(128,70232,6)):
        bc=Benchmark(Contract(n_causal=ncausal,causal_seed=seed,n_k_basis=nbasis))
        th,cv=bc.calibrate(n_null=600,n_cell=120)
        pos=bc.evaluate(bc.two(.100,.045,.100,.195),th,240,f"conv-{ncausal}-{seed}-{nbasis}")
        con=bc.evaluate(bc.one(.040,.105,1.),th,240,f"conv-c-{ncausal}-{seed}-{nbasis}")
        convergence.append({"n_causal":ncausal,"seed":seed,"n_k_basis":nbasis,"flexible_rank":bc.flexible_rank,
                            "thresholds":th,"positive_two_scale_ST1_fail":pos["ST1 fail"],
                            "constant_pass":con["ST1/ST2 pass"],"coverage":cv})

    output={"seed":BASE_SEED,"n_calibration_mocks":{"null":1200,"per_composite_cell":240,
                                                     "causal_calibration_cells":N_CAUSAL_COVERAGE_CELLS,
                                                     "independent_causal_validation_cells":N_CAUSAL_COVERAGE_CELLS},
            "n_evaluation_mocks_per_injection":400,"power_target_for_classified_status":POWER_TARGET,
            "power_definitions":{"nonnull_detection_power":"probability of passing the non-null threshold",
                                 "ST1_power":"probability of rejecting one-turnover compression for a predeclared resolved non-null injection",
                                 "conditional_ST2_power":"probability of rejecting the constant-scale branch conditional on non-null, ST1-compatible entry",
                                 "complete_pass_fraction":"probability of the final ST1/ST2-pass category after all preceding decisions"},
            "nuisance_rank":b.nuisance_rank,"flexible_rank":b.flexible_rank,"contract":short_contract(b.c),
            "injections":{"constant physical scale":{"beta":.040,"mass":.105,"power":1.},
                          "evolving scale":{"beta":.080,"mass":.105,"power":2.},
                          "resolved positive two-scale":{"beta1":.100,"mass1":.045,"beta2":.100,"mass2":.195},
                          "compressed positive two-scale":{"beta1":.020,"mass1":.095,"beta2":.020,"mass2":.120}},
            "thresholds":threshold,"composite_coverage":coverage,"classifications":classifications,
            "noise_operating_characteristics":noise_rows,"epsilon_ST2_power":epsilon_rows,
            "finite_bank_convergence":convergence,
            "time_step_convergence":b.time_step_convergence(),
            "response_cosines":b.response_cosines(),
            "scope":"indicative semi-numerical benchmark; not a survey forecast or Boltzmann validation"}
    RESULTS.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    plot_classification(output, italian=False); plot_classification(output, italian=True)
    plot_diagnostics(output, italian=False); plot_diagnostics(output, italian=True)
    print(json.dumps(output,indent=2))


def plot_classification(output, italian=False):
    rows=output["classifications"]; labels=list(rows)
    cats=["null","deferred","ST1/ST2 pass","ST2 fail","ST1 fail"]
    colors=["#6c757d","#8e7cc3","#0f8b7e","#c46a00","#bd1e24"]
    display=(["nullo","rinviato","supera ST1/ST2","fallisce ST2","fallisce ST1"] if italian else cats)
    bottom=np.zeros(len(labels)); fig,ax=plt.subplots(figsize=(9.1,4.8))
    for cat,color,name in zip(cats,colors,display):
        vals=np.array([rows[x]["fractions"].get(cat,0.) for x in labels])
        ax.bar(range(len(labels)),vals,bottom=bottom,color=color,label=name); bottom+=vals
    xlabels=(["GR/nullo","scala fisica\ncostante","scala evolutiva","due scale\npositive risolte","due scale\npositive compresse"] if italian else
             ["GR/null","constant\nphysical scale","evolving\nscale","resolved positive\ntwo-scale","compressed positive\ntwo-scale"])
    ax.set_xticks(range(len(labels)),xlabels); ax.set_ylim(0,1); ax.grid(axis="y",alpha=.25)
    ax.set_ylabel("Frazione di classificazione recuperata" if italian else "Recovered classification fraction")
    ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.23)); fig.tight_layout()
    path=FIG_ITA if italian else FIG; path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)


def plot_diagnostics(output, italian=False):
    noise=output["noise_operating_characteristics"]; eps=output["epsilon_ST2_power"]; conv=output["finite_bank_convergence"]
    fig,axs=plt.subplots(1,3,figsize=(12,3.65))
    axs[0].plot([r["noise_multiplier"] for r in noise],[r["detection_power"] for r in noise],"o-",color="#0f8b7e")
    axs[0].axhline(POWER_TARGET,color="#8e7cc3",ls="--",lw=1); axs[0].set_xscale("log",base=2); axs[0].set_ylim(0,1.05)
    axs[0].set_xlabel("Moltiplicatore del rumore" if italian else "Noise multiplier"); axs[0].set_ylabel("Potenza di rivelazione" if italian else "Detection power")
    axs[1].plot([r["epsilon"] for r in eps],[r["ST2_conditional_power"] for r in eps],"o-",color="#c46a00")
    axs[1].axhline(POWER_TARGET,color="#8e7cc3",ls="--",lw=1); axs[1].set_ylim(0,1.05); axs[1].set_xlabel(r"$\epsilon$ in $q\propto a^{1+\epsilon}$")
    axs[1].set_ylabel("Potenza condizionale ST2" if italian else "Conditional ST2 power")
    axs[2].plot(range(len(conv)),[r["positive_two_scale_ST1_fail"] for r in conv],"o-",color="#bd1e24")
    axs[2].set_xticks(range(len(conv)),[f"{r['n_causal']}/{r['n_k_basis']}" for r in conv]); axs[2].set_ylim(0,1.05)
    axs[2].set_xlabel("Storie causali / base $k$" if italian else "Causal histories / k basis")
    axs[2].set_ylabel("Frazione ST1 fail" if italian else "ST1-fail fraction")
    for ax in axs: ax.grid(alpha=.25)
    fig.tight_layout(); path=DIAG_ITA if italian else DIAG; path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
