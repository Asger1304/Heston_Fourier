import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
"""
Benchmark: Germano analytical-gradient vs finite-difference least squares.

For each trial:
  1. Draw a random Heston parameter set (Feller-satisfying) via heston_param_generator.
  2. Generate synthetic option prices on a fixed strikes × maturities grid.
  3. Calibrate with both methods from the same random starting point.
  4. Record wall time, function evaluations (nfev), gradient evaluations (njev), and final cost.

Usage:
    python benchmark_calibration.py            # 20 trials, default grid
    python benchmark_calibration.py --n 50     # more trials
"""

import argparse
import time
import numpy as np
import pandas as pd

from src.calibration.heston_calibration import HestonCalibrator, HestonParams, heston_param_generator
from src.pricing.fourier_pricing import compute_lewis_strikes

# ###########################################################################
# market grid
# ###########################################################################
S0    = 100.0
R     = 0.03
Q     = 0.01
STRIKES     = [60, 70, 80, 90, 100, 110, 120, 130, 140]
MATURITIES  = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0]   # years


NOISE_STD = 0.01  # relative noise level applied to each price

def generate_target(params: HestonParams, seed=None) -> pd.DataFrame:
    """Synthetic option prices on the fixed grid with multiplicative noise."""
    cal = HestonCalibrator(S0=S0, r=R, q=Q)
    cal.fit(params)
    rng = np.random.default_rng(seed)
    rows = []
    for T in MATURITIES:
        cf     = cal.create_cf(T)
        prices = compute_lewis_strikes(cf, S0, T, STRIKES, R, Q)
        for K, price in zip(STRIKES, prices ):
            rows.append({"K": K, "T": T, "r": R, "target": price})
    return pd.DataFrame(rows)


# ###########################################################################
# Single trial
# ###########################################################################
def run_trial(true_params: HestonParams, x0: HestonParams, trial_idx: int = 0):
    """function to run a simple trial
    """
    target_df = generate_target(true_params, seed=trial_idx)

    fitted_params = {}
    results = {}
    for method, label in [
        ("quad",         "GK (Num Grad)"),
        ("quad_germano", "GK (Anal Grad)"),
        ("ls",           "Trapezoid (Num Grad)"),
        ("germano",      "Trapezoid (Anal Grad)"),
    ]:
        cal = HestonCalibrator(S0=S0, r=R, q=Q)
        t0  = time.perf_counter()
        if method == "quad":
            params, res = cal.fit_least_squares_quad(target_df, x0=x0, verbose=0)
        elif method == "quad_germano":
            params, res = cal.fit_germano_quad(target_df, x0=x0, verbose=0)
        elif method == "ls":
            params, res = cal.fit_least_squares(target_df, x0=x0, verbose=0)
        else:
            params, res = cal.fit_germano(target_df, x0=x0, verbose=0)
        elapsed = time.perf_counter() - t0
        fitted_params[label] = params.to_array()
        results[label] = {
            "time_s": elapsed,
            "cost":   res.cost,
        }

    ref = fitted_params["GK (Num Grad)"]
    for label in fitted_params:
        results[label]["param_dist"] = np.sum(np.abs(fitted_params[label] - ref))
    results["GK (Num Grad)"]["param_dist"] = 0.0

    return results


####################################################################################################
# Main
####################################################################################################
def experiment(n=None, seed=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=20, help="number of trials")
    parser.add_argument("--seed", type=int, default=42,  help="RNG seed for reproducibility")
    args, _ = parser.parse_known_args()
    if n    is not None: args.n    = n
    if seed is not None: args.seed = seed

    true_gen  = heston_param_generator(seed=args.seed)
    start_gen = heston_param_generator(seed=args.seed + 1000)

    all_results: dict[str, list] = {}

    print(f"Strikes    : {STRIKES}")
    print(f"Maturities : {[round(t, 4) for t in MATURITIES]}")
    print(f"Grid       : {len(STRIKES)} × {len(MATURITIES)} = {len(STRIKES)*len(MATURITIES)} options per trial")
    ORDERED = ["GK (Num Grad)", "GK (Anal Grad)", "Trapezoid (Num Grad)", "Trapezoid (Anal Grad)"]
    SHORT   = ["GK FD",         "GK AD",          "Trap FD",              "Trap AD"]
    T_W, D_W, C_W = 8, 8, 10
    BLOCK = T_W + 3 + D_W + 3 + C_W

    def _dist(d): return f"{'---':>{D_W}}" if d == 0.0 else f"{d:{D_W}.4f}"

    h1  = f"{'':5}" + "".join(f" | {n:^{BLOCK}}" for n in SHORT)
    h2  = f"{'Trial':>5}" + (f" | {'time(s)':>{T_W}} | {'Σ|Δθ|':>{D_W}} | {'cost':>{C_W}}") * len(SHORT)
    sep = "-" * 5 + (f"+{'-'*(T_W+2)}+{'-'*(D_W+2)}+{'-'*(C_W+2)}") * len(SHORT)

    print(f"S0={S0},  r={R},  q={Q},  noise={NOISE_STD*100:.0f}%,  trials={args.n}\n")
    print(h1)
    print(h2)
    print(sep)

    for i in range(args.n):
        true_params = next(true_gen)
        x0          = next(start_gen)

        trial = run_trial(true_params, x0, trial_idx=i)
        for label, stats in trial.items():
            if label not in all_results:
                all_results[label] = []
            all_results[label].append(stats)

        row = f"{i+1:>5}"
        for label in ORDERED:
            s = trial[label]
            row += f" | {s['time_s']:{T_W}.3f} | {_dist(s['param_dist'])} | {s['cost']:{C_W}.4f}"
        print(row)


    # Speedup relative to GK (Num Grad) baseline
    t_ref = np.mean([s["time_s"] for s in all_results["GK (Num Grad)"]])
    for label, stats_list in all_results.items():
        if label == "GK (Num Grad)":
            continue
        t = np.mean([s["time_s"] for s in stats_list])
        print(f"\nSpeedup ({label} vs GK (Num Grad)): {t_ref / t:.2f}×")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=20, help="number of trials")
    parser.add_argument("--seed", type=int, default=42,  help="Seed")
    experiment(parser)
