import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import datetime as dt
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.data_utils import clean_options_df, apply_arbitrage_filters, build_call_target, generate_rq_df
from src.pricing.fourier_pricing import implied_vol
from src.calibration.heston_calibration import HestonCalibrator


def load_data(path="data/optionsdata.csv"):
    raw = pd.read_csv(path, sep=";", decimal=",")
    cleaned = clean_options_df(raw)
    calls = cleaned[cleaned["cp_flag"] == "C"]
    puts = cleaned[cleaned["cp_flag"] == "P"]
    return calls, puts


def fit_rates(puts, calls, s0, t_0):
    flipped_put = puts.pivot_table(index="exdate", columns="Strike", values="Mid", aggfunc="first")
    flipped_call = calls.pivot_table(index="exdate", columns="Strike", values="Mid", aggfunc="first")
    r_series, q = generate_rq_df(flipped_put, flipped_call, s0, t_0)
    print(f"fitted q = {q:.4f},  r range = [{r_series.min():.4f}, {r_series.max():.4f}]")
    return r_series, q


def build_target(calls, puts, s0, r_series, q, t_0):
    calls_f = apply_arbitrage_filters(calls, s0, r_series, q, t_0)
    puts_f = apply_arbitrage_filters(puts, s0, r_series, q, t_0)
    target_df = build_call_target(calls_f, puts_f, s0, q)
    n_call = (target_df["source"] == "C").sum()
    n_put = (target_df["source"] == "P").sum()
    print(f"calibration set: {len(target_df)} options ({n_call} OTM calls + {n_put} OTM puts via parity) across {target_df['T'].nunique()} maturities")
    return target_df


_RUN_COLORS = {
    "unweighted":         "Reds",
    "relative (% error)": "Greens",
}


def run_calibrations(target_df, s0, q, runs):
    fitted_by_run = {}
    for label, weights in runs.items():
        calibrator = HestonCalibrator(S0=s0, q=q)
        print(f"\ncalibration — {label}:")
        t0 = time.perf_counter()
        fitted, res = calibrator.fit_least_squares_quad(target_df, weights=weights)
        elapsed = time.perf_counter() - t0
        print(f"  params: {fitted}")
        print(f"  cost:   {res.cost:.4f}  time: {elapsed:.2f}s")
        calibrator.compute_prices(target_df)
        fitted_by_run[label] = (calibrator, fitted, res, target_df[["K", "T", "r", "target", "prediction"]].copy())
    return fitted_by_run


def compute_iv_surfaces(target_df, fitted_by_run, s0, q):
    target_df["target_iv"] = target_df.apply(
        lambda row: implied_vol(row["target"], s0, row["K"], row["r"], row["T"], q),
        axis=1,
    )
    iv_df = target_df.dropna(subset=["target_iv"]).copy()
    iv_df["log_moneyness"] = np.log(iv_df["K"] / s0)

    for label, (_, _, _, df_run) in fitted_by_run.items():
        col_name = f"pred_iv__{label}"
        df_run[col_name] = df_run.apply(
            lambda row: implied_vol(row["prediction"], s0, row["K"], row["r"], row["T"], q),
            axis=1,
        )
        iv_df = iv_df.merge(df_run[["K", "T", col_name]], on=["K", "T"], how="left")

    return iv_df


def plot_iv_surfaces(iv_df, runs):
    def _surface(df, values_col, axis_col):
        s = df.pivot_table(index="T", columns=axis_col, values=values_col, aggfunc="mean")
        return s.interpolate(axis=1, limit_area="inside").interpolate(axis=0, limit_area="inside")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("IV surface vs Strike", "IV surface vs Log Moneyness"),
    )

    def _add_all(axis_col, col):
        market = _surface(iv_df, "target_iv", axis_col)
        fig.add_trace(go.Surface(
            x=market.columns.values, y=market.index.values, z=market.values,
            name="Market", colorscale="Blues", opacity=0.85, showscale=False,
            legendgroup="market", showlegend=(col == 1),
        ), row=1, col=col)
        for label in runs.keys():
            colorscale = _RUN_COLORS.get(label, "Reds")
            surf = _surface(iv_df, f"pred_iv__{label}", axis_col)
            fig.add_trace(go.Surface(
                x=surf.columns.values, y=surf.index.values, z=surf.values,
                name=f"Heston ({label})", colorscale=colorscale, opacity=0.55, showscale=False,
                legendgroup=label, showlegend=(col == 1),
            ), row=1, col=col)

    _add_all("K", col=1)
    _add_all("log_moneyness", col=2)

    fig.update_layout(
        title="Heston calibration — market vs fitted IV surfaces",
        scene=dict(xaxis_title="Strike", yaxis_title="Maturity (yrs)", zaxis_title="Implied vol"),
        scene2=dict(xaxis_title="log(K/S₀)", yaxis_title="Maturity (yrs)", zaxis_title="Implied vol"),
        height=750,
        width=1500,
    )
    fig.show()


def main():
    t_0 = dt.date(2019, 11, 13)
    s0 = 3094

    calls, puts = load_data()
    r_series, q = fit_rates(puts, calls, s0, t_0)
    target_df = build_target(calls, puts, s0, r_series, q, t_0)

    runs = {
        "unweighted":         None,
        "relative (% error)": lambda df: 1.0 / df["target"].values,
    }

    all_fitted = run_calibrations(target_df, s0, q, runs)

    iv_df = compute_iv_surfaces(target_df, all_fitted, s0, q)
    plot_iv_surfaces(iv_df, all_fitted)


if __name__ == "__main__":
    main()
