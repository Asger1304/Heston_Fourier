import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import least_squares


def frac_of_year(t_0, t):
    """get the fraction if the year, we dont account for leap years"""
    delta_days = (t - t_0).days
    return delta_days / 365


def clean_options_df(options_df, min_price=0.1, max_rel_spread=0.5):
    """apply first part of data cleaning steps"""
    df = options_df.copy()
    df["exdate"] = pd.to_datetime(df["exdate"], format="%d.%m.%Y").dt.date
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y").dt.date

    df = df[df["best_bid"] > 0]
    df = df[df["best_offer"] > df["best_bid"]]
    df = df[df["Mid"] >= min_price]

    spread = (df["best_offer"] - df["best_bid"]) / df["Mid"]
    df = df[spread <= max_rel_spread]

    return df.reset_index(drop=True)


def apply_arbitrage_filters(
    options_df,
    s0,
    r_series,
    q,
    t_0,
    moneyness_cap=0.3,
    t_min=7 / 365,
    t_max=2.0,
):
    """ apply arbitrage filters
    """
    df = options_df.copy()
    df["T"] = df["exdate"].map(lambda d: frac_of_year(t_0, d))
    df = df[(df["T"] >= t_min) & (df["T"] <= t_max)]

    df["r"] = df["exdate"].map(r_series).astype(float)
    df = df.dropna(subset=["r"])

    F = s0 * np.exp((df["r"] - q) * df["T"])
    df["log_moneyness"] = np.log(df["Strike"] / F)
    df = df[df["log_moneyness"].abs() <= moneyness_cap]

    disc_K = df["Strike"] * np.exp(-df["r"] * df["T"])
    disc_S = s0 * np.exp(-q * df["T"])

    is_call = df["cp_flag"] == "C"
    lower = np.where(is_call, np.maximum(disc_S - disc_K, 0.0), np.maximum(disc_K - disc_S, 0.0))
    upper = np.where(is_call, disc_S, disc_K)
    df = df[(df["Mid"] >= lower) & (df["Mid"] <= upper)]

    return df.reset_index(drop=True)


def build_call_target(calls_df, puts_df, s0, q):
    """we build the"""
    calls = calls_df.copy()
    calls["F"] = s0 * np.exp((calls["r"] - q) * calls["T"])
    otm_calls = calls.loc[calls["Strike"] >= calls["F"]].copy()
    otm_calls["target"] = otm_calls["Mid"]
    otm_calls["source"] = "C"

    puts = puts_df.copy()
    puts["F"] = s0 * np.exp((puts["r"] - q) * puts["T"])
    otm_puts = puts.loc[puts["Strike"] < puts["F"]].copy()
    otm_puts["target"] = (
        otm_puts["Mid"]
        + s0 * np.exp(-q * otm_puts["T"])
        - otm_puts["Strike"] * np.exp(-otm_puts["r"] * otm_puts["T"])
    )
    otm_puts["source"] = "P"

    cols = ["Strike", "T", "r", "target", "source"]
    out = pd.concat([otm_calls[cols], otm_puts[cols]], ignore_index=True)
    return (
        out.rename(columns={"Strike": "K"})
        .sort_values(["T", "K"])
        .drop_duplicates(subset=["K", "T"])
        .reset_index(drop=True)
    )


def generate_rq_df(put_df: pd.DataFrame, call_df: pd.DataFrame, s0, t_0: dt.date):
    def residuals(x, t_0, synths: dict, s0):
        q = x[-1]
        r = x[:-1]
        reses = []
        for i, (t, syn) in enumerate(synths.items()):
            tau = frac_of_year(t_0=t_0, t=t)
            res = syn.values - syn.index.values * np.exp(-r[i] * tau) + s0 * np.exp(-q * tau)
            reses.append(res)
        return np.concatenate(reses)

    synths = {}
    for date, p_row in put_df.iterrows():
        if date not in call_df.index:
            continue
        synth = p_row - call_df.iloc[call_df.index.get_loc(date)]
        synth = synth.dropna()
        if len(synth) == 0:
            continue
        synths[date] = synth

    x0 = np.zeros(len(synths) + 1)
    res = least_squares(residuals, x0, args=(t_0, synths, s0), bounds=(0, np.inf))
    fitted_x = res.x

    dates = list(synths.keys())
    r = pd.Series(fitted_x[:-1], dates)
    q = fitted_x[-1]
    return r, q


def check_dropped_maturities(options_df, r_series, filtered_df=None, stage_label="after arb filters"):
    all_maturities = set(options_df["exdate"].unique())
    covered = set(r_series.index)
    dropped_rq = all_maturities - covered
    print(f"Total maturities in cleaned data : {len(all_maturities)}")
    print(f"With r/q estimate (paired put+call): {len(covered)}")
    if dropped_rq:
        print(f"Dropped at r/q stage ({len(dropped_rq)}):")
        for d in sorted(dropped_rq):
            n = (options_df["exdate"] == d).sum()
            print(f"  {d}  ({n} options)")
    else:
        print(f"Dropped at r/q stage: 0")

    if filtered_df is not None:
        surviving = set(filtered_df["exdate"].unique()) if "exdate" in filtered_df.columns else set()
        dropped_filter = covered - surviving
        print(f"Surviving {stage_label}: {len(surviving)}")
        if dropped_filter:
            print(f"Dropped {stage_label} ({len(dropped_filter)}):")
            for d in sorted(dropped_filter):
                print(f"  {d}")
        else:
            print(f"Dropped {stage_label}: 0")


if __name__=="__main__":
    t_0 = dt.date(year=2019, month=11, day=13)
    s0  = 3094

    t_df = pd.read_csv('data/optionsdata.csv', sep=';', decimal=',')
    t_df = clean_options_df(t_df)

    calls = t_df[t_df["cp_flag"] == "C"]
    puts  = t_df[t_df["cp_flag"] == "P"]

    put_pivot  = puts.pivot_table(index="exdate", columns="Strike", values="Mid", aggfunc="first")
    call_pivot = calls.pivot_table(index="exdate", columns="Strike", values="Mid", aggfunc="first")

    r_series, q = generate_rq_df(put_pivot, call_pivot, s0, t_0)
    print(f"q = {q:.4f}\n")

    calls_f = apply_arbitrage_filters(calls, s0, r_series, q, t_0)
    puts_f  = apply_arbitrage_filters(puts,  s0, r_series, q, t_0)
    filtered = pd.concat([calls_f, puts_f])

    check_dropped_maturities(t_df, r_series, filtered_df=filtered)