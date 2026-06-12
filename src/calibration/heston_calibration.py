import numpy as np
from scipy.optimize import least_squares

from src.pricing.fourier_pricing import (
    compute_lewis, compute_lewis_strikes,
    compute_lewis_prices_and_gradient,
    compute_lewis_strikes_quad, compute_lewis_prices_and_gradient_quad,
)

_DEFAULT_BOUNDS = (
    [1e-4, 0.1, 1e-4, 1e-3, -0.99],
    [1.0, 20.0, 1.0, 2.0, 0.0],
)
# h order [v0,theta,rho,kappa,sigma] → HestonParams.to_array() order [v0,kappa,theta,sigma,rho]
_H_TO_ARR = [0, 3, 1, 4, 2]


class HestonParams:
    """utils"""
    def __init__(self, v0, kappa, theta, sigma, rho):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def __str__(self):
        return f"v0: {self.v0}, kappa: {self.kappa}, theta: {self.theta}, sigma: {self.sigma}, rho: {self.rho}"

    def to_array(self):
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    @classmethod
    def from_array(cls, x):
        return cls(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])


class HestonCalibrator:
    """Primary calibrator object"""
    def __init__(self, S0, r=0.0, q=0.0):
        self.S0 = S0
        self.r = r
        self.q = q

    def fit(self, params: HestonParams):
        self.v0 = params.v0
        self.kappa = params.kappa
        self.theta = params.theta
        self.sigma = params.sigma
        self.rho = params.rho

    #CF's

    def create_cf(self, t, r=None):
        if r is None:
            r = self.r
        x0, drift = np.log(self.S0), r - self.q

        def phi(u):
            u = np.asarray(u, dtype=complex)
            d = np.sqrt((self.rho * self.sigma * 1j * u - self.kappa)**2 + self.sigma**2 * (u**2 + 1j * u))
            g = (self.kappa - self.rho * self.sigma * 1j * u - d) / (self.kappa - self.rho * self.sigma * 1j * u + d)
            C = (
                1j * u * (x0 + drift * t)
                + (self.kappa * self.theta / self.sigma**2)
                * ((self.kappa - self.rho * self.sigma * 1j * u - d) * t
                   - 2.0 * np.log((1.0 - g * np.exp(-d * t)) / (1.0 - g)))
            )
            D = ((self.kappa - self.rho * self.sigma * 1j * u - d) / self.sigma**2
                 * ((1.0 - np.exp(-d * t)) / (1.0 - g * np.exp(-d * t))))
            return np.exp(C + D * self.v0)

        return phi

    def create_germano_cf(self, t, r=None):
        """Germano et al. (2017) Eq. 18 — sinh/cosh form, numerically continuous."""
        if r is None:
            r = self.r
        kappa, theta, sigma, rho, v0 = self.kappa, self.theta, self.sigma, self.rho, self.v0
        x0, drift = np.log(self.S0), r - self.q

        def phi(u):
            u  = np.asarray(u, dtype=complex)
            fi = kappa - 1j * sigma * rho * u
            d  = np.sqrt(fi**2 + sigma**2 * (u**2 + 1j * u))
            d  = np.where(np.real(d) < 0, -d, d)
            s, c = np.sinh(d * t / 2), np.cosh(d * t / 2)
            A_1 = (u**2 + 1j * u) * s
            A_2 = d / v0 * c + fi / v0 * s
            A   = A_1 / A_2
            D   = (np.log(d / v0) + (kappa - d) * t / 2
                   - np.log((d + fi) / (2 * v0) + (d - fi) / (2 * v0) * np.exp(-d * t)))
            return np.exp(
                1j * u * (x0 + drift * t)
                - kappa * theta * rho * t * 1j * u / sigma
                - A + 2 * kappa * theta / sigma**2 * D
            )
        return phi

    def create_germano_h(self, t, r=None):
        """Analytical gradient h(u) s.t. ∇φ = φ·h. Theorem 1 / Eqs. 23–30."""
        if r is None:
            r = self.r
        kappa, theta, sigma, rho, v0 = self.kappa, self.theta, self.sigma, self.rho, self.v0

        def h(u):
            u  = np.asarray(u, dtype=complex)
            fi = kappa - 1j * sigma * rho * u
            d  = np.sqrt(fi**2 + sigma**2 * (u**2 + 1j * u))
            d  = np.where(np.real(d) < 0, -d, d)

            x = d * t / 2
            s, c = (np.exp(x) - np.exp(-x)) / 2, (np.exp(x) + np.exp(-x)) / 2
            A_1 = (u**2 + 1j * u) * s
            A_2 = d / v0 * c + fi / v0 * s
            A   = A_1 / A_2
            B   = d * np.exp(kappa * t / 2) / (v0 * A_2)
            D   = (np.log(d / v0) + (kappa - d) * t / 2
                   - np.log((d + fi) / (2 * v0) + (d - fi) / (2 * v0) * np.exp(-d * t)))

            dd_drho   = -fi * sigma * 1j * u / d
            dA1_drho  = -1j * u * (u**2 + 1j * u) * t * fi * sigma / (2 * d) * c
            dA2_drho  = -sigma * 1j * u * (2 + fi * t) / (2 * d * v0) * (fi * c + d * s)
            dA_drho   = dA1_drho / A_2 - A / A_2 * dA2_drho
            dB_drho   = B / d * dd_drho - B / A_2 * dA2_drho
            dB_dkappa = 1j / (sigma * u) * dB_drho + B * t / 2

            dfi_dsigma = -1j * rho * u
            dd_dsigma  = (fi * dfi_dsigma + sigma * (u**2 + 1j * u)) / d
            dA1_dsigma = (u**2 + 1j * u) * t / 2 * dd_dsigma * c
            dA2_dsigma = (dd_dsigma * (c + (fi * c + d * s) * t / 2) + dfi_dsigma * s) / v0
            dA_dsigma  = dA1_dsigma / A_2 - A / A_2 * dA2_dsigma

            h_1 = -A / v0
            h_2 = 2 * kappa / sigma**2 * D - kappa * rho * t * 1j * u / sigma
            h_3 = (-dA_drho
                   + 2 * kappa * theta / (sigma**2 * d) * (dd_drho - d / A_2 * dA2_drho)
                   - kappa * theta * t * 1j * u / sigma)
            h_4 = (1 / (sigma * 1j * u) * dA_drho
                   + 2 * theta / sigma**2 * D
                   + 2 * kappa * theta / (sigma**2 * B) * dB_dkappa
                   - theta * rho * t * 1j * u / sigma)
            h_5 = (-dA_dsigma
                   - 4 * kappa * theta / sigma**3 * D
                   + 2 * kappa * theta / (sigma**2 * d) * (dd_dsigma - d / A_2 * dA2_dsigma)
                   + kappa * theta * rho * t * 1j * u / sigma**2)

            return np.stack([h_1, h_2, h_3, h_4, h_5])
        return h

    #Pricing

    def compute_lewis_heston(self, strike, t, r=None):
        if r is None:
            r = self.r
        return compute_lewis(self.create_cf(t, r), self.S0, t, strike, r, self.q)

    def compute_prices(self, df):
        has_r = "r" in df.columns
        group_cols = ["T", "r"] if has_r else ["T"]
        df["prediction"] = np.nan
        for keys, sub in df.groupby(group_cols, sort=False):
            T, r = (keys if has_r else (keys, self.r))
            cf = self.create_cf(T, r)
            df.loc[sub.index, "prediction"] = compute_lewis_strikes_quad(cf, self.S0, T, sub["K"].values, r, self.q)
        return df

    #Helpers

    @staticmethod
    def _prep_weights(weights, target_df):
        """lets us use a general weight, could be a function or a vector"""
        if weights is None:
            return None
        return np.asarray(weights(target_df) if callable(weights) else weights, dtype=float)

    @staticmethod
    def _prep_x0(x0):
        """so called, start up check"""
        if x0 is None:
            x0 = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.4, rho=-0.7)
        return x0.to_array() if isinstance(x0, HestonParams) else x0

    @staticmethod
    def _group_positions(target_df):
        """grouping maturities and their risk free rates when applicable"""
        has_r = "r" in target_df.columns
        group_cols = ["T", "r"] if has_r else ["T"]
        groups    = [(keys, sub) for keys, sub in target_df.groupby(group_cols, sort=False)]
        positions = [target_df.index.get_indexer(sub.index) for _, sub in groups]
        return groups, positions, has_r


    #fitting with the analytical gradient
    def _germano_fit(self, pricing_fn, target_df, x0, bounds, weights, verbose):
        """Shared core for fit_germano / fit_germano_quad."""
        #prep
        x0 = self._prep_x0(x0)
        if bounds is None:
            bounds = _DEFAULT_BOUNDS
        w = self._prep_weights(weights, target_df)

        target = target_df["target"].values
        groups, positions, has_r = self._group_positions(target_df)
        _cache = {"x": None, "prices": None, "grad": None}

        def _recompute(x):
            self.fit(HestonParams.from_array(x))
            prices = np.empty(len(target_df))
            grad   = np.empty((len(target_df), 5))
            for (keys, sub), pos in zip(groups, positions):
                T, r = (keys if has_r else (keys, self.r))
                p, J = pricing_fn(self.create_germano_cf(T, r), self.create_germano_h(T, r),
                                  self.S0, T, sub["K"].values, r, self.q)
                prices[pos] = p
                grad[pos]   = J[:, _H_TO_ARR]
            _cache.update(x=x.copy(), prices=prices, grad=grad)
        #caching lets us compute prices and gradiant at the same time
        def residuals(x):
            if _cache["x"] is None or not np.array_equal(x, _cache["x"]):
                _recompute(x)
            target_df["prediction"] = _cache["prices"]
            diff = target - _cache["prices"]
            return diff if w is None else w * diff
        

        def gradient(x):
            if _cache["x"] is None or not np.array_equal(x, _cache["x"]):
                _recompute(x)
            return -(w[:, None] * _cache["grad"]) if w is not None else -_cache["grad"]

        res    = least_squares(residuals, x0, jac=gradient, bounds=bounds, method="trf", verbose=verbose)
        fitted = HestonParams.from_array(res.x)
        self.fit(fitted)
        return fitted, res

    #Single-start fitting

    def fit_least_squares(self, target_df, x0=None, bounds=None, weights=None, verbose=1):
        x0 = self._prep_x0(x0)
        if bounds is None:
            bounds = _DEFAULT_BOUNDS
        w      = self._prep_weights(weights, target_df)
        target = target_df["target"].values
        groups, positions, has_r = self._group_positions(target_df)

        def residuals(x):
            self.fit(HestonParams.from_array(x))
            prices = np.empty(len(target_df))
            for (keys, sub), pos in zip(groups, positions):
                T, r = (keys if has_r else (keys, self.r))
                prices[pos] = compute_lewis_strikes(self.create_cf(T, r), self.S0, T, sub["K"].values, r, self.q)
            target_df["prediction"] = prices
            diff = target - prices
            return diff if w is None else w * diff

        res    = least_squares(residuals, x0, bounds=bounds, method="trf", verbose=verbose)
        fitted = HestonParams.from_array(res.x)
        self.fit(fitted)
        return fitted, res

    def fit_least_squares_quad(self, target_df, x0=None, bounds=None, weights=None, verbose=1):
        """Same as fit_least_squares but prices each strike with Gauss-Kronrod quad."""
        x0 = self._prep_x0(x0)
        if bounds is None:
            bounds = _DEFAULT_BOUNDS
        w      = self._prep_weights(weights, target_df)
        target = target_df["target"].values
        groups, positions, has_r = self._group_positions(target_df)

        def residuals(x):
            self.fit(HestonParams.from_array(x))
            prices = np.empty(len(target_df))
            for (keys, sub), pos in zip(groups, positions):
                T, r = (keys if has_r else (keys, self.r))
                prices[pos] = compute_lewis_strikes_quad(self.create_cf(T, r), self.S0, T, sub["K"].values, r, self.q)
            target_df["prediction"] = prices
            diff = target - prices
            return diff if w is None else w * diff

        res    = least_squares(residuals, x0, bounds=bounds, method="trf", verbose=verbose)
        fitted = HestonParams.from_array(res.x)
        self.fit(fitted)
        return fitted, res

    def fit_germano(self, target_df, x0=None, bounds=None, weights=None, verbose=1):
        """LM with analytical gradient (trapezoid integration)."""
        return self._germano_fit(compute_lewis_prices_and_gradient, target_df, x0, bounds, weights, verbose)

    def fit_germano_quad(self, target_df, x0=None, bounds=None, weights=None, verbose=1):
        """LM with analytical gradient and Gauss-Kronrod integration."""
        return self._germano_fit(compute_lewis_prices_and_gradient_quad, target_df, x0, bounds, weights, verbose)

    #Multi-start fitting

    def _fit_multi_start(self, fit_fn, target_df, n_starts, bounds, weights, seed, verbose):
        """We create a way to a fit with multiple random starts in addition to the default one"""
        gen = heston_param_generator(seed=seed)
        best_params, best_res = None, None
        for i in range(n_starts):
            x0 = next(gen)
            try:
                params, res = fit_fn(target_df, x0=x0, bounds=bounds, weights=weights, verbose=verbose)
            except Exception as e:
                print(f"  start {i + 1:>2}/{n_starts}: failed ({e})")
                continue
            print(f"  start {i + 1:>2}/{n_starts}: cost={res.cost:.4f}  {params}")
            if best_res is None or res.cost < best_res.cost:
                best_params, best_res = params, res
        self.fit(best_params)
        return best_params, best_res

    def fit_multi_start(self, target_df, n_starts=10, bounds=None, weights=None, seed=None, verbose=0):
        return self._fit_multi_start(self.fit_least_squares, target_df, n_starts, bounds, weights, seed, verbose)

    def fit_multi_start_quad(self, target_df, n_starts=10, bounds=None, weights=None, seed=None, verbose=0):
        return self._fit_multi_start(self.fit_least_squares_quad, target_df, n_starts, bounds, weights, seed, verbose)

    def fit_multi_start_germano(self, target_df, n_starts=10, bounds=None, weights=None, seed=None, verbose=0):
        return self._fit_multi_start(self.fit_germano, target_df, n_starts, bounds, weights, seed, verbose)

    def fit_multi_start_germano_quad(self, target_df, n_starts=10, bounds=None, weights=None, seed=None, verbose=0):
        return self._fit_multi_start(self.fit_germano_quad, target_df, n_starts, bounds, weights, seed, verbose)

    # --- Visualisation ---

    def make_plot(self, df):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        df = df.sort_values(["T", "K"])
        K_vals         = df["K"].unique()
        T_vals         = df["T"].unique()
        target_surface = df.pivot(index="T", columns="K", values="target").values
        pred_surface   = df.pivot(index="T", columns="K", values="prediction").values

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "surface"}, {"type": "surface"}]],
            subplot_titles=("Target Surface", "Predicted Surface"),
        )
        fig.add_trace(go.Surface(x=K_vals, y=T_vals, z=target_surface), row=1, col=1)
        fig.add_trace(go.Surface(x=K_vals, y=T_vals, z=pred_surface),   row=1, col=2)
        fig.update_layout(
            title="Implied Volatility Surfaces",
            scene =dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="Price"),
            scene2=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="Price"),
        )
        fig.show()



#Utilities


def heston_param_generator(seed=None):
    if seed is not None:
        np.random.seed(seed)
    while True:
        v0    = np.random.uniform(0.05, 0.95)
        vbar  = np.random.uniform(0.05, 0.95)
        kappa = np.random.uniform(0.5, 5.0)
        sigma = np.random.uniform(0.5, 0.95)
        rho   = np.random.uniform(-0.95, -0.1)
        if 2 * kappa * vbar > sigma**2:
            yield HestonParams(v0=v0, kappa=kappa, theta=vbar, sigma=sigma, rho=rho)



