"""A model for actually holding a Heston model, this is made primarily for simulation """

import numpy as np
from src.pricing.fourier_pricing import compute_lewis, compute_lewis_strikes_quad


class HestonModel:
    def __init__(self, S0, v0, r,q, kappa, theta, sigma, rho, T, N, n_paths, seed=None):
        """initialize the model with parameters"""
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.q=q
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.T = T
        self.N = N
        self.n_paths = n_paths
        self.seed = seed

    def create_cf(self):
        """default heston CF for pricing VS simulation"""
        x0 = np.log(self.S0)
        r = self.r
        T = self.T
        q=self.q

        def phi(u):
            u = np.asarray(u, dtype=complex)
            d = np.sqrt((self.rho * self.sigma * 1j * u - self.kappa)**2 + self.sigma**2 * (u**2 + 1j * u))
            g = (self.kappa - self.rho * self.sigma * 1j * u - d) / (self.kappa - self.rho * self.sigma * 1j * u + d)
            C = (
                1j * u * (x0 + (r-q) * T)
                + (self.kappa * self.theta / self.sigma**2)
                * (
                    (self.kappa - self.rho * self.sigma * 1j * u - d) * T
                    - 2.0 * np.log((1.0 - g * np.exp(-d * T)) / (1.0 - g))
                )
            )
            D = (
                (self.kappa - self.rho * self.sigma * 1j * u - d) / self.sigma**2
                * ((1.0 - np.exp(-d * T)) / (1.0 - g * np.exp(-d * T)))
            )
            return np.exp(C + D * self.v0)

        return phi
    


    

    def create_simulation(self):
        """creating a heston model simulation, we use the a milstein corrected euler scheme"""
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.N
        sqrt_dt = np.sqrt(dt)
        half_dt = 0.5 * dt
        rho_perp = np.sqrt(1.0 - self.rho**2)
        kappa_dt = self.kappa * dt
        sigma_sqrt_dt = self.sigma * sqrt_dt
        sigma2_4_dt = 0.25 * self.sigma**2 * dt
        r_q_dt = (self.r - self.q) * dt


        X = np.empty((self.N + 1, self.n_paths))
        v = np.empty((self.N + 1, self.n_paths))
        X[0] = np.log(self.S0)
        v[0] = self.v0

        from tqdm import tqdm
        for n in tqdm(range(self.N)):
            z1 = rng.standard_normal(self.n_paths)
            z2 = rng.standard_normal(self.n_paths)
            eps_s = self.rho * z1 + rho_perp * z2
            v_pos = np.maximum(v[n], 0.0) #we force the variance process to be non negative
            sqrt_v_pos = np.sqrt(v_pos)
            v[n + 1] = np.maximum(
                v[n] + kappa_dt * (self.theta - v_pos) + sigma_sqrt_dt * sqrt_v_pos * z1 + sigma2_4_dt * (z1**2 - 1),
                0.0
            )
            X[n + 1] = X[n] + r_q_dt - half_dt * v_pos + sqrt_v_pos * sqrt_dt * eps_s

        self.value_path = np.exp(X).T
        self.volatility_path = v.T

    def plot_simulation(self):
        """ Plot the simulations
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        t = np.linspace(0, 1, self.value_path.shape[1])
        vol = np.sqrt(self.volatility_path)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Price Paths", "Volatility Paths"))
        for i in range(5):
            fig.add_trace(go.Scatter(x=t, y=self.value_path[i], mode='lines', showlegend=False, line=dict(color=colors[i])), row=1, col=1)
        for i in range(5):
            fig.add_trace(go.Scatter(x=t, y=vol[i], mode='lines', showlegend=False, line=dict(color=colors[i])), row=1, col=2)

        fig.update_layout(title="Heston Model Simulation", height=500, width=1000)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Volatility", row=1, col=2)
        fig.show()

    def plot_convergence(self, strike):
        import plotly.graph_objects as go

        payoffs = np.maximum(self.value_path[:, -1] - strike, 0.0) * np.exp(-self.r * self.T)
        running_mean = np.cumsum(payoffs) / np.arange(1, self.n_paths + 1)

        fourier_price = self.compute_lewis_heston_price(strike)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, self.n_paths + 1),
            y=running_mean,
            mode='lines',
            name='running mean',
            line=dict(color='steelblue'),
        ))
        fig.add_hline(
            y=fourier_price,
            line_dash='dash',
            line_color='firebrick',
            annotation_text=f'Lewis: {fourier_price:.4f}',
            annotation_position='top right',
        )
        fig.update_layout(
            title=f'Price Convergence  (K={strike}, T={self.T})',
            xaxis_title='Number of paths',
            yaxis_title='Call price',
            height=450,
            width=900,
        )
        fig.show()

    def get_simulated_price(self, strike):
        return np.mean(np.maximum(self.value_path[:, -1] - strike, 0.0)) * np.exp(-self.r * self.T)

    def compute_lewis_heston_price(self, strike):
        return compute_lewis(self.create_cf(), self.S0, self.T, strike, self.r, self.q)

    def compute_lewis_heston_prices(self, strikes):
        return compute_lewis_strikes_quad(self.create_cf(), self.S0, self.T, strikes, self.r, self.q)

    def _compute_ivs(self, strikes):
        from src.pricing.fourier_pricing import implied_vol
        lewis_prices = self.compute_lewis_heston_prices(strikes)
        lewis_ivs = [implied_vol(price, self.S0, K, self.r, self.T, self.q) for K, price in zip(strikes, lewis_prices)]
        mc_ivs = [implied_vol(self.get_simulated_price(K), self.S0, K, self.r, self.T, self.q) for K in strikes]
        return lewis_ivs, mc_ivs

    def plot_iv(self, strikes):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        lewis_ivs, mc_ivs = self._compute_ivs(strikes)
        log_moneyness = [np.log(K / self.S0) for K in strikes]

        lewis_color = '#1f77b4'
        mc_color = '#ff7f0e'

        fig = make_subplots(rows=1, cols=2, subplot_titles=('IV vs Strike', 'IV vs Log Moneyness'))
        fig.add_trace(go.Scatter(x=strikes, y=lewis_ivs, mode='lines+markers', name='Lewis (Fourier)', legendgroup='lewis', line=dict(color=lewis_color), marker=dict(color=lewis_color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=strikes, y=mc_ivs, mode='lines+markers', name='Monte Carlo', legendgroup='mc', line=dict(color=mc_color), marker=dict(color=mc_color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=log_moneyness, y=lewis_ivs, mode='lines+markers', name='Lewis (Fourier)', legendgroup='lewis', showlegend=False, line=dict(color=lewis_color), marker=dict(color=lewis_color)), row=1, col=2)
        fig.add_trace(go.Scatter(x=log_moneyness, y=mc_ivs, mode='lines+markers', name='Monte Carlo', legendgroup='mc', showlegend=False, line=dict(color=mc_color), marker=dict(color=mc_color)), row=1, col=2)
        fig.update_xaxes(title_text='Strike', row=1, col=1)
        fig.update_xaxes(title_text='Log Moneyness', row=1, col=2)
        fig.update_yaxes(title_text='Implied Volatility', row=1, col=1)
        fig.update_yaxes(title_text='Implied Volatility', row=1, col=2)
        fig.update_layout(title=f'Heston Implied Volatility  (T={self.T})', height=500, width=1100)
        fig.show()


if __name__ =="__main__":
    model = HestonModel(
    S0=100.0, v0=0.04, r=0.05,q=0,
    kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
    T=1, N=512, n_paths=20000, seed=None)

    model.create_simulation()
    print(f"Simulated price: {model.get_simulated_price(strike=100)}")
    print(f"lewis price:    {model.compute_lewis_heston_price(strike=100)}")
    model.plot_simulation()
    strikes = [60, 80, 100, 120, 140, 160, 180, 200,220]
    print([model.get_simulated_price(k) for k in strikes])
    print(f"lewis prices (vectorized): {model.compute_lewis_heston_prices(strikes)}")
    model.plot_iv(strikes)
