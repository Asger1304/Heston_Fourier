"""Module containing the fourier pricing formulas as functions"""

import numpy as np
from scipy.integrate import quad, trapezoid, quad_vec
from scipy.stats import norm
from scipy.optimize import brentq

#calculate BSM call price
#this is used to both invert implied volatilities when that is necessary and in the beginning to validate output of 
#the lewis style formula formula when i was implimenting it
def compute_bsm(s_0, k, r, t, sigma, q=0.0):
    d1 = (np.log(s_0 / k) + (r - q + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s_0 * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)




def bsm_cf(sigma, s, t, r, q):
    def phi(u):
        x = np.log(s) + (r - q - 0.5 * sigma**2) * t
        return np.exp(1j * u * x - 0.5 * sigma**2 * u**2 * t)
    return phi



def implied_vol(price, s0, k, r, t, q=0.0):
    """numerically inverting the BSM to get the implied vol out"""
    def objective(sigma):
        return compute_bsm(s0, k, r, t, sigma, q) - price
    try:
        return brentq(objective, 1e-6, 10.0)
    except ValueError:
        return np.nan


def compute_lewis(cf, s, t, k, r, q=0.0,u_max =400):
    """Computes a single lewis style price for an arbetrary CF, using the gauss-kronrod quadrature which is the default in this library"""
    kl = np.log(k)

    def integrand(u):
        val = np.exp(-1j * u * kl) * cf(u - 0.5j) / (u**2 + 0.25)
        return np.real(val)

    integral_value, _ = quad(integrand, 1e-8, u_max, limit=200, epsabs=1e-10, epsrel=1e-10)
    return s * np.exp(-q * t) - np.exp(kl / 2 - r*t) / np.pi * integral_value


def compute_lewis_strikes_quad(cf, s, t, strikes, r, q=0.0, u_max=np.inf):
    """Vectorised  lewis pricer.
    
    same aproach as the single integral case just optimized a bit for speed in python
    """
    kl = np.log(np.asarray(strikes, dtype=float))

    def integrand(u):
        return np.real(np.exp(-1j * u * kl) * cf(u - 0.5j) / (u**2 + 0.25))

    integral, _ = quad_vec(integrand, 1e-8, u_max, epsabs=1e-10, epsrel=1e-10)
    return s * np.exp(-q * t) - np.exp(kl / 2 - r * t) / np.pi * integral


def compute_lewis_prices_and_gradient_quad(cf, h_fn, s, t, strikes, r, q=0.0, u_max=300.0):
    """Prices and ∂C/∂θ in one quad_vec pass — GK with the analytical gradiant.

    Returns:
        prices : (n_strikes,1) call prices
        grad    : (n_strikes, 5) gradient in order [v0,theta,rho,kappa,sigma]
    """
    kl = np.log(np.asarray(strikes, dtype=float))      
    n_K = len(kl)
    price_scale = np.exp(kl / 2 - r * t) / np.pi       

    def integrand(u):
        u_s = u - 0.5j
        phi = cf(u_s)                                  
        h   = h_fn(u_s)                                
        w   = 1.0 / (u**2 + 0.25)
        e   = np.exp(-1j * u * kl)                      
        phi_w_e = np.real(phi * w * e)                   
        grad_vec = np.real(phi * w * h[:, None] * e[None, :])  
        return np.concatenate([phi_w_e, grad_vec.ravel()])  #(n_k, 6)

    result, _ = quad_vec(integrand, 1e-8, u_max, epsabs=1e-10, epsrel=1e-10)
    prices = s * np.exp(-q * t) - price_scale * result[:n_K] # (n_k,1) covered call to call price
    grad    = (-price_scale * result[n_K:].reshape(5, n_K)).T  # (n_K, 5)
    return prices, grad






def compute_lewis_strikes(cf, s, t, strikes, r, q=0.0, u_max=300.0, n_u=4096):
    """Computing the lewis prices with a trapazoid rule approach

    we get to "vectorize" this by hand, this is just matrix algebra.
    """
    kl = np.log(np.asarray(strikes, dtype=float))
    u = np.linspace(1e-8, u_max, n_u)

    val = np.exp(-1j * np.outer(u, kl)) * cf(u - 0.5j)[:, None] / (u**2 + 0.25)[:, None]
    integral_value = trapezoid(np.real(val), u, axis=0)

    return s * np.exp(-q * t) - np.exp(kl / 2 - r * t) / np.pi * integral_value




def compute_lewis_prices_and_gradient(cf, h_fn, s, t, strikes, r, q=0.0, u_max=300.0, n_u=4096):
    """Prices and ∂C/∂θ mixing both the analytical gradiant and trapazoid approach, using the trapazoid approach for both prices and the gradiant

    Returns:
        prices : (n_strikes,) call prices
        grad    : (n_strikes, 5) gradient in order [v0,theta,rho,kappa,sigma]
    """
    kl  = np.log(np.asarray(strikes, dtype=float))    # (n_K,1)
    u   = np.linspace(1e-8, u_max, n_u)               # (n_u,1)
    u_s = u - 0.5j

    phi     = cf(u_s)                                  # (n_u,)
    h       = h_fn(u_s)                                # (5, n_u)
    w       = 1.0 / (u**2 + 0.25)                      # (n_u,)
    exp_kl  = np.exp(-1j * np.outer(u, kl))            # (n_u, n_K)
    phi_w   = phi * w                                   # (n_u,)

    price_scale = np.exp(kl / 2 - r * t) / np.pi      
    grad_scale   = -price_scale                       

    # Prices
    price_integrands = np.real(exp_kl * phi_w[:, None])          # (n_u, n_K)
    prices = s * np.exp(-q * t) - price_scale * trapezoid(price_integrands, u, axis=0)

    phi_h_w    = phi_w[None, :] * h                               # (5, n_u)
    grad_integrands = np.real(phi_h_w[:, :, None] * exp_kl[None, :, :])  # (5, n_u, n_K)
    grad = (grad_scale * trapezoid(grad_integrands, u, axis=1)).T    # (n_K, 5)

    return prices, grad
