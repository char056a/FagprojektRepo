import numpy as np
from scipy.optimize import root_scalar
import json
from .utils import ReLU

with open('diabetessims/config.json', 'r') as f:
    data = json.load(f) #læs json-fil
    params = data["HM"]

params["model"] = "HM"

def get_FR(p, G = None):
    """Returns FR"""
    if G is None:
        G = p.G
    return ReLU(0.003 * (G - 9) * p.VG * p.BW)

def get_F01c(p, G = None):
    """Returns FR"""
    if G is None:
        G = p.G
    return min(1, G/4.5) * p.F01 * p.BW


def sys(p, d = 0, uI = 0, uP = 0):
    """
    Solves dx = f(x, u, d)

    Parameters
    ----------
    u : int or float 
        Insulin injection rate.
    d : int or float 
        Meal ingestion rate.
    
    Returns
    -------
    dx : numpy array
        Solution to system of differential equations. 
    """
    G = p.Q1/(p.VG * p.BW)
    

    D = 1000 * d/p.MwG

    F01c = get_F01c(p, p.G)
    FR = get_FR(p, p.G)

    UG = p.D2 / p.taus
    UI = p.S2/p.taus

    dG = (G - p.G)/p.tauig
    dGsc = (p.G - p.Gsc) / p.tausc
    dQ1 = UG - F01c - FR - p.x1 * p.Q1 + p.k12 * p.Q2 + p.BW * p.EGP0 * (1 - p.x3)
    dQ2 = p.Q1 * p.x1 - (p.k12 + p.x2)*p.Q2
    dS1 = uI - p.S1 / p.taus
    dS2 = ((p.S1 - p.S2)/p.taus)
    dI = ((uP + UI) / (p.VI * p.BW) - p.ke * p.I) # Den her kan være wack
    dx1 = p.kb1 * p.I - p.ka1 * p.x1
    dx2 = p.kb2 * p.I - p.ka2 * p.x2
    dx3 = p.kb3 * p.I - p.ka3 * p.x3
    dD1 = p.AG * D - p.D1 / p.taud
    dD2 = (p.D1 - p.D2)/p.taud
    dx = np.array([dG, dGsc, dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2])
    return dx


def steadystate(p, G = None, uI = None, uP = 0):
    if uI is None:
        uI = ssinv(p = p, G = G, uP = uP)
    return ss(p, uI = uI, uP = uP), uI

def ss(p, uI = 0, uP = 0):
    u = uI + uP
    G = G_from_u(p, u)
    I = u/(p.VI * p.BW * p.ke)
    x1 = p.kb1/p.ka1 * I
    x2 = p.kb2/p.ka2 * I
    x3 = p.kb3/p.ka3 * I
    Q1 = G * p.VG * p.BW
    Q2 = x1 * Q1/(p.k12 + x2)
    S = uI * p.taus
    x0 = np.array([G, G, Q1, Q2, S, S, I, x1, x2, x3, 0, 0])
    return x0


def ssinv(p, G = None, uP = 0):
    if G is None:
        G = p.Gbar
    F01c = get_F01c(p,G)
    FR = get_FR(p,G)
    Q1 = G * p.VG * p.BW
    k1 = p.kb1/p.ka1
    k2 = p.kb2/p.ka2
    k3 = p.kb3/p.ka3
    a = -k2 * p.BW * (G * p.VG * k1 + p.EGP0 * k3)
    b = p.BW * p.EGP0 *(-p.k12 * k3 + k2)  - k2 * (F01c + FR)
    c = p.k12 * (p.BW * p.EGP0 - F01c - FR)
    d = np.sqrt(b**2 - 4*a*c)
    sol1 = (-b + d) / (2 * a)
    sol2 = (-b - d) / (2 * a)
    I = max(sol1, sol2)
    uIuP = I * p.VI * p.BW * p.ke
    uI = uIuP - uP
    return uI

def G_from_u(p, u):
    I = u/(p.VI * p.BW * p.ke)
    x1 = p.kb1/p.ka1 * I
    x2 = p.kb2/p.ka2 * I
    x3 = p.kb3/p.ka3 * I
    G_mat = np.zeros((2,2))
    for k in range(2):
        for c in range(2):
            G_mat[k, c] = (-p.F01 * (1-k) + 0.027 * p.VG * c + p.EGP0 * (1 -x3))/(p.F01 / 4.5 * k + p.VG * ( 0.003 * c + x1  - x1*p.k12 /(p.k12 + x2)))
    # Find valid value
    G_res = np.empty((2,2), dtype=bool)
    G_res[0] = G_mat[0] >= 4.5
    G_res[1] = G_mat[1] <= 4.5
    G_res[:,0] *= G_mat[:,0] <= 9
    G_res[:, 1] *= G_mat[:, 1] >= 9
    G = G_mat[np.where(G_res)][0]
    return G
