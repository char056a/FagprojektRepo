import numpy as np
import json

with open('diabetessims/config.json', 'r') as f:
    data = json.load(f) #læs json-fil
    params = data["MVP"]

params["model"] = "MVP"

def sys(p, d = 0, uI = 0, uP = 0, HR = None):
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
    dD1 = d - p.D1/p.taum
    dD2 = (p.D1 - p.D2)/p.taum
    dIsc = uI/(p.tau1 * p.CI) - p.Isc/p.tau1
    dIp = (p.Isc - p.Ip + uP/p.CI)/p.tau2
    dIeff = -p.p2 * p.Ieff + p.p2 * p.SI * p.Ip
    dG = - (p.GEZI + p.Ieff) * p.G + p.EGP0 + 1000/18 * p.D2 / (p.VG * p.taum)
    dGsc = (p.G - p.Gsc) / p.tausc

    dx = np.array([dD1, dD2, dIsc, dIp, dIeff, dG, dGsc])
    return dx

def steadystate(p, G = None, uI = None, uP = 0):
    if uI is None:
        uI = ssinv(p = p, G = G, uP = uP)
    if G is None:
        G = ss(p = p, uI = uI, uP = uP)
    Isc = uI / p.CI
    Ip = Isc + uP / p.CI
    Ieff = p.SI * Ip
    x0 = np.array([0, 0, Isc, Ip,Ieff, G, G])
    return x0, uI

def G_from_u(p, u):
    return p.EGP0/(p.GEZI + p.SI / p.CI * u)

def ss(p, uI = 0, uP = 0):
    u = uI + uP
    G = p.EGP0/(p.GEZI + p.SI / p.CI * u)
    Isc = uI / p.CI
    Ip = Isc + uP / p.CI
    Ieff = p.SI * Ip
    x0 = np.array([0, 0, Isc, Ip,Ieff, G, G])    
    return x0

def ssinv(p, G = None, uP = 0):
    if G is None:
        G = p.Gbar
    uI = p.CI/p.SI * (p.EGP0 / G - p.GEZI) - uP
    return uI
