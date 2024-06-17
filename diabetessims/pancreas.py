import numpy as np
from diabetessims.odeclass import ODE
import json

    
class PKPM(ODE):
    def __init__(self, patient_type = 0, Gbar = None, **kwargs):
        with open('diabetessims/config.json', 'r') as f:
            defaults = json.load(f)["PKPM"]

        if patient_type == 2 :
            remove = "normal"
            keep = "T2"
        else:
            remove = "T2"
            keep = "normal"

        del defaults[remove]
        for key, item in defaults[keep].items():
            defaults[key] = item
        del defaults[keep]

        defaults.update(kwargs)
        super().__init__(defaults)
        if Gbar is not None: # if a desired glucose level is given
            x0, _ = self.steadystate(Gbar) # find steady state with given parameters
            self.update_state(x0) # set to steady state
            for key in self.state_keys: # also set "x0" values
                setattr(self, key+"0", getattr(self, key))

    def get_ISR(self, G, **kwargs):
        if G <= self.Gl: # if glucose is low
            f = self.fb
        else: # if glucose is high
            f = self.fb + (1 - self.fb) *  (G - self.Gl) / (self.Kf +  G - self.Gl)
        I0 = kwargs.get("I0", self.I0)
        rho = kwargs.get("rho", self.rho)
        DIR = kwargs.get("DIR", self.DIR)
        N = kwargs.get("N", self.N)
        return self.W * max(I0 * rho * DIR * f * N,0) # do not let isr be negative

    def get_dependant_vars(self, G):
        if G <= self.Gl: # if glucose is low
            alpha2 = 0
            idx = 0 # use first value for params with two values
        else: # if glucose is high
            idx = 1 # use second value for params with two values
            if G <= self.Gu: # if glucose is below the upper level
                alpha2 = self.hhat * (G - self.Gl)/(self.Gu - self.Gl) 
            else: # if glucose is above the upper level
                alpha2 = self.hhat
        alpha1 = self.alpha1[idx]
        delta1 = self.delta1[idx]
        v = self.v[idx]
        return v, delta1, alpha1, alpha2


    def sys(self, G):
        v, delta1, alpha1, alpha2 = self.get_dependant_vars(G)
        # ode
        dM = alpha1 - delta1 * self.M
        dP = v * self.M - self.delta2 * self.P - self.k * self.P * self.rho * self.DIR
        dR =  self.k * self.P * self.rho * self.DIR - self.gamma * self.R
        dgamma = self.eta * (-self.gamma + self.gammab + alpha2)
        dD = self.gamma * self.R - self.k1p * (self.CT - self.DIR) * self.D + self.k1m * self.DIR
        dDIR = self.k1p * (self.CT - self.DIR) * self.D - self.k1m * self.DIR - self.rho * self.DIR
        drho = self.zeta * (-self.rho + self.rhob + self.krho * (self.gamma - self.gammab))
        dx = np.array([dM, dP, dR, dgamma, dD, dDIR, drho])
        ISR = self.get_ISR(G)
        return dx, ISR


    def steadystate(self, G):
        v, delta1, alpha1, alpha2 = self.get_dependant_vars(G)
        # ode
        M = alpha1/delta1
        gamma = self.gammab + alpha2
        rho = self.rhob + self.krho * (gamma - self.gammab)
        P = 1/self.k
        R = (v * M - P*self.delta2)/gamma
        DIR = R * gamma / rho
        D = (self.k1m * DIR + rho * DIR)/ (self.k1p * (self.CT - DIR))
        x0 = np.array([M, P, R, gamma, D, DIR, rho])
        ISR = self.get_ISR(G, rho = rho, DIR = DIR)
        return x0, ISR
    

    def eval(self, G):
        dx, ISR = self.sys(G)
        x_new = dx * self.timestep + self.get_state()
        x_new = x_new * (x_new > 0)
        self.update_state(x_new)
        return ISR


class SD(ODE):
    def __init__(self, alpha, gamma, h, KD, ybar, timestep):
        data = {
            "state_keys" : ["SRs", "yprev"],
            "SRs" : 0,
            "ybar" : ybar,
            "yprev" : ybar,
            "alpha" : alpha,
            "gamma" : gamma,
            "h" : h,
            "KD" : KD,
            "timestep" : timestep
        }
        super().__init__(data)

    
    def eval(self,y):
        dy = (y - self.yprev)/self.timestep # approx derivative of y
        dSRs = -self.alpha * (self.SRs + self.gamma * (self.h - y))
        SRD = max(dy * self.KD, 0)
        res = max(self.SRs + SRD,0)

        self.SRs += dSRs * self.timestep
        self.yprev = y
        return res


class PID(ODE):
    def __init__(self, Kp, Td, Ti, ybar, timestep):
        data = {
            "state_keys" : ["I", "yprev"],
            "I" : 0,
            "yprev" : ybar,
            "Kp" : Kp,
            "Td" : Td,
            "Ti" : Ti,
            "ybar" : ybar,
            "timestep" : timestep
        }
        super().__init__(data)

    def eval(self,y):
        dy = (y - self.yprev)/self.timestep
        ek = y - self.ybar

        P = self.Kp * ek 
        dI = P/self.Ti # 
        D = self.Kp * self.Td * dy

        res = P + self.I + D

        self.yprev = y 
        self.I += dI * self.timestep # Updates integral term
        return res        
