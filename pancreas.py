import numpy as np
from odeclass import ODE
import json

    
class PKPM(ODE):
    def __init__(self, **kwargs):
        with open('config.json', 'r') as f:
            defaults = json.load(f)["PKPM"]
        defaults.update(kwargs)
        super().__init__(defaults)

    def ode(self, G):
        if G <= self.Gu: # if glucose is low
            f = self.fb
            alpha2 = 0
            idx = 0 # use first value for params with two values
        else: # if glucose is high
            idx = 1 # use second value for params with two values
            f = self.fb + (1 - self.fb) *  (G - self.Gu) / (self.Kf +  G - self.Gu)
            if G <= self.Gl: # if glucose is below the upper level
                alpha2 = self.hhat * (G - self.Gu)/(G - self.Gu) 
            else: # if glucose is above the upper level
                alpha2 = self.hhat
        # ode
        dM = self.alpha1[idx] - self.delta1[idx] * self.M
        dP = self.v[idx] * self.M - self.delta2 * self.P - self.k * self.P * self.rho * self.DIR
        dR =  self.k * self.P * self.rho * self.DIR - self.gamma * self.R
        dgamma = self.eta * (-self.gamma + self.gammab + alpha2)
        dD = self.gamma * self.R - self.k1p * (self.CT - self.DIR) * self.D - self.k1m * self.DIR - self.rho * self.DIR
        dDIR = self.k1p * (self.CT - self.DIR) * self.D - self.k1m * self.DIR - self.rho * self.DIR
        drho = self.zeta * (-self.rho + self.rhob + self.krho * (self.gamma - self.gammab))
        
        dx = np.array([dM, dP, dR, dgamma, dD, dDIR, drho])
        ISR = self.get_ISR(f)
        return dx, ISR

    def get_ISR(self, f, **kwargs):
        I0 = kwargs.get("I0", self.I0)
        rho = kwargs.get("rho", self.rho)
        DIR = kwargs.get("DIR", self.DIR)
        N = kwargs.get("N", self.N)

        return max(I0 * rho * DIR * f * N,0) # do not let isr be negative

    def eval(self, G):
        dx, ISR = self.ode(G)
        x_new = dx * self.timestep + self.get_state()
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

