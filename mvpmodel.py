import numpy as np
# import matplotlib.pyplot as plt

class MVPmodel:
    def __init__(self, x0, params):
        """
        HEJ
        Parameters
        ----------
        x0 : numpy array.
            Initial state vector.
            x = [D1, D2, Isc, Ip, Ieff, G, Gsc]
        params : numpy array.
            Model parameters.
            params = [tau1, tau2, C1, p2, S1, gezi, egp0, Vg, taum, tausc]
        """
        self.x0 = x0
        self.x = x0
        self.D1, self.D2, self.Isc, self.Ip, self.Ieff, self.G, self.Gsc = x0

        self.tau1, self.tau2, self.C1, self.p2, self.S1,\
        self.gezi, self.egp0, self.Vg, self.taum, self.tausc = params

    def update_state(self, x_new):
        """Update state vector to values given by input"""
        self.x = x_new
        self.D1, self.D2, self.Isc, self.Ip, self.Ieff, self.G, self.Gsc = x_new

    def f(self, u, d):
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
        dD1 = d - self.D1/self.taum
        dD2 = (self.D1 - self.D2)/self.taum
        dIsc = u/(self.tau1 * self.C1) - self.Isc/self.tau1
        dIp = (self.Isc - self.Ip)/self.tau2
        dIeff = -self.p2 * self.Ieff + self.p2 * self.S1 * self.Ip
        dG = - (self.gezi + self.Ieff) * self.G + self.egp0 + 1000 * self.D2 / (self.Vg * self.taum)
        dGsc = (self.G - self.Gsc) / self.tausc

        dx = np.array([dD1, dD2, dIsc, dIp, dIeff, dG, dGsc])
        return dx


    def euler_step(self, dx):
        """
        Updates state using state vector derivative and one step of eulers method.
        
        Parameters
        ----------
        dx : numpy array
            Derivative of state vector.
        """
        x_new = self.x + dx * self.tausc
        self.update_state(x_new)


    def iterate(self, us, ds):
        state_list = []
        state_list.append(self.x)
        for u, d in zip(us, ds):
            x_change = self.f(u, d)
            self.euler_step(x_change)
            state_list.append(self.x)
        return np.array(state_list)

    def PID_controller(self, I, r, y, y_prev, us, Kp, Ti, Td, Ts):
        """
        :input I: Integral term
        :input r: Glucose concentration target
        :input y: Current blood glucose
        :input y_prev: Previous blood glucose
        :input us: Insulin steady state

        Tuning parameters
        :input Kp: Range 0-0.5
        :input Ti: 100-1000 minutes
        :input Td : 0-30 min
        """
        ek = y - r
        Pk = Kp * ek
        Ki = Kp * Ts / Ti
        Ikp1 = I + Ki * ek
        Kd = Kp * Td / Ts
        Dk = Kd * (y - y_prev)
        uk = us + Pk + I + Dk
        return uk, Ikp1

    def simulate(self, ds, r, us, Kp, Ti, Td):
        """
        :input ds: Array of "meal size" for every timestep.
        :input r: Glucose concentration target
        :input us: Insulin steady state

        Tuning parameters
        :input Kp: Range (0-0.5)
        :input Ti: 100-1000 (minutes)
        :input Td : 0-30 (smin)
        """
        u = us
        y = [self.G]
        I = 0
        for d in ds:
            dx = self.f(u, d)
            self.euler_step(dx)     
            y.append(self.G) 
            u, I = self.PID_controller(I, r, y[-1], y[-2], us, Kp, Ti, Td, self.tausc)
        return y
