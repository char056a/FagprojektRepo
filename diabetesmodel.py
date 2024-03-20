import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class MVPmodel:
    def __init__(self,**kwargs):
        with open('config.json', 'r') as f:
            defaults = json.load(f)
        
        defaults.update(kwargs)
        
        for key, value in defaults.items():
            setattr(self, key, value)

        self.x0 = [self.D1, self.D2, self.Isc, self.Ip, self.Ieff, self.G, self.Gsc]
        self.x = [self.D1, self.D2, self.Isc, self.Ip, self.Ieff, self.G, self.Gsc]

    def __str__(self):
        return str(self.__dict__)
        


    def update_state(self, x_new):
        """Update state vector to values given by input"""
        self.x = x_new
        self.D1, self.D2, self.Isc, self.Ip, self.Ieff, self.G, self.Gsc = x_new
        return

    def reset(self):
        """Resets state to x0"""
        self.update_state(self.x0)
        return

    def time_arr(self, length):
        return np.arange(0, length*self.tausc, self.tausc)

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

    def PID_controller(self, I, y, y_prev):
        """
        :input I: Integral term
        :input Gbar: Glucose concentration target
        :input y: Current blood glucose
        :input y_prev: Previous blood glucose
        :input us: Insulin steady state

        Tuning parameters
        :input Kp: Range 0-0.5
        :input Ti: 100-1000 minutes
        :input Td : 0-30 min
        """
        ek = y - self.Gbar
        Pk = self.Kp * ek
        Ki = self.Kp * self.tausc / self.Ti
        Ikp1 = I + Ki * ek
        Kd = self.Kp * self.Td / self.tausc
        Dk = Kd * (y - y_prev)
        uk = self.us + Pk + I + Dk
        uk = max(uk, 0)
        return uk, Ikp1

    def glucose_penalty(self, G = "Default"):
        """
        Calculates penalty given blood glucose.
        p = 1/2 (G - Gbar)**2 + kappa/2 * max(Gmin - G, 0)**2

        Parameters
        ----------
        Gbar : int or float 
            Desired blood glucose
        kappa : int or float 
            Penalty weight
        Gmin : int or float
            Threshold for hypoglycemia
        G : int or float, default: False
            Current GLucose. If not set, use current state.
        
        Returns
        -------
        p(G) : float
            Penalty
        """
        if G == "Default":
            G = self.G
        return 1/2 * (G - self.Gbar)**2 + self.kappa/2 * max((self.Gmin - G), 0)**2
 

    def bolus_sim(self, bolus, meal_size, meal_idx = 0, iterations = 100, plot = False):
        ds = np.zeros(iterations)
        us = np.ones(iterations) * self.us

        ds[meal_idx] = meal_size / self.tausc # Ingestion 
        us[0] += bolus * 1000 / self.tausc

        states = self.iterate(us, ds)
        Gt = states[:, 5]
        p = np.array([self.glucose_penalty(G = G) for G in Gt])
        t = self.time_arr(iterations + 1)
        phi = simpson(p, x = t)
        if plot:
            fig, ax = plt.subplots(1,2)
            ax[0].plot(t, p)
            ax[1].plot(t, Gt)
            ax[0].set_title("Penalty Function")
            ax[1].set_title("Blood Glucose")
            plt.show()
        return phi, p, Gt

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
        return


    def iterate(self, u_list, d_list):
        """
        Simulation given a list of insulin injection and carb ingestion rates.
        """
        state_list = []
        state_list.append(self.x)
        for u, d in zip(u_list, d_list):
            x_change = self.f(u, d)
            self.euler_step(x_change)
            state_list.append(self.x)
        return np.array(state_list)    
        
    def simulate(self, ds):
        """
        :input ds: Array of "meal size" for every timestep.
        :input Gbar: Glucose concentration target
        :input us: Insulin steady state
        """
        u = self.us

        y = [self.G]
        res = [self.x]
        I = 0
        u_list = [u]
        for d in ds:
            dx = self.f(u, d)
            self.euler_step(dx)     
            y.append(self.G) 
            res.append(self.x)
            u, I = self.PID_controller(I, y[-1], y[-2])
            u_list.append(u)
        return np.array(res), u_list

    def plot(self, data, u, d):
        t = self.time_arr(data.shape[0])/60
        fig, ax = plt.subplots(2,2,figsize=(7,5))
        ax[0,0].scatter(t, data[:,5], label= "Blood glucose")
        ax[0,0].scatter(t, data[:,6], label = "Subcutaneous glucose")
        ax[0,0].scatter(t, [self.Gbar]*data.shape[0], label= "Target")
        ax[0,1].scatter(t, data[:,0], label= "D1")
        ax[0,1].scatter(t, data[:,1], label= "D2")
        ax[0,1].scatter(t[:-1], d, label= "d")
        ax[1,0].scatter(t, data[:,2], label= "Isc")
        ax[1,0].scatter(t, data[:,3], label= "Ip")
        ax[1,0].scatter(t, data[:,4], label= "Ieff")
        ax[1,1].scatter(t[:-1], u[:len(t) - 1], label= "Insulin Injection Rate") # Lidt bøvet måde at håndtere det her på, men det går nok.

        ax[0,0].set_ylabel("mg/dL")
        for i in range(4):
            ax[i//2,i%2].legend()
            ax[i//2, i%2].set_xlabel("time (h)")
        fig.tight_layout()
        return
    
    def plot2(self, data, u, d):
        t = self.time_arr(data.shape[0])/60
        fig = plt.figure(constrained_layout=True,figsize=(10,7))
        subplots = fig.subfigures(2,2)

        axG = subplots[0,0].subplots(1,1)
        axG.scatter(t, data[:,5], label= "Blood glucose")
        axG.scatter(t, data[:,6], label = "Subcutaneous glucose")
        axG.scatter(t, [self.Gbar]*data.shape[0], label= "Target")
        axG.set_xlabel("time (h)")

        axd = subplots[0,1].subplots()
        axd2 = axd.twinx()
        axd2.scatter(t[:-1], d, label= "d", color="red")

        axd.scatter(t, data[:,0], label= "D1")
        axd.scatter(t, data[:,1], label= "D2")
        axd.set_xlabel("time (h)")
        axd.set_ylabel("g")
        axd2.set_ylabel("g/min CHO")
        axd.legend()
        axd2.legend()

        axI = subplots[1,0].subplots()
        axI2 = axI.twinx()

        axI.scatter(t, data[:,2], label= "Isc")
        axI.scatter(t, data[:,3], label= "Ip")
        axI2.scatter(t, data[:,4], label= "Ieff", color="g")
        axI.set_xlabel("time (h)")
        axI.set_ylabel("mg/dL")
        axI2.set_ylabel("mg/dL")
        axI.legend()
        axI2.legend()

        axu = subplots[1,1].subplots()
        axu.scatter(t[:-1], u[:len(t) - 1], label= "Insulin Injection Rate") # Lidt bøvet måde at håndtere det her på, men det går nok.

        axu.set_xlabel("time (h)")
        axu.set_ylabel("mg/dL")
        axI2.set_ylabel("mg/dL")
        axI.legend()
        axG.set_ylabel("mg/dL")
    
        fig.tight_layout()
        return

    def optimal_bolus(self, meal_idx = 0, min_U = 0, max_U = 75, min_meal = 30, max_meal = 150, n = 50):
        Us = np.linspace(min_U, max_U, n)
        meals = np.linspace(min_meal, max_meal, n)

        res = np.empty((len(meals), len(Us), 3))

        for i, d0 in enumerate(meals):
            for j, U in enumerate(Us):
                self.reset()
                phi, _, _ = self.bolus_sim(U, d0, meal_idx=meal_idx)
                res[n - 1 - j ,i] = [phi]*3

        best = np.argmin(res[:,:,0], axis=0)

        r = 1 / (res.max() - res.min())
        res = r * (res - res.min())

        for i,j in enumerate(best):
            res[j,i] = [1,0,0]

        plt.imshow(res, extent = [min_meal, max_meal,min_U, max_U], aspect="auto")
        plt.ylabel("Bolus Size (U)")
        plt.xlabel("Meal Size (g. CHO)")


        best_us = [Us[n - 1- i] for i in best]
        return meals, best_us