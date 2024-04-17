import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class patient:
    def __init__(self, model = "EHM", patient_type = 1,pancreas_model = "SD",**kwargs):
        self.model = model.upper()
        self.pancreas_model = pancreas_model.upper()
        defaults = {}
        with open('config.json', 'r') as f:
            data = json.load(f)
        defaults.update(data["general"])
        defaults.update(data[self.model])

        #defaults.update(data[self.pancreas_model])

        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)

        for key in self.state_keys:
            setattr(self, key+"0", defaults[key])

        if self.model == "EHM":
            self.f_func = lambda *args : EHM(self, *args)
        if self.model == "MVP":
            self.f_func = lambda *args : MVP(self, *args)
            

    def __str__(self):
        return str(self.__dict__)
        
    def get_state(self):
        """Update state vector to values given by input"""
        x = np.array([getattr(self,key) for key in self.state_keys])
        return x

    def get_attr(self, states, attr):
        """Returns column of state matrix with idx matching given attribute"""
        idx = self.state_keys.index(attr) # Finds the index of desired attribute
        return states[:,idx]

    def update_state(self, x_new):
        """Update state vector to values given by input"""
        for key, val in zip(self.state_keys, x_new):
            setattr(self, key, val)
        return

    def reset(self):
        """Resets state to x0"""
        x0 = [getattr(self,key+"0") for key in self.state_keys]
        self.update_state(x0)
        return

    def time_arr(self, length):
        return np.arange(0, length*self.timestep, self.timestep)

    def euler_step(self, dx):
        """
        Updates state using state vector derivative and one step of eulers method.
        
        Parameters
        ----------
        dx : numpy array
            Derivative of state vector.
        """
        x_new = self.get_state() + dx * self.timestep
        self.update_state(x_new)
        return



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
        Ki = self.Kp * self.timestep / self.Ti
        Ikp1 = I + Ki * ek
        Kd = self.Kp * self.Td / self.timestep
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
        ds[meal_idx] = meal_size / self.timestep # Ingestion 
        us[0] += bolus * 1000 / self.timestep
        states, _ = self.simulate(ds, us)
        Gt = states[:, 5]
        p = np.array([self.glucose_penalty(G = G) for G in Gt])
        t = self.time_arr(iterations + 1)/60
        phi = simpson(p, x = t)
        if plot:
            fig, ax = plt.subplots(1,2)
            ax[0].plot(t, p)
            ax[1].plot(t, Gt)
            ax[0].set_xlabel("time(h)")
            ax[1].set_xlabel("time(h)")
            ax[1].set_ylabel("g")

            ax[0].set_title("Penalty Function")
            ax[1].set_title("Blood Glucose")
            plt.show()
        return phi, p, Gt


    def simulate(self, ds, u_func=None):
        """
        Simulates patient.

        Parameters
        ----------
        ds : numpy array
            Ingestion rate
        u_func : Default = None, int, float, numpy array, list or "PID"
            Specifies insulin injection rate.
            If None; uses steady state insulin rate.
            If "PID"; uses PID controller.
        
        Returns
        -------
        states : numpy array
            State vector in each time step
        u_list : numpy array
            Insulin injection rate for each time step
        """
        res = np.empty((len(ds)+1, len(self.state_keys)))
        res[0, :] = self.get_state()
        inp = 0
        u = self.us
        if isinstance(u_func, (np.ndarray, list)):
            u = u_func[0]
            inp = 1
            def get_u(inp):
                u = u_func[inp]
                return u, inp+1

        elif u_func == "PID":
            inp = (0, self.G)
            def get_u(inp):
                I, y_prev = inp
                u, I = self.PID_controller(I, self.Gsc, y_prev)
                return u, (I, self.G)

        elif isinstance(u_func, (float, int)):
            u = u_func
            def get_u(inp):
                return u_func, inp

        else:
            def get_u(inp):
                return self.us, inp

        u_list = [u]

        for i,d in enumerate(ds[:-1]):
            dx = self.f_func(u, d)
            self.euler_step(dx)     
            res[i+1,:] = self.get_state()
            u, inp = get_u(inp)
            u_list.append(u)
        dx = self.f_func(u, d)
        self.euler_step(dx)     
        res[i+2, :] = self.get_state()
        return np.array(res), u_list



    def plot(self, data, u, d):
        t = self.time_arr(data.shape[0])/60
        fig, ax = plt.subplots(3,2,figsize=(9,7))

        ax[0,0].set_title("Glucose")
        ax[0,0].plot(t, data[:,5], label= "Blood")
        ax[0,0].plot(t, data[:,6], label = "Subcutaneous")
        ax[0,0].plot(t, [self.Gbar]*data.shape[0], label= "Target")
        ax[0,0].legend()

        ax[0,1].set_title("Insulin Injection Rate")
        ax[0,1].plot(t[:-1], u[:len(t) - 1]) # Lidt bøvet måde at håndtere det her på, men det går nok.
        ax[1,0].set_title("Carbs")
        ax[1,0].plot(t, data[:,0], label= "D1")
        ax[1,0].plot(t, data[:,1], label= "D2")
        ax[1,0].legend()
        ax[1,1].set_title("Carb Ingestion (d)")
        ax[1,1].plot(t[:-1], d)
        ax[2,0].set_title("Insulin")
        ax[2,0].plot(t, data[:,2], label= "Subcutaneous")
        ax[2,0].plot(t, data[:,3], label= "Plasma")
        ax[2,0].legend()
        ax[2,1].plot(t, data[:,4])
        ax[2,1].set_title("Effective Insulin")

        ax[0,0].set_ylabel("mg/dL")
        ax[0,1].set_ylabel("mU/min")
        ax[1,0].set_ylabel("g")
        ax[1,1].set_ylabel("g CHO/min")
        ax[2,0].set_ylabel("mU/dL")
        ax[2,1].set_ylabel("mU/dL")

        for i in range(6):
            ax[i//2, i%2].set_xlabel("time (h)")
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



def MVP(self, u, d):
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
    dGsc = (self.G - self.Gsc) / self.timestep

    dx = np.array([dD1, dD2, dIsc, dIp, dIeff, dG, dGsc])
    return dx


def EHM(self, uI, d, uG = 0):
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
    G = self.Q1/self.VG
    RA = self.f*self.D2/self.TauD
    D = 1000 * d/self.MwG
    HRmax = 220 - self.age
    HR = self.HRR * (HRmax - self.HR0) + self.HR0
    fE1x = self.E1/(self.a * self.HR0)**self.n
    fE1 = fE1x/(1+fE1x)
    F01c = min(self.F01, self.F01 * G / 4.5)
    FR = max(0.003 * (G - 9) * self.VG, 0) 

    dx = []
    dx.append((G - self.GI)/self.TauIG)
    dx.append(-F01c - FR - self.x1 * self.Q1 + self.k12 * self.Q2 + RA + self.EGP0 * (1 - self.x3) \
        + self.Kglu * self.VG * self.Z2 - self.alpha * self.E2**2 * self.x1 * self.Q1)
    dx.append(self.x1 * self.Q1 - (self.k12 + self.x2) * self.Q2 + self.alpha * self.E2**2 * self.x1 * self.Q1 \
        - self.alpha * self.E2**2 * self.x2 * self.Q2 - self.beta * self.E1 / self.HR0)

    dx.append(uI - self.S1 / self.TauS)
    dx.append((self.S1 - self.S2)/self.TauS)
    dx.append(self.S2 / (self.VI * self.TauS) - self.ke * self.I)
    dx.append(self.kb1 * self.I - self.ka1 * self.x1)
    dx.append(self.kb2 * self.I - self.ka2 * self.x2)
    dx.append(self.kb3 * self.I - self.ka3 * self.x3)
    dx.append(self.AG * D - self.D1 / self.TauD)
    dx.append((self.D1 - self.D2)/self.TauD)
    dx.append(uG - self.Z1 / self.Tauglu)
    dx.append((self.Z1 - self.Z2)/self.Tauglu)
    dx.append((HR - self.HR0 - self.E1)/self.TauHR)
    dx.append(-(fE1/self.Tauin - 1/self.TE) * self.E2 + fE1 * self.TE /(self.c1 + self.c2))
    dx.append((self.c1 * fE1 + self.c2 - self.TE)/self.Tauex)
    return np.array(dx)