import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from odeclass import ODE
import pancreas
from scipy.optimize import root_scalar
import utils

def MVP(self, d = 0, uI = 0, uP = 0, HR = None):
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
    dIsc = uI/(self.tau1 * self.CI) - self.Isc/self.tau1
    dIp = (self.Isc - self.Ip + uP/self.CI)/self.tau2
    dIeff = -self.p2 * self.Ieff + self.p2 * self.SI * self.Ip
    dG = - (self.GEZI + self.Ieff) * self.G + self.EGP0 + 1000 * self.D2 / (self.VG * self.taum)
    dGsc = (self.G - self.Gsc) / self.tausc

    dx = np.array([dD1, dD2, dIsc, dIp, dIeff, dG, dGsc])
    return dx

def MVP_steadystate(self, uP = 0):
    uI = self.CI/self.SI * (self.EGP0 / self.Gbar - self.GEZI) - uP
    Isc = uI / self.CI
    Ip = Isc + uP / self.CI
    Ieff = self.SI * Ip
    x0 = np.array([0, 0, Isc, Ip,Ieff, self.Gbar, self.Gbar])
    return x0, uI

def MVP_G_from_u(self, u):
    return self.EGP0/(self.GEZI + self.SI / self.CI * u)


def EHM(self, d = 0, uI = 0, uP = 0):
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
    G = self.Q1/(self.VG * self.BW)
    

    D = 1000 * d/self.MwG

    F01c = min(1, self.G / 4.5) * self.F01 * self.BW
    FR = max(0.003 * (self.G - 9) * self.VG * self.BW, 0) 

    UG = self.D2 / self.taus
    UI = self.S2/self.taus

    dG = (G - self.G)/self.tauig
    dGsc = (self.G - self.Gsc) / self.tausc
    dQ1 = UG - F01c - FR - self.x1 * self.Q1 + self.k12 * self.Q2 + self.BW * self.EGP0 * (1 - self.x3)
    dQ2 = self.Q1 * self.x1 - (self.k12 + self.x2)*self.Q2
    dS1 = uI - self.S1 / self.taus
    dS2 = ((self.S1 - self.S2)/self.taus)
    dI = ((uP + UI) / (self.VI * self.BW) - self.ke * self.I) # Den her kan være wack
    dx1 = self.kb1 * self.I - self.ka1 * self.x1
    dx2 = self.kb2 * self.I - self.ka2 * self.x2
    dx3 = self.kb3 * self.I - self.ka3 * self.x3
    dD1 = self.AG * D - self.D1 / self.taud
    dD2 = (self.D1 - self.D2)/self.taud
    dx = np.array([dG, dGsc, dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2])
    return dx

def HM_steadystate(self, uP = 0):
    G = self.Gbar
    Q1 = G * self.VG * self.BW
    F01c = self.F01 * self.BW

    k1 = self.kb1/self.ka1
    k2 = self.kb2/self.ka2
    k3 = self.kb3/self.ka3

    eq = lambda I0 : -F01c + Q1 * I0 * k1 * (-1 + self.k12 / (self.k12 + k2 * I0)) + self.BW * self.EGP0 * (1 - k3 * I0)
    sol = root_scalar(eq, bracket=[0, 20])
    I = sol.root
    
    uIuP = I * self.VI * self.BW * self.ke
    uI = uIuP - uP
    x1 = k1 * I
    x2 = k2 * I
    x3 = k3 * I

    Q2 = x1 * Q1/(self.k12 + x2)
    S1 = S2 = uI * self.taus
    x0 = np.array([G, G, Q1, Q2, S1, S2, I, x1, x2, x3, 0, 0])
    return x0, uI

def HM_G_from_u(self, u):
    I = u/(self.VI * self.BW * self.ke)
    x1 = self.kb1/self.ka1 * I
    x2 = self.kb2/self.ka2 * I
    x3 = self.kb3/self.ka3 * I

    # Calculate G for any combination of F01c and FR
    G_mat = np.zeros((2,2))
    for k in range(2):
        for c in range(2):
            G_mat[k, c] = (-self.F01 * (1-k) + 0.027 * self.VG * c + self.EGP0 * (1 -x3))/(self.F01 / 4.5 * k + self.VG * ( 0.003 * c + x1  - x1*self.k12 /(self.k12 + x2)))
    # Find valid value
    G_res = np.empty((2,2), dtype=bool)
    G_res[0] = G_mat[0] >= 4.5
    G_res[1] = G_mat[1] <= 4.5
    G_res[:,0] *= G_mat[:,0] <= 9
    G_res[:, 1] *= G_mat[:, 1] >= 9
    G = G_mat[np.where(G_res)][0]
    return G


class Patient(ODE):
    def __init__(self, patient_type, model = "EHM", **kwargs):
        self.model = model.upper()
        self.type = patient_type
        defaults = {} #tomt dictionary 
        with open('config.json', 'r') as f:
            data = json.load(f) #læs json-fil
        defaults.update(data["general"]) #tilføj "general" til dictionary(defaults)
        defaults.update(data[self.model]) #tilføj modelegenskaberne til dictionary(defaults)
        defaults.update(kwargs) #tilføj keywordarguments til dictionary(defaults)

        super().__init__(defaults)
        if self.model == "MVP":
            self.f_func = lambda **kwargs: MVP(self,**kwargs) #caller MVP-modellen
        else:
            self.f_func = lambda **kwargs: EHM(self,**kwargs) #caller EHM-modellen

        if patient_type != 1:
            self.pancreasObj = pancreas.PKPM(patient_type = patient_type, timestep=self.timestep/self.pancreas_n, Gbar=self.Gbar, **kwargs.get("pancreas_param", {}))
        if patient_type != 0:
            self.pumpObj = pancreas.PID(Kp = self.Kp, Td = self.Td, Ti = self.Ti, ybar = self.Gbar, timestep=self.timestep)

    def pump(self, G = None):
        if self.type == 0:
            return 0
        if G is None:
            G == self.Gsc
        return max(0, self.pumpObj.eval(G) + self.us)
    
    def pancreas(self, G):
        if self.type == 1:
            return 0
        u = 0
        for i in range(self.pancreas_n):
            u += self.pancreasObj.eval(G)
        return max(0, u/self.pancreas_n)
        
    def full_reset(self):
        self.reset()
        if self.type != 1:
            self.pancreasObj.reset()
        if self.type != 0:
            self.pumpObj.reset()

    def steadystate(self, uP):
        if self.model == "EHM":
            return HM_steadystate(self, uP=uP)
        if self.model == "MVP":
            return MVP_steadystate(self, uP=uP)
        return

    def G_from_u(self, u):
        if self.model == "EHM":
            return HM_G_from_u(self, u)
        if self.model == "MVP":
            return MVP_G_from_u(self, u)
        return

    def glucose_penalty(self, G = None):
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
        G : int, float, np.ndarray, list, default: None
            Glucose to evaluate penalty for. If not set, use current state.
        
        Returns
        -------
        p(G) : float
            Penalty
        """
        if G is None: # If G is not specified, use current G
            G = self.G
        func = lambda g :  1/2 * (18 * (g - self.Gbar))**2 + self.kappa/2 * max((18*(self.Gmin - g)), 0)**2
        if isinstance(G, (np.ndarray, list)):
            return np.array([func(Gi) for Gi in G])
        return func(G)
 
    def bolus_sim(self, bolus, meal_size, meal_idx = 0, iterations = 100, plot = False):
        ds = np.zeros(iterations)
        us = np.ones(iterations) * self.us
        ds[meal_idx] = meal_size / self.timestep # Ingestion 
        us[0] += bolus * 1000 / self.timestep
        info = self.simulate(ds, us)
        Gt = info["G"]
        p = self.glucose_penalty(Gt)
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


    def simulate(self, ds = None, uIs = None, uPs = None, iterations = None):
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
        if iterations is None:
            iterations = 0
            for arr in [ds, uIs, uPs]:
                if arr is not None:
                    iterations = max(len(arr), iterations)
            if iterations == 0:
                iterations = int(24 * 60 / self.timestep)
        if ds is None: # if no meal is given, set to zero.
            ds = np.zeros(iterations)
            dn = iterations
        else:
            ds = np.array([ds]).flatten()
            dn = len(ds)


        uPs = np.array([uPs]).flatten()
        uIs = np.array([uIs]).flatten()

        def uP_func(i):
            u_panc = self.pancreas(self.G)
            u_arr = uPs[i%len(uPs)]
            if u_arr is None:
                return u_panc
            if np.isnan(u_arr):
                return u_panc
            return u_arr
        
        def uI_func(i):
            u_pump = self.pump(self.Gsc)
            u_arr = uIs[i%len(uIs)]
            if u_arr is None:
                return u_pump
            if np.isnan(u_arr):
                return u_pump
            return u_arr

        info = dict()
        for i in self.state_keys:
            info[i]=np.empty(iterations+1)
            info[i][0]=getattr(self,i)
        info["t"] = self.time_arr(iterations+1)
        info["uP"] = []
        info["uI"] = []
        for i in range(iterations):
            d = ds[i%dn]
            uP = uP_func(i)
            uI = uI_func(i)
            dx = self.f_func(d = d, uI = uI, uP = uP)
            self.euler_step(dx)
            x = utils.ReLU(self.get_state())     
            self.update_state(x)
            for k in self.state_keys:
                info[k][i+1]=getattr(self,k)
            info["uP"].append(uP)
            info["uI"].append(uI)
        info["pens"]=self.glucose_penalty(info["G"])
        return info

    def meal_bolus(self, meal_size = 0, min_U = 0, max_U = 50, n = 50, iterations = 1000):
        Us = np.linspace(min_U, max_U, n)
        phi_best = 10**18
        for j, U in enumerate(Us):
            self.full_reset()
            phi, p, Gt = self.bolus_sim(U, meal_size, meal_idx=0, iterations=iterations)
            if phi < phi_best:
                p_best = p
                G_best = Gt
                phi_best = phi
                best_u = U
        return best_u, phi_best, p_best, G_best


    def optimal_bolus(self, meal_idx = 0, min_U = 0, max_U = 50, min_meal = 20, max_meal = 150, n = 50, iterations = 1000):
        Us = np.linspace(min_U, max_U, n)
        meals = np.linspace(min_meal, max_meal, n)
        res = np.empty((len(meals), len(Us)))
        for i, d0 in enumerate(meals):
            for j, U in enumerate(Us):
                self.full_reset()
                phi, _, _ = self.bolus_sim(U, d0, meal_idx=meal_idx, iterations=iterations)
                res[n - 1 - j ,i] = phi
        best = np.argmin(res, axis=0)
        best_us = [Us[n - 1- i] for i in best]
        return meals, best_us, res
    
    def day_sim(self, ds, bolus_data):
        n = len(ds)
        uIs = np.empty(n)
        uIs[:] = np.nan
        for m in bolus_data:
            uIs[m[0]] = 1000 * m[1] / self.timestep
            uIs[m[0]+1:m[2]] = 0
        return self.simulate(ds = ds, uIs = uIs)
    
    def hist(self,G_arr):
        Gbar_s=str(np.round(self.Gbar,3))
        Gmin_s=str(np.round(self.Gmin,3))
        bin_place=np.empty(len(G_arr))
        for i, G in enumerate(G_arr):
            if G<3:
                bin_place[i]=0
            elif 3<=G<self.Gmin:
                bin_place[i]=1
            elif self.Gmin<=G<self.Gbar-0.5:
                bin_place[i]=2
            elif self.Gbar-0.5<=self.Gbar+0.5:
                bin_place[i]=3
            elif self.Gbar+0.5<=G<10:
                bin_place[i]=4
            elif 10 <=G<13:
                bin_place[i]=5
            elif 13<=G:
                bin_place[i]=6
        plt.figure(figsize=(10,10))
        n,bins,patches=plt.hist(bin_place,bins=range(8),orientation="horizontal",align="left",density=True)
        colors=["#020249","#050578","#3c3cf4","#00FF7F","#f82828","#7e0202","#5a0000"]
        for c, p in zip(colors, patches):
            p.set_facecolor(c)
        plt.yticks(ticks=[0,1,2,3,4,5,6],labels=["Very high (13 < G )","high ( 10 < G <13)","Moderately high (" +Gbar_s+ " + 0.5 < G < 10)","Ideal (" +Gbar_s+" -0.5 < G < " +Gbar_s+" +0.5)","moderately low (" +Gmin_s+ " < G <"+  Gbar_s + "-0.5)","Low (3 < G <"+ Gmin_s+")","very low (G < 3)"])
        plt.tick_params(axis='y', labelsize=5)
        plt.title("Percent of time spent at different glucose levels      Steady state G is " + Gbar_s)
        plt.show()
        return

    def statePlot(self,infodict,shape,size,keylist,fonts):

        """ 
        Makes plot of different states. 
        Parameters
        ----------

        infodict: Dictionary of all states/disturbances that could possibly be plotted  from a given simulation

        shape: tuple or list indicating layout of plots ("number of rows", "number of columns")

        size: tuple of list indicating size of figure ("length", "width")

        keylist: A list of lists in row-major order of where to put each plot. 
        
        Returns
        -------

        plots of given states

        Example
        -------
        For example: statePLot( self, info, (1,3), (20,20) , [["D1","D2],["Isc"],["x1","x2","x3"]]
        Creates a plot with  D1 and D2 in one figure, Isc in another and x1, x2 and x3 together in a third figure, in a 1 X 3 layout and 20X20 size.  
        """


        fig,ax=plt.subplots(nrows=shape[0],ncols=shape[1],figsize=size)
        ax=ax.flatten()
        colorlist=["#0B31A5","#D3004C","#107C10"]
        titles={
            "MVP":{
            "D1":["D2","[mmol]"],
            "D2":["D2","[mmol]"],
            "Isc": ["Subc. insulin","[mU/L]"],
            "Ip" : ["Insulin in plasma","[mU/L]"],
            "Ieff": ["Effective insulin","[mU/L]"],
            "G" : ["Blood glucose","[mmol/L]"],
            "Gsc": ["Subc. glucose","[mmol/L]"], 
            "pens": ["Penalty function", " "],
            "uI" : ["Injected Insulin", "[mU/min]"],
            "uP" : ["Insulin from pancreas", "[mU/min]"]
            },

            "EHM": {
            "G" : ["Blood Glucose","[mmol/L]"],
            "Gsc": ["Subc. glucose","[mmol/L]"], 

            "Q1" : ["Main bloods gluc","[mmol]"],
            "Q2" : ["Gluc in peripheral tissue","[mmol]"],
            "S1" : ["Subc. insulin variable 1","[mU]"],
            "S2" : ["Subc. insulin variable 2","[mU]"],
            "I"  : ["Plasma insulin conc.","[mU/L]"],
            "x1": ["I eff. on gluc 1","[1/min]"],
            "x2": ["I eff. on gluc 2","[1/min]"],
            "x3": ["I eff. on gluc 3","[1/min]"],
            "D1": ["Meal Glucose 1","[mmol]"],
            "D2": ["Meal Glucose 2","[mmol]"],
            "Z1": ["Subc. Glucagon","[μg]"],
            "Z2": ["plasma Glucagon","[μg]"],
            "E1": ["Short-term exercise eff.","[min]"],
            "E2": ["Long-term exercise eff.","[min]"],
            "pens": ["Penalty function", " "],
            "uI" : ["Injected Insulin", "[mU/min]"],
            "uP" : ["Insulin from pancreas", "[mU/min]"]
            }
        }
        
        for i,l in enumerate(keylist):
                title=""
                for c, k in enumerate(l):
                    if c==0:
                        title+=titles[self.model][k][0]
                    elif 0 < c < len(l)-1:
                        title+=", " + titles[self.model][k][0]
                    elif c==len(l)-1:
                        title+=" and "+titles[self.model][k][0]
                    max_l = min(len(infodict["t"]), len(infodict[k]))

                    if k=="G":
                        ax[i].plot(infodict["t"][:max_l]/60,4.44*np.ones(max_l),"--",color="#998F85",label="minimum glucose")
                    ax[i].plot((infodict["t"][:max_l])/60,infodict[k][:max_l],".",label=k,color=colorlist[c])
                    ax[i].set_title(title,fontsize=fonts)
                    ax[i].set_xlabel("Time [h]",fontsize=fonts)
                    ax[i].set_ylabel(titles[self.model][k][1])
                    ax[i].set_xlim(0,infodict["t"][:max_l][-1]/60)
                    ax[i].set_xticks(np.linspace(0,infodict["t"][:max_l][-1]/60,5))
                    ax[i].tick_params(axis='x', labelsize=5)
                ax[i].legend(loc="lower left")
        plt.show()
        fig.tight_layout()
        return

def find_ss(model = "EHM", **kwargs): 
    p = Patient(patient_type = 0, model = model, **kwargs)   
    def cost(G):
        _, u = p.pancreasObj.steadystate(G)
        G_res = p.G_from_u(u)
        return G_res-G
    return root_scalar(cost,  bracket= [0.5,20])

def baseline_patient(patient_type = 1, model = "EHM", **kwargs):
        Gbar = kwargs.get("Gbar" , find_ss(model, **kwargs).root)
        patient = Patient(patient_type = patient_type, model = model, Gbar = Gbar, **kwargs)
        uP = patient.pancreas(Gbar)
        x0, uI = patient.steadystate(uP)
        patient.us = max(0,uI)
        patient.update_state(x0) # set to steady state
        for key in patient.state_keys: # also set "x0" values
            setattr(patient, key+"0", getattr(patient, key))
        return patient


