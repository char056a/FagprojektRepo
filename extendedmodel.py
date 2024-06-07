import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from odeclass import ODE
import pancreas

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
    dIsc = uI/(self.tau1 * self.C1) - self.Isc/self.tau1
    dIp = (self.Isc - self.Ip)/self.tau2 + uP/self.C1
    dIeff = -self.p2 * self.Ieff + self.p2 * self.S1 * self.Ip
    dG = - (self.gezi + self.Ieff) * self.G + self.egp0 + 1000 * self.D2 / (self.Vg * self.taum)
    dGsc = (self.G - self.Gsc) / self.timestep

    dx = np.array([dD1, dD2, dIsc, dIp, dIeff, dG, dGsc])
    return dx



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

    F01c = min(self.F01, self.F01 * self.G / 4.5) * self.BW
    FR = max(0.003 * (self.G - 9) * self.VG * self.BW, 0) 

    UG = self.D2 / self.TauD
    UI = self.S2/self.TauS

    dG = (G - self.G)/self.TauIG
    dQ1 = UG - F01c - FR - self.x1 * self.Q1 + self.k12 * self.Q2 + self.BW * self.EGP0 * (1 - self.x3)
    dQ2 = self.Q1 * self.x1 - (self.k12 + self.x2)*self.Q2
    dS1 = (uI - self.S1 / self.TauS)
    dS2 = ((self.S1 - self.S2)/self.TauS)
    dI = ((uP + UI) / (self.VI * self.BW) - self.ke * self.I) # Den her kan være wack
    dx1 = self.kb1 * self.I - self.ka1 * self.x1
    dx2 = self.kb2 * self.I - self.ka2 * self.x2
    dx3 = self.kb3 * self.I - self.ka3 * self.x3
    dD1 = self.AG * D - self.D1 / self.TauD
    dD2 = (self.D1 - self.D2)/self.TauD
    dx = np.array([dG, dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2])
    return dx



class Patient(ODE):
    def __init__(self, patient_type, model = "EHM", **kwargs):
        self.model = model.upper()
        
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
            self.pancreasObj = pancreas.PKPM(timestep=self.timestep, Gbar=self.Gbar)
        if patient_type != 0:
            self.pumpObj = pancreas.PID(Kp = self.Kp, Td = self.Td, Ti = self.Ti, ybar = self.Gbar, timestep=self.timestep)
            
        if patient_type == 0:
            self.pancreas = lambda G : self.pancreasObj.eval(G)
            self.pump = lambda G : 0
        if patient_type == 1:
            self.pancreas = lambda G : 0
            self.pump = lambda G : max(self.pumpObj.eval(G) + self.us,0)
        if patient_type == 2:
            self.pancreas = lambda G : self.W * self.pancreasObj(G)
            self.pump = lambda G : max(self.pumpObj.eval(G) + self.us,0)
     
            

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
        func = lambda g :  1/2 * (g - self.Gbar)**2 + self.kappa/2 * max((self.Gmin - g), 0)**2
        if isinstance(G, (np.ndarray, list)):
            return np.array([func(Gi) for Gi in G])
        return func(G)
 
    def bolus_sim(self, bolus, meal_size, meal_idx = 0, iterations = 100, plot = False):
        ds = np.zeros(iterations)
        us = np.ones(iterations) * self.us
        ds[meal_idx] = meal_size / self.timestep # Ingestion 
        us[0] += bolus * 1000 / self.timestep
        states, _ = self.simulate(ds, us)
        Gt = self.get_attr(states, "G")
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
        if ds is None: # if no meal is given, set to zero.
            if iterations is None:
                iterations = 24 * 60 // self.timestep # if no iteration is given, set to 24h
            ds = np.zeros(iterations)
            dn = iterations
        else:
            ds = np.array([ds]).flatten()
            dn = len(ds)
            if iterations is None:
                iterations = dn
        
        if uPs is None:
            uP_func = lambda i : self.pancreas(self.G)
        else:
            uPs = np.array([uPs]).flatten()
            uP_func = lambda i : uPs[i%len(uPs)]

        if uIs is None:
            uI_func = lambda i : self.pump(self.G)
        else:
            uIs = np.array([uIs]).flatten()
            uI_func = lambda i : uIs[i%len(uIs)]


        info = dict()
        for i in self.state_keys:
            info[i]=np.empty(iterations+1)
            info[i][0]=getattr(self,i)
        info["t"] = self.time_arr(iterations+1)

        info["uP"] = []
        info["uI"] = []
        for i in range(iterations):
            d = ds[i%dn]
            #uP = self.pancreas(self.G)
            #uI = self.pump(self.G)
            uP = uP_func(i)
            uI = uI_func(i)
            dx = self.f_func(d = d, uI = uI, uP = uP)
            print(dx)
            self.euler_step(dx)     

            for k in self.state_keys:
                info[k][i+1]=getattr(self,k)
            info["uP"].append(uP)
            info["uI"].append(uI)

        info["pens"]=self.glucose_penalty()
        return info



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
            "pens": ["Penalty function", " "]
            },

            "EHM": {
            "G" : ["Blood Glucose","[mmol/L]"],
            "Q1" : ["Main bloods gluc","[mmol]"],
            "Q2" : ["Gluc in peripheral tissue","[mmol]"],
            "S1" : ["Subc. insulin variable 1","[mU]"],
            "S2" : ["Subc. insulin variable 2","[mU]"],
            "I"  : ["Plasma insulin conc.","[mU/L]"],
            "x1": ["I effect on gluc distrib/transp","[1/min]"],
            "x2": ["I effect on gluc disposal","[1/min]"],
            "x3": ["I effect on endogenous gluc prod","[1/min]"],
            "D1": ["Meal Glucose 1","[mmol]"],
            "D2": ["Meal Glucose 2","[mmol]"],
            "Z1": ["Subc. Glucagon","[μg]"],
            "Z2": ["plasma Glucagon","[μg]"],
            "E1": ["Short-term exercise eff.","[min]"],
            "E2": ["Long-term exercise eff.","[min]"],
            "pens": ["Penalty function", " "]
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
                    if k=="G":
                        ax[i].plot(infodict["t"]/60,4.44*np.ones(len(infodict["t"])),"--",color="#998F85",label="minimum glucose")
                    ax[i].plot((infodict["t"])/60,infodict[k],".",label=k,color=colorlist[c])
                    ax[i].set_title(title,fontsize=fonts)
                    ax[i].set_xlabel("Time [h]",fontsize=fonts)
                    ax[i].set_ylabel(titles[self.model][k][1])
                    ax[i].set_xlim(0,infodict["t"][-1]/60)
                    ax[i].set_xticks(np.linspace(0,infodict["t"][-1]/60,5))
                    ax[i].tick_params(axis='x', labelsize=7)
                ax[i].legend(loc="lower left")
        plt.show()
        return




p = Patient(1, "EHM")
info = p.simulate()
print(info["I"])
p.statePlot(infodict=info,shape=(2,2), size=(5,15),  keylist=[["I"],["Q1", "Q2"],["D1","D2"]],fonts=7)
#p.optimal_bolus()