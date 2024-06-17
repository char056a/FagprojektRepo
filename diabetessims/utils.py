from scipy.optimize import minimize
import numpy as np


def ReLU(x):
    return x * (x > 0)

def cohen_coon(ybar, t, delay):
        Kp = t/(ybar * delay) * (4/3 + delay/(4 * t))
        Ti = delay * (32 + 6 * delay/t)/(13+8*delay/t)
        Td = 4 * delay / (11 + 2*delay / t)
        return Kp, Ti, Td

def piecewise_linear(b1, a1, b2, a2, x_split):
    def func(x):
        if x < x_split:
            return a1 * x + b1
        else:
            return a2 * x + b2
    def func_arr(xs):
        if isinstance(xs, (float, int)):
            return func(xs)
        ys = []
        for x in xs:
            ys.append(func(x))
        return ys
    return func_arr

def piecewise_linear_fit(x, y):
    n = len(x)
    A = np.array([np.ones(n), x]).T
    def cost(split):
        sol1, res1, _, _ = np.linalg.lstsq(A[:split], y[:split], rcond=None)
        sol2, res2, _, _ = np.linalg.lstsq(A[split:], y[split:], rcond=None)
        return res1 + res2, sol1, sol2
    
    min_res = 200000000

    for spl in range(n-1):
        res, sol1, sol2 = cost(spl)
        if len(res):
            if min_res >= res:
                best_spl = spl
                func = piecewise_linear(*sol1, *sol2, x[spl])
                min_res = res
    return min_res, func, best_spl

def timestamp_arr(data, timestep, fill = 0, h=24):
    n = int(h * 60 / timestep)
    arr = np.empty(n)
    arr[:] = fill
    for m in data:
        if len(m) == 2:
            idx = int(m[1] / timestep * 60)
            arr[idx] = m[0]/ timestep
        else:
            idx = (np.array(m)[1:3] / timestep * 60).astype(int)

            arr[idx[0]:idx[1]] = m[0]/(idx[1]-idx[0])/timestep
    return arr

def filter(arr, minval=None, maxval=None):
    n = len(x)
    for i,x in enumerate(arr):
        if x is None:
            x[i] = x[(i-1)%n]
        if x == np.nan:
            x[i] = x[(i-1)%n]
        if minval is not None:
            if x < minval:
                x[i] = x[(i-1)%n]
        if maxval is not None:
            if x > maxval:
                x[i] = x[(i-1)%n]
    return x

def generate_table(meals, bolus):
    bolus = bolus.T
    start = meals[:, 1] 
    end = meals[:, 2]
    n = 3
    if bolus.ndim == 2:
        n += bolus.shape[1] - 1
    inner = "Time & Meal size (g) & Bolus Size (mU) \\\\ \\hline \n"
    for m,b in zip(meals, bolus):
        start, end = [f"{str(int(m[i+1])).zfill(2)}:{str(int(60*(m[i+1]%1))).zfill(2)}" for i in range(2)]
        row1 = f"{start}-{end} & {int(m[0])} "
        if bolus.ndim == 2:
            row2 = "".join([f"& {int(b[i])} "  for i in range(n-2)]) 
        else:
            row2 = f"% {int(b)}"
        row3 = "\\\\ \\hline \n"
        inner += row1 + row2 + row3
    return "\\begin{table}[]\n\\begin{tabular}{|"+"".join(["r|" for i in range(n)])+"}\\hline \n"+inner+"\\end{tabular}\n\\end{table}"


class Wrapper:
    def __init__(self, mod, instance):
        self.mod = mod
        self.instance = instance

    def __getattr__(self, name):
        func = getattr(self.mod, name)
        if callable(func):
            def wrapper(*args, **kwargs):
                return func(self.instance, *args, **kwargs)
            return wrapper
        return func
