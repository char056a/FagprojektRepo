from scipy.optimize import minimize
import numpy as np

def piecewise_linear(b1, a1, b2, a2, x_split):
    def func(x):
        if x < x_split:
            return a1 * x + b1
        else:
            return a2 * x + b2
    def func_arr(xs):
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
