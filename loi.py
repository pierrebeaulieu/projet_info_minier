import numpy as np

def loi_func(loi:str, moyenne, bande, n):
    if loi == "triangular":
        return np.array( n* [np.random.triangular(moyenne - moyenne*bande/100, moyenne, moyenne + moyenne*bande/100)])
        
    if loi == "normal":
        return np.array(n * [np.random.normal(moyenne, bande*moyenne)])

    if loi == "uniform":
        return np.array(n * [np.random.uniform(moyenne-bande*moyenne/200, moyenne+bande*moyenne/200)])