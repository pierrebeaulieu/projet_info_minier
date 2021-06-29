import numpy as np

def loi_func(loi:str, moyenne, bande, n, bool = True):
    if loi == "triangular":
        sortie = np.array( n* [np.random.triangular(moyenne - moyenne*bande/100, moyenne, moyenne + moyenne*bande/100)])
        
    if loi == "normal":
        sortie = np.array(n * [np.random.normal(moyenne, bande*moyenne)])

    if loi == "uniform":
        sortie =  np.array(n * [np.random.uniform(moyenne-bande*moyenne/200, moyenne+bande*moyenne/200)])
    if not bool:
        sortie[1:] = 0
    return sortie
        
