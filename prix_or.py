from math import sqrt
from scipy.stats import norm
import numpy as np


def brownian(x0, n, dt, delta, out=None):

    x0 = np.asarray(x0)

    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    if out is None:
        out = np.empty(r.shape)

    np.cumsum(r, axis=-1, out=out)

    out += np.expand_dims(x0, axis=-1)

    return out


n = 12


def prix(prix_init,nb_annees, n,delta=1, sigma=0.15, alpha=0.12, To=0):
    """
    En entree, le prix au début de l'etude, le nombre d'années sur lesquelles l'etude porte et n est le nombre d'intervalles
    sur lesquelles une annee est partitionnee. 
    Dans ce modele la variation suit un brownien geometrique, delta est la vitesse du mouvement brownien. sigma est la volatilite du
    prix de l'or.
    En sortie, le prix de l'or à chaque année selon le modèle.
    """
    prix = prix_init
    liste_prix = np.zeros(nb_annees)
    for i in range(nb_annees):
        liste_prix[i] = prix
        W = 0.01*np.exp(brownian(0, n, 1/n, delta))
        #calcul de l'intégrale par méthode des rectangles:
        integrale = 0
        for j in range(n):
            integrale += sigma*np.exp(-alpha*(To-j*(1/n)))*W[j]
        prix = prix*np.exp(integrale - ((sigma**2)/(4*alpha))*np.exp(-2*alpha*To)*(np.exp(2*alpha)-1))
    return liste_prix
