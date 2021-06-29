from math import sqrt
from scipy.stats import norm
import numpy as np


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

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
