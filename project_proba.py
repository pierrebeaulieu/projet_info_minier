from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from ipywidgets import interact, FloatText
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# Pour récupérer les données
import numpy as np
import matplotlib.pyplot as plt
import csv

# n est les nombre d'années sur lesquelles on veut effectuer le calcul

def app():
    def extraction_csv(nom_fichier, n):

        # ouverture du fichier

        fichier = open(nom_fichier, 'r')

        # Lecture du fichier

        contenu = csv.reader(fichier, delimiter=";")

        # Définition de nos tableau d'entree

        investissement = []
        cout_tonne_remuee = []
        ratio_sterile = []
        cout_traitement = []
        charges_fixes = []
        prix_or = []
        taux_recuperation_or = []
        prop_or_paye_dore = []
        nombre_grammes_once = []
        taux_actualisation = []
        tonnage_geol = []
        teneur_minerai_geol = []
        taux_recup = []
        dilution_minerai = []
        rythme_prod_annee = []
        premiere_annee_prod = []

        # Récupération des données sous forme de listes

        for row in contenu:

            if row[0] == "Tonnage de minerai geologique":
                tonnage_geol.append(float(row[2]))
            if row[0] == "Teneur du minerai geologique":
                teneur_minerai_geol.append(float(row[2]))
            if row[0] == "Taux de recuperation du gisement":
                taux_recup.append(float(row[2]))
            if row[0] == "Dilution du minerai":
                dilution_minerai.append(float(row[2]))
            if row[0] == "Rythme de production annuelle de minerai":
                rythme_prod_annee.append(float(row[2]))
            if row[0] == "Premiere annee de production":
                premiere_annee_prod.append(float(row[2]))
            if row[4] == "Investissement initial pour l'exploitation":
                investissement.append(float(row[6]))
            if row[4] == "Cout par tonne remuee dans la mine a ciel ouvert":
                cout_tonne_remuee.append(float(row[6]))
            if row[4] == "Ratio sterile sur minerai dans la mine a ciel ouvert":
                ratio_sterile.append(float(row[6]))
            if row[4] == "Cout de traitement par tonne de minerai":

                cout_traitement.append(float(row[6]))
            if row[4] == "Charges fixes annuelles":
                charges_fixes.append(float(row[6]))
            if row[4] == "Prix de vente de l'or":
                prix_or.append(float(row[6]))
            if row[4] == "Taux de recuperation de l'or dans le traitement":
                taux_recuperation_or.append(float(row[6]))
            if row[4] == "Proportion d'or paye dans le dore":
                prop_or_paye_dore.append(float(row[6]))
            if row[4] == "Nombre de grammes dans une once":
                nombre_grammes_once.append(float(row[6]))
            if row[4] == "Taux d'actualisation":
                taux_actualisation.append(float(row[6]))
        fichier.close()

        # On prend des valeurs constantes dans un premier temps sur n années.
        for i in range(n-1):
            investissement.append(investissement[0])
        investissement = np.array(investissement)
        cout_tonne_remuee = np.array(n*cout_tonne_remuee)
        ratio_sterile = np.array(n*ratio_sterile)
        cout_traitement = np.array(n*cout_traitement)
        charges_fixes = np.array(n*charges_fixes)
        prix_or = np.array(n*prix_or)
        taux_recuperation_or = np.array(n*taux_recuperation_or)
        prop_or_paye_dore = np.array(n*prop_or_paye_dore)
        nombre_grammes_once = np.array(n*nombre_grammes_once)
        taux_actualisation = np.array(n*taux_actualisation)
        tonnage_geol = np.array(n*tonnage_geol)
        teneur_minerai_geol = np.array(n*teneur_minerai_geol)
        taux_recup = np.array(n*taux_recup)
        dilution_minerai = np.array(n*dilution_minerai)
        rythme_prod_annee = np.array(n*rythme_prod_annee)
        premiere_annee_prod = np.array(n*premiere_annee_prod)

        # Creation du dictionnaire

        entree = {}
        entree["investissement"] = investissement
        entree["cout_tonne_remuee"] = cout_tonne_remuee
        entree["ratio_sterile"] = ratio_sterile
        entree["cout_traitement"] = cout_traitement
        entree["charges_fixes"] = charges_fixes
        entree["prix_or"] = prix_or
        entree["taux_recuperation_or"] = taux_recuperation_or
        entree["prop_or_paye_dore"] = prop_or_paye_dore
        entree["nombre_grammes_once"] = nombre_grammes_once
        entree["taux_actualisation"] = taux_actualisation
        entree["tonnage_geol"] = tonnage_geol
        entree["teneur_minerai_geol"] = teneur_minerai_geol
        entree["taux_recup"] = taux_recup
        entree["dilution_minerai"] = dilution_minerai
        entree["rythme_prod_annee"] = rythme_prod_annee
        entree["premiere_annee_prod"] = premiere_annee_prod
    
        return entree


    class entree:
        def __init__(self, params):
            self.n = len(params.get('ratio_sterile'))
            self.investissement = params.get('investissement')
            self.cout_tonne_remuee = params.get('cout_tonne_remuee')
            self.ratio_sterile = params.get('ratio_sterile')
            self.cout_traitement = params.get('cout_traitement')
            self.charges_fixes = params.get('charges_fixes')
            self.prix_or = params.get('prix_or')
            self.taux_recuperation_or = params.get('taux_recuperation_or')
            self.prop_or_paye_dore = params.get('prop_or_paye_dore')
            self.nombre_grammes_once = params.get('nombre_grammes_once')
            self.taux_actualisation = params.get('taux_actualisation')
            self.tonnage_geol = params.get('tonnage_geol')
            self.teneur_minerai_geol = params.get('teneur_minerai_geol')
            self.taux_recup = params.get('taux_recup')
            self.dilution_minerai = params.get('dilution_minerai')
            self.rythme_prod_annee = params.get('rythme_prod_annee')
            self.premiere_annee_prod = params.get('premiere_annee_prod')

        def __repr__(self):
            return f"** investissement initiale pour l'exploitation ** : {self.investissement[0]} M$  , \n \n  ** prix de l'or ** = {self.prix_or[0]} $/oz, \n \
                \n ** Ratio sterile sur minerai dans la mine à ciel ouvert ** = {self.ratio_sterile[0]},\n \
        \n  ** Teneur du minerai geologique ** = {self.teneur_minerai_geol[0]} g/t,  \n \n  ** Taux de récuperation du gisement ** = {self.taux_recup[0]} %,\n \n ** Dilution du minerai ** = {self.dilution_minerai[0]}, \n \n  **  Rythme de production annuelle de minerai ** = {self.rythme_prod_annee[0]} Mt/an,\n \n ** Coût par tonne remuee dans la mine à ciel ouvert ** = {self.cout_tonne_remuee[0]} $/t roche, \n \n ** Taux d'actualisation ** = {self.taux_actualisation[0]} % \n \n ** Charges fixes annuelles ** = {self.charges_fixes[0]} M$/an. "

        def cout_exploitation_tonne_minerai(self):
            return self.cout_tonne_remuee * (self.ratio_sterile + np.ones(self.n))

        def teneur_minerai_industriel(self):
            return self.teneur_minerai_geol/(self.dilution_minerai/100 + np.ones(self.n))

        def tonnage_industriel(self):
            return self.tonnage_geol * self.taux_recup/100 * (self.dilution_minerai/100 + np.ones(self.n))

        def qte_or_paye(self):
            return self.teneur_minerai_industriel() * self.taux_recuperation_or/100 * self.prop_or_paye_dore/100

        def recette_tonne_minerai(self):
            return self.qte_or_paye() * self.prix_or / self.nombre_grammes_once

        def dep_operatoiresMCO(self):
            return self.rythme_prod_annee * self.cout_exploitation_tonne_minerai()

        def dep_operatoiresTraitement(self):
            return self.rythme_prod_annee * self.cout_traitement

        def charges_fixes_(self):
            return self.charges_fixes

        def dep_operatoiresTotale(self):
            return self.charges_fixes_() + self.dep_operatoiresTraitement() + self.dep_operatoiresMCO()

        def recette(self):
            return self.rythme_prod_annee * self.recette_tonne_minerai()

        def cash_flow(self):
            cash = self.recette() - self.dep_operatoiresTotale() - self.cout_traitement
            cash[0] -= self.investissement[0]
            return cash

    # investissement pas pris en compte, revoir formule

        def cumul_cash_flow(self):
            cumul = self.cash_flow()
            sortie = np.zeros(self.n)
            N = len(cumul)
            for i in range(N):
                sortie[i] = np.sum(cumul[:i])
            return sortie

        def taux_actualisation_(self):
            actua = 1 / (np.ones(self.n) + self.taux_actualisation/100)
            final = np.zeros(self.n)
            N = len(actua)
            for i in range(N):
                final[i] = np.product(actua[:i])
            return final

        def cash_flow_actu(self):
            CF = self.cash_flow()
            N = len(CF)
            facteur = self.taux_actualisation_()
            CFA = np.copy(CF)
            return facteur * CF

        def cumul_cash_flow_actu(self):
            cumul = self.cash_flow_actu()
            sortie = np.zeros(self.n)
            N = len(cumul)
            for i in range(N):
                sortie[i] = np.sum(cumul[:i])
            return sortie

        def dri(self):
            cumul = self.cumul_cash_flow_actu()
            dr = 0
            while dr < self.n:
                if cumul[dr] >= 0:
                    return dr
                else:
                    dr += 1
            return -1         # -1 pour signaler que le délai de retour n'est jamais atteint

        def tri(self):
            N = self.n
            Tri=[]
            CF = self.cash_flow()
            for i in range(N):
                a = 0.0
                b = 0.35
                c = (a+b)/2
                epsilon = 0.1
                k = 0 #nombre itération
                p = 100
                def f(x):
                    taux = np.zeros(self.n) #initialisation
                    for k in range(N):
                        taux[k] = 1/(1+x)**k
                    actu = taux * CF
                    return np.sum(actu[:i])
                if f(b)*f(a) < 0:
                    while abs(f(c)) > epsilon and k <p:
                        if f(c)*f(a) > 0 : 
                            a = c
                            c = (b+c)/2

                        else :
                            b = c 
                            c= (a+c)/2
                        k += 1
                    tri = c
                    Tri.append(tri)
                else : Tri.append(0)
            return Tri


    n=15
    # on extrait les données du tableau de base et modifie en fonction du choix du client
    valeurs = entree(extraction_csv("donnees_entree_proj_minier.csv", n))

    valeurs_non_modif = deepcopy(valeurs)

    st.header("Modélisation des lois de probabilité des paramètres incertains")

    # On modélise la loi de probabilité de l'investissement en tenant compte du choix du client

    st.subheader("Loi de probabilité suivie par l'investissement")

    loi_invest = st.selectbox("loi de l'investissement", ['triangular', 'normal', 'uniform'])

    if loi_invest == "triangular":
        centre_triang = st.number_input(f"L'investissement est modélisé par une loi triangulaire centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.investissement[0])
        bande_triang = st.number_input("Pourcentage de la bande de l'investissement:", 1, 50, 10, 1)
        st.write(f'On considère donc une loi triangulaire centrée sur {centre_triang}M$ et de bande {bande_triang}%')

        arr = np.random.triangular(centre_triang - centre_triang*bande_triang/100, centre_triang, centre_triang + centre_triang*bande_triang/100, size=10000)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=200)
        st.pyplot(fig)

        valeurs.investissement = np.array(n * [np.random.triangular(centre_triang - centre_triang*bande_triang/100, centre_triang, centre_triang + centre_triang*bande_triang/100)])
        st.write("Les nouvelles valeurs pour l'investissement sont (pour une simulation):", valeurs.investissement)

    if loi_invest == "normal":
        moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
        ecart_type = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
        st.write(f"On considère donc une loi normale de moyenne {moyenne_invest} et d'ecart_type {ecart_type}")

        valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, ecart_type*moyenne_invest)])


    # On modélise la loi de probabilité du coût total par une loi uniforme

    st.subheader("Loi de probabilité suivie par le coût")

    centre_unif = st.number_input(f"Le coût est modélisé par une loi uniforme centrée sur la valeur {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
    bande_unif = st.number_input("Pourcentage de la bande du coût:", 1, 50, 10, 1)
    st.write(f'On considère donc une loi uniforme centrée sur {centre_unif}$/t et de bande {bande_unif}%')

    valeurs.cout_tonne_remuee = np.array(n * [np.random.uniform(centre_unif-centre_unif*bande_unif/200, centre_unif+centre_unif*bande_unif/200)])

    # La teneur suit une loi normale

    st.subheader("Loi de probabilité suivie par la teneur en minerai")

    moyenne_teneur = st.number_input(f"La teneur est modélisée par une loi normale de moyenne {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
    ecart_type = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
    st.write(f"On considère donc une loi normale de moyenne {moyenne_teneur} et d'ecart_type {ecart_type}")

    valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, ecart_type*moyenne_teneur)])

    # De même pour le tonnage

    st.subheader("Loi de probabilité suivie par le tonnage")

    moyenne_tonnage = st.number_input(f"Le tonnage est modélisé par une loi normale de moyenne {valeurs.tonnage_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.tonnage_geol[0])
    ecart_type = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du tonnage", 1/10)
    st.write(f"On considère donc une loi normale de moyenne {moyenne_tonnage} et d'ecart_type {ecart_type}")

    valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, ecart_type*moyenne_tonnage)])

    # On trace la sortie voulue du client

    st.header("Analyse des résultats avec ces nouvelles lois")

    sortie_voulue = st.selectbox("Résultat", ['cash_flow_actu', 'cumul_cash_flow', 'tri'], ) 

    abscisse = np.arange(1, n)
    fig, ax = plt.subplots()
    plt.style.use('seaborn')
    evaluable = "valeurs."+f"{sortie_voulue}"+"()"
    sortie = eval(evaluable)[1:]
    ax = plt.scatter(abscisse, sortie,
                    label=f"valeur issue d'une seule simulation")

    plt.ylabel('$ (en millions) ')
    plt.xlabel("annee")
    plt.title(f"{sortie_voulue}")
    plt.legend()

    st.pyplot(fig)

    # On fait maintenant de même pour un grand nombre de simulations

    st.write("Voyons maintenant ce que ça donne sur un grand nombre de simulations")

    # On demande le nombre de simulations 

    nb_simu = st.number_input("Nombre de simulations", 1, 10000, 1000, 1)

    i = 1
    

    while i < nb_simu:
        i+=1
        valeurs.investissement += np.array(n * [np.random.triangular(centre_triang - centre_triang*bande_triang/100, centre_triang, centre_triang + centre_triang*bande_triang/100)])
        valeurs.cout_tonne_remuee += np.array(n* [np.random.uniform(centre_unif-centre_unif*bande_unif/200, centre_unif+centre_unif*bande_unif/200)])
        valeurs.tonnage_geol += np.array(n * [np.random.normal(moyenne_tonnage, ecart_type*moyenne_tonnage)])
        valeurs.teneur_minerai_geol += np.array(n * [np.random.normal(moyenne_teneur, ecart_type*moyenne_teneur)])

    valeurs.investissement *= 1/nb_simu
    valeurs.cout_tonne_remuee *= 1/nb_simu
    valeurs.tonnage_geol *= 1/nb_simu
    valeurs.teneur_minerai_geol *= 1/nb_simu

    fig, ax = plt.subplots()
    plt.style.use('seaborn')
    sortie = eval(evaluable)[1:]

    evaluable_non_modif = "valeurs_non_modif."+f"{sortie_voulue}"+"()"
    sortie_non_modif = eval(evaluable_non_modif)[1:]
    ax = plt.scatter(abscisse, sortie,
                    label=f"valeur moyenne issue d'un grand nombre de simulations")

    ax = plt.scatter(abscisse, sortie_non_modif,
                    label=f"réalisation à partir des valeurs déterministes")


    plt.ylabel('$ (en millions) ')
    plt.xlabel("annee")
    plt.title(f"{sortie_voulue}")
    plt.legend()

    st.pyplot(fig)

    st.write("On vérifie bien que la moyenne issue des simulations coïncide étroitement avec le calcul déterministe")

    # Demandons maintenant un TRI objectif

    st.header('Statistiques de rentabilité interne')

    tri_goal = st.number_input("Objectif de TRI", 0.005, 0.5, 0.05, 0.005)
    annee = st.number_input("A l'année:", 1, 15, 15, 1)

    i = 0
    iter_tri = 0

    while i < nb_simu:
        i+=1
        valeurs.investissement = np.array(n * [np.random.triangular(centre_triang - centre_triang*bande_triang/100, centre_triang, centre_triang + centre_triang*bande_triang/100)])
        valeurs.cout_tonne_remuee = np.array(n* [np.random.uniform(centre_unif-centre_unif*bande_unif/200, centre_unif+centre_unif*bande_unif/200)])
        valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, ecart_type*moyenne_tonnage)])
        valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, ecart_type*moyenne_teneur)])

        # pour la proba que tri>tri_goal
        if valeurs.tri()[annee-1] > tri_goal:
            iter_tri += 1

    iter_tri *= 1/nb_simu

    st.write(f"La probabilité que le TRI soit supérieur à {tri_goal} à l'année {annee} vaut {iter_tri}")

    # Calculons alors le risque de rentabilité du projet

    st.header("Risque de rentabilité de l'investissement")

    i = 0
    iter_van = 0
    annee = st.number_input("Caculons la probabilité que l'investissement ne soit pas rentable à l'année:", 1, 15, 15, 1)
    esperance_perte = 0

    while i < nb_simu:
        i+=1
        valeurs.investissement = np.array(n * [np.random.triangular(centre_triang - centre_triang*bande_triang/100, centre_triang, centre_triang + centre_triang*bande_triang/100)])
        valeurs.cout_tonne_remuee = np.array(n* [np.random.uniform(centre_unif-centre_unif*bande_unif/200, centre_unif+centre_unif*bande_unif/200)])
        valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, ecart_type*moyenne_tonnage)])
        valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, ecart_type*moyenne_teneur)])

        # pour la proba que van<0
        if valeurs.cumul_cash_flow()[annee-1] < 0:
            iter_van += 1
            esperance_perte += valeurs.cumul_cash_flow()[annee-1]

    esperance_perte *= 1/nb_simu
    iter_van *= 1/nb_simu

    st.write(f"La probabilité que le VAN soit négatif à l'année {annee} vaut {iter_van}")

    st.write(f"De plus, l'espérance de perte à l'année {annee} vaut {esperance_perte }M$")
