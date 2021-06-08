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


    n=1
    # on extrait les données du tableau de base et modifie en fonction du choix du client
    valeurs = entree(extraction_csv("donnees_entree_proj_minier.csv", n))



    st.markdown("# Projet informatique sur l'économie d'une exploitation minière # ")

    st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
    st.text(" ")
    st.text(' ')


    if st.checkbox("Voir les données d'entrées", value = True): 
        st.subheader("Données d'entrée")
        st.markdown(valeurs)


    #détermine le nombre de paramètre à modifier
    u = st.number_input("nombre d'entrées à modifier", 1,5,step=1)
    annees = st.number_input("nombre d'années considérées", 0, 50, 15)

    st.markdown(
        "### Dans un premier temps, on veut déterminer la conséquence de la modification de chaque paramètre d'entrée toute chose égale par ailleurs ### \n  \n \n  ")
    st.markdown("\n \n ")
    def user_input(k):
        coeff = [1,2,3,4,5,6]
        entre = [1,2,3,4,5,6]
        data = dict()
        for i in range(k):
            entre[i] = st.selectbox(f'paramètres {i+1} avec incertitude', ["investissement", "cout_tonne_remuee", "ratio_sterile", "cout_traitement", "charges_fixes", "prix_or", "taux_recuperation_or",
                                                                        "prop_or_paye_dore", "taux_actualisation", "tonnage_geol", "teneur_minerai_geol", "taux_recup", "dilution_minerai", "rythme_prod_annee", "premiere_annee_prod"], index=1)
            coeff[i] = st.slider(
            f"coefficient modificateur de l'entrée {i+1} modifiée", 0.5, 1.5, 1.)
            data[f"coefficient{i+1}"] = coeff[i]
            data[f"entrees{i+1}"] = entre[i]
        

        
        options = st.selectbox("Résultat", ['cash_flow_actu', 'tri', "cumul_cash_flow_actu"], )
        data['options']=  options
        data['annees'] = annees 
        parametres = pd.DataFrame(data, index=["1ère valeure modifiée"])
        return parametres


    df = user_input(u)

    #sortie que le client souhaite observer
    sortie_voulue = df.loc['1ère valeure modifiée', 'options']

    # paramètres à modifier par le client
    n = df.loc['1ère valeure modifiée', 'annees'] + 1

    # tableau d'entrée avec le bon nombre d'années
    valeurs = entree(extraction_csv("donnees_entree_proj_minier.csv", n))


    st.write(df) # à enlever à la fin mais permet de visualiser le tableau


    abscisse = np.arange(1,n) #on fait partir à 1 car pour l'année 0, seulement investissement initiale
    fig, ax = plt.subplots()
    plt.style.use('seaborn')  # pour avoir un autre style de graphique plus frais
    evaluable = "valeurs."+f"{sortie_voulue}"+"()"
    sortie = eval(evaluable)[1:]
    st.write(f" Le cumul des cash flow avec les données initiales vaut {valeurs.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {n-1}")
    #tracé avec les valeurs initiales
    ax = plt.scatter(abscisse, sortie,
                    label=f"valeur initiale")

    for i in range(u): 
        # on extrait les données du tableau de base et modifie en fonction du choix du client
        valeur = deepcopy(valeurs)
        

        #i-ème paramètre modifié
        entree_modif = df.loc['1ère valeure modifiée', f'entrees{i+1}']
        coeff = df.loc['1ère valeure modifiée', f'coefficient{i+1}']

        params_modif = eval(f"valeur.{entree_modif}")
        params_modif *= coeff
        change = f"valeur.{entree_modif}"
        changement = eval(change)
        st.markdown(
        f" Le cumul des cash flow avec {entree_modif} modifiées vaut {valeur.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {n-1} ")
        sortie_modif = eval("valeur."+f"{sortie_voulue}"+"()")[1:]
        
        ax = plt.scatter(abscisse, sortie_modif,
                    label=f"{entree_modif} modifiée facteur {coeff}")





    plt.ylabel('$ (en millions) ')
    plt.xlabel("annee")
    plt.title(f"{sortie_voulue}")
    plt.legend()

    st.pyplot(fig)


    #partie visualisation des tendances liées aux modifications de paramètres

    st.markdown("## Analyse de sensibilité ##")

    p = st.number_input("nombre d'années d'exploitation", 1, 30, 15)

    #on détermine les entrées avec le bon nombre d'années considérées
    valeurs = entree(extraction_csv("donnees_entree_proj_minier.csv", p))


    pourcentage = np.linspace(0.5, 1.5, 100)


    fig, ax = plt.subplots(figsize  = (15,10))



    for variable in ["taux_actualisation"]:
        sortie = []
        for pourcent in pourcentage:
            valeur = deepcopy(valeurs)
            params_modif = eval(f"valeur.{variable}")
            params_modif *= pourcent
            sortie.append(eval("valeur."+"cumul_cash_flow_actu"+"()")[-1])
        plt.plot(pourcentage, sortie, label = f"{variable}")

    plt.xlabel('pourcentage')
    plt.ylabel('cumul cash flow actualisé')
    plt.title('variation du cumul du cash flow actualisée selon le pourcentage et selon les variables modifiées')
    plt.legend()
    st.pyplot(fig)
        





