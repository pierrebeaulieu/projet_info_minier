from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from ipywidgets import interact, FloatText
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import io
from copy import deepcopy

# Pour récupérer les données
import numpy as np
import matplotlib.pyplot as plt
import csv


""" GROS PROBLEME AVEC LES ANNEES """


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
            if row[0] == "Investissement initial pour l'exploitation":
                investissement.append(float(row[2]))
            if row[0] == "Cout par tonne remuee dans la mine a ciel ouvert":
                cout_tonne_remuee.append(float(row[2]))
            if row[0] == "Ratio sterile sur minerai dans la mine a ciel ouvert":
                ratio_sterile.append(float(row[2]))
            if row[0] == "Cout de traitement par tonne de minerai":

                cout_traitement.append(float(row[2]))
            if row[0] == "Charges fixes annuelles":
                charges_fixes.append(float(row[2]))
            if row[0] == "Prix de vente de l'or":
                prix_or.append(float(row[2]))
            if row[0] == "Taux de recuperation de l'or dans le traitement":
                taux_recuperation_or.append(float(row[2]))
            if row[0] == "Proportion d'or paye dans le dore":
                prop_or_paye_dore.append(float(row[2]))
            if row[0] == "Nombre de grammes dans une once":
                nombre_grammes_once.append(float(row[2]))
            if row[0] == "Taux d'actualisation":
                taux_actualisation.append(float(row[2]))
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


    



    st.markdown("# 1 - Affichage des données # ")

    st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
    st.text(" ")
    st.text(' ')

    type_entree = st.selectbox("Rentrer les données à la main ou par un .csv", [".csv", "Manuellement"], )

    if type_entree == ".csv":

        st.write(f'Voici le template à télécharger pour renseigner les valeurs en entrée si vous ne le possédez pas déjà')

        download=st.button('Télécharger le Template')

        if download:
            df = pd.read_csv("donnees_entree_proj_minier.csv")
            fichiercsv = df.to_csv(index=False)
            b64 = base64.b64encode(fichiercsv.encode()).decode()  # some strings
            linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
            st.markdown(linko, unsafe_allow_html=True)

        st.write(f'Veuillez renseigner le chemin accès à votre .csv pour continuer.')
        st.write(f'Pour se faire, faites un clique droit sur votre .csv, puis cliquez sur propriétés, puis copiez le chemin et ajoutez /nomfichier.csv')

        file_path= st.text_input('Chemin accès du fichier')

        if len(file_path) >1:

            st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
            st.text(" ")
            st.text(' ')

        

            #détermine le nombre d'années

            annees = st.number_input("nombre d'années considérées", 0, 50, 15)
            valeurs = entree(extraction_csv(file_path, annees))
            n = annees
            if st.checkbox("Voir les données d'entrées", value = True): 
                st.subheader("Données d'entrée")
                st.markdown(valeurs)

            st.markdown(
                "### Tracé de la valeur choisie en fonction du paramètre choisi ### \n  \n \n  ")
            st.markdown("\n \n ")
            
            # Ce que l'on doit tracer en fonction des années

            sortie = st.selectbox("Axe des ordonnées", ['cash_flow_actu', 'tri', "cumul_cash_flow_actu"], )



            # tableau d'entrée avec le bon nombre d'années
            
            abscisse = np.arange(1,annees)
            evaluable = "valeurs."+f"{sortie}"+"()"
            ordonne = eval(evaluable)[1:]
            fig, ax = plt.subplots()
            plt.style.use('seaborn')  # pour avoir un autre style de graphique plus frais
            
            st.write(f" Le cumul des cash flow avec les données initiales vaut {valeurs.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {n-1}")
            #tracé avec les valeurs initiales
            ax = plt.scatter(abscisse, ordonne,
                            label=f"valeur initiale")

            plt.ylabel('$ (en millions) ')
            plt.xlabel("annee")
            plt.title(f"{sortie}")
            plt.legend()

            st.pyplot(fig)
    else :
        st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
        st.text(" ")
        st.text(' ')



        #détermine le nombre d'années

        annees = st.number_input("nombre d'années considérées", 0, 50, 15)
        valeurs = entree(extraction_csv("donnees_entree_proj_minier.csv", annees))
        n = annees
        
        st.subheader("Données d'entrée")
        valeurs.n = annees
        valeurs.investissement = np.array([st.number_input('investissement',270)]*valeurs.n)
        valeurs.cout_tonne_remuee = np.array([st.number_input('cout_tonne_remuee',4)]*valeurs.n,)
        valeurs.ratio_sterile = np.array([st.number_input('ratio_sterile',4)]*valeurs.n)
        valeurs.cout_traitement = np.array([st.number_input('cout_traitement',15)]*valeurs.n)
        valeurs.charges_fixes = np.array([st.number_input('charges_fixes',18)]*valeurs.n)
        valeurs.prix_or = np.array([st.number_input('prix_or',1400)]*valeurs.n)
        valeurs.taux_recuperation_or = np.array([st.number_input('taux_recuperation_or',95)]*valeurs.n)
        valeurs.prop_or_paye_dore = np.array([st.number_input('prop_or_paye_dore',98)]*valeurs.n)
        valeurs.nombre_grammes_once = np.array([st.number_input('nombre_grammes_once',31)]*valeurs.n)
        valeurs.taux_actualisation = np.array([st.number_input('taux_actualisation',5)]*valeurs.n)
        valeurs.tonnage_geol = np.array([st.number_input('tonnage_geol',11)]*valeurs.n)
        valeurs.teneur_minerai_geol = np.array([st.number_input('teneur_minerai_geol',3)]*valeurs.n)
        valeurs.taux_recup = np.array([st.number_input('taux_recup',95)]*valeurs.n)
        valeurs.dilution_minerai = np.array([st.number_input('dilution_minerai',15)]*valeurs.n)
        valeurs.rythme_prod_annee = np.array([st.number_input('rythme_prod_annee',0.8)]*valeurs.n)

        st.markdown(
            "### Tracé de la valeur choisie en fonction du paramètre choisi ### \n  \n \n  ")
        st.markdown("\n \n ")
        
        # Ce que l'on doit tracer en fonction des années

        sortie = st.selectbox("Axe des ordonnées", ['cash_flow_actu', 'tri', "cumul_cash_flow_actu"], )



        # tableau d'entrée avec le bon nombre d'années
        
        abscisse = np.arange(1,annees)
        evaluable = "valeurs."+f"{sortie}"+"()"
        ordonne = eval(evaluable)[1:]
        fig, ax = plt.subplots()
        plt.style.use('seaborn')  # pour avoir un autre style de graphique plus frais
        
        st.write(f" Le cumul des cash flow avec les données initiales vaut {valeurs.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {n-1}")
        #tracé avec les valeurs initiales
        ax = plt.scatter(abscisse, ordonne,
                        label=f"valeur initiale")

        plt.ylabel('$ (en millions) ')
        plt.xlabel("annee")
        plt.title(f"{sortie}")
        plt.legend()

        st.pyplot(fig) 

        download=st.button('Télécharger les variables sous forme de .csv')

        if download:
            df = pd.read_csv("donnees_entree_proj_minier.csv",sep=';')
            older = [11,3,95,15,12.02,2.61,0.8,270,4,4,20,15,18,1400,95,98,2.43,31,109.68,5]
            new = [valeurs.tonnage_geol[0],valeurs.teneur_minerai_geol[0],valeurs.taux_recup[0],valeurs.dilution_minerai[0],valeurs.tonnage_industriel()[0],valeurs.teneur_minerai_industriel()[0],valeurs.rythme_prod_annee[0],valeurs.investissement[0],valeurs.cout_tonne_remuee[0],valeurs.ratio_sterile[0],valeurs.cout_exploitation_tonne_minerai()[0],valeurs.cout_traitement[0],valeurs.charges_fixes[0],valeurs.prix_or[0],valeurs.taux_recuperation_or[0],valeurs.prop_or_paye_dore[0],valeurs.qte_or_paye()[0],valeurs.nombre_grammes_once[0],valeurs.recette_tonne_minerai()[0],valeurs.taux_actualisation[0]]
            df["valeur"]=df["valeur"].replace(older,new)
            
            fichiercsv = df.to_csv(index=False,sep=";")
            b64 = base64.b64encode(fichiercsv.encode()).decode()  # some strings
            linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
            st.markdown(linko, unsafe_allow_html=True)

        

        