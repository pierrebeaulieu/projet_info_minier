from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from ipywidgets import interact, FloatText
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import base64

from copy import deepcopy

# Pour récupérer les données
import numpy as np
import matplotlib.pyplot as plt
import class_entree

from loi import loi_func



def app():

    # 0- ENTREE DES DONNEES
    st.markdown("# Projet informatique sur l'économie d'une exploitation minière # ")

    st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
    st.text(" ")
    st.text(' ')

    st.write(f"Petit guide de l'application:")
    st.write(f"- Pour télécharger un graphique faire clique droit dessus, et cliquer sur enregistrer l'image sous.")
    st.write(f"- Il suffit de renseigner une seule fois les données au début dans un des onglets pour naviguer partout ensuite.")
    st.write(f"- Enregistrement des données possible sous format csv")

    st.markdown("# 0 - Entrée des données pour (1) Affichage des données # ")

    st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
    st.text(" ")
    st.text(' ')

    type_entree = st.selectbox("Rentrer les données à la main ou par un .csv", [".csv", "Manuellement"], )

    if type_entree == ".csv":

        st.write(f'Voici le template à télécharger pour renseigner les valeurs en entrée si vous ne le possédez pas déjà:')

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
            n = annees
            valeurs = class_entree.entree(class_entree.extraction_csv(file_path, annees))

            if st.checkbox("Voir les données d'entrées", value = True,key="1"): 
                st.subheader("Données d'entrée")
                st.markdown(valeurs)

            st.markdown(
                "### Tracé de la valeur choisie en fonction du paramètre choisi ### \n  \n \n  ")
            st.markdown("\n \n ")
            
            # Ce que l'on doit tracer en fonction des années

            sortie = st.selectbox("Axe des ordonnées", ['cash_flow_actu', 'tri', "cumul_cash_flow_actu","cash_flow","cumul_cash_flow",], )



            # tableau d'entrée avec le bon nombre d'années
            
            abscisse = np.arange(1,annees)
            
            evaluable = "valeurs."+f"{sortie}"+"()"
            ordonne = eval(evaluable)[1:]
            fig, ax = plt.subplots()
            plt.style.use('seaborn')  # pour avoir un autre style de graphique plus frais
            
            st.write(f" Le cumul des cash flow avec les données initiales vaut {valeurs.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {annees-1}")
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
        valeurs = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", annees))
        
            
        n = annees
        
        st.subheader("Données d'entrée")
        valeurs.n = annees
        valeurs.investissement = np.array([st.number_input('investissement',value = 270)]+(valeurs.n-1)*[0])
        valeurs.cout_tonne_remuee = np.array([st.number_input('cout_tonne_remuee',value = 4)]*valeurs.n,)
        valeurs.ratio_sterile = np.array([st.number_input('ratio_sterile',value = 4)]*valeurs.n)
        valeurs.cout_traitement = np.array([st.number_input('cout_traitement',value = 15)]*valeurs.n)
        valeurs.charges_fixes = np.array([st.number_input('charges_fixes',value = 18)]*valeurs.n)
        valeurs.prix_or = np.array([st.number_input('prix_or',value = 1400)]*valeurs.n)
        valeurs.taux_recuperation_or = np.array([st.number_input('taux_recuperation_or',value = 95)]*valeurs.n)
        valeurs.prop_or_paye_dore = np.array([st.number_input('prop_or_paye_dore',value = 98)]*valeurs.n)
        valeurs.nombre_grammes_once = np.array([st.number_input('nombre_grammes_once',value = 31)]*valeurs.n)
        valeurs.taux_actualisation = np.array([st.number_input('taux_actualisation',value = 5)]*valeurs.n)
        valeurs.tonnage_geol = np.array([st.number_input('tonnage_geol',value = 11)]*valeurs.n)
        valeurs.teneur_minerai_geol = np.array([st.number_input('teneur_minerai_geol',value = 3)]*valeurs.n)
        valeurs.taux_recup = np.array([st.number_input('taux_recup',value = 95)]*valeurs.n)
        valeurs.dilution_minerai = np.array([st.number_input('dilution_minerai',value = 15)]*valeurs.n)
        valeurs.rythme_prod_annee = np.array([st.number_input('rythme_prod_annee',value = 0.8)]*valeurs.n)
        for j in range(0,int(valeurs.premiere_annee_prod[0])):
            valeurs.rythme_prod_annee[j]=0
        retour = np.zeros(valeurs.n)
        tonnage_indus = valeurs.tonnage_geol * valeurs.taux_recup/100 * (valeurs.dilution_minerai/100 + np.ones(valeurs.n))
        retour[0] += tonnage_indus[0]
    
        k=1
        while k<n and (retour[k-1]-valeurs.rythme_prod_annee[k-1])>=0:
            retour [k]=retour[k-1]-valeurs.rythme_prod_annee[k-1]
            k+=1
        if retour[k-1]-valeurs.rythme_prod_annee[k-1]<0:
            valeurs.rythme_prod_annee[k-1]=retour[k-1]
            for i in range(k,valeurs.n):
                retour[i]=0
                valeurs.rythme_prod_annee[i]=0

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

        # 1-AFFICHAGE DES DONNEES

       

        st.markdown(
            "### Tracé de la valeur choisie en fonction du paramètre choisi ### \n  \n \n  ")
        st.markdown("\n \n ")
        
        # Ce que l'on doit tracer en fonction des années

        sortie = st.selectbox("Axe des ordonnées", ['cash_flow_actu', 'tri', "cumul_cash_flow_actu","cash_flow",], )



        # tableau d'entrée avec le bon nombre d'années
        
        abscisse = np.arange(1,annees)
        evaluable = "valeurs."+f"{sortie}"+"()"
        ordonne = eval(evaluable)[1:]
        fig, ax = plt.subplots()
        plt.style.use('seaborn')  # pour avoir un autre style de graphique plus frais
        
        st.write(f" Le cumul des cash flow avec les données initiales vaut {valeurs.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {n}")
        #tracé avec les valeurs initiales
        ax = plt.scatter(abscisse, ordonne,
                        label=f"valeur initiale")

        plt.ylabel('$ (en millions) ')
        plt.xlabel("annee")
        plt.title(f"{sortie}")
        plt.legend()

        st.pyplot(fig) 