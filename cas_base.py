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

    st.markdown("# 0 - Entrée des données # ")

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
            n = annees
            valeurs = class_entree.entree(class_entree.extraction_csv(file_path, annees))

            if st.checkbox("Voir les données d'entrées", value = True,key="1"): 
                st.subheader("Données d'entrée")
                st.markdown(valeurs)

            if st.checkbox("1 - Affichage des données", value = False,key="2") :

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
                
                st.write(f" Le cumul des cash flow avec les données initiales vaut {valeurs.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {annees-1}")
                #tracé avec les valeurs initiales
                ax = plt.scatter(abscisse, ordonne,
                                label=f"valeur initiale")

                plt.ylabel('$ (en millions) ')
                plt.xlabel("annee")
                plt.title(f"{sortie}")
                plt.legend()

                st.pyplot(fig)
                downloadbis=st.button('Télécharger le graphique sous forme de .png')

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

        # 1-AFFICHAGE DES DONNEES

        if st.checkbox("1 - Affichage des données", value = False,key="2") :

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

            downloadbis=st.button('Télécharger le graphique sous forme de .png')
