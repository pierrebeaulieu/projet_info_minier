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

          # on extrait les données du tableau de base et modifie en fonction du choix du client

            st.markdown("# Projet informatique sur l'économie d'une exploitation minière # ")

            st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
            st.text(" ")
            st.text(' ')

            #détermine le nombre de paramètre à modifier
            u = st.number_input("nombre d'entrées à modifier", 1,5,step=1)
            

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
            valeurs = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", n))

            if st.checkbox("Voir les données d'entrées", value = True): 
                st.subheader("Données d'entrée")
                st.markdown(valeurs)


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
            valeurs = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", p))


            pourcentage = np.linspace(0.5, 1.5, 100)


            fig, ax = plt.subplots(figsize  = (15,10))

            variable = st.multiselect("Variable à analyser",["investissement", "cout_tonne_remuee", "ratio_sterile", "cout_traitement", "charges_fixes", "prix_or", "taux_recuperation_or","prop_or_paye_dore", "taux_actualisation", "tonnage_geol", "teneur_minerai_geol", "taux_recup", "dilution_minerai", "rythme_prod_annee", "premiere_annee_prod"])

            for variable in variable:
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

        # 1-AFFICHAGE SENSIBILITE

        # on extrait les données du tableau de base et modifie en fonction du choix du client

        st.markdown("# 2 - Analyse de Sensibilité # ")

        st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
        st.text(" ")
        st.text(' ')

        #détermine le nombre de paramètre à modifier
        u = st.number_input("nombre d'entrées à modifier", 1,5,step=1)

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
        valeurs = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", n))

        if st.checkbox("Voir les données d'entrées", value = True): 
            st.subheader("Données d'entrée")
            st.markdown(valeurs)


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
        valeurs = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", p))


        pourcentage = np.linspace(0.5, 1.5, 100)


        fig, ax = plt.subplots(figsize  = (15,10))

        variable = st.multiselect("Variable à analyser",["investissement", "cout_tonne_remuee", "ratio_sterile", "cout_traitement", "charges_fixes", "prix_or", "taux_recuperation_or","prop_or_paye_dore", "taux_actualisation", "tonnage_geol", "teneur_minerai_geol", "taux_recup", "dilution_minerai", "rythme_prod_annee", "premiere_annee_prod"])

        for variable in variable:
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

            