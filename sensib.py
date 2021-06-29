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

    st.markdown("# 0 - Entrée des données pour (2) Analyse de sensibilité # ")

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
                token = False
                coeff = [1,2,3,4,5,6]
                entre = [1,2,3,4,5,6]
                data = dict()
                for i in range(k):
                    entre[i] = st.selectbox(f'paramètres {i+1} avec incertitude', ["investissement", "cout_tonne_remuee", "ratio_sterile", "cout_traitement", "charges_fixes", "prix_or", "taux_recuperation_or",
                                                                                "prop_or_paye_dore", "taux_actualisation", "tonnage_geol", "teneur_minerai_geol", "taux_recup", "dilution_minerai", "rythme_prod_annee", "premiere_annee_prod"], index=1)
                    if entre[i]=="taux_recup":
                        coeff[i]=st.slider(f"coefficient modificateur de l'entrée {i+1} modifiée", 0., int(100*100/valeurs.taux_recup[0])/100, 1.)
                    elif entre[i]=="taux_recuperation_or":
                        coeff[i]=st.slider(f"coefficient modificateur de l'entrée {i+1} modifiée", 0., int(100*100/valeurs.taux_recuperation_or[0])/100, 1.)
                    elif entre[i]=="investissement":
                        coeff[i] = st.slider(f"coefficient modificateur de l'entrée {i+1} modifiée", 0.5, 1.5, 1.)
                        token=True
                    else:
                        coeff[i] = st.slider(f"coefficient modificateur de l'entrée {i+1} modifiée", 0.5, 1.5, 1.)
                    data[f"coefficient{i+1}"] = coeff[i]
                    data[f"entrees{i+1}"] = entre[i]
                

                
                options = st.selectbox("Résultat", ['cash_flow_actu', 'tri', "cumul_cash_flow_actu"], )
                data['options']=  options
                data['annees'] = annees 
                parametres = pd.DataFrame(data, index=["1ère valeure modifiée"])
                return parametres,token


            df,token = user_input(u)

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

            if token: 
                st.write(f"Pour des raisons d'échelle, nous traçons nos valeurs à partir de l'année 1. Donc en modifiant l'investissement nous ne voyons pas d'impact sur le cashflow car celui ci n'est modifié qu'en annee 0. Sa valeur à l'année 0 est donc {valeur.cash_flow_actu()[0]} M$")


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
            token = False
            coeff = [1,2,3,4,5,6]
            entre = [1,2,3,4,5,6]
            data = dict()
            for i in range(k):
                entre[i] = st.selectbox(f'paramètres {i+1} avec incertitude', ["investissement", "cout_tonne_remuee", "ratio_sterile", "cout_traitement", "charges_fixes", "prix_or", "taux_recuperation_or",
                                                                            "prop_or_paye_dore", "taux_actualisation", "tonnage_geol", "teneur_minerai_geol", "taux_recup", "dilution_minerai", "rythme_prod_annee", "premiere_annee_prod"], index=1)
                if entre[i]=="taux_recup":
                    coeff[i]=st.slider(f"coefficient modificateur de l'entrée {i+1} modifiée", 0, int(100*100/valeurs.taux_recup[0])/100, 1.)
                elif entre[i]=="taux_recuperation_or":
                    coeff[i]=st.slider(f"coefficient modificateur de l'entrée {i+1} modifiée", 0, int(100*100/valeurs.taux_recuperation_or[0])/100, 1.)
                elif entre[i]=="investissement":
                    coeff[i] = st.slider(
                    f"coefficient modificateur de l'entrée {i+1} modifiée", 0.5, 1.5, 1.)
                    token=True
                else:
                    coeff[i] = st.slider(
                    f"coefficient modificateur de l'entrée {i+1} modifiée", 0.5, 1.5, 1.)
                data[f"coefficient{i+1}"] = coeff[i]
                data[f"entrees{i+1}"] = entre[i]
            

            
            options = st.selectbox("Résultat", ['cash_flow_actu', 'tri', "cumul_cash_flow_actu"], )
            data['options']=  options
            data['annees'] = annees 
            parametres = pd.DataFrame(data, index=["1ère valeure modifiée"])
            return parametres,token


        df,token = user_input(u)

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
            sortie_modif = eval("valeur."+f"{sortie_voulue}"+"()")[1:]
            
            ax = plt.scatter(abscisse, sortie_modif,
                        label=f"{entree_modif} modifiée facteur {coeff}")

            if token: 
                st.write(f"Pour des raisons d'échelle, nous traçons nos valeurs à partir de l'année 1. Donc en modifiant l'investissement nous ne voyons pas d'impact sur le cashflow car celui ci n'est modifié qu'en annee 0. Sa valeur à l'année 0 est donc {valeur.cash_flow_actu()[0]} M$")





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

            