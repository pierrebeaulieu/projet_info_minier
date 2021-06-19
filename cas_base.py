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
            val_p1 = class_entree.entree(class_entree.extraction_csv(file_path, annees))
            val_p2 = class_entree.entree(class_entree.extraction_csv(file_path, annees))


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
                # 2-ETUDE DE SENSIBILITE
            
            if st.checkbox("2-Etude de Sensibilité", value = False,key="3") :
            
                st.markdown("# Projet informatique sur l'économie d'une exploitation minière # ")

                st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
                st.text(" ")
                st.text(' ')


                if st.checkbox("Voir les données d'entrées", value = True,key="3"): 
                    st.subheader("Données d'entrée")
                    st.markdown(valeurs)


                #détermine le nombre de paramètre à modifier
                u = st.number_input("nombre d'entrées à modifier", 1,5,step=1,key="b")
                
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

                st.write(df) # à enlever à la fin mais permet de visualiser le tableau


                abscisse = np.arange(1,annees) #on fait partir à 1 car pour l'année 0, seulement investissement initiale
                fig, ax = plt.subplots()
                plt.style.use('seaborn')  # pour avoir un autre style de graphique plus frais
                evaluable = "val_p1."+f"{sortie_voulue}"+"()"
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

                    params_modif = eval(f"val_p1.{entree_modif}")
                    params_modif *= coeff
                    change = f"val_p1.{entree_modif}"
                    changement = eval(change)
                    st.markdown(
                    f" Le cumul des cash flow avec {entree_modif} modifiées vaut {val_p1.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {n-1} ")
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


                #on détermine les entrées avec le bon nombre d'années considérées


                pourcentage = np.linspace(0.5, 1.5, 100)


                fig, ax = plt.subplots(figsize  = (15,10))

                variable = st.multiselect("Variable à analyser",["investissement", "cout_tonne_remuee", "ratio_sterile", "cout_traitement", "charges_fixes", "prix_or", "taux_recuperation_or","prop_or_paye_dore", "taux_actualisation", "tonnage_geol", "teneur_minerai_geol", "taux_recup", "dilution_minerai", "rythme_prod_annee", "premiere_annee_prod"])

                for variable in variable:
                    sortie = []
                    for pourcent in pourcentage:
                        params_modif = eval(f"val_p1.{variable}")
                        params_modif *= pourcent
                        sortie.append(eval("val_p1."+"cumul_cash_flow_actu"+"()")[-1])
                    plt.plot(pourcentage, sortie, label = f"{variable}")

                plt.xlabel('pourcentage')
                plt.ylabel('cumul cash flow actualisé')
                plt.title('variation du cumul du cash flow actualisée selon le pourcentage et selon les variables modifiées')
                plt.legend()
                st.pyplot(fig)
                    
            if st.checkbox("3-Etude Probabiliste", value = False,key="4"):

                valeurs_non_modif = deepcopy(valeurs)

                st.header("Modélisation des lois de probabilité des paramètres incertains")

                # On modélise la loi de probabilité de l'investissement en tenant compte du choix du client

                st.subheader("Loi de probabilité suivie par l'investissement")

                #Début choix loi investissement
                
                loi_invest = st.selectbox("loi de l'investissement", ['triangular', 'normal', 'uniform'])

                if loi_invest == "triangular":
                    moyenne_invest = st.number_input(f"L'investissement est modélisé par une loi triangulaire centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.investissement[0])
                    bande_invest = st.number_input("Pourcentage de la bande de l'investissement:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_invest}M$ et de bande {bande_invest}%')

                    arr = np.random.triangular(moyenne_invest - moyenne_invest*bande_invest/100, moyenne_invest, moyenne_invest + moyenne_invest*bande_invest/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.investissement = np.array(n * [np.random.triangular(moyenne_invest - moyenne_invest*bande_invest/100, moyenne_invest, moyenne_invest + moyenne_invest*bande_invest/100)])
                

                if loi_invest == "normal":
                    moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_invest = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_invest} et d'ecart_type {bande_invest}")
                    
                    arr = np.random.normal(moyenne_invest, bande_invest*moyenne_invest, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])

                if loi_invest == "uniform":
                    moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_invest = st.number_input(f"Pourcentage de la bande de l'investissement:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_invest} et de bande {bande_invest}")
                    
                    arr = np.random.uniform(moyenne_invest-moyenne_invest*bande_invest/200, moyenne_invest+moyenne_invest*bande_invest/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])

                # Fin choix loi investissement
                
                # Début choix loi coût
                
                st.subheader("Loi de probabilité suivie par le coût")
                loi_cout= st.selectbox("loi du coût", ['triangular', 'normal', 'uniform'])

                if loi_cout == "triangular":
                    moyenne_cout = st.number_input(f"Le cout est modélisé par une loi triangulaire centrée sur la valeur {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                    bande_cout = st.number_input("Pourcentage de la bande du cout:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_cout}M$ et de bande {bande_cout}%')

                    arr = np.random.triangular(moyenne_cout - moyenne_cout*bande_cout/100, moyenne_cout, moyenne_cout + moyenne_cout*bande_cout/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.cout_tonne_remuee = np.array(n * [np.random.triangular(moyenne_cout - moyenne_cout*bande_cout/100, moyenne_cout, moyenne_cout + moyenne_cout*bande_cout/100)])

                    
                if loi_cout == "normal":
                    moyenne_cout = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                    bande_cout = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du cout", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_cout} et d'ecart_type {bande_cout}")
                    
                    arr = np.random.normal(moyenne_cout, bande_cout*moyenne_cout, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.cout_tonne_remuee = np.array(n * [np.random.normal(moyenne_cout, bande_cout*moyenne_cout)])

                if loi_cout == "uniform":
                    moyenne_cout = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                    bande_cout = st.number_input(f"Pourcentage de la bande du cout:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_cout} et de bande {bande_cout}")
                    
                    arr = np.random.uniform(moyenne_cout-moyenne_cout*bande_cout/200, moyenne_cout+moyenne_cout*bande_cout/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.cout_tonne_remuee = np.array(n * [np.random.normal(moyenne_cout, bande_cout*moyenne_cout)])

                # Fin choix loi coût    

                # Début choix loi teneur
                st.subheader("Loi de probabilité suivie par la teneur")
                
                loi_teneur = st.selectbox("loi de la teneur", ['triangular', 'normal', 'uniform'])

                if loi_teneur == "triangular":
                    moyenne_teneur = st.number_input(f"La teneur est modélisé par une loi triangulaire centrée sur la valeur {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_teneur = st.number_input("Pourcentage de la bande de la teneur:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_teneur}M$ et de bande {bande_teneur}%')

                    arr = np.random.triangular(moyenne_teneur - moyenne_teneur*bande_teneur/100, moyenne_teneur, moyenne_teneur + moyenne_teneur*bande_teneur/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.teneur_minerai_geol = np.array(n * [np.random.triangular(moyenne_teneur - moyenne_teneur*bande_teneur/100, moyenne_teneur, moyenne_teneur + moyenne_teneur*bande_teneur/100)])
                    
                    st.write("Les nouvelles valeurs pour la teneur du minerai géologique sont (pour une simulation):", valeurs.investissement)

                if loi_teneur == "normal":
                    moyenne_teneur = st.number_input(f"La teneur est modélisée par une loi normale de moyenne {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_teneur = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_teneur} et d'ecart_type {bande_teneur}")
                    
                    arr = np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur)])

                if loi_teneur == "uniform":
                    moyenne_teneur = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_teneur = st.number_input(f"Pourcentage de la bande de la teneur:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_teneur} et de bande {bande_teneur}")
                    
                    arr = np.random.uniform(moyenne_teneur-moyenne_teneur*bande_teneur/200, moyenne_teneur+moyenne_teneur*bande_teneur/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur)])

                # Fin choix loi teneur
                
                # Début choix loi tonnage
                
                st.subheader("Loi de probabilité suivie par le tonnage")

                loi_tonnage = st.selectbox("loi du tonnage", ['triangular', 'normal', 'uniform'])

                if loi_tonnage == "triangular":
                    moyenne_tonnage = st.number_input(f"Le tonnage est modélisé par une loi triangulaire centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.investissement[0])
                    bande_tonnage = st.number_input("Pourcentage de la bande du tonnage:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_tonnage}M$ et de bande {bande_tonnage}%')

                    arr = np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.tonnage_geol = np.array(n * [np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100)])
                    

                if loi_tonnage == "normal":
                    moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi normale de moyenne {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_tonnage = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du tonnage", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_tonnage} et d'ecart_type {bande_tonnage}")
                    
                    arr = np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

                if loi_tonnage == "uniform":
                    moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi uniforme centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_tonnage = st.number_input(f"Pourcentage de la bande du tonnage:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_tonnage} et de bande {bande_tonnage}")
                    
                    arr = np.random.uniform(moyenne_tonnage-moyenne_tonnage*bande_tonnage/200, moyenne_tonnage+moyenne_tonnage*bande_tonnage/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    st.pyplot(fig)

                    valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

                # Fin choix loi tonnage

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
                    valeurs.investissement += loi_func(loi_invest, moyenne_invest, bande_invest, n)
                    valeurs.cout_tonne_remuee += loi_func(loi_cout, moyenne_cout, bande_cout, n)
                    valeurs.tonnage_geol += loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
                    valeurs.teneur_minerai_geol += loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

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
                tri_simu = np.zeros(nb_simu)

                while i < nb_simu:
                    valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n)
                    valeurs.cout_tonne_remuee = loi_func(loi_cout, moyenne_cout, bande_cout, n)
                    valeurs.tonnage_geol = loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
                    valeurs.teneur_minerai_geol = loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

                    # pour la proba que tri>tri_goal
                    if valeurs.tri()[annee-1] > tri_goal:
                        iter_tri += 1
                    tri_simu[i] = valeurs.tri()[annee-1]
                    i+=1

                # On affiche la répartition des valeurs du TRI à l'aide d'un histogramme

                fig, ax = plt.subplots()
                ax.hist(tri_simu, bins=25)
                plt.title(f"Répartition des valeurs du TRI à l'année {annee}")
                st.pyplot(fig)

                iter_tri *= 1/nb_simu

                st.write(f"La probabilité que le TRI soit supérieur à {tri_goal} à l'année {annee} vaut {iter_tri}")

                # Calculons alors le risque de rentabilité du projet

                st.header("Risque de rentabilité de l'investissement")

                i = 0
                iter_van = 0
                annee = st.number_input("Caculons la probabilité que l'investissement ne soit pas rentable à l'année:", 1, 15, 15, 1)
                esperance_perte = 0
                van_simu = np.zeros(nb_simu)

                while i < nb_simu:
                    valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n)
                    valeurs.cout_tonne_remuee = loi_func(loi_cout, moyenne_cout, bande_cout, n)
                    valeurs.tonnage_geol = loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
                    valeurs.teneur_minerai_geol = loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

                    # pour la proba que van<0
                    if valeurs.cumul_cash_flow()[annees-1] < 0:
                        iter_van += 1
                        esperance_perte += valeurs.cumul_cash_flow()[annees-1]
                    van_simu[i] = valeurs.cumul_cash_flow()[annees-1]
                    i+=1
                
                esperance_perte *= 1/nb_simu
                iter_van *= 1/nb_simu

                # Affichage de la répartition du van

                fig, ax = plt.subplots()
                ax.hist(van_simu, 25)
                plt.title(f"Répartition des valeurs du VAN à l'année {annees}")
                st.pyplot(fig)

                st.write(f"La probabilité que le VAN soit négatif à l'année {annees} vaut {iter_van}")

                st.write(f"De plus, l'espérance de perte à l'année {annees} vaut {esperance_perte }M$")

    else :
        st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
        st.text(" ")
        st.text(' ')



        #détermine le nombre d'années

        annees = st.number_input("nombre d'années considérées", 0, 50, 15)
        valeurs = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", annees))
        val_p1 = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", annees))
        
            
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

        val_p1 = deepcopy(valeurs)
        val_p2  = deepcopy(valeurs) 


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


    # 2-ETUDE DE SENSIBILITE
        
        if st.checkbox("2-Etude de Sensibilité", value = False,key="3") :
        
            st.markdown("# Projet informatique sur l'économie d'une exploitation minière # ")

            st.text(" ")   # Pour sauter une ligne, sinon le texte se superpose
            st.text(" ")
            st.text(' ')


            if st.checkbox("Voir les données d'entrées", value = True,key="3"): 
                st.subheader("Données d'entrée")
                st.markdown(valeurs)


            #détermine le nombre de paramètre à modifier
            u = st.number_input("nombre d'entrées à modifier", 1,5,step=1,key="b")
            
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

            st.write(df) # à enlever à la fin mais permet de visualiser le tableau


            abscisse = np.arange(1,annees) #on fait partir à 1 car pour l'année 0, seulement investissement initiale
            fig, ax = plt.subplots()
            plt.style.use('seaborn')  # pour avoir un autre style de graphique plus frais
            evaluable = "val_p1."+f"{sortie_voulue}"+"()"
            sortie = eval(evaluable)[1:]


            st.write(f" Le cumul des cash flow avec les données initiales vaut {valeurs.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {annees-1}")
            #tracé avec les valeurs initiales
            ax = plt.scatter(abscisse, sortie,
                            label=f"valeur initiale")

            for i in range(u): 
                # on extrait les données du tableau de base et modifie en fonction du choix du client

                

                #i-ème paramètre modifié
                entree_modif = df.loc['1ère valeure modifiée', f'entrees{i+1}']
                coeff = df.loc['1ère valeure modifiée', f'coefficient{i+1}']

                params_modif = eval(f"val_p1.{entree_modif}")
                params_modif *= coeff
                change = f"val_p1.{entree_modif}"
                changement = eval(change)
                st.markdown(
                f" Le cumul des cash flow avec {entree_modif} modifiées vaut {val_p1.cumul_cash_flow_actu()[-1] :.2f} M$ après l'année {n-1} ")
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


            #on détermine les entrées avec le bon nombre d'années considérées


            pourcentage = np.linspace(0.5, 1.5, 100)


            fig, ax = plt.subplots(figsize  = (15,10))

            variable = st.multiselect("Variable à analyser",["investissement", "cout_tonne_remuee", "ratio_sterile", "cout_traitement", "charges_fixes", "prix_or", "taux_recuperation_or","prop_or_paye_dore", "taux_actualisation", "tonnage_geol", "teneur_minerai_geol", "taux_recup", "dilution_minerai", "rythme_prod_annee", "premiere_annee_prod"])

            for variable in variable:
                sortie = []
                for pourcent in pourcentage:
                    params_modif = eval(f"val_p1.{variable}")
                    params_modif *= pourcent
                    sortie.append(eval("val_p1."+"cumul_cash_flow_actu"+"()")[-1])
                plt.plot(pourcentage, sortie, label = f"{variable}")

            plt.xlabel('pourcentage')
            plt.ylabel('cumul cash flow actualisé')
            plt.title('variation du cumul du cash flow actualisée selon le pourcentage et selon les variables modifiées')
            plt.legend()
            st.pyplot(fig)
        
        # 3- Etude Probabiliste 

        if st.checkbox("3-Etude Probabiliste", value = False,key="4"):

            valeurs_non_modif = deepcopy(valeurs)

            st.header("Modélisation des lois de probabilité des paramètres incertains")

            # On modélise la loi de probabilité de l'investissement en tenant compte du choix du client

            st.subheader("Loi de probabilité suivie par l'investissement")

            #Début choix loi investissement
            
            loi_invest = st.selectbox("loi de l'investissement", ['triangular', 'normal', 'uniform'])

            if loi_invest == "triangular":
                moyenne_invest = st.number_input(f"L'investissement est modélisé par une loi triangulaire centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.investissement[0])
                bande_invest = st.number_input("Pourcentage de la bande de l'investissement:", 1, 50, 10, 1)
                st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_invest}M$ et de bande {bande_invest}%')

                arr = np.random.triangular(moyenne_invest - moyenne_invest*bande_invest/100, moyenne_invest, moyenne_invest + moyenne_invest*bande_invest/100, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.investissement = np.array(n * [np.random.triangular(moyenne_invest - moyenne_invest*bande_invest/100, moyenne_invest, moyenne_invest + moyenne_invest*bande_invest/100)])
            

            if loi_invest == "normal":
                moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                bande_invest = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
                st.write(f"On considère donc une loi normale de moyenne {moyenne_invest} et d'ecart_type {bande_invest}")
                
                arr = np.random.normal(moyenne_invest, bande_invest*moyenne_invest, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])

            if loi_invest == "uniform":
                moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                bande_invest = st.number_input(f"Pourcentage de la bande de l'investissement:", 0.1)
                st.write(f"On considère donc une loi uniforme centrée sur {moyenne_invest} et de bande {bande_invest}")
                
                arr = np.random.uniform(moyenne_invest-moyenne_invest*bande_invest/200, moyenne_invest+moyenne_invest*bande_invest/200, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])

            # Fin choix loi investissement
            
            # Début choix loi coût
            
            st.subheader("Loi de probabilité suivie par le coût")
            loi_cout= st.selectbox("loi du coût", ['triangular', 'normal', 'uniform'])

            if loi_cout == "triangular":
                moyenne_cout = st.number_input(f"Le cout est modélisé par une loi triangulaire centrée sur la valeur {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                bande_cout = st.number_input("Pourcentage de la bande du cout:", 1, 50, 10, 1)
                st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_cout}M$ et de bande {bande_cout}%')

                arr = np.random.triangular(moyenne_cout - moyenne_cout*bande_cout/100, moyenne_cout, moyenne_cout + moyenne_cout*bande_cout/100, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.cout_tonne_remuee = np.array(n * [np.random.triangular(moyenne_cout - moyenne_cout*bande_cout/100, moyenne_cout, moyenne_cout + moyenne_cout*bande_cout/100)])

                
            if loi_cout == "normal":
                moyenne_cout = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                bande_cout = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du cout", 1/10)
                st.write(f"On considère donc une loi normale de moyenne {moyenne_cout} et d'ecart_type {bande_cout}")
                
                arr = np.random.normal(moyenne_cout, bande_cout*moyenne_cout, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.cout_tonne_remuee = np.array(n * [np.random.normal(moyenne_cout, bande_cout*moyenne_cout)])

            if loi_cout == "uniform":
                moyenne_cout = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                bande_cout = st.number_input(f"Pourcentage de la bande du cout:", 0.1)
                st.write(f"On considère donc une loi uniforme centrée sur {moyenne_cout} et de bande {bande_cout}")
                
                arr = np.random.uniform(moyenne_cout-moyenne_cout*bande_cout/200, moyenne_cout+moyenne_cout*bande_cout/200, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.cout_tonne_remuee = np.array(n * [np.random.normal(moyenne_cout, bande_cout*moyenne_cout)])

            # Fin choix loi coût    

            # Début choix loi teneur
            st.subheader("Loi de probabilité suivie par la teneur")
            
            loi_teneur = st.selectbox("loi de la teneur", ['triangular', 'normal', 'uniform'])

            if loi_teneur == "triangular":
                moyenne_teneur = st.number_input(f"La teneur est modélisé par une loi triangulaire centrée sur la valeur {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                bande_teneur = st.number_input("Pourcentage de la bande de la teneur:", 1, 50, 10, 1)
                st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_teneur}M$ et de bande {bande_teneur}%')

                arr = np.random.triangular(moyenne_teneur - moyenne_teneur*bande_teneur/100, moyenne_teneur, moyenne_teneur + moyenne_teneur*bande_teneur/100, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.teneur_minerai_geol = np.array(n * [np.random.triangular(moyenne_teneur - moyenne_teneur*bande_teneur/100, moyenne_teneur, moyenne_teneur + moyenne_teneur*bande_teneur/100)])
                
                st.write("Les nouvelles valeurs pour la teneur du minerai géologique sont (pour une simulation):", valeurs.investissement)

            if loi_teneur == "normal":
                moyenne_teneur = st.number_input(f"La teneur est modélisée par une loi normale de moyenne {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                bande_teneur = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
                st.write(f"On considère donc une loi normale de moyenne {moyenne_teneur} et d'ecart_type {bande_teneur}")
                
                arr = np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur)])

            if loi_teneur == "uniform":
                moyenne_teneur = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                bande_teneur = st.number_input(f"Pourcentage de la bande de la teneur:", 0.1)
                st.write(f"On considère donc une loi uniforme centrée sur {moyenne_teneur} et de bande {bande_teneur}")
                
                arr = np.random.uniform(moyenne_teneur-moyenne_teneur*bande_teneur/200, moyenne_teneur+moyenne_teneur*bande_teneur/200, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur)])

            # Fin choix loi teneur
            
            # Début choix loi tonnage
            
            st.subheader("Loi de probabilité suivie par le tonnage")

            loi_tonnage = st.selectbox("loi du tonnage", ['triangular', 'normal', 'uniform'])

            if loi_tonnage == "triangular":
                moyenne_tonnage = st.number_input(f"Le tonnage est modélisé par une loi triangulaire centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.investissement[0])
                bande_tonnage = st.number_input("Pourcentage de la bande du tonnage:", 1, 50, 10, 1)
                st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_tonnage}M$ et de bande {bande_tonnage}%')

                arr = np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.tonnage_geol = np.array(n * [np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100)])
                

            if loi_tonnage == "normal":
                moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi normale de moyenne {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                bande_tonnage = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du tonnage", 1/10)
                st.write(f"On considère donc une loi normale de moyenne {moyenne_tonnage} et d'ecart_type {bande_tonnage}")
                
                arr = np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

            if loi_tonnage == "uniform":
                moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi uniforme centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                bande_tonnage = st.number_input(f"Pourcentage de la bande du tonnage:", 0.1)
                st.write(f"On considère donc une loi uniforme centrée sur {moyenne_tonnage} et de bande {bande_tonnage}")
                
                arr = np.random.uniform(moyenne_tonnage-moyenne_tonnage*bande_tonnage/200, moyenne_tonnage+moyenne_tonnage*bande_tonnage/200, size=10000)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=200)
                st.pyplot(fig)

                valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

            # Fin choix loi tonnage

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
                valeurs.investissement += loi_func(loi_invest, moyenne_invest, bande_invest, n)
                valeurs.cout_tonne_remuee += loi_func(loi_cout, moyenne_cout, bande_cout, n)
                valeurs.tonnage_geol += loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
                valeurs.teneur_minerai_geol += loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

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
            tri_simu = np.zeros(nb_simu)

            while i < nb_simu:
                valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n)
                valeurs.cout_tonne_remuee = loi_func(loi_cout, moyenne_cout, bande_cout, n)
                valeurs.tonnage_geol = loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
                valeurs.teneur_minerai_geol = loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

                # pour la proba que tri>tri_goal
                if valeurs.tri()[annee-1] > tri_goal:
                    iter_tri += 1
                tri_simu[i] = valeurs.tri()[annee-1]
                i+=1

            # On affiche la répartition des valeurs du TRI à l'aide d'un histogramme

            fig, ax = plt.subplots()
            ax.hist(tri_simu, bins=25)
            plt.title(f"Répartition des valeurs du TRI à l'année {annee}")
            st.pyplot(fig)

            iter_tri *= 1/nb_simu

            st.write(f"La probabilité que le TRI soit supérieur à {tri_goal} à l'année {annee} vaut {iter_tri}")

            # Calculons alors le risque de rentabilité du projet

            st.header("Risque de rentabilité de l'investissement")

            i = 0
            iter_van = 0
            annee = st.number_input("Caculons la probabilité que l'investissement ne soit pas rentable à l'année:", 1, 15, 15, 1)
            esperance_perte = 0
            van_simu = np.zeros(nb_simu)

            while i < nb_simu:
                valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n)
                valeurs.cout_tonne_remuee = loi_func(loi_cout, moyenne_cout, bande_cout, n)
                valeurs.tonnage_geol = loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
                valeurs.teneur_minerai_geol = loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

                # pour la proba que van<0
                if valeurs.cumul_cash_flow()[annees-1] < 0:
                    iter_van += 1
                    esperance_perte += valeurs.cumul_cash_flow()[annees-1]
                van_simu[i] = valeurs.cumul_cash_flow()[annees-1]
                i+=1
            
            esperance_perte *= 1/nb_simu
            iter_van *= 1/nb_simu

            # Affichage de la répartition du van

            fig, ax = plt.subplots()
            ax.hist(van_simu, 25)
            plt.title(f"Répartition des valeurs du VAN à l'année {annees}")
            st.pyplot(fig)

            st.write(f"La probabilité que le VAN soit négatif à l'année {annees} vaut {iter_van}")

            st.write(f"De plus, l'espérance de perte à l'année {annees} vaut {esperance_perte }M$")