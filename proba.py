from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from ipywidgets import interact, FloatText
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prix_or import prix
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

            
            valeurs_non_modif = deepcopy(valeurs)

            st.header("Modélisation des lois de probabilité des paramètres incertains")

            # On modélise la loi de probabilité de l'investissement en tenant compte du choix du client

            variables = st.multiselect("variables soumises à des incertitudes",["investissement", "coût", "teneur", "tonnage"])

            #Début choix loi investissement

            if "investissement" in variables:
            
                st.subheader("Loi de probabilité suivie par l'investissement")

                loi_invest = st.selectbox("loi de l'investissement", ['triangular', 'normal', 'uniform'])

                if loi_invest == "triangular":
                    moyenne_invest = st.number_input(f"L'investissement est modélisé par une loi triangulaire centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.investissement[0])
                    bande_invest = st.number_input("Pourcentage de la bande de l'investissement:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_invest :.2f}M$ et de bande {bande_invest}%')

                    arr = np.random.triangular(moyenne_invest - moyenne_invest*bande_invest/100, moyenne_invest, moyenne_invest + moyenne_invest*bande_invest/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("investissement")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.investissement = np.array(n * [np.random.triangular(moyenne_invest - moyenne_invest*bande_invest/100, moyenne_invest, moyenne_invest + moyenne_invest*bande_invest/100)])
                    valeurs.investissement[1:]= 0

                if loi_invest == "normal":
                    moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_invest = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_invest} et d'ecart_type {bande_invest}")
                    
                    arr = np.random.normal(moyenne_invest, bande_invest*moyenne_invest, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("investissement")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])
                    valeurs.investissement[1:]= 0

                if loi_invest == "uniform":
                    moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_invest = st.number_input(f"Pourcentage de la bande de l'investissement:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_invest} et de bande {bande_invest}")
                    
                    arr = np.random.uniform(moyenne_invest-moyenne_invest*bande_invest/200, moyenne_invest+moyenne_invest*bande_invest/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("investissement")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])
                    valeurs.investissement[1:]= 0

                # Fin choix loi investissement
            
            if "investissement" not in variables:
                loi_invest = "uniform"
                bande_invest = 0
                moyenne_invest = valeurs.investissement[0]
            
            # Début choix loi coût
            if "coût" in variables :

                st.subheader("Loi de probabilité suivie par le coût")
                loi_cout= st.selectbox("loi du coût", ['triangular', 'normal', 'uniform'])

                if loi_cout == "triangular":
                    moyenne_cout = st.number_input(f"Le cout est modélisé par une loi triangulaire centrée sur la valeur {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                    bande_cout = st.number_input("Pourcentage de la bande du cout:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_cout}M$ et de bande {bande_cout}%')

                    arr = np.random.triangular(moyenne_cout - moyenne_cout*bande_cout/100, moyenne_cout, moyenne_cout + moyenne_cout*bande_cout/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("coût")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.cout_tonne_remuee = np.array(n * [np.random.triangular(moyenne_cout - moyenne_cout*bande_cout/100, moyenne_cout, moyenne_cout + moyenne_cout*bande_cout/100)])

                    
                if loi_cout == "normal":
                    moyenne_cout = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                    bande_cout = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du cout", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_cout} et d'ecart_type {bande_cout}")
                    
                    arr = np.random.normal(moyenne_cout, bande_cout*moyenne_cout, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("coût")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.cout_tonne_remuee = np.array(n * [np.random.normal(moyenne_cout, bande_cout*moyenne_cout)])

                if loi_cout == "uniform":
                    moyenne_cout = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.cout_tonne_remuee[0]}. Possibilité de changer cette valeur:", value = valeurs.cout_tonne_remuee[0])
                    bande_cout = st.number_input(f"Pourcentage de la bande du cout:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_cout} et de bande {bande_cout}")
                    
                    arr = np.random.uniform(moyenne_cout-moyenne_cout*bande_cout/200, moyenne_cout+moyenne_cout*bande_cout/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("coût")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.cout_tonne_remuee = np.array(n * [np.random.normal(moyenne_cout, bande_cout*moyenne_cout)])

                # Fin choix loi coût    
            if "coût" not in variables:
                loi_cout = "uniform"
                bande_cout = 0
                moyenne_cout = valeurs.cout_tonne_remuee[0]

            # Début choix loi teneur
            if "teneur" in variables:

                st.subheader("Loi de probabilité suivie par la teneur")
                
                loi_teneur = st.selectbox("loi de la teneur", ['triangular', 'normal', 'uniform'])

                if loi_teneur == "triangular":
                    moyenne_teneur = st.number_input(f"La teneur est modélisé par une loi triangulaire centrée sur la valeur {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_teneur = st.number_input("Pourcentage de la bande de la teneur:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_teneur}M$ et de bande {bande_teneur}%')

                    arr = np.random.triangular(moyenne_teneur - moyenne_teneur*bande_teneur/100, moyenne_teneur, moyenne_teneur + moyenne_teneur*bande_teneur/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("teneur")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.teneur_minerai_geol = np.array(n * [np.random.triangular(moyenne_teneur - moyenne_teneur*bande_teneur/100, moyenne_teneur, moyenne_teneur + moyenne_teneur*bande_teneur/100)])
                    
    

                if loi_teneur == "normal":
                    moyenne_teneur = st.number_input(f"La teneur est modélisée par une loi normale de moyenne {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_teneur = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_teneur} et d'ecart_type {bande_teneur}")
                    
                    arr = np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("teneur")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur)])

                if loi_teneur == "uniform":
                    moyenne_teneur = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.teneur_minerai_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_teneur = st.number_input(f"Pourcentage de la bande de la teneur:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_teneur} et de bande {bande_teneur}")
                    
                    arr = np.random.uniform(moyenne_teneur-moyenne_teneur*bande_teneur/200, moyenne_teneur+moyenne_teneur*bande_teneur/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("teneur")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.teneur_minerai_geol = np.array(n * [np.random.normal(moyenne_teneur, bande_teneur*moyenne_teneur)])

                # Fin choix loi teneur
            if "teneur" not in variables:
                loi_teneur = "uniform"
                bande_teneur = 0
                moyenne_teneur = valeurs.teneur_minerai_geol[0]
            
            # Début choix loi tonnage

            if "tonnage" in variables:
                
                st.subheader("Loi de probabilité suivie par le tonnage")

                loi_tonnage = st.selectbox("loi du tonnage", ['triangular', 'normal', 'uniform'])

                if loi_tonnage == "triangular":
                    moyenne_tonnage = st.number_input(f"Le tonnage est modélisé par une loi triangulaire centrée sur la valeur {valeurs.tonnage_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.tonnage_geol[0])
                    bande_tonnage = st.number_input("Pourcentage de la bande du tonnage:", 1, 50, 10, 1)
                    st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_tonnage}M$ et de bande {bande_tonnage}%')

                    arr = np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("tonnage")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.tonnage_geol = np.array(n * [np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100)])
                    

                if loi_tonnage == "normal":
                    moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi normale de moyenne {valeurs.tonnage_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_tonnage = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du tonnage", 1/10)
                    st.write(f"On considère donc une loi normale de moyenne {moyenne_tonnage} et d'ecart_type {bande_tonnage}")
                    
                    arr = np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("tonnage")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

                if loi_tonnage == "uniform":
                    moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi uniforme centrée sur la valeur {valeurs.tonnage_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
                    bande_tonnage = st.number_input(f"Pourcentage de la bande du tonnage:", 0.1)
                    st.write(f"On considère donc une loi uniforme centrée sur {moyenne_tonnage} et de bande {bande_tonnage}")
                    
                    arr = np.random.uniform(moyenne_tonnage-moyenne_tonnage*bande_tonnage/200, moyenne_tonnage+moyenne_tonnage*bande_tonnage/200, size=10000)
                    fig, ax = plt.subplots()
                    ax.hist(arr, bins=200)
                    plt.xlabel("tonnage")
                    plt.ylabel("nombre d'occurences lors de la simulation")
                    plt.legend()
                    st.pyplot(fig)

                    valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

                # Fin choix loi tonnage
            if "tonnage" not in variables:
                loi_tonnage = "uniform"
                bande_tonnage = 0
                moyenne_tonnage = valeurs.tonnage_geol[0]

            # Debut du choix du prix de l'or:

            st.subheader("Loi de probabilité suivie par le prix de l'or")

            loi_prix_or = st.selectbox("loi du prix de l'or", ['constante','modele de Heath-Jarrow-Morton' ])

            if loi_prix_or == "modele de Heath-Jarrow-Morton":
                prix_or_init = st.number_input(f"Le prix de l'or est modélisé par le modele de Heath-Jarrow-Morton partant de la valeur {valeurs.prix_or[0]}. Possibilité de changer cette valeur:", value = valeurs.prix_or[0])
                Alpha = st.number_input("Paramètre exponentiel du model:", 0.1, 0.3, 0.2, 0.01)
                Sigma = st.number_input("Ecart type du prix de l'or:", 0.05, 0.15, 0.1, 0.01)

                st.write("voici un exemple d'une suite possible des valeurs du prix de l'or selon ce modèle")

                arr = np.array(prix(prix_or_init, n, 12, alpha=Alpha,sigma=Sigma))
                fig, ax = plt.subplots()
                ax.plot(arr, 'o')
                plt.style.use('seaborn')
                st.pyplot(fig)


                valeurs.prix_or = np.array(prix(prix_or_init, n, 12, alpha=Alpha,sigma=Sigma))

            if loi_prix_or == "constante":
                prix_or_init = st.number_input(f"Le prix de l'or est considéré constant de valeur {valeurs.prix_or[0]}. Possibilité de changer cette valeur:", value = valeurs.prix_or[0])
                valeurs.prix_or = np.array(n * [prix_or_init])
            
            #Fin du choix du prix de l'or


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
                valeurs.investissement += loi_func(loi_invest, moyenne_invest, bande_invest, n, bool = False)
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
                valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n, bool = False)
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

            st.write(f"La probabilité que le TRI soit supérieur à {tri_goal} à l'année {annee} vaut {iter_tri :.2f}")

            # Calculons alors le risque de rentabilité du projet

            st.header("Risque de rentabilité de l'investissement")

            i = 0
            iter_van = 0
            annee = st.number_input("Caculons la probabilité que l'investissement ne soit pas rentable à l'année:", 1, 15, 15, 1)
            esperance_perte = 0
            van_simu = np.zeros(nb_simu)

            while i < nb_simu:
                valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n, bool = False)
                valeurs.cout_tonne_remuee = loi_func(loi_cout, moyenne_cout, bande_cout, n)
                valeurs.tonnage_geol = loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
                valeurs.teneur_minerai_geol = loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

                # pour la proba que van<0
                if valeurs.cumul_cash_flow()[annee-1] < 0:
                    iter_van += 1
                    esperance_perte += valeurs.cumul_cash_flow()[annee-1]
                van_simu[i] = valeurs.cumul_cash_flow()[annee-1]
                i+=1
            
            esperance_perte *= 1/nb_simu
            iter_van *= 1/nb_simu

            # Affichage de la répartition du van

            fig, ax = plt.subplots()
            ax.hist(van_simu, 25)
            plt.title(f"Répartition des valeurs du VAN à l'année {annee}")
            st.pyplot(fig)

            st.write(f"La probabilité que le VAN soit négatif à l'année {annee} vaut {iter_van :.2f}")

            st.write(f"De plus, l'espérance de perte à l'année {annee} vaut {esperance_perte :.2f }M$")


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
        valeurs.investissement[1:] = 0
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

        
        valeurs_non_modif = deepcopy(valeurs)

        st.header("Modélisation des lois de probabilité des paramètres incertains")
        variables = st.multiselect("variables soumises à des incertitudes",["investissement", "coût", "teneur", "tonnage"])

            #Début choix loi investissement

             


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
            valeurs.investissement[1:] = 0

        if loi_invest == "normal":
            moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi normale de moyenne {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
            bande_invest = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne de la teneur", 1/10)
            st.write(f"On considère donc une loi normale de moyenne {moyenne_invest} et d'ecart_type {bande_invest}")
            
            arr = np.random.normal(moyenne_invest, bande_invest*moyenne_invest, size=10000)
            fig, ax = plt.subplots()
            ax.hist(arr, bins=200)
            st.pyplot(fig)

            valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])
            valeurs.investissement[1:] = 0

        if loi_invest == "uniform":
            moyenne_invest = st.number_input(f"L'investissement est modélisée par une loi uniforme centrée sur la valeur {valeurs.investissement[0]}. Possibilité de changer cette valeur:", value = valeurs.investissement[0])
            bande_invest = st.number_input(f"Pourcentage de la bande de l'investissement:", 0.1)
            st.write(f"On considère donc une loi uniforme centrée sur {moyenne_invest} et de bande {bande_invest}")
            
            arr = np.random.uniform(moyenne_invest-moyenne_invest*bande_invest/200, moyenne_invest+moyenne_invest*bande_invest/200, size=10000)
            fig, ax = plt.subplots()
            ax.hist(arr, bins=200)
            st.pyplot(fig)

            valeurs.investissement = np.array(n * [np.random.normal(moyenne_invest, bande_invest*moyenne_invest)])
            valeurs.investissement[1:] = 0

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
            moyenne_tonnage = st.number_input(f"Le tonnage est modélisé par une loi triangulaire centrée sur la valeur {valeurs.tonnage_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.tonnage_geol[0])
            bande_tonnage = st.number_input("Pourcentage de la bande du tonnage:", 1, 50, 10, 1)
            st.write(f'On considère donc une loi triangulaire centrée sur {moyenne_tonnage}M$ et de bande {bande_tonnage}%')

            arr = np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100, size=10000)
            fig, ax = plt.subplots()
            ax.hist(arr, bins=200)
            st.pyplot(fig)

            valeurs.tonnage_geol = np.array(n * [np.random.triangular(moyenne_tonnage - moyenne_tonnage*bande_tonnage/100, moyenne_tonnage, moyenne_tonnage + moyenne_tonnage*bande_tonnage/100)])
            

        if loi_tonnage == "normal":
            moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi normale de moyenne {valeurs.tonnage_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
            bande_tonnage = st.number_input(f"... et d'écart-type 1/10*la valeur moyenne du tonnage", 1/10)
            st.write(f"On considère donc une loi normale de moyenne {moyenne_tonnage} et d'ecart_type {bande_tonnage}")
            
            arr = np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage, size=10000)
            fig, ax = plt.subplots()
            ax.hist(arr, bins=200)
            st.pyplot(fig)

            valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

        if loi_tonnage == "uniform":
            moyenne_tonnage = st.number_input(f"Le tonnage est modélisée par une loi uniforme centrée sur la valeur {valeurs.tonnage_geol[0]}. Possibilité de changer cette valeur:", value = valeurs.teneur_minerai_geol[0])
            bande_tonnage = st.number_input(f"Pourcentage de la bande du tonnage:", 0.1)
            st.write(f"On considère donc une loi uniforme centrée sur {moyenne_tonnage} et de bande {bande_tonnage}")
            
            arr = np.random.uniform(moyenne_tonnage-moyenne_tonnage*bande_tonnage/200, moyenne_tonnage+moyenne_tonnage*bande_tonnage/200, size=10000)
            fig, ax = plt.subplots()
            ax.hist(arr, bins=200)
            st.pyplot(fig)

            valeurs.tonnage_geol = np.array(n * [np.random.normal(moyenne_tonnage, bande_tonnage*moyenne_tonnage)])

        # Fin choix loi tonnage
         # Debut du choix du prix de l'or:

        st.subheader("Loi de probabilité suivie par le prix de l'or")

        loi_prix_or = st.selectbox("loi du prix de l'or", ['constante','modele de Heath-Jarrow-Morton' ])

        if loi_prix_or == "modele de Heath-Jarrow-Morton":
            prix_or_init = st.number_input(f"Le prix de l'or est modélisé par le modele de Heath-Jarrow-Morton partant de la valeur {valeurs.prix_or[0]}. Possibilité de changer cette valeur:", value = valeurs.prix_or[0])
            Alpha = st.number_input("Paramètre exponentiel du model:", 0.02, 0.12, 0.06, 0.01)
            Sigma = st.number_input("Ecart type du prix de l'or:", 0.05, 0.15, 0.1, 0.01)

            st.write("voici un exemple d'une suite possible des valeurs du prix de l'or selon ce modèle")

            arr = np.array(prix(prix_or_init, n, 12, alpha=Alpha,sigma=Sigma))
            fig, ax = plt.subplots()
            ax.plot(arr, 'o')
            plt.style.use('seaborn')
            st.pyplot(fig)


            valeurs.prix_or = np.array(prix(prix_or_init, n, 12, alpha=Alpha,sigma=Sigma))

        if loi_prix_or == "constante":
            prix_or_init = st.number_input(f"Le prix de l'or est considéré constant de valeur {valeurs.prix_or[0]}. Possibilité de changer cette valeur:", value = valeurs.prix_or[0])
            valeurs.prix_or = np.array(n * [prix_or_init])
        
        #Fin du choix du prix de l'or


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
            valeurs.investissement[1:] = 0
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
            valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n, bool = False)
            valeurs.investissement[1:] = 0
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

        st.write(f"La probabilité que le TRI soit supérieur à {tri_goal} à l'année {annee} vaut {iter_tri :.2f}")

        # Calculons alors le risque de rentabilité du projet

        st.header("Risque de rentabilité de l'investissement")

        i = 0
        iter_van = 0
        annee = st.number_input("Caculons la probabilité que l'investissement ne soit pas rentable à l'année:", 1, 15, 15, 1)
        esperance_perte = 0
        van_simu = np.zeros(nb_simu)

        while i < nb_simu:
            valeurs.investissement = loi_func(loi_invest, moyenne_invest, bande_invest, n)
            valeurs.investissement[1:] = 0
            valeurs.cout_tonne_remuee = loi_func(loi_cout, moyenne_cout, bande_cout, n)
            valeurs.tonnage_geol = loi_func(loi_tonnage, moyenne_tonnage, bande_tonnage, n)
            valeurs.teneur_minerai_geol = loi_func(loi_teneur, moyenne_teneur, bande_teneur, n)

            # pour la proba que van<0
            if valeurs.cumul_cash_flow()[annee-1] < 0:
                iter_van += 1
                esperance_perte += valeurs.cumul_cash_flow()[annee-1]
            van_simu[i] = valeurs.cumul_cash_flow()[annee-1]
            i+=1
        
        esperance_perte *= 1/nb_simu
        iter_van *= 1/nb_simu

        # Affichage de la répartition du van

        fig, ax = plt.subplots()
        ax.hist(van_simu, 25)
        plt.title(f"Répartition des valeurs du VAN à l'année {annee}")
        st.pyplot(fig)

        st.write(f"La probabilité que le VAN soit négatif à l'année {annee} vaut {iter_van :.2f}")

        st.write(f"De plus, l'espérance de perte à l'année {annee} vaut {esperance_perte:.2f}M$")
