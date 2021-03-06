from loi import loi_func
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

import class_entree

# n est les nombre d'années sur lesquelles on veut effectuer le calcul

def app():
    
    n=15
    # on extrait les données du tableau de base et modifie en fonction du choix du client
    valeurs = class_entree.entree(class_entree.extraction_csv("donnees_entree_proj_minier.csv", n))

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

    st.write(f"La probabilité que le VAN soit négatif à l'année {annee} vaut {iter_van}")

    st.write(f"De plus, l'espérance de perte à l'année {annee} vaut {esperance_perte }M$")
