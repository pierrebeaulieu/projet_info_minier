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