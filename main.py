import project_proba
import project
import cas_base
import sensib
import streamlit as st
import proba


PAGES = {
    "Affichage des données" : cas_base,
    "Analyse de sensibilité": sensib,
    "Analyse Probabiliste": proba,
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

