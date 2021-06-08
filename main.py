import project
import cas_base
import project_proba
import streamlit as st
PAGES = {
    "Tracé des données" : cas_base,
    "analyse de sensibilité": project,
    "Analyse Stochastique" : project_proba,
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()