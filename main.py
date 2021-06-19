
import cas_base

import streamlit as st



PAGES = {
    "Analyse économique de projets miniers" : cas_base,
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

