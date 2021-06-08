import project
import streamlit as st
PAGES = {
    "analyse de sensibilité": project,
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()