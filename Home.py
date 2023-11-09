import streamlit as st
from sidebar_logo import add_logo
from streamlit_extras.switch_page_button import switch_page

### Miscallaneous Side bar stuff
add_logo("xai_logo.jpg")
st.sidebar.markdown(
    "This web app was created by Rizwan Ye as part of his Honours Project. You can find him on [GitHub](https://github.com/Rizwanye). \n\n**Supervisors From UTS:**\n- Shoujin Wang \n- Jianlong Zhou "
)

## Style of button click
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ed6c01;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #ffffff;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)


### Main page description of project

st.title("XAI Bank: A Holistic Way To Understand Your Loan Application")
click = st.button("Access XAI Loan Application")
if click:
    switch_page("Loan Application")

### Bottom information



