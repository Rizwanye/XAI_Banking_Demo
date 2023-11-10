import streamlit as st
from sidebar_logo import add_logo
from streamlit_extras.switch_page_button import switch_page
from PIL import Image

### Miscallaneous Side bar stuff
favicon = "images/favicon.png"
st.set_page_config(
    page_icon=favicon
)

add_logo("images/xai_logo.jpg")
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
click = st.button("Click Here To Access XAI Loan Application")
if click:
    switch_page("Loan Application")

### Bottom information

st.markdown(
    """
    <div style="padding-top: 20px;">
        Ever since banks have witnessed the 2008 financial crisis, loan applications have adopted ever increasing complexity using tools such as:
        <ul style="padding-top: 10px;">
            <li>Credit scoring models</li>
            <li>Machine learning algorithms</li>
            <li>Behavioral analytics</li>
            <li>Predictive analytics</li>
            <li>and more</li>
        </ul>
        Often loans are approved or declined without much reasoning or deep understanding.
    </div>
    """,
    unsafe_allow_html=True
)
######
st.markdown("""
# How does XAI bank help?
""")

image = Image.open('images/loan_explained.jpg')
st.image(image, caption='Image Source By Knime')


st.markdown("""
XAI Bank was developed using a decision tree, and the decision-making process was explained through the LIME model. This approach enhances transparency by providing interpretable insights into complex machine learning models.

To improve user comprehension, a part of this development utilized theoretical principles like cognitive load to understand how to appropriately present results for end users.

Based on an empirical survey, XAI Bank utilizes radar charts and a language model to further aid users in understanding the features used within the machine learning model. In this context, the web app helps the end user understand the outcome of their loan application, which often in the real world is not transparent due to the complex systems utilized.

This web app is the outcome of extensive empirical research conducted as part of the UTS Honors project.
""")

