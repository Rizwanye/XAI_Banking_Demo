import streamlit as st
import csv
import pandas as pd
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sidebar_logo import add_logo
add_logo("xai_logo.jpg")
st.markdown("# Home")

print("hello world!")
st.title("Banking APP")
st.write(""""
# Explore the reason why you recieve or decline loan?
The model will show you the reason why your loan was approved
""")

st.sidebar.markdown(
    "This web app was created by Rizwan Ye as part of his Honours Project. You can find him on [GitHub](https://github.com/Rizwanye). \n\n**Supervisors From UTS:**\n- Shoujin Wang \n- Jianlong Zhou "
)
