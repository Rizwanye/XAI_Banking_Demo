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

print("hello world!")
st.title("Banking APP")
st.write(""""
# Explore the reason why you recieve or decline loan?
The model will show you the reason why your loan was approved
""")

# Load your dataset
file_path = "Banking_Dataset.csv"
df = pd.read_csv(file_path)

# Preprocess the data as you did in your code
banking_dataset = df.drop(columns="Loan_ID")
banking_dataset = banking_dataset.dropna()
banking_dataset = pd.get_dummies(banking_dataset, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
banking_dataset['Loan_Status'] = banking_dataset['Loan_Status'].map({'Y': 1, 'N': 0})
X = banking_dataset.drop(columns=['Loan_Status'])
y = banking_dataset['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Define the list of categorical indices based on your column names
categorical_features = [i for i, col in enumerate(X.columns) if col.startswith(('Gender_', 'Married_', 'Dependents_', 'Education_', 'Self_Employed_', 'Property_Area_'))]

# Train your Decision Tree model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


st.sidebar.slider("Income Per Year", 0, 10000)
st.sidebar.slider("Co-Applicant-Income Per Year", 0, 10000)
st.sidebar.slider("Loan Amount", 0, 10000)
st.sidebar.selectbox("Loan Term Duration", (120, 180, 240, 300, 360, 480))