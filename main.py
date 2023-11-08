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

dataset_name = st.sidebar.selectbox("Select Dataset", ("Banking Dataset", "Breast Cancer"))
st.write(dataset_name)

file_path = "Banking_Dataset.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Remove the "Loan_ID" column
banking_dataset = df.drop(columns="Loan_ID")

# Drop rows with missing values
banking_dataset = banking_dataset.dropna()
banking_dataset = df.drop(columns="Loan_ID")

banking_dataset = pd.get_dummies(banking_dataset, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
banking_dataset['Loan_Status'] = banking_dataset['Loan_Status'].map({'Y': 1, 'N': 0})
X = banking_dataset.drop(columns=['Loan_Status'])
y = banking_dataset['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))