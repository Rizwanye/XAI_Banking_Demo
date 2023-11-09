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

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
file_path = "Banking_Dataset.csv"
df = pd.read_csv(file_path)

# Preprocess the data as you did in your code
banking_dataset = df.drop(columns="Loan_ID")
banking_dataset = banking_dataset.dropna()
banking_dataset = pd.get_dummies(banking_dataset, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
banking_dataset['Loan_Status'] = banking_dataset['Loan_Status'].map({'Y': 1, 'N': 0})
X = banking_dataset.drop(columns=['Loan_Status', 'Credit_History'])
y = banking_dataset['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Define the list of categorical indices based on your column names
categorical_features = [i for i, col in enumerate(X.columns) if col.startswith(('Gender_', 'Married_', 'Dependents_', 'Education_', 'Self_Employed_', 'Property_Area_'))]

# Train your Decision Tree model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

income = st.sidebar.slider("Income Per Year", 0, 10000)
co_applicant_income = st.sidebar.slider("Co-Applicant-Income Per Year", 0, 10000)
loan_amount = st.sidebar.slider("Loan Amount", 0, 10000)
loan_term = st.sidebar.selectbox("Loan Term Duration", (120, 180, 240, 300, 360, 480))
gender = st.sidebar.selectbox("Gender", ("Male","Female"))
married = st.sidebar.selectbox("Married", ("Yes","No"))
dependents = st.sidebar.selectbox("Dependents", (1,2,3,4))
graduated = st.sidebar.selectbox("Graduated From Uni", ("Yes","No"))
employment = st.sidebar.selectbox("Self employment", ("Yes","No"))
area = st.sidebar.selectbox("Property Area", ("Rural","Semiurban","Urban"))

# Define input values based on user input
input_values = {
    'ApplicantIncome': income,
    'CoapplicantIncome': co_applicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Gender_Female': 1 if gender == "Female" else 0,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Married_No': 1 if married == "No" else 0,
    'Married_Yes': 1 if married == "No" == "Yes" else 0,
    'Dependents_0': 1 if dependents == 0 else 0,
    'Dependents_1': 1 if dependents == 1 else 0,
    'Dependents_2': 1 if dependents == 2 else 0,
    'Dependents_3+': 1 if dependents == 4 else 0,
    'Education_Graduate': 1 if graduated == "Yes" else 0,
    'Education_Not Graduate': 1 if graduated == "No" else 0,
    'Self_Employed_No': 1 if employment == "No" else 0,
    'Self_Employed_Yes': 1 if employment== "Yes" else 0,
    'Property_Area_Rural': 1 if area == "Rural" else 0,
    'Property_Area_Semiurban': 1 if area == "Semiurban" else 0,
    'Property_Area_Urban': 1 if area == "Urban" else 0,
}

# Create a Pandas DataFrame from the input values
input_data = pd.DataFrame([input_values])

# Use the trained classifier to make a prediction
prediction = clf.predict(input_data)

# Map the prediction to its corresponding label
prediction_label = 'Y' if prediction[0] == 1 else 'N'

# Display the prediction in the Streamlit app
st.write("Hello world!", prediction_label)
st.write(input_values)
