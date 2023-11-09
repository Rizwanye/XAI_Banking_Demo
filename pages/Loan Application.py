import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sidebar_logo import add_logo

#sidebar logo
add_logo("xai_logo.jpg")


# Back End Decision Tree Development
file_path = "Banking_Dataset.csv"
df = pd.read_csv(file_path)

banking_dataset = df.drop(columns="Loan_ID")
banking_dataset = banking_dataset.dropna()
banking_dataset = pd.get_dummies(banking_dataset, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
banking_dataset['Loan_Status'] = banking_dataset['Loan_Status'].map({'Y': 1, 'N': 0})
X = banking_dataset.drop(columns=['Loan_Status', 'Credit_History'])
y = banking_dataset['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

categorical_features = [i for i, col in enumerate(X.columns) if col.startswith(('Gender_', 'Married_', 'Dependents_', 'Education_', 'Self_Employed_', 'Property_Area_'))]

# Training
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## side bar information
st.sidebar.markdown("# Applicant Information")
gender = st.sidebar.selectbox("Gender", ("Male","Female"))
married = st.sidebar.selectbox("Married", ("Yes","No"))
graduated = st.sidebar.selectbox("Graduated From Uni", ("Yes","No"))
dependents = st.sidebar.selectbox("Dependents", (1,2,3,4))

st.sidebar.markdown("# Financial Details")
income = st.sidebar.slider("Income Per Year", 0, 10000)
co_applicant_income = st.sidebar.slider("Co-Applicant-Income Per Year", 0, 10000)
employment = st.sidebar.selectbox("Self employment", ("Yes","No"))

st.sidebar.markdown("# Loan Request")
loan_amount = st.sidebar.slider("Loan Amount", 0, 10000)
loan_term = st.sidebar.selectbox("Loan Term Duration", (120, 180, 240, 300, 360, 480))
area = st.sidebar.selectbox("Property Area", ("Rural","Semiurban","Urban"))

# User details to be pushed into decision classifier
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
prediction_label = 'Your loan has been succesfully approved' if prediction[0] == 1 else 'We apologise your loan is not acceptable'

# Display the prediction in the Streamlit app
st.write("Our machine learning algorithm has determined that: ", prediction_label)
st.write(input_values)

from lime.lime_tabular import LimeTabularExplainer

# Create a Pandas DataFrame from the input values
input_data = pd.DataFrame([input_values])

# Load your trained classifier (clf) and data for Lime explanation (X_train, categorical_features)

# Initialize a LimeTabularExplainer
explainer = LimeTabularExplainer(training_data=X_train.values, mode="classification", feature_names=X.columns)

# Predict using your model (make sure 'Credit_History' is removed as previously mentioned)

# Get the prediction probability for the positive class (Y)

# Create a function to explain the prediction
def explain_prediction(input_data, clf):
    # Ensure that the feature names in input_data match the model's training data
    feature_names = X_train.columns.tolist()
    input_data = input_data[feature_names]

    # Explain the prediction
    explanation = explainer.explain_instance(input_data.values[0], clf.predict_proba)
    return explanation

# Create a Streamlit app
st.title("Loan Prediction")
prediction_label = 'Y'
st.write(f"Prediction for the given input: {prediction_label}")
explanation = explain_prediction(input_data, clf)
st.pyplot(explanation.as_pyplot_figure())

st.title("Bank Loan Application Explainer")