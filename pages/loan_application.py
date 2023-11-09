import streamlit as st
from sidebar_logo import add_logo
from model import predict_loan_approval, explain_prediction, clf
add_logo("xai_logo.jpg")

st.sidebar.markdown("# Applicant Information")
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
married = st.sidebar.selectbox("Married", ("Yes", "No"))
graduated = st.sidebar.selectbox("Graduated From Uni", ("Yes", "No"))
dependents = st.sidebar.selectbox("Dependents", (1, 2, 3, 4))

st.sidebar.markdown("# Financial Details")
income = st.sidebar.slider("Income Per Year", 0, 10000)
co_applicant_income = st.sidebar.slider("Co-Applicant-Income Per Year", 0, 10000)
employment = st.sidebar.selectbox("Self employment", ("Yes", "No"))

st.sidebar.markdown("# Loan Request")
loan_amount = st.sidebar.slider("Loan Amount", 0, 10000)
loan_term = st.sidebar.selectbox("Loan Term Duration", (120, 180, 240, 300, 360, 480))
area = st.sidebar.selectbox("Property Area", ("Rural", "Semiurban", "Urban"))

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
        'Property_Area_Urban': 1 if area == "Urban" else 0
}

st.title("Bank Loan Application Explainer")

# Make predictions using the model
prediction = predict_loan_approval(input_values, clf)

# Map the prediction to its corresponding label
prediction_label = 'Your loan has been successfully approved' if prediction[0] == 1 else 'We apologize, your loan is not acceptable'

# Display the prediction in the Streamlit app
st.write("Our machine learning algorithm has determined that: ", prediction_label)
st.write(input_values)

## side bar information



# Map the prediction to its corresponding label
prediction_label = 'Your loan has been succesfully approved' if prediction[0] == 1 else 'We apologise your loan is not acceptable'

# Display the prediction in the Streamlit app
st.write("Our machine learning algorithm has determined that: ", prediction_label)
st.write(input_values)

# Create a Streamlit app
st.title("Loan Prediction")
#prediction_label = 'Y'
st.write(f"Prediction for the given input: {prediction_label}")
explanation = explain_prediction(input_values, clf)
st.pyplot(explanation.as_pyplot_figure())