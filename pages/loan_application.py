import streamlit as st
from sidebar_logo import add_logo
from model import predict_loan_approval, explain_prediction, clf, explained_list

favicon = "favicon.png"
st.set_page_config(
    page_icon=favicon
)
add_logo("xai_logo.jpg")

st.sidebar.markdown("# Applicant Information")
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
married = st.sidebar.selectbox("Married", ("Yes", "No"))
graduated = st.sidebar.selectbox("Graduated From Uni", ("Yes", "No"))
dependents = st.sidebar.selectbox("Dependents", (0, 1, 2, 3, 4))

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
st.markdown("<p style='font-size: 20px;'>This visualization shows the most important features used to determine the Loan application outcome.</p>", unsafe_allow_html=True)

# Make predictions using the model
prediction = predict_loan_approval(input_values, clf)

prediction_label = 'Your loan has been successfully approved' if prediction[0] == 1 else 'We apologize, your loan is not acceptable'

st.write(f"Prediction for the given input: {prediction_label}")

# Create a Streamlit app
st.title("Loan Prediction")

explanation = explain_prediction(input_values, clf)
st.pyplot(explanation.as_pyplot_figure())

##########################################################################
# Explain the prediction for the input data

# Now you have the simplified feature names and weights in a list of tuples
# You can display it or further process it as needed



def explanation_list_output():
        prediction = predict_loan_approval(input_values, clf)
        if prediction[0] == 0:
                inverted_explanation_list = explained_list(input_values, clf)
                update = [(key, value) for key, value in inverted_explanation_list if
                                 not (key.startswith("Dependents_") and input_values.get(key, 0) == 0)]
                explanation_list= [(feature, -value) for feature, value in update]
        else:
                explanation_list_1 = explained_list(input_values, clf)
                explanation_list = [(key, value) for key, value in explanation_list_1 if
                  not (key.startswith("Dependents_") and input_values.get(key, 0) == 0)]

        return explanation_list


from radar_chart import plot_feature_importance
explanation_list = explanation_list_output()
plot_feature_importance(explanation_list)