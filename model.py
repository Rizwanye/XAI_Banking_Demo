import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

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

def predict_loan_approval(input_values, clf):
    input_data = pd.DataFrame([input_values])
    prediction = clf.predict(input_data)
    return prediction

def explain_prediction(input_values, clf):
    input_data = pd.DataFrame([input_values])
    feature_names = X_train.columns.tolist()
    explainer = LimeTabularExplainer(training_data=X_train.values, mode="classification", feature_names=X.columns)

    explanation = explainer.explain_instance(input_data.values[0], clf.predict_proba)
    return explanation

def explained_list(input_values, clf):
    input_data = pd.DataFrame([input_values])
    explainer = LimeTabularExplainer(training_data=X_train.values, mode="classification", feature_names=X.columns)
    explanation = explainer.explain_instance(input_data.values[0], clf.predict_proba, num_samples=1000, top_labels=1, num_features=10)

    label_indexes = explanation.available_labels()
    first_label_index = label_indexes[0]

    explanation_list = []

    for feature, weight in explanation.as_list(label=first_label_index):
        feature_name = feature.split(' <= ')[0].split(' < ')[-1]
        explanation_list.append((feature_name, weight))
    return explanation_list
