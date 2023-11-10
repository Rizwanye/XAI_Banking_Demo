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


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap


explanation_list = explanation_list_output()

# Sort the features by their weights for better visualization
explanation_list.sort(key=lambda x: x[1], reverse=True)

# Values for the x axis
ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(explanation_list), endpoint=False)

# Feature weights
WEIGHTS = [weight for _, weight in explanation_list]

# Feature names
FEATURES = [feature for feature, _ in explanation_list]

# Colors
COLORS = ["#430161", "#683380", "#ee7A1A", "#ED6C01"]

# Colormap
cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)

# Normalizer
norm = mpl.colors.Normalize(vmin=min(WEIGHTS), vmax=max(WEIGHTS))

# Normalized colors. Each weight is mapped to a color in the color scale 'cmap'
COLORS = cmap(norm(WEIGHTS))

# Initialize layout in polar coordinates
fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})

# Set background color to white, both axis and figure.
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_theta_offset(1.2 * np.pi / 2)
ax.set_ylim(min(WEIGHTS) - 0.1, max(WEIGHTS) + 0.1)

# Add bars to represent the feature weights
ax.bar(ANGLES, WEIGHTS, color=COLORS, alpha=0.9, width=0.52, zorder=10)

# Add dashed vertical lines as references
ax.vlines(ANGLES, min(WEIGHTS), max(WEIGHTS), color="grey", ls=(0, (1, 3)), zorder=11)
# Add labels for the features
FEATURES = ["\n".join(wrap(f, 5, break_long_words=False)) for f in FEATURES]

# Set the labels
ax.set_xticks(ANGLES)
ax.set_xticklabels(FEATURES, size=12)

ax.xaxis.grid(False)
ax.set_yticklabels([])
ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
ax.spines["start"].set_color("none")
ax.spines["polar"].set_color("none")
XTICKS = ax.xaxis.get_major_ticks()
for tick in XTICKS:
    tick.set_pad(10)


# Add legend -----------------------------------------------------

# First, make some room for the legend and the caption in the bottom.
fig.subplots_adjust(bottom=0.175)

# Create an inset axes.
# Width and height are given by the (0.35 and 0.01) in the
# bbox_to_anchor
cbaxes = inset_axes(
    ax,
    width="100%",
    height="100%",
    loc="center",
    bbox_to_anchor=(0.325, 0.1, 0.35, 0.01),
    bbox_transform=fig.transFigure # Note it uses the figure.
)

# Create a new norm, which is discrete
bounds = [0, 100, 150, 200, 250, 300]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Create the colorbar
cb = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    cax=cbaxes, # Use the inset_axes created above
    orientation = "horizontal",
    ticks=[100, 150, 200, 250]
)
# Customize the tick labels
cb.set_ticks([100, 150, 200, 250])
cb.set_ticklabels(['Negative       ', '   Medium', '', 'Positive'])

# Remove the outline of the colorbar
cb.outline.set_visible(False)

# Remove tick marks
cb.ax.xaxis.set_tick_params(size=0)

# Set legend label and move it to the top (instead of default bottom)
cb.set_label("Feature Importance Score", size=14, labelpad=-40)

# Add annotations ------------------------------------------------

# Make some room for the title and subtitle above.
fig.subplots_adjust(top=0.8)

# Define title, subtitle, and caption
title = "\nHiking Locations in Washington"
subtitle = "\n".join([
    "This Visualisation shows the cummulative length of tracks,",
    "the amount of tracks and the mean gain in elevation per location.\n",
    "If you are an experienced hiker, you might want to go",
    "to the North Cascades since there are a lot of tracks,",
    "higher elevations and total length to overcome."
])

# And finally, add them to the plot.
fig.text(0.1, 0.93, title, fontsize=25, weight="bold", ha="left", va="baseline")
fig.text(0.1, 0.9, subtitle, fontsize=14, ha="left", va="top")

# Note: you can use `fig.savefig("plot.png", dpi=300)` to save it with in hihg-quality.

st.pyplot(fig)
