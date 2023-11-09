import matplotlib.pyplot as plt
import numpy as np
from model import predict_loan_approval, explain_prediction, clf

# Assuming you already have the Lime explanation_list
explanation_list

# Sort the features by their weights for better visualization
explanation_list.sort(key=lambda x: x[1], reverse=True)

# Values for the x axis
ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(explanation_list), endpoint=False)

# Feature weights
WEIGHTS = [weight for _, weight in explanation_list]

# Feature names
FEATURES = [feature for feature, _ in explanation_list]

# Colors
good = ["#430161", "#F2984D", "#ee7A1A", "#ED6C01"]
bad = ["#ED6C01", "#ee7A1A", "#F2984D", "#430161"]
COLORS = good if prediction[0] == 1 else bad

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
cb.set_label("Amount of tracks", size=12, labelpad=-40)

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
