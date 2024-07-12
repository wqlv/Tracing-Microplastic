import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

data = pd.read_csv("  ")

data.columns = data.columns.astype(str)

y = data.iloc[:, 0]
X = data.iloc[:, 1:]

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

class_means = []
for label in pd.unique(y):
    class_means.append(np.mean(X_lda[y == label], axis=0))

class_ranges = []
for label in pd.unique(y):
    class_ranges.append((np.min(X_lda[y == label], axis=0), np.max(X_lda[y == label], axis=0)))

plt.figure(figsize=(10, 8))
for label, mean, (min_vals, max_vals) in zip(pd.unique(y), class_means, class_ranges):
    width = max_vals[0] - min_vals[0]
    height = max_vals[1] - min_vals[1]
    ellipse = Ellipse(xy=mean, width=width, height=height, alpha=0.5)
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=label)
    plt.gca().add_patch(ellipse)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.title('LDA Projection with Range Circles')
plt.savefig("  ")  # Save the plot as an image

# Create a DataFrame to store LDA coordinates
lda_df = pd.DataFrame(data=X_lda, columns=['LD1', 'LD2'])

# Add class labels to the DataFrame
lda_df['Class'] = y.values

# Calculate explained variance ratio
explained_variance_ratio = lda.explained_variance_ratio_

# Calculate LD1 and LD2 contribution
ld1_contribution = explained_variance_ratio[0] * 100
ld2_contribution = explained_variance_ratio[1] * 100

# Save LDA coordinates to Excel
lda_df.to_excel("  ", index=False)

# Print LD1 and LD2 contribution
print("LD1 Contribution: {:.2f}%".format(ld1_contribution))
print("LD2 Contribution: {:.2f}%".format(ld2_contribution))
