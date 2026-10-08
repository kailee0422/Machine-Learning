# Experiment Report

## Introduction

This report examines two classification techniques, **K-Nearest Neighbors (KNN)** and
**Nearest Centroid Classification**, using scikit-learn. The focus is on understanding
the decision boundaries and the impact of hyperparameters — weights and `k` for KNN, and
`shrink_threshold` for Nearest Centroid — on classification accuracy, using the Iris
dataset.

## Materials and Methods

- **Dataset**: Iris (150 samples, 3 species); only sepal length and sepal width are used
  for visualization.
- **KNN**: `KNeighborsClassifier` wrapped in a `Pipeline` with `StandardScaler`; the
  `k` value is tuned by experimentation, with decision boundaries plotted for both
  `"uniform"` and `"distance"` weighting schemes.
- **Nearest Centroid**: `NearestCentroid` classifier with the `shrink_threshold`
  parameter swept to find the optimal value, comparing decision boundaries with and
  without shrinkage.

## Results

**KNN**: uniform weighting consistently outperforms distance weighting across all
tested `k` values — best accuracy **0.7632** at `k=3` (uniform) vs. **0.6579** at `k=9`
(distance). Uniform weights fluctuate more with `k` but reach a higher peak at small
`k`; distance weights are more stable but cap out lower.

**Nearest Centroid**: accuracy peaks around **0.82** for `shrink_threshold` between 0.2
and 0.6, starting at ~0.814 with no shrinkage and dropping to its lowest as the
threshold approaches 1.0. The decision boundary at `shrink_threshold=0.20` gives the
best performance; `0.86` gives the worst.

## Discussion

- **KNN**: with uniform weights, all neighbors contribute equally, which helps when `k`
  is small but lets noise from distant neighbors in as `k` grows. Distance weighting
  reduces that noise by downweighting far neighbors, trading accuracy for stability.
- **Nearest Centroid**: moderate shrinkage lets the model ignore noisy/less-important
  centroids and generalize better, but excessive shrinkage removes decision-relevant
  information and hurts accuracy — hence the 0.2–0.6 sweet spot.

## Conclusion

- Uniform weights generally outperform distance weights in KNN; the `k` value should be
  tuned to the dataset to find the optimum.
- Shrinkage improves Nearest Centroid accuracy only within a moderate range
  (0.2–0.6 here) — beyond that it degrades performance.

## Code

You can run on colab or local

<a target="_blank" href="https://colab.research.google.com/github/kailee0422/Machine-Learning/blob/main/HW1/ML_HW1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
