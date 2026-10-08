# Experiment Report

## Introduction

This study compares four tree-based ensemble classifiers — **Decision Tree**, **Random
Forest**, **Extra Trees**, and **AdaBoost** — on the Iris dataset, examining decision
boundaries across every feature pair and the effect of `max_depth` and `n_estimators`
on accuracy.

## Materials and Methods

- **Dataset**: Iris (150 samples, 3 classes: Setosa, Versicolor, Virginica).
- Each of the four classifiers was trained on all six feature pairs (sepal
  length/width, petal length/width, and their cross combinations), with decision
  boundaries plotted for each.
- `max_depth` (1–100) and `n_estimators` (1–100) were swept in turn to find the
  configuration giving the highest accuracy per feature pair, starting from a baseline
  of `max_depth=3, n_estimators=30`.

## Results

**Baseline (`max_depth=3`, `n_estimators=30`)** — all four models score identically per
feature pair, since none of the three ensemble methods needed deeper trees or more
estimators to separate this baseline split:

| Feature pair | DecisionTree | RandomForest | ExtraTrees | AdaBoost |
|---|---|---|---|---|
| [0,1] (sepal L/W) | 0.9267 | 0.9267 | 0.9267 | 0.8200 |
| [0,2] (sepal L / petal L) | **0.9933** | **0.9933** | **0.9933** | **0.9933** |
| [0,3] (sepal L / petal W) | 0.9733 | 0.9733 | 0.9733 | 0.9667 |
| [1,2] (sepal W / petal L) | 0.9867 | 0.9867 | 0.9867 | 0.9867 |
| [1,3] (sepal W / petal W) | 0.9800 | 0.9800 | 0.9800 | 0.9800 |
| [2,3] (petal L/W) | 0.9933 | 0.9933 | 0.9933 | 0.9867 |

**After tuning `max_depth`** (AdaBoost improves on [0,1] to 0.9267; other feature pairs
unchanged) → next, `n_estimators` was swept instead, holding `max_depth=3`:

| Feature pair | DecisionTree | RandomForest | ExtraTrees | AdaBoost |
|---|---|---|---|---|
| [0,1] | 0.9267 | 0.9267 | 0.9267 | 0.9200 |
| [0,2] | 0.9933 | 0.9933 | 0.9933 | 0.9933 |
| [0,3] | 0.9733 | 0.9733 | 0.9733 | 0.9733 |
| [1,2] | 0.9867 | 0.9867 | 0.9867 | 0.9867 |
| [1,3] | 0.9800 | 0.9800 | 0.9800 | 0.9800 |
| [2,3] | 0.9933 | 0.9933 | 0.9933 | 0.9933 |

`n_estimators` tuning gave no further improvement over the `max_depth`-tuned results, so
`max_depth` was swept again (now with `n_estimators` fixed at 100) — final best accuracy
per feature pair, AdaBoost-tuned:

| Feature pair | DecisionTree | RandomForest | ExtraTrees | AdaBoost |
|---|---|---|---|---|
| [0,1] | 0.9267 | 0.9267 | 0.9267 | **0.9267** |
| [0,2] | 0.9933 | 0.9933 | 0.9933 | **0.9933** |
| [0,3] | 0.9733 | 0.9733 | 0.9733 | **0.9733** |
| [1,2] | 0.9867 | 0.9867 | 0.9867 | **0.9867** |
| [1,3] | 0.9800 | 0.9800 | 0.9800 | **0.9800** |
| [2,3] | 0.9933 | 0.9933 | 0.9933 | **0.9933** |

## Discussion

When feature pairs are highly correlated (e.g. sepal length vs. petal length),
adjusting `n_estimators` or `max_depth` makes little difference — the classes are
already well separated. For less-correlated pairs (e.g. sepal width vs. sepal length),
moderate tuning helps a bit before plateauing, suggesting the models are near their
practical accuracy ceiling for this dataset.

## Conclusion

Random Forest and Extra Trees gave the most stable, accurate results overall,
especially on the sepal-length/petal-length feature pair, which reached near-perfect
classification. AdaBoost was more sensitive to under-tuned `max_depth`/`n_estimators`
but matched the other three once properly tuned. Future work could increase the sample
size and explore finer-grained parameter search.

## Code

You can run on colab or local

<a target="_blank" href="https://colab.research.google.com/github/kailee0422/Machine-Learning/blob/main/HW2/ML_HW2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
