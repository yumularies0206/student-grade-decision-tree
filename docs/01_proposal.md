# Proposal / Initial Write-up

**Proposed Title:** Predicting Student At-Risk Status Using a Decision Tree Classifier on Behavioural Study Data

## Short Description of the Topic
**a. Model to be studied.** Decision Tree Classifier (CART, Gini impurity) for binary classification: Pass (A–C) vs. At-risk/Fail (D–F).

**b. Real-world context / motivation.** Teachers in large classes cannot individually track every student. An interpretable
early-warning rule based on easily collected habits (weekly self-study hours, attendance, participation) lets them identify
struggling students before final grades are released and target tutoring or consultation (supports SDG 4 – Quality Education).

**c. Preliminary literature, datasets, references.**
- Kaggle – Student Performance Dataset (1,000,000 records; study hours, attendance, participation, total score, grade)
- Breiman, Friedman, Olshen & Stone (1984). *Classification and Regression Trees.*
- Russell & Norvig (2022). *Artificial Intelligence: A Modern Approach* (4th ed.) – decision-tree learning.
- Géron (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.).
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.

## Planned Method
Stratified 80/20 split → grid-search cross-validation → evaluate with precision, recall, F1, ROC-AUC, PR-AUC →
compare with logistic regression and majority-class baselines → interpret rules and feature importance → ethical reflection.
