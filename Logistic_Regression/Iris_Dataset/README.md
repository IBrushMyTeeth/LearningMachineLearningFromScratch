# Multi-class Logistic Regression on the Iris Dataset

## Project Overview

This project implements a **multi-class logistic regression** model to predict the species of flowers in the Iris dataset.

The dataset used is the **`load_iris`** dataset from `scikit-learn`, which contains 150 samples of iris flowers categorized into three species: *setosa*, *versicolor*, and *virginica*.

### Main Objectives

- Understand and implement multi-class logistic regression.
- Utilize PyTorch's modules and tools for model building.
- Calculate and interpret a **confusion matrix**.
- Generate and analyze the **ROC (Receiver Operating Characteristic) curve** for multi-class classification.

## Key Learnings

- **Multi-class logistic regression** involves creating separate linear models for each class, with the outputs passed through the **softmax function** (generalization of the sigmoid function for multi-class problems).  
- The **weight matrix** has shape `(n_features, n_classes)` and is optimized to maximize the probability of the true class.  
- The **confusion matrix** provides a summary of prediction results by comparing predicted labels against true labels. It helps identify which classes are often misclassified.  
- For the **ROC curve**, multi-class classification must be converted into a **one-vs-rest (OvR)** binary framework for each class. The ROC curve visualizes the trade-off between the **true positive rate (sensitivity)** and **false positive rate (1 - specificity)**, allowing assessment of the model's discriminative ability.  
- PyTorch facilitates building and training the logistic regression model efficiently while providing utilities for evaluation metrics.



