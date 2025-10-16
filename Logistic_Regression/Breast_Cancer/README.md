# Binary Classification on the Breast Cancer Dataset

## Project Overview

This project focuses on building and understanding a **binary classification model** using **logistic regression** to predict whether a patient has breast cancer.  
The dataset used is the **Breast Cancer Wisconsin Dataset** from `scikit-learn`.

The main objectives were:
- To **implement logistic regression** using **PyTorch**, gaining a deeper understanding of classification while also learning PyTorch fundamentals.  

- To **move from manual gradient descent implementations** to using **`nn.Module`**, **`torch.optim`**, and **automatic differentiation** for training models.  

- To **analyze and improve model performance** through optimization, regularization, and careful debugging.

---

## Key Learnings

### 🔹 Logistic Regression Concepts
- Logistic regression, though similar to linear regression, is used for **classification**, not prediction of continuous values. 

- The **sigmoid function** converts linear outputs into probabilities between 0 and 1, allowing interpretation as classification likelihoods.

### 🔹 Training and Optimization
- Implemented **manual cross-entropy loss** to understand its mathematical foundation before using PyTorch’s built-in functions.

- Transitioned from **manual gradient descent** to PyTorch’s **`torch.optim`** for efficient and modular training.  

- Learned how to build models using **`nn.Module`**, improving code readability and scalability.  

- The model initially performed poorly, but **tuning the learning rate**, adding **regularization**, and **tracking the loss function** over iterations significantly improved accuracy.  

- Learned the importance of **tensor shape alignment**, as PyTorch can silently broadcast mismatched dimensions.

### 🔹 Regularization Techniques
- Implemented both **L1 (Lasso)** and **L2 (Ridge)** regularization. 

- Experimented with a hybrid approach combining both to control overfitting while maintaining model flexibility.

---

## Tools & Technologies
- **Python 3**
- **PyTorch** – for model implementation, auto-differentiation, and optimization  
- **scikit-learn** – for dataset loading and preprocessing  

---

## Challenges Faced
- Transitioning from manual training loops to PyTorch’s **module and optimizer system** required rethinking the workflow, getting familiar with the library pytorch and what it offers.

- PyTorch’s **silent broadcasting** and tensor shape mismatches caused confusing bugs — fixed by verifying tensor dimensions at every step. 

- Balancing the **learning rate** and **regularization strength** was critical for achieving stable convergence.  

- Understanding the **mathematical link** between cross-entropy loss and maximum likelihood estimation helped clarify why logistic regression works so effectively for classification.

---

