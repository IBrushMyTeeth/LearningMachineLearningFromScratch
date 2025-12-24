# Probabilistic Learning

## Overview  
This folder explores **probabilistic approaches to machine learning**, with a focus on explicitly modeling uncertainty using probability distributions rather than relying solely on point estimates.

The projects contrast **Maximum Likelihood Estimation (MLE)** with **Bayesian inference** by implementing models with closed-form solutions. In particular, Bayesian polynomial regression is used to study how priors, likelihoods, and data interact to produce posterior distributions and uncertainty-aware predictions.

The emphasis is on analytical understanding, transparency, and visualization rather than approximate inference or large-scale probabilistic models.

---

## Key Learnings  
- Implemented **Maximum Likelihood Estimation (MLE)** as a point-estimate baseline.
- Built **Bayesian polynomial regression** with a Gaussian prior and closed-form posterior.
- Computed posterior **mean and covariance** analytically.
- Separated **model uncertainty** (epistemic) from observation noise (aleatoric).
- Visualized **predictive uncertainty** through posterior predictive variance.
- Compared frequentist and Bayesian perspectives on confidence and generalization.
- Observed how MLE yields constant predictive uncertainty, while Bayesian regression produces input-dependent uncertainty driven by data density.

---

## Topics Covered  
- Likelihood-based modeling  
- Maximum Likelihood Estimation  
- Bayesian linear regression  
- Prior and posterior distributions  
- Predictive mean and variance  
- Uncertainty visualization  

---

## Motivation  
Many machine learning models produce confident point predictions even when data is limited or noisy. Probabilistic learning provides a principled way to:

- Quantify uncertainty in predictions  
- Incorporate prior knowledge into models  
- Understand how uncertainty changes with data  

This folder serves as a conceptual foundation for more advanced probabilistic models and approximate inference methods explored in future work.

---

## Next Steps  
- Sample from the posterior and posterior predictive distribution  
- Implement **MAP estimation** and relate it to regularization  
- Extend Bayesian regression to higher-dimensional feature spaces  
- Explore approximate inference methods (e.g. variational inference)  
- Connect Bayesian regression to Gaussian Processes  
