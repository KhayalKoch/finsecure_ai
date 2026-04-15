# 🛡️FinSecure AI: Quantitative Risk & Adversarial Resilience
**AI-driven Fraud Detection with Financial Loss Quantification**
- FinSecure AI is a high-performance framework designed to bridge the gap between Machine Learning metrics and Financial Risk Management in the banking sector.

## Week 1: High-Fidelity Foundation
In the first phase, we successfully simulated a realistic fintech environment and established a robust baseline detection system.

### 📊 Baseline Performance
- Our model, built on **XGBoost**, has been optimized to handle the extreme class imbalance typical of financial fraud.
* **Accuracy:** 96.99%
* **CV F1-Score (Weighted):** 0.9718
* **Fraud Detection F1-Score:** 0.98
* **Customer Friction:** Extremely low (Only 8 false positives out of ~35k test samples).

### 📈 Performance Visualization
- Below is the official output from our baseline model, showcasing the precision-recall balance and the confusion matrix.

![Model Performance Results](./assets/model_results.png)

## Technical Implementation
* **Realistic Simulation:** Uses Log-normal distribution for transaction amounts and pattern-based fraud injection (Salami attacks, Velocity attacks).
* **Advanced Preprocessing:** Robust scaling to handle financial outliers and categorical encoding for transaction types.
* **Model Engine:** XGBoost with `scale_pos_weight` for handling imbalanced datasets.

## 📁 Repository Structure
* `data/`: Data generation and preprocessing scripts.
* `model/`: ML model definitions and training logic.
* `assets/`: Visual evidence and performance reports.
* `requirements.txt`: Project dependencies.

## 🗓️ What's Next?
* **Week 2:** Adversarial Attacks - Simulating sophisticated fraud patterns to bypass the baseline model.
* **Week 3:** Financial Loss Quantification - Calculating the monetary impact of undetected fraud.