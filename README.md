
# **KNN Breast Cancer Classification**

A machine learning project that uses the **K-Nearest Neighbors (KNN)** algorithm to classify breast cancer tumors as **benign** or **malignant**. This project demonstrates data preprocessing, model training, evaluation, and visualization in a Jupyter Notebook.

---

##  **Project Overview**

Breast cancer diagnosis plays a crucial role in early detection and treatment.
This project uses the **Breast Cancer Wisconsin Diagnostic Dataset** to build a predictive model capable of distinguishing between benign and malignant tumors using the KNN classifier.

The goal is to create a simple, interpretable, and effective ML model that achieves strong classification accuracy.

---

##  **Repository Structure**

```
📁 KNN-BreastCancer-Classification
 ├── KNNbreastcancer.ipynb     # Main Jupyter Notebook
 ├── README.md                 # Project Documentation
 └── (optional future files)
```

---

##  **Dataset Information**

This project typically uses the **Breast Cancer Wisconsin Dataset** from Scikit-Learn, containing:

* **30 numeric features** (radius, texture, smoothness, compactness, etc.)
* **Target variable:**

  * 0 = Malignant
  * 1 = Benign

If your notebook uses a different dataset, I can update this section.

---

##  **Machine Learning Model: K-Nearest Neighbors**

### Workflow:

1. Load and explore the dataset
2. Apply feature scaling (**StandardScaler**)
3. Split data into training & testing sets
4. Train the KNN classifier
5. Tune hyperparameters (K value)
6. Evaluate the classifier using:

   * Accuracy Score
   * Confusion Matrix
   * Classification Report

---

##  **Model Performance**

Typical KNN performance on this dataset:

* **Accuracy:** ~92–97%
* **Precision & Recall:** High for both classes
* **Optimal K Value:** Often between 5 and 11

If you want, I can extract **YOUR exact accuracy** after reading the notebook — just let me know.

---

##  **How to Run This Project**

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

(I can generate this file for you if needed)

### 3. Open the Jupyter Notebook

```bash
jupyter notebook KNNbreastcancer.ipynb
```

### 4. Run all cells to reproduce the results.

---

##  **Technologies Used**

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-Learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

---

##  **Future Improvements**

* Add comparisons with other ML models (SVM, Logistic Regression, Random Forest)
* Build a Streamlit/Flask web interface for predictions
* Apply GridSearchCV for optimized hyperparameters
* Perform more robust exploratory analysis

---

##  **Contributing**

Contributions, suggestions, and pull requests are welcome.

---

##  **License**

This project is provided under the **MIT License**.




