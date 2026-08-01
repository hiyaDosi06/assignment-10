# Heart Disease Risk Prediction System

An end-to-end Machine Learning pipeline and RESTful API deployment designed to assess patient clinical metrics and predict the risk of cardiovascular disease. The solution features data preprocessing, Random Forest classification model training, a Flask web microservice, and continuous integration configured for cloud deployment.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Features](#dataset--features)
3. [Repository Structure](#repository-structure)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Local Setup & Installation](#local-setup--installation)
6. [API Endpoints & Usage](#api-endpoints--usage)
7. [Cloud Deployment](#cloud-deployment)
8. [Conclusion & MLOps Insights](#conclusion--mlops-insights)

---

## 🏥 Project Overview
Cardiovascular diseases are one of the leading causes of global mortality. Early diagnosis using clinical features can significantly improve patient outcomes. This project builds a machine learning inference engine wrapped in a Flask REST API, exposing endpoints that allow healthcare systems to pass patient physiological parameters and receive real-time risk predictions.

---

## 📊 Dataset & Features
The model is trained on the Kaggle Heart Disease Dataset, which consists of 14 clinical attributes used to predict whether a patient has heart disease (`target = 1`) or not (`target = 0`).

### Feature Descriptions:
* **age**: Age of the patient in years
* **sex**: Gender (1 = male, 0 = female)
* **cp**: Chest pain type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic)
* **trestbps**: Resting blood pressure (in mm Hg on admission)
* **chol**: Serum cholesterol level in mg/dl
* **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
* **restecg**: Resting electrocardiographic results (0, 1, or 2)
* **thalach**: Maximum heart rate achieved during exercise
* **exang**: Exercise-induced angina (1 = yes, 0 = no)
* **oldpeak**: ST depression induced by exercise relative to rest
* **slope**: Slope of the peak exercise ST segment (0, 1, or 2)
* **ca**: Number of major vessels (0–3) colored by fluoroscopy
* **thal**: Thalassemia disorder indicator (1 = normal, 2 = fixed defect, 3 = reversible defect)
* **target**: Diagnosis result (0 = No Heart Disease, 1 = Heart Disease Detected)

---

## 📂 Repository Structure
```text
HeartDiseaseDeployment/
│
├── app.py           # REST API web application using Flask
├── train_model.py   # Machine Learning pipeline script (Data loading, training, saving)
├── model.pkl        # Serialized Random Forest machine learning model
├── heart.csv        # Source clinical dataset (Kaggle)
├── requirements.txt # Python dependencies for deployment
└── README.md        # Technical project documentation
