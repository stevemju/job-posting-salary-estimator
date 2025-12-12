---
title: Job Posting Salary Estimator
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# 💼 Job Posting Salary Estimator

This data science project predicts a realistic salary range for a job posting based on its title, description, company, and location.

The application uses a ML model (CatBoost) for prediction and a local LLM (using Ollama) for real-time feature extraction from unstructured text. The project is packaged as an interactive web application using Streamlit, managed with Poetry and DVC, and deployed on Hugging Face Spaces.


**➡️ [View the Demo Here](https://huggingface.co/spaces/stevemju/job-salary-estimator) ⬅️**


<img width="1415" height="639" alt="Capture d’écran 2025-11-03 à 21 27 28" src="https://github.com/user-attachments/assets/8b8b4b2a-d277-4ae0-be27-c31b7a260ffb" />


## 🛠️ Tech Stack

* **Package Management:** `Poetry`

* **Version Control:** `Git` & `Git LFS`

* **Core Stack:** `Python`, `Pandas`, `NumPy`

* **Data Ops:** `DVC` (Data Version Control) for pipeline and artifact management.

* **Machine Learning:** `CatBoost` (using Quantile Regression to estimate percentiles), `scikit-learn`

* **Hyperparameter Tuning:** `Optuna`

* **LLM (Local):** `Ollama` (running `phi3:mini` for inference but `llama3:8b` for feature extraction during training)

* **LLM Interfacing:** `instructor`, `pydantic`

* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)

* **Web App:** `Streamlit`

* **Deployment:** `Hugging Face Spaces` (using a Docker setup)
  

## ✨ Features

* **Salary Range Prediction:** Instead of a single prediction, it provides a realistic salary range by predicting the 35th and 65th percentiles.

* **LLM-Powered Feature Extraction:** Uses a local LLM (e.g., `phi3:mini`) to read unstructured job descriptions and extract key features:

  * Technical Skills (e.g., "Python", "AWS")

  * Domain Skills (e.g., "Market Access", "Pricing")

  * Soft Skills (e.g., "Leadership", "Communication")

  * Required Experience (e.g., `5` years)

  * Education Level (e.g., `Bachelor's`)

* **Advanced Feature Engineering:** Creates over 1,000 features, including:

  * Semantic embeddings for skills (mean & max pooling).

  * Semantic embeddings for the categorized job function.

  * Interaction features (e.g. location + title, title + years of experience, job function + title etc...).

* **Interactive UI:** A simple web interface built with Streamlit to make the model demoable.

* **Reproducible Pipeline:** Fully automated data processing and training pipeline managed by DVC.


## 🚀 How It Works

The project uses a two-model (quantile regression) approach to generate its predictions:

1. **Feature Extraction:** When a user inputs a job description, the app sends it to a locally running Ollama model. The LLM extracts a structured JSON of skills, experience, and education.

2. **Feature Engineering:** The app's pipeline combines these extracted features with other engineered features (like `job_function`, `seniority`, and `location_category`). It then uses saved embedding caches to create over 1,000 semantic and interaction features.

3. **Dual Prediction** (specific bound values are subject to change):

   * **Lower Bound Model:** A CatBoost model trained with a `Quantile:alpha=0.35` loss function predicts the 35th percentile of the salary.

   * **Upper Bound Model:** A separate CatBoost model trained with a `Quantile:alpha=0.65` loss function predicts the 65th percentile.

4. **Display:** The two resulting predictions are presented to the user as a realistic salary range.

## 🚢 Deployment

This application is configured for deployment on Hugging Face Spaces using a custom Docker setup. The deployment is defined by the following files:

* `Dockerfile`: Defines the build environment.

* `docker_startup.sh`: The startup script that launches the Ollama server and the Streamlit app.

* `requirements.txt`: A list of all Python packages.

* `packages.txt`: A list of all system-level (Linux) packages.

The project is automatically built and deployed on every push to the main branch on Hugging Face.
