import streamlit as st
import numpy as np
import sys
import yaml

from embeddings.job_function import load_job_function_embedding_cache
from embeddings.skills import load_skill_cache
from llm.ollama_setup import get_client
from model.save import load_model
from predictions.inference import predict_salary
from predictions.features import categorical_features, all_features


sys.path.append('src')

@st.cache_resource
def load_artifacts():
    print("Loading models and artifacts.")

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    lower_model = load_model("data/models/lower_catboost_2025-11-03_21:04.cbm")
    upper_model = load_model("data/models/upper_catboost_2025-11-03_21:36.cbm")

    # load embedding cache
    job_function_cache = load_job_function_embedding_cache(params['embedding_paths']['job_function_cache'])
    skill_cache = load_skill_cache(params['embedding_paths']['skill_cache'])

    return lower_model, upper_model, job_function_cache, skill_cache, params['models']['decoder_model_name']

lower_model, upper_model, job_function_cache, skill_cache, decoder_model_name = load_artifacts()

st.set_page_config(layout="wide")
st.title("💼 US Job Posting Salary Estimator")
st.markdown("Enter the job details below to predict a salary range.")

# --- User Inputs ---
with st.form(key="job_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Job Title")
        company_name = st.text_input("Company Name")
        location = st.text_input("Location")

    with col2:
        description = st.text_area("Job Description", height=210)
    
    submit_button = st.form_submit_button(label="Estimate Salary Range")

# --- Prediction Logic ---
if submit_button:
    if not title or not description:
        st.error("Please fill in at least the 'Job Title' and 'Job Description' fields.")
    else:
        with st.spinner("Analyzing job posting... (This may take a moment as the LLM is running)"):
            try:

                client = get_client()

                lower_salary = predict_salary(
                    title, 
                    company_name, 
                    location, 
                    description, 
                    lower_model, 
                    client, 
                    decoder_model_name, 
                    all_features, 
                    categorical_features,
                    job_function_cache,
                    skill_cache
                )
                
                upper_salary = predict_salary(
                    title, 
                    company_name, 
                    location, 
                    description, 
                    upper_model, 
                    client, 
                    decoder_model_name, 
                    all_features, 
                    categorical_features,
                    job_function_cache,
                    skill_cache
                )

                # --- Display Results ---
                st.header("Predicted Salary Range")
                lower_formatted = f"${int(np.round(lower_salary, -2)):,}"
                upper_formatted = f"${int(np.round(upper_salary, -2)):,}"
                st.metric(label="Estimated Range", value=f"{lower_formatted} - {upper_formatted}")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.error("Please ensure the Ollama server in this Space is running and the model is available.")
