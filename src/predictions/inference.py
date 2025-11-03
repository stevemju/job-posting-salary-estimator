import numpy as np
import pandas as pd

from typing import Dict, List
from catboost import CatBoostRegressor
from openai import OpenAI

from embeddings.job_function import compute_job_function_embedding, load_job_function_embedding_cache
from embeddings.skills import get_aggregated_skill_embeddings, load_skill_cache
from feature_cleaning.education_level import clean_and_categorize_education
from feature_cleaning.location import clean_and_standardize_location
from feature_cleaning.skills import clean_skill_list
from feature_extraction.job_function import extract_job_function_from_title
from feature_extraction.seniority import extract_seniority_from_title
from llm.job_details import get_job_details
from embeddings.skills import mean_skill_emb_prefix, max_skill_emb_prefix
from embeddings.job_function import job_function_emb_prefix
from model.save import load_model
from llm.ollama_setup import get_client, is_ollama_server_running
from model.train import load_initial_dataset
from src.predictions.features import categorical_features, all_features
from src.llm.model import decoder_model_name



def compute_features(
        title: str,
        company_name: str,
        location: str,
        description: str,
        client: OpenAI,
        decoder_model_name: str,
        job_function_cache: Dict,
        skill_cache: Dict,
        ) -> Dict:
    if not is_ollama_server_running(client):
        raise Exception("LLM client is not running.")

    cleaned_location = clean_and_standardize_location(location)
    seniority = extract_seniority_from_title(title)
    job_function = extract_job_function_from_title(title)

    _, job_details = get_job_details(description, 0, client, decoder_model_name)

    categorized_education_level = clean_and_categorize_education(job_details['education_level'])
    experience_years_required = job_details['experience_years_required']
    cleaned_skills = clean_skill_list(job_details['technical_skills'] + job_details['soft_skills'] + job_details['domain_skills'])

    # embeddings
    mean_skill_emb, max_skill_emb = get_aggregated_skill_embeddings(cleaned_skills, skill_cache)
    job_function_embedding = compute_job_function_embedding(job_function, job_function_cache)

    # explode embeddings
    mean_skill_emb_exploded = {f"{mean_skill_emb_prefix}{k}": v for k, v in enumerate(mean_skill_emb)}
    max_skill_emb_exploded = {f"{max_skill_emb_prefix}{k}": v for k, v in enumerate(max_skill_emb)}
    job_function_embedding_exploded = {f"{job_function_emb_prefix}{k}": v for k, v in enumerate(job_function_embedding)}

    # interaction features
    seniority_job_function = f"{seniority}_{job_function}"
    location_job_function = f"{cleaned_location}_{job_function}"
    seniority_function_location = f"{seniority}_{cleaned_location}"
    company_experience = f"{company_name}_{experience_years_required}"
    job_function_experience = f"{job_function}_{experience_years_required}"
    seniority_function_experience = f"{seniority}_{job_function}_{experience_years_required}"

    base_features =  {
        'categorized_education_level': categorized_education_level,
        'seniority': seniority,
        'job_function': job_function,
        'cleaned_location': cleaned_location,
        'company_name': company_name,
        'seniority_job_function': seniority_job_function,
        'location_job_function': location_job_function,
        'seniority_function_location': seniority_function_location,
        'company_experience': company_experience,
        'job_function_experience': job_function_experience,
        'seniority_function_experience': seniority_function_experience,
        'experience_years_required': experience_years_required,
    }

    merged_features = base_features | mean_skill_emb_exploded | max_skill_emb_exploded | job_function_embedding_exploded

    assert sorted(merged_features.keys()) == sorted(all_features)
    return dict(sorted(merged_features.items()))

def predict_salary(
    title: str, company_name: str, location: str, description: str,
    model: CatBoostRegressor,
    client: OpenAI,
    decoder_model_name: str,
    all_features: List[str],
    categorical_features: List[str],
    job_function_cache: Dict,
    skill_cache: Dict
) -> float:
    feature_dict = compute_features(title, company_name, location, description, client, decoder_model_name, job_function_cache, skill_cache)
    inference_df = pd.DataFrame([feature_dict])

    inference_df = inference_df.reindex(columns=all_features)

    for col in categorical_features:
        if col in inference_df.columns:
            inference_df[col] = inference_df[col].astype('category')

    prediction_log = model.predict(inference_df)
    prediction_dollars = np.expm1(prediction_log)

    return prediction_dollars[0]
