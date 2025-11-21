import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import yaml

from embeddings.job_function import compute_job_function_embedding_df, load_job_function_embedding_cache
from embeddings.skills import compute_skills_embeddings_df, load_skill_cache
from feature_cleaning.education_level import clean_and_categorize_education
from feature_cleaning.location import clean_and_standardize_location


if __name__ == '__main__':
    processed = pd.read_csv('data/datasets/postings_processed.csv', index_col=0)

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    # clean basic features
    processed['cleaned_location'] = processed.location.apply(clean_and_standardize_location)
    processed['cleaned_education_level'] = processed['education_level'].apply(clean_and_categorize_education)

    # build interaction features
    processed['seniority_job_function'] = processed['seniority'] + '_' + processed['job_function']
    processed['location_job_function'] = processed['cleaned_location'] + '_' + processed['job_function']
    processed['company_experience'] = processed.apply(lambda row: row['company_name'] + '_' + str(row['experience_years_required']), axis=1)
    processed['job_function_experience'] = processed.apply(lambda row: row['job_function'] + '_' + str(row['experience_years_required']), axis=1)
    processed['seniority_function_location'] = processed['seniority'] + '_' + processed['job_function'] + '_' + processed['cleaned_location']
    processed['seniority_function_experience'] = processed.apply(lambda row: row['seniority'] + '_' + row['job_function'] + '_' + str(row['experience_years_required']), axis=1)

    # build embeddings from cache
    model = SentenceTransformer(params['models']['encoder_model_name'])
    
    job_function_embedding_cache = load_job_function_embedding_cache(params['embedding_paths']['job_function_cache'])
    skill_embedding_cache = load_skill_cache(params['embedding_paths']['skill_cache'])
    processed = compute_job_function_embedding_df(processed, 'job_function', model, job_function_embedding_cache)
    processed = compute_skills_embeddings_df(processed, 'skills', model, skill_embedding_cache)

    print(processed.info())
    processed.to_csv(args.output_path, index=True)
