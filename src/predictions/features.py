from embeddings.utils import get_embedding_dimension
from embeddings.skills import mean_skill_emb_prefix, max_skill_emb_prefix
from embeddings.job_function import job_function_emb_prefix
from llm.model import encoder_model_name


categorical_features = [
    'categorized_education_level',
    'seniority',
    'job_function',
    'cleaned_location',
    'company_name',
    'seniority_job_function',
    'location_job_function',
    'seniority_function_location',
    'company_experience',
    'job_function_experience',
    'seniority_function_experience'
    ]

numerical_features = ['experience_years_required']

mean_skill_embedding_features = [f"{mean_skill_emb_prefix}{i}" for i in range(get_embedding_dimension(encoder_model_name))]
max_skill_embedding_features = [f"{max_skill_emb_prefix}{i}" for i in range(get_embedding_dimension(encoder_model_name))]
job_function_embedding_features = [f"{job_function_emb_prefix}{i}" for i in range(get_embedding_dimension(encoder_model_name))]

all_features = sorted(categorical_features + numerical_features + mean_skill_embedding_features + max_skill_embedding_features + job_function_embedding_features)

target_column = 'target_salary'

excluded_features = [
    'old_experience_years_required',
    'experience_years_required_old',
    'cleaned_education_level',
    'cleaned_skills',
    'title',
    'description',
    'location',
    'skills', 
    'education_level', 
    'normalized_salary',
    'soft_skills', 
    'domain_skills', 
    'technical_skills'
    ]
