import yaml
from embeddings.job_function import load_job_function_embedding_cache
from embeddings.skills import load_skill_cache

from model.save import load_model
from llm.ollama_setup import get_client
from predictions.inference import predict_salary
from predictions.features import categorical_features, all_features


final_lower_model = load_model('data/models/lower_catboost_2025-11-03_19:50.cbm')
final_upper_model = load_model('data/models/upper_catboost_2025-11-03_19:51.cbm')

print("models loaded")

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

job_function_cache = load_job_function_embedding_cache(params['embedding_paths']['job_function_cache'])
skill_cache = load_skill_cache(params['embedding_paths']['skill_cache'])

new_job = {
    "title": "Senior Data Scientist",
    "company_name": "Tech Inc",
    "location": "New York, NY",
    "description": "Looking for a Data Scientist with Python and SQL."
}

print(f"Predicting salary for: {new_job['title']}")

# Predict the lower bound
lower_salary = predict_salary(
    **new_job,
    model=final_lower_model,
    client=get_client(),
    decoder_model_name='phi3:mini', 
    all_features=all_features,
    categorical_features=categorical_features,
    job_function_cache=job_function_cache,
    skill_cache=skill_cache
)

# Predict the upper bound
upper_salary = predict_salary(
    **new_job,
    model=final_upper_model,
    client=get_client(),
    decoder_model_name='phi3:mini',
    all_features=all_features,
    categorical_features=categorical_features,
    job_function_cache=job_function_cache,
    skill_cache=skill_cache
)

print("\n--- PREDICTION RESULT ---")
print(f"Predicted Salary Range: ${lower_salary:,.0f} - ${upper_salary:,.0f}")
