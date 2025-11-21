import argparse
import pandas as pd
import yaml

from embeddings.skills import create_skill_embedding_cache


if __name__ == '__main__':
    df = pd.read_csv('data/datasets/postings_processed.csv', usecols=['cleaned_skills'])

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--skill-cache-output', type=str, required=True)
    args = parser.parse_args()

    create_skill_embedding_cache(
        df_input = df,
        skill_column = 'cleaned_skills',
        encoder_model_name = params['models']['encoder_model_name'],
        output_cache_path = args.skill_cache_output
    )
