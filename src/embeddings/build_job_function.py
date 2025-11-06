import argparse
import pandas as pd
import yaml

from embeddings.job_function import create_function_embedding_cache


if __name__ == '__main__':
    df = pd.read_csv('data/datasets/postings_final.csv', usecols=['job_function'])

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-function-cache-output', type=str, required=True)
    args = parser.parse_args()

    create_function_embedding_cache(
        df_input = df,
        function_column = 'job_function',
        encoder_model_name = params['models']['encoder_model_name'],
        output_cache_path = args.job_function_cache_output
    )
