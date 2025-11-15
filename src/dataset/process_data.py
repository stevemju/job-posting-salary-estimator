import argparse
import pandas as pd
import yaml

from llm.batch_processor import process_in_batches
from llm.ollama_setup import get_client, is_ollama_server_running


if __name__ == '__main__':
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-file-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv('data/datasets/postings_cleaned.csv', index_col=0)

    client = get_client()
    is_ollama_server_running(client)

    processed = process_in_batches(
        df=df, 
        output_filepath=args.checkpoint_file_path,
        client=client,
        decoder_model_name=params['models']['decoder_model_name'],
        batch_size=params['llm_processing']['batch_size'],
        max_workers=params['llm_processing']['max_workers']
        )
    
    print(processed.info())
    processed.to_csv('data/datasets/postings_processed.csv', index=True)
    