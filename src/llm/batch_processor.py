import pandas as pd
import os

from tqdm import tqdm
from openai import OpenAI
from llm.job_details import get_job_details
from llm.ollama_setup import is_ollama_server_running
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_in_batches(
    df: pd.DataFrame,
    output_filepath: str,
    client: OpenAI,
    decoder_model_name: str,
    batch_size: int = 200,
    max_workers: int = 16
):
    if not is_ollama_server_running(client):
        return None

    print(f"Starting job. Results will be saved to '{output_filepath}'.")
    print(f"Total rows: {len(df)}, Batch size: {batch_size}")

    processed_df = pd.DataFrame()

    # --- 1. RESUME LOGIC ---
    if os.path.exists(output_filepath):
        print(f"Checkpoint file found at '{output_filepath}'. Loading previous results.")
        processed_df = pd.read_parquet(output_filepath)
        # Get the indices of rows that are already processed
        processed_indices = set(processed_df.index)
        print(f"Found {len(processed_indices)} previously processed rows.")
    else:
        processed_indices = set()

    # Filter the main DataFrame to get only the rows that need processing
    df_to_process = df[~df.index.isin(processed_indices)]

    if df_to_process.empty:
        print("All rows have already been processed. Nothing to do.")
        return processed_df

    print(f"Starting processing for {len(df_to_process)} remaining rows...")

    # --- 2. BATCH PROCESSING ---
    for i in tqdm(range(0, len(df_to_process), batch_size), desc="Overall Progress"):
        batch = df_to_process.iloc[i:i + batch_size]
        batch_results = [None] * len(batch)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks with original index and description
            futures = [
                executor.submit(get_job_details, row['description'], index, client, decoder_model_name)
                for index, row in batch.iterrows()
            ]

            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size + 1}", leave=False):
                original_index, result_data = future.result()
                result_data['original_index'] = original_index
                batch_results[batch.index.get_loc(original_index)] = result_data

        # --- 3. SAVE CHECKPOINT ---
        batch_results_df = pd.DataFrame(batch_results)
        batch_results_df = batch_results_df.set_index('original_index')

        processed_df = pd.concat([processed_df, batch_results_df])

        processed_df.to_parquet(output_filepath)

    print("Processing complete.")

    final_df = df.join(processed_df)
    return final_df
