import os
import pickle
import numpy as np
import pandas as pd
import yaml

from typing import Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

job_function_emb_prefix = 'job_func_emb_'


def load_job_function_embedding_cache(
        cache_path: str = 'data/embedding_cache/job_function_embedding_cache.pkl'
        ) -> Dict:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Embedding cache file not found at '{cache_path}'. Please run the training function first.")

    with open(cache_path, 'rb') as f:
        embedding_cache = pickle.load(f)

    return embedding_cache


def compute_job_function_embedding(
    job_function: str,
    embedding_cache: Dict,
) -> pd.DataFrame:
    # Get the embedding dimension from the first item in the cache
    embedding_dim = len(next(iter(embedding_cache.values())))

    # Look up the function in the cache, return zero vector if not found
    embedding_vector = embedding_cache.get(job_function, np.zeros(embedding_dim))
    return embedding_vector


def create_function_embedding_cache(        
        df_input: pd.DataFrame,
        function_column: str = 'job_function',
        encoder_model_name: str = 'all-MiniLM-L6-v2',
        output_cache_path: str = 'data/embedding_cache/job_function_embedding_cache.pkl'
    ) -> pd.DataFrame:
    """
    Creates embeddings for the job_function feature, saves the learned
    embedding mapping to a file.
    """
    df = df_input.copy()

    # Get unique job functions from the training data
    unique_functions = df[function_column].dropna().unique()
    print(f"Found {len(unique_functions)} unique job functions to embed.")

    # Embed the categories
    model = SentenceTransformer(encoder_model_name)
    function_embeddings_array = model.encode(unique_functions, show_progress_bar=True)

    # Create the mapping cache
    embedding_cache = {func: emb for func, emb in zip(unique_functions, function_embeddings_array)}
    print("Job function embeddings created.")

    # Export
    with open(output_cache_path, 'wb') as f:
        pickle.dump(embedding_cache, f)
    print(f"Embedding cache saved to '{output_cache_path}'")


def compute_job_function_embedding_df(
    df_input: pd.DataFrame,
    function_column: str,
    model: SentenceTransformer,
    embedding_cache
) -> pd.DataFrame:
    df = df_input.copy()

    tqdm.pandas()
    
    print("Applying embeddings to the DataFrame...")
    embedding_series = df[function_column].progress_apply(lambda func: embedding_cache.get(func, np.zeros(model.get_sentence_embedding_dimension())))

    embedding_df = pd.DataFrame(embedding_series.to_list(), index=df.index)
    embedding_df.columns = [f'{job_function_emb_prefix}{i}' for i in range(embedding_df.shape[1])]

    return df.join(embedding_df)
