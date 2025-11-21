import os
import pickle
import numpy as np
import pandas as pd
import yaml

from typing import Dict, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from feature_cleaning.utils import parse_stringified_list

mean_skill_emb_prefix = 'mean_skill_emb_'
max_skill_emb_prefix = 'max_skill_emb_'


def load_skill_cache(cache_path: str) -> Dict:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Embedding cache file not found at '{cache_path}'. Please run the training function first.")

    with open(cache_path, 'rb') as f:
        embedding_cache = pickle.load(f)
    
    return embedding_cache


def compute_aggregated_skill_embeddings(
        cleaned_skill_list: List,
        embedding_cache: Dict
    ):
    """
    Looks up embeddings once and computes both mean and max.
    Returns a tuple: (mean_vector, max_vector).
    """
    embedding_dim = len(next(iter(embedding_cache.values())))

    if not isinstance(cleaned_skill_list, list):
        return (np.zeros(embedding_dim), np.zeros(embedding_dim))

    embeddings = [embedding_cache[skill] for skill in cleaned_skill_list if skill in embedding_cache]
    if not embeddings:
        return (np.zeros(embedding_dim), np.zeros(embedding_dim))

    return (np.mean(embeddings, axis=0), np.max(embeddings, axis=0))


def create_skill_embedding_cache( 
    df_input: pd.DataFrame,
    encoder_model_name: str,
    output_cache_path: str,
    skill_column: str = 'cleaned_skills',
  ) -> pd.DataFrame:
    df = df_input.copy()
    df[skill_column] = df[skill_column].apply(parse_stringified_list)

    unique_skills = df[skill_column].explode().dropna().unique()
    all_skills = [skill for skill in unique_skills if isinstance(skill, str)]

    print(f"Found {len(all_skills)} unique skills to embed.")
    model = SentenceTransformer(encoder_model_name)

    unique_skill_embeddings = model.encode(all_skills, show_progress_bar=True)
    embedding_cache = {skill: emb for skill, emb in zip(all_skills, unique_skill_embeddings)}
    print("Embeddings cached.")

    with open(output_cache_path, 'wb') as f:
        pickle.dump(embedding_cache, f)
    print(f"Embedding cache saved to '{output_cache_path}'")


def compute_skills_embeddings_df(
    df_input: pd.DataFrame,
    skill_column: str,
    model: SentenceTransformer,
    embedding_cache
  ) -> pd.DataFrame:
    df = df_input.copy()
    
    tqdm.pandas()

    embedding_dim = model.get_sentence_embedding_dimension()

    def get_aggregated_vectors(skill_list: List):
        if not isinstance(skill_list, list):
            return (np.zeros(embedding_dim), np.zeros(embedding_dim))

        embeddings = [embedding_cache[skill] for skill in skill_list if skill in embedding_cache]
        if not embeddings:
            return (np.zeros(embedding_dim), np.zeros(embedding_dim))

        return (np.mean(embeddings, axis=0), np.max(embeddings, axis=0))

    print("Aggregating mean and max embeddings for each job posting...")

    results_series = df[skill_column].progress_apply(get_aggregated_vectors)

    mean_aggregated_vectors = [result[0] for result in results_series]
    max_aggregated_vectors = [result[1] for result in results_series]

    mean_df = pd.DataFrame(mean_aggregated_vectors, index=df.index).add_prefix(mean_skill_emb_prefix)
    max_df = pd.DataFrame(max_aggregated_vectors, index=df.index).add_prefix(max_skill_emb_prefix)

    return df.join(mean_df).join(max_df)
