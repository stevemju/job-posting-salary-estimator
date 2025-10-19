import instructor
import pandas as pd

from pydantic import BaseModel, Field
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from openai import OpenAI

from llm.ollama_setup import is_ollama_server_running


class JobLocation(BaseModel):
    city: str = Field(..., description='City of the job posting as a string. If the job is remote, try to find the city where the company is based.')
    state: str = Field(..., description='The 2-letter state abbreviation of the job posting as a string. E.g. CA, NY If the job is remote, try to find the state where the company is based.')

def get_job_location(description: str, index: int, client: OpenAI, decoder_model_name: str) -> Tuple[int, Dict]:
    if not isinstance(description, str):
        return index, {'refined_location': 'unknown'}

    try:
        instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
        job_location = instructor_client.chat.completions.create(
            model=decoder_model_name,
            response_model=JobLocation,
            messages=[
                {"role": "user", "content": f"Extract the city and state from the following job description. If no location is mentioned, return 'unknown'. Description: {description}"}
            ],
            max_retries=2,
        )

        city = job_location.city.lower()
        state = job_location.state.lower()

        if city == 'unknown' and state == 'unknown':
            return index, {'refined_location': 'other_us'}
        elif city != 'unknown' and state != 'unknown':
            refined_location = f"{city}, {state}"
            return index, {'refined_location': refined_location}
        elif city == 'unknown':
            return index, {'refined_location': state}
        elif state == 'unknown':
            return index, {'refined_location': city}

    except Exception as e:
        print(f"Error processing index {index}: {e}")
        return index, {'refined_location': 'unknown'}
    
def refine_locations_concurrently(
    df: pd.DataFrame,
    client: OpenAI,
    decoder_model_name: str,
    location_column: str = 'cleaned_location',
    description_column: str = 'description',
    max_workers: int = 16
):
    if not is_ollama_server_running():
        return None

    df_to_process = df[(df[location_column] == 'other_us') | (df[location_column] == 'unknown')].copy()

    if df_to_process.empty:
        print("No 'other_us' or 'unknown' locations to refine. DataFrame is already clean.")
        return df

    print(f"Found {len(df_to_process[df_to_process[location_column] == 'other_us'])} rows with location as 'other_us'.")
    print(f"Found {len(df_to_process[df_to_process[location_column] == 'unknown'])} rows with location as 'unknown'. Starting LLM refinement...")

    results_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_job_location, row[description_column], index, client, decoder_model_name): index
            for index, row in df_to_process.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Refining Locations"):
            original_index, result_data = future.result()
            result_data['original_index'] = original_index
            results_list.append(result_data)

    if not results_list:
        print("No results were generated.")
        return df

    # --- MERGE RESULTS BACK ---
    results_df = pd.DataFrame(results_list).set_index('original_index')
    print(results_df)

    # Update the 'cleaned_location' column in the original DataFrame
    # where the new refined_location is not 'unknown' or 'other_us'
    final_df = df.copy()
    # Create a boolean mask for valid, non-unknown refined locations
    valid_results = results_df[(results_df['refined_location'] != 'unknown') & (results_df['refined_location'] != 'other_us')]

    final_df['refined_location'] = final_df[location_column]

    # Use .loc to safely update the original DataFrame based on the index
    if not valid_results.empty:
        print(f"Updating {len(valid_results)} locations with new LLM-extracted data.")
        final_df.loc[valid_results.index, 'refined_location'] = valid_results['refined_location']

    print("Location refinement complete.")
    return final_df
