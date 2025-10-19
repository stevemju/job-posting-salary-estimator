from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
import instructor

class JobDetails(BaseModel):
    """
    A Pydantic model to hold the structured details extracted from a job description.
    The overall instruction for the LLM is to act as an HR analyst and fill these fields.
    """
    technical_skills: List[str] = Field(
        ...,
        description="""
        A list of specific, granular TECHNICAL skills that are **explicitly mentioned in the job description**.
        - Look for keywords related to programming languages (e.g., Python, SQL), cloud platforms (e.g., AWS, Azure), or software tools (e.g., Tableau, Docker).
        - Extract only the keywords (e.g., if it appears, "experience with Amazon Web Services" should be "AWS").
        - **CRITICAL:** If the text does not mention any technical skills, you **must** return an empty list `[]`. Do not guess, invent skills, or copy the examples from these instructions.
        """
    )

    soft_skills: List[str] = Field(
        ...,
        description="""
        A list of key SOFT SKILLS and professional competencies that are **explicitly mentioned in the job description**.
        - **Examples:** Leadership, Communication, Teamwork, Project Management, Problem-Solving.
        - Extract general concepts (e.g., "Effective communication skills" -> "Communication").
        - **CRITICAL:** If the text does not mention any soft skills, you **must** return an empty list `[]`. Do not guess, invent skills, or copy the examples from these instructions.
        """
    )

    domain_skills: List[str] = Field(
        ...,
        description="""
        A list of DOMAIN-SPECIFIC business or professional skills related to the job's function that are **explicitly mentioned in the job description**. Your goal is to break down sentences and phrases into a granular list of core, reusable skills.
        - These are skills that require specific industry or functional knowledge but are not programming or IT skills.
        - **CRITICAL:** Your output MUST be a list of individual, single-concept strings. Do not return long phrases or sentences. If the text does not mention any domain skills, you **must** return an empty list `[]`. Do not guess, invent skills, or copy the examples from these instructions.
        """
    )

    experience_years_required: int = Field(
        ...,
        description="""
        Extract the required years of professional experience as a single integer.
        - "at least 5 years" -> 5
        - "5-7 years of experience" -> 5 (take the lower bound)
        - "three years" -> 3
        - **CRITICAL:** Round decimals to the nearest whole number. Output SHOULD be an integer, NOT a float. For instance: "4.5 years of experience" -> return 5
        - **If no experience is mentioned, return -1.**
        """
    )

    education_level: str = Field(
        ...,
        description="""
        Extract the highest level of education required.
        - Standardize the output to one of the following: "High School", "Associate's", "Bachelor's", "Master's", "PhD".
        - "B.S. in Computer Science" -> "Bachelor's"
        - "Master's degree preferred, Bachelor's required" -> "Master's" (take the highest mentioned)
        - **If no education level is mentioned, return "Unspecified".**
        """
    )


def get_job_details(description: str, index: int, client: OpenAI, decoder_model_name: str) -> Tuple[int, Dict]:
    if not isinstance(description, str) or len(description.strip()) < 20:
        print(f"Invalid description.")
        return index, {
            'technical_skills': [],
            'soft_skills': [],
            'domain_skills': [],
            'skills': [],
            'experience_years_required': None,
            'education_level': None
        }

    try:
        instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)

        prompt_with_instruction = (
            "You are an expert HR analyst. Your task is to extract the required job details "
            f"from the following job description:\n\n{description}"
        )

        job_details = instructor_client.chat.completions.create(
            model=decoder_model_name,
            response_model=JobDetails,
            messages=[
                {"role": "user", "content": prompt_with_instruction}
            ],
            max_retries=3,
        )

        output_dict = job_details.model_dump()
        combined_skills = (job_details.technical_skills or []) + (job_details.soft_skills or []) + (job_details.domain_skills or [])
        output_dict['skills'] = list(set(combined_skills))

        return index, output_dict

    except Exception as e:
        print(f"Error processing row {index}: {e}")

        return index, {
            'technical_skills': [],
            'soft_skills': [],
            'domain_skills': [],
            'skills': [],
            'experience_years_required': None,
            'education_level': None
        }
