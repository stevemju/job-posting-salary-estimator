import re
from typing import Dict


education_map = {
        'doctorate': ['phd', 'doctorate', 'md', 'dds', 'jd', 'dvm', 'do'],
        'master': ['master', 'mba', 'msc', 'meng', 'ma', 'ms'],
        'bachelor': ['bachelor', 'btech', 'bs', 'ba', 'bfa', 'bsc'],
        'associate': ['associate', 'vocational', 'trade school', 'paralegal certificate'],
        'high school': ['high school', 'ged'],
    }

def clean_and_categorize_education(education_level: str, education_map: Dict = education_map) -> str:
    if not isinstance(education_level, str):
        return 'unspecified'

    education_level = education_level.lower()
    education_level = re.sub(r'[^a-z0-9\s]', '', education_level)

    for category, keywords in education_map.items():
        pattern = '|'.join(keywords)

        if re.search(pattern, education_level):
            return category

    return 'unspecified'
