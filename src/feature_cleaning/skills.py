import numpy as np

from typing import List


def clean_skill_list(skills) -> List:
    if not isinstance(skills, (list, np.ndarray)):
        print(f"Non-list type encountered: {type(skills)}")
        return []

    return [str(skill).strip().lower() for skill in skills]
