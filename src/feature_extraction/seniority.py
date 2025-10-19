def extract_seniority_from_title(title: str) -> str:
    if not isinstance(title, str):
        return "unknown"

    title_lower = title.lower()

    # The order is important here: check for the most senior roles first. Except for cook to not be catched as coo
    if any(keyword in title_lower for keyword in ['cook']):
        level = "junior"
    elif any(keyword in title_lower for keyword in ['chief', 'c-level', 'c suite', 'ceo', 'cfo', 'cto', 'coo']):
        level = "chief"
    elif any(keyword in title_lower for keyword in ['vp', 'vice president', 'partner', 'executive']):
        level = "vp"
    elif 'director' in title_lower:
        level = "director"
    elif any(keyword in title_lower for keyword in ['manager', 'lead', 'supervisor', 'head', 'foreman', 'superintendent']):
        level = "manager_lead"
    elif any(keyword in title_lower for keyword in ['senior', 'sr.', 'sr', 'principal']):
        level = "senior"
    # second check is to prevent internal medicine to be catched as an internship
    elif any(keyword in title_lower for keyword in ['intern', 'trainee', 'graduate']) and 'internal' not in title_lower and 'teaching' not in title_lower:
        level = "intern"
    elif any(keyword in title_lower for keyword in ['entry', 'junior', 'jr', 'associate']):
        level = "entry_junior"
    elif any(keyword in title_lower for keyword in ['attorney']):
        level = 'attorney'
    else:
        level = "individual_contributor"

    return level
