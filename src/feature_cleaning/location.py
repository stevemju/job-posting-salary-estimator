import re


def clean_and_standardize_location(location: str) -> str:
    """
    Cleans a raw location string and maps it to a standardized region.
    Prioritizes major metropolitan areas and falls back to the state level.
    """
    if not isinstance(location, str):
        return "unknown"

    # --- IDEMPOTENCY CHECK ---
    # If the location is already in a clean format, return it immediately.
    if re.match(r'^(metro_|state_)', location) or location in ['remote', 'other_us', 'unknown']:
        return location

    loc_lower = location.lower()

    # --- 1. Handle Remote first ---
    if 'remote' in loc_lower:
        return "remote"

    # --- 2. Define mappings for major metropolitan areas ---
    metro_map = {
        'new york': 'metro_nyc', 'nyc': 'metro_nyc', 'jersey city': 'metro_nyc', 'stamford': 'metro_nyc', 'brooklyn': 'metro_nyc', 'queens': 'metro_nyc', 'newark': 'metro_nyc', 'albany, ny': 'metro_albany',
        'sf': 'metro_sf_bay', 'san francisco': 'metro_sf_bay', 'bay area': 'metro_sf_bay', 'cupertino': 'metro_sf_bay', 'palo alto': 'metro_sf_bay', 'sunnyvale': 'metro_sf_bay', 'mountain view': 'metro_sf_bay', 'santa clara': 'metro_sf_bay', 'redwood city': 'metro_sf_bay', 'livermore': 'metro_sf_bay',
        'los angeles': 'metro_la', 'burbank': 'metro_la', 'anaheim': 'metro_la', 'malibu': 'metro_la', 'culver city': 'metro_la', 'glendale': 'metro_la', 'pasadena': 'metro_la', 'downey': 'metro_la', 'orange county': 'metro_la',
        'boston': 'metro_boston', 'cambridge': 'metro_boston',
        'seattle': 'metro_seattle', 'issaquah': 'metro_seattle',
        'chicago': 'metro_chicago',
        'austin': 'metro_austin',
        'dallas': 'metro_dfw', 'fort worth': 'metro_dfw', 'plano': 'metro_dfw', 'dfw': 'metro_dfw',
        'washington, dc': 'metro_dc', 'ashburn': 'metro_dc', 'falls church': 'metro_dc',
        'san diego': 'metro_san_diego', 'la jolla': 'metro_san_diego', 'coronado': 'metro_san_diego',
        'denver': 'metro_denver', 'aurora': 'metro_denver',
        'atlanta': 'metro_atlanta', 'alpharetta': 'metro_atlanta',
        'miami': 'metro_miami', 'boca raton': 'metro_miami',
        'phoenix': 'metro_phoenix', 'gilbert': 'metro_phoenix',
        'raleigh': 'metro_raleigh_durham', 'durham': 'metro_raleigh_durham', 'chapel hill': 'metro_raleigh_durham',
        'houston': 'metro_houston',
        'philadelphia': 'metro_philly', 'king of prussia': 'metro_philly',
    }

    for keyword, metro_name in metro_map.items():
        if keyword in loc_lower:
            return metro_name

    # --- 3. Fallback to State level using regex ---
    states = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
        'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
        'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
        'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA',
        'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
        'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM',
        'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
        'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
        'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
        'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
        'district of columbia': 'DC'
    }
    state_abbrs = list(states.values())
    state_names = list(states.keys())

    # Create regex patterns: \b means word boundary
    state_abbr_pattern = re.compile(r'\b(' + '|'.join(state_abbrs) + r')\b', re.IGNORECASE)
    state_name_pattern = re.compile(r'\b(' + '|'.join(state_names) + r')\b', re.IGNORECASE)

    # First, look for abbreviations (more reliable, e.g., 'CA' vs 'Washington')
    match_abbr = state_abbr_pattern.search(location)
    if match_abbr:
        return f"state_{match_abbr.group(1).upper()}"

    # If no abbreviation, look for full state name
    match_name = state_name_pattern.search(loc_lower)
    if match_name:
        state_abbr = states[match_name.group(1)]
        return f"state_{state_abbr}"

    # --- 4. If all else fails, categorize ---
    # if 'united states' in loc_lower or 'usa' in loc_lower:
    #     return 'other_us'

    # the dataset contains only listings for the US
    return "other_us"
