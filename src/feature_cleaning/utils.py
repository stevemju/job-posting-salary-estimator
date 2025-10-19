import ast
import pandas as pd

def parse_stringified_list(val):
    """
    Safely parses a string that looks like a list back into a Python list.
    Handles potential errors like empty or malformed cells.
    """
    if pd.isna(val):
        return []
    try:
        # ast.literal_eval is a safe way to evaluate a string containing a
        # Python literal or container display.
        evaluated = ast.literal_eval(val)
        if isinstance(evaluated, list):
            return evaluated
        else:
            return []
    except (ValueError, SyntaxError):
        # If the string is not a valid list literal (e.g., just a plain word),
        # return an empty list or handle as you see fit.
        return []
