import pandas as pd

def get_french_month_mapping():
    """Returns a dictionary mapping French month abbreviations to numeric months."""
    return {
        'janv': '01', 'févr': '02', 'mars': '03', 'avr': '04',
        'mai': '05', 'juin': '06', 'juil': '07', 'août': '08',
        'sept': '09', 'oct': '10', 'nov': '11', 'déc': '12'
    }

def convert_french_date(date_str):
    """
    Converts a date string from French format (e.g., 'janv-20') to ISO format ('2020-01-01').
    
    Args:
        date_str (str): Date string in format 'mmm-yy' where mmm is French month abbreviation
    
    Returns:
        str: Date string in format 'YYYY-MM-01'
    """
    month, year = date_str.split('-')
    month = month.lower()
    year = '20' + year  
    month_num = get_french_month_mapping()[month]
    return f"{year}-{month_num}-01"

