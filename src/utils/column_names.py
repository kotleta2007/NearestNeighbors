import pandas as pd

def clean_column_names(df):
    """
    Renames columns by removing special characters and joining words together.

    Parameters:
    df (pandas.DataFrame): DataFrame with complex column names

    Returns:
    pandas.DataFrame: DataFrame with cleaned column names
    """
    # Create a copy to avoid modifying the original DataFrame
    df_cleaned = df.copy()

    # Function to clean individual column name
    def clean_name(col):
        # Remove {, }, ", and spaces
        cleaned = col.replace('{', '').replace('}', '').replace('"', '').replace(' ', '')
        # Split by commas and join
        parts = cleaned.split(',')
        return ''.join(parts)

    # Create mapping of old names to new names
    name_mapping = {col: clean_name(col) for col in df.columns}

    # Rename the columns
    df_cleaned.rename(columns=name_mapping, inplace=True)

    return df_cleaned
