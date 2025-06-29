import pandas as pd
import io

def load_legend(legend_df):
    """
    Loads instrument types from the provided pandas DataFrame.
    Looks for 'Instrument Type' or 'Instrument_Type' column.

    Args:
        legend_df (pd.DataFrame): DataFrame loaded from the legend Excel/CSV.
                                 This DataFrame is passed directly from Flask's file stream.

    Returns:
        set: A set of unique instrument type strings (e.g., {'PT', 'TT', 'FC'}).
             Returns an empty set if the required column is not found.
    """
    column_name = None
    # Check for common column names for instrument type, prioritizing 'Instrument Type'
    if 'Instrument Type' in legend_df.columns:
        column_name = 'Instrument Type'
    elif 'Instrument_Type' in legend_df.columns:
        column_name = 'Instrument_Type'
    else:
        # If neither column is found, print a warning and return an empty set
        print("Warning: Legend file must contain 'Instrument Type' or 'Instrument_Type' column.")
        return set()

    # Filter out any empty (NaN) values, convert to string, and then to uppercase
    # Using .dropna() ensures we only process cells that have content.
    # .astype(str) ensures all entries are strings before .str.upper().
    valid_types = set(legend_df[column_name].dropna().astype(str).str.upper())
    return valid_types

def generate_excel(data, filename):
    """
    Generates an Excel (.xlsx) file from a list of dictionaries in memory.
    This allows the web app to create the file and send it directly without saving to disk first.

    Args:
        data (list): A list of dictionaries, where each dictionary represents an extracted instrument.
                     Example: [{'P&ID Filename': 'file.pdf', 'Instrument Tag': 'P-101', ...}, ...]
        filename (str): The desired name for the Excel file (e.g., "instrument_index.xlsx").

    Returns:
        io.BytesIO: A BytesIO object containing the Excel file content. This can be directly
                    sent as a file download by Flask.
    """
    if not data:
        # If no data is provided, create an empty DataFrame with the expected columns.
        # This ensures that even if no instruments are found, a valid (empty) Excel file is generated.
        df = pd.DataFrame(columns=[
            'P&ID Filename', 'Instrument Tag', 'Instrument Type',
            'Circle Center X', 'Circle Center Y', 'Circle Radius', 'OCR Raw Text (In ROI)'
        ])
    else:
        # Create a pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(data)

    # Create an in-memory binary stream to hold the Excel file content
    output = io.BytesIO()

    # Use pandas' ExcelWriter to write the DataFrame to the in-memory stream
    # engine='openpyxl' is necessary for .xlsx format.
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Instrument Index') # Write without row numbers

    # Rewind the stream to the beginning so that Flask can read its entire content when sending
    output.seek(0) 
    return output