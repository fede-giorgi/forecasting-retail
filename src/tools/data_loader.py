import pandas as pd


def load_raw_data(input_path):
    """
    Loads raw excel data from all sheets and concatenates them.
    """
    print(f"Loading data from: {input_path}...")
    
    # Read all sheets into a dictionary of DataFrames
    all_sheets = pd.read_excel(input_path, sheet_name=None)
    
    # Concatenate all DataFrames into one
    df = pd.concat(all_sheets.values(), ignore_index=True)
    
    # Ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    
    print(f"Data loaded successfully. Combined Shape: {df.shape}")

    return df