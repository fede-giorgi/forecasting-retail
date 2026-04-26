import pandas as pd


def load_raw_data(input_path):
    """
    Loads raw excel data.
    """
    print(f"Loading data from: {input_path}...")
    df = pd.read_excel(input_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    
    print(f"Data loaded successfully. Shape: {df.shape}")

    return df