from pathlib import Path
import pandas as pd


def load_raw_data(input_path: str | Path) -> pd.DataFrame:
    """
    Load the Online Retail II dataset from an Excel file.
    Reads all available sheets and concatenates them into a single DataFrame.
    """
    path = Path(input_path)
    
    # Load all sheets into a dictionary
    sheets = pd.read_excel(
        path,
        sheet_name=None,
        dtype={"Invoice": str, "StockCode": str},
    )
    
    # Concatenate all sheets (Year 2009-2010 and 2010-2011)
    df = pd.concat(sheets.values(), ignore_index=True)
    
    # Ensure InvoiceDate is in datetime format for temporal analysis
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    
    return df

def load_processed_data(file_path: str | Path) -> pd.DataFrame:
    """Loads the processed data from a parquet file."""
    print("Loading processed data...")
    return pd.read_parquet(file_path)