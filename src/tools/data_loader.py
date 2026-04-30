from pathlib import Path
import pandas as pd


def load_raw_data(input_path: str | Path) -> pd.DataFrame:
    """
    Load Online Retail II xlsx. Reads BOTH sheets and concatenates
    with a SourceSheet column for traceability.
    """
    path = Path(input_path)
    sheets = pd.read_excel(
        path,
        sheet_name=None,
        dtype={"Invoice": str, "StockCode": str},
    )
    df = pd.concat(
        [s.assign(SourceSheet=name) for name, s in sheets.items()],
        ignore_index=True,
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df
