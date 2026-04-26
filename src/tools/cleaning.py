import pandas as pd

def clean_raw_data(df):
    
    # Remove cancellations and invalid lines
    df = df.dropna(subset=["InvoiceDate", "StockCode", "Invoice"]).copy()
    df["Invoice"] = df["Invoice"].astype(str)

    # "C..." invoices are cancellations
    df = df[~df["Invoice"].str.startswith("C")].copy()
    df = df[df["Quantity"] > 0].copy()
    df = df[df["Price"] > 0].copy()

    # Revenue
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Monday-aligned week
    df["Week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time

    # Consistent types
    df["StockCode"] = df["StockCode"].astype(str)

    return df