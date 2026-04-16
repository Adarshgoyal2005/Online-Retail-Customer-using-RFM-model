"""
data_processor.py — Generic column-agnostic data processor.

Supports: .xlsx, .xls, .csv, .tsv, .txt (tab-separated)
Auto-detects the following semantic roles from any tabular dataset:
  customer_id  – unique customer identifier
  invoice_id   – transaction / order identifier
  date         – purchase / invoice date
  quantity     – numeric quantity of items
  price        – unit price per item
  total        – pre-computed total amount (optional fallback)

If TotalAmount cannot be computed from quantity × price, it falls back
to a pre-existing total column, or quantity alone as proxy.
"""

import re
import pandas as pd
import numpy as np


# ── Column synonym maps ────────────────────────────────────────────────────────
# Each key is a semantic role; values are ordered regex patterns (case-insensitive).
COLUMN_PATTERNS = {
    "customer_id": [
        r"customer.?id", r"cust.?id", r"client.?id", r"user.?id",
        r"customer.?no", r"cust.?no", r"member.?id", r"buyer.?id",
        r"customerid", r"custid", r"userid",
    ],
    "invoice_id": [
        r"invoice.?no", r"invoice.?id", r"invoice", r"order.?id",
        r"order.?no", r"transaction.?id", r"txn.?id", r"bill.?no",
        r"receipt.?id", r"sale.?id", r"orderid", r"invoiceid",
    ],
    "date": [
        r"invoice.?date", r"order.?date", r"purchase.?date",
        r"transaction.?date", r"sale.?date", r"date", r"timestamp",
        r"created.?at", r"order.?time", r"bill.?date",
    ],
    "quantity": [
        r"^quantity$", r"^qty$", r"^units?$", r"^amount$",
        r"quantity.?sold", r"qty.?sold", r"no.?of.?items",
        r"item.?count", r"pieces", r"count",
    ],
    "price": [
        r"unit.?price", r"price", r"rate", r"cost", r"unit.?cost",
        r"selling.?price", r"mrp", r"value",
    ],
    "total": [
        r"total.?amount", r"total.?price", r"total.?cost",
        r"line.?total", r"net.?amount", r"revenue", r"sales",
        r"total.?value", r"gross.?amount",
    ],
}


def _detect_column(df_cols, role):
    """Return the first DataFrame column name matching any pattern for 'role'."""
    cols_lower = {c: c.lower().strip() for c in df_cols}
    for pattern in COLUMN_PATTERNS[role]:
        for original, lower in cols_lower.items():
            if re.search(pattern, lower):
                return original
    return None


def detect_columns(df):
    """
    Auto-detect semantic column mapping for df.
    Returns a dict: {role: column_name_or_None}.
    """
    mapping = {}
    for role in COLUMN_PATTERNS:
        mapping[role] = _detect_column(df.columns.tolist(), role)
    return mapping


def load_and_clean(filepath):
    """
    Load a tabular file (Excel, CSV, TSV, TXT) and apply generic cleaning.
    Returns (cleaned_df, column_mapping).
    """
    ext = filepath.rsplit(".", 1)[-1].lower()
    if ext in ("csv", "txt"):
        # Try common separators
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(filepath, sep=sep, encoding="utf-8", low_memory=False)
                if df.shape[1] > 1:
                    break
            except Exception:
                continue
        else:
            df = pd.read_csv(filepath, low_memory=False)
    elif ext == "tsv":
        df = pd.read_csv(filepath, sep="\t", low_memory=False)
    else:
        df = pd.read_excel(filepath, engine="openpyxl")
    mapping = detect_columns(df)

    # ── Rename detected columns to standard names ────────────────────────────
    rename = {}
    if mapping["customer_id"]:
        rename[mapping["customer_id"]] = "Customer ID"
    if mapping["invoice_id"]:
        rename[mapping["invoice_id"]] = "Invoice"
    if mapping["date"]:
        rename[mapping["date"]] = "InvoiceDate"
    if mapping["quantity"]:
        rename[mapping["quantity"]] = "Quantity"
    if mapping["price"]:
        rename[mapping["price"]] = "Price"
    if mapping["total"] and mapping["total"] not in rename:
        rename[mapping["total"]] = "TotalAmount_raw"

    df = df.rename(columns=rename)

    # ── Customer ID ──────────────────────────────────────────────────────────
    if "Customer ID" in df.columns:
        df = df.dropna(subset=["Customer ID"])
    else:
        # If no customer column, create a synthetic one from invoice or row index
        if "Invoice" in df.columns:
            df["Customer ID"] = df["Invoice"].astype(str)
        else:
            df["Customer ID"] = df.index.astype(str)

    # ── Date ─────────────────────────────────────────────────────────────────
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.dropna(subset=["InvoiceDate"])
    else:
        # Synthetic date: look for any datetime column
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    df["InvoiceDate"] = parsed
                    df = df.dropna(subset=["InvoiceDate"])
                    break
            except Exception:
                pass
        else:
            # Last resort: assign a single reference date to all rows
            df["InvoiceDate"] = pd.Timestamp("2020-01-01")

    # ── Invoice ID ───────────────────────────────────────────────────────────
    if "Invoice" not in df.columns:
        if "Customer ID" in df.columns:
            df["Invoice"] = (
                df["Customer ID"].astype(str) + "_" + df.index.astype(str)
            )
        else:
            df["Invoice"] = df.index.astype(str)

    # ── Quantity & Price → TotalAmount ───────────────────────────────────────
    has_qty   = "Quantity" in df.columns
    has_price = "Price" in df.columns

    if has_qty:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
        df = df[df["Quantity"] > 0]

    if has_price:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        df = df[df["Price"] > 0]

    if has_qty and has_price:
        df["TotalAmount"] = df["Quantity"] * df["Price"]
    elif "TotalAmount_raw" in df.columns:
        df["TotalAmount"] = pd.to_numeric(df["TotalAmount_raw"], errors="coerce").fillna(0)
        df = df[df["TotalAmount"] > 0]
    elif has_qty:
        df["TotalAmount"] = df["Quantity"]   # use quantity as proxy
    elif has_price:
        df["TotalAmount"] = df["Price"]
    else:
        # Try first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df["TotalAmount"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce").fillna(0)
        else:
            df["TotalAmount"] = 1.0          # absolute fallback

    df = df.drop_duplicates()
    df = df.dropna(subset=["TotalAmount"])
    df = df[df["TotalAmount"] > 0]

    return df, mapping


def compute_rfm(df):
    """Compute Recency, Frequency, Monetary for each customer."""
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum"),
    ).reset_index()

    return rfm
