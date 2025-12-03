# app.py — Event Sales Dashboard + DFPL "EXE Not Working" Tab

import os
import json
import io
import requests
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(page_title="Event Sales – Live (API Version)", page_icon="📈", layout="wide")

API_URL = "https://lugtmmcpcgzyytkzqozn.supabase.co/rest/v1/orders"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1Z3RtbWNwY2d6eXl0a3pxb3puIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg3MTcwNTcsImV4cCI6MjA3NDI5MzA1N30.FmZV8z8XXm1x_cex8CxRPRYt0RT_L9Mrm0qCc03zcj8"

LOCAL_CSV_CANDIDATES = ["orders_rows.csv", "./data/orders_rows.csv"]


# ------------------------
# Helpers
# ------------------------
def _parse_datetime(dt_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except Exception:
        return datetime.utcnow().replace(tzinfo=timezone.utc)


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") if series is not None else series


def _load_local_csv() -> Optional[pd.DataFrame]:
    for p in LOCAL_CSV_CANDIDATES:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None


# ------------------------
# Data Fetch (Supabase REST)
# ------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_orders(min_dt: Optional[datetime] = None, max_dt: Optional[datetime] = None) -> pd.DataFrame:
    headers = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}
    params = {"select": "*"}
    if min_dt:
        params["created_at"] = f"gte.{min_dt.isoformat()}"
    if max_dt:
        if "created_at" in params:
            params["and"] = f"(created_at.lte.{max_dt.isoformat()})"
        else:
            params["created_at"] = f"lte.{max_dt.isoformat()}"

    try:
        r = requests.get(API_URL, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception:
        st.warning("API error — falling back to local CSV.")
        tmp = _load_local_csv()
        df = tmp if tmp is not None else pd.DataFrame()

    if df.empty:
        return df

    # Normalize
    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].apply(_parse_datetime)

    for c in ["items_total", "delivery_charge", "grand_total"]:
        if c in df.columns:
            df[c] = _safe_to_numeric(df[c])

    # Parse items
    if "items" in df.columns:
        def parse_items(x):
            if isinstance(x, (list, dict)):
                return x
            try:
                return json.loads(x)
            except Exception:
                return []
        df["items_parsed"] = df["items"].apply(parse_items)
        df["items_count"] = df["items_parsed"].apply(
            lambda lst: sum(int(i.get("quantity", 0) or 0) for i in (lst or []))
        )
    else:
        df["items_parsed"] = [[] for _ in range(len(df))]
        df["items_count"] = 0

    # Defaults
    for col, default in [
        ("branch", "N/A"),
        ("status", "unknown"),
        ("order_type", "N/A"),
        ("payment_method", "N/A"),
        ("customer_address", ""),
        ("customer_name", ""),
        ("customer_phone", ""),
        ("cashier_name", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


# ------------------------
# SIDEBAR FILTERS
# ------------------------
st.sidebar.header("Filters")

today_utc = datetime.utcnow().date()
start = st.sidebar.date_input("Start date", today_utc)
end = st.sidebar.date_input("End date", today_utc)

start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
end_dt = datetime.combine(end, datetime.max.time()).replace(tzinfo=timezone.utc)

address_query = st.sidebar.text_input("Filter by address (contains)", "").strip()
customer_query = st.sidebar.text_input("Customer name/phone (contains)", "").strip()
min_amount = st.sidebar.number_input("Min grand total (Rs)", min_value=0, value=0)

if st.sidebar.button("🔄 Refresh data"):
    fetch_orders.clear()

st.sidebar.caption("API: " + API_URL)


# ------------------------
# LOAD DATA
# ------------------------
df = fetch_orders(start_dt, end_dt)

if df.empty:
    st.warning("No data returned.")
    st.stop()


# Dropdown filter values
branch_opts = sorted(df["branch"].astype(str).unique())
order_type_opts = sorted(df["order_type"].astype(str).unique())
payment_opts = sorted(df["payment_method"].astype(str).unique())
status_opts = sorted(df["status"].astype(str).str.title().unique())


# Extra filters
with st.sidebar.expander("More filters"):
    sel_branches = st.multiselect("Branch", branch_opts, branch_opts)
    sel_order_types = st.multiselect("Order type", order_type_opts, order_type_opts)
    sel_payments = st.multiselect("Payment method", payment_opts, payment_opts)
    sel_status = st.multiselect("Status", status_opts, status_opts)


# APPLY FILTERS
fdf = df.copy()
fdf = fdf[fdf["branch"].isin(sel_branches)]
fdf = fdf[fdf["order_type"].isin(sel_order_types)]
fdf = fdf[fdf["payment_method"].isin(sel_payments)]
fdf = fdf[fdf["status"].str.title().isin(sel_status)]
fdf = fdf[fdf["grand_total"].fillna(0) >= min_amount]

if address_query:
    fdf = fdf[fdf["customer_address"].str.contains(address_query, case=False, na=False)]

if customer_query:
    m = (
        fdf["customer_name"].str.contains(customer_query, case=False, na=False) |
        fdf["customer_phone"].str.contains(customer_query, case=False, na=False)
    )
    fdf = fdf[m]

if fdf.empty:
    st.warning("No rows after filters.")
    st.stop()


# ------------------------
# TABS
# ------------------------
main_tab, dfpl_tab = st.tabs(["📊 Dashboard", "🧾 DFPL – EXE Not Working"])


# ===========================
# MAIN DASHBOARD TAB
# ===========================
with main_tab:

    st.title("📊 Event Sales – Live Dashboard (API Version)")

    # KPIs
    total_orders = len(fdf)
    total_gmv = fdf["grand_total"].sum()
    avg_ticket = total_gmv / total_orders if total_orders else 0
    total_items = fdf["items_count"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders", f"{total_orders:,}")
    k2.metric("GMV", f"Rs {total_gmv:,.0f}")
    k3.metric("Avg Ticket", f"Rs {avg_ticket:,.0f}")
    k4.metric("Items", f"{total_items:,}")

    # RECENT ORDERS
    st.subheader("Recent Orders")
    cols = [
        "created_at","order_number","branch","order_type","payment_method",
        "grand_total","status","cashier_name","customer_name","customer_phone","customer_address"
    ]
    cols = [c for c in cols if c in fdf.columns]

    st.dataframe(
        fdf.sort_values("created_at", ascending=False).head(100)[cols],
        use_container_width=True,
        hide_index=True
    )


# ===========================
# DFPL TAB — ONLY EXE NOT WORKING
# ===========================
with dfpl_tab:

    st.header("🧾 DFPL – Only EXE Not Working Orders")
    st.info("Showing only orders where **customer_address contains 'EXE Not Working'**.")

    dfpl_df = fdf[
        fdf["customer_address"].str.contains("EXE Not Working", case=False, na=False)
    ].copy()

    # Search inputs
    search_order_no = st.text_input("Order Number (exact match)")
    search_text = st.text_input("Name / Phone / Address (contains)")
    search_min_amount = st.number_input("Min Amount (Rs)", 0, value=0)

    # Apply search
    if search_order_no:
        dfpl_df = dfpl_df[dfpl_df["order_number"].astype(str) == search_order_no]

    if search_text:
        txt = search_text.strip()
        m = (
            dfpl_df["customer_name"].str.contains(txt, case=False, na=False) |
            dfpl_df["customer_phone"].str.contains(txt, case=False, na=False) |
            dfpl_df["customer_address"].str.contains(txt, case=False, na=False)
        )
        dfpl_df = dfpl_df[m]

    dfpl_df = dfpl_df[dfpl_df["grand_total"].fillna(0) >= search_min_amount]

    if dfpl_df.empty:
        st.warning("No EXE Not Working orders match the criteria.")
    else:
        st.success(f"{len(dfpl_df):,} matching orders found.")

        view_cols = [
            "created_at","order_number","branch","order_type",
            "payment_method","grand_total","status",
            "cashier_name","customer_name","customer_phone","customer_address"
        ]
        view_cols = [c for c in view_cols if c in dfpl_df.columns]

        st.dataframe(
            dfpl_df.sort_values("created_at", ascending=False)[view_cols],
            use_container_width=True,
            hide_index=True
        )

        # Export
        csv_buf = io.StringIO()
        dfpl_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download DFPL CSV",
            csv_buf.getvalue(),
            "dfpl_exe_not_working_orders.csv",
            "text/csv"
        )


st.caption(f"Connected to Supabase REST: {API_URL}")
