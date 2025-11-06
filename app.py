# app.py — API-based dashboard with address & rich filters

import os
import json
import io
import requests
from datetime import datetime, timezone
from typing import Optional, List
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(page_title="Event Sales – Live (API Version)", page_icon="📈", layout="wide")

API_URL = "https://lugtmmcpcgzyytkzqozn.supabase.co/rest/v1/orders"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1Z3RtbWNwY2d6eXl0a3pxb3puIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkzODk0MDQsImV4cCI6MjA3NDk2NTQwNH0.uSEDsRNpH_QGwgGxrrxuYKCkuH3lszd8O9w7GN9INpE"

# Optional local dev CSV fallback (kept because you mentioned sharing a CSV)
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

    # Try API first
    try:
        r = requests.get(API_URL, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
    except Exception as e:
        st.warning(f"API fetch failed ({e}). Falling back to local CSV if available.")
        df = _load_local_csv() or pd.DataFrame()

    if df.empty:
        return df

    # Normalize
    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].apply(_parse_datetime)
    else:
        df["created_at"] = pd.Timestamp.utcnow().tz_convert(None)

    for c in ["items_total", "delivery_charge", "grand_total"]:
        if c in df.columns:
            df[c] = _safe_to_numeric(df[c])

    # items parsing
    if "items" in df.columns:
        def parse_items(x):
            if isinstance(x, (list, dict)):
                return x
            if pd.isna(x):
                return []
            try:
                return json.loads(x)
            except Exception:
                return []
        df["items_parsed"] = df["items"].apply(parse_items)
        df["items_count"] = df["items_parsed"].apply(lambda lst: sum(int(i.get("quantity", 0) or 0) for i in (lst or [])))
    else:
        df["items_parsed"] = [[] for _ in range(len(df))]
        df["items_count"] = 0

    # fill basics if missing
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
# Sidebar Filters
# ------------------------
st.sidebar.header("Filters")

today_utc = datetime.utcnow().date()
start = st.sidebar.date_input("Start date", value=today_utc)
end = st.sidebar.date_input("End date", value=today_utc)

start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
end_dt = datetime.combine(end, datetime.max.time()).replace(tzinfo=timezone.utc)

address_query = st.sidebar.text_input("Filter by address (contains)", value="").strip()
customer_query = st.sidebar.text_input("Customer name/phone (contains)", value="").strip()

min_amount = st.sidebar.number_input("Min grand total (Rs)", min_value=0, value=0, step=100)

if st.sidebar.button("🔄 Refresh data"):
    fetch_orders.clear()

st.sidebar.markdown("---")
st.sidebar.caption("API Source")
st.sidebar.text_input("Endpoint", value=API_URL, disabled=True)

# ------------------------
# Load
# ------------------------
df = fetch_orders(start_dt, end_dt)

st.title("📊 Event Sales – Live Dashboard (API Version)")
st.write("Real-time data pulled via Supabase REST. Cache TTL = 30s.")

if df.empty:
    st.warning("No data found. Verify API / CSV or date range.")
    st.stop()

# dynamic filter options
branch_opts = sorted([b for b in df["branch"].astype(str).unique() if b])
order_type_opts = sorted([o for o in df["order_type"].astype(str).unique() if o])
payment_opts = sorted([p for p in df["payment_method"].astype(str).unique() if p])
status_opts = sorted([s for s in df["status"].astype(str).str.title().unique() if s])

with st.sidebar.expander("More filters", expanded=False):
    sel_branches = st.multiselect("Branch", options=branch_opts, default=branch_opts)
    sel_order_types = st.multiselect("Order type", options=order_type_opts, default=order_type_opts)
    sel_payments = st.multiselect("Payment method", options=payment_opts, default=payment_opts)
    sel_status = st.multiselect("Status", options=status_opts, default=status_opts)

# apply filters
fdf = df.copy()
fdf = fdf[fdf["branch"].astype(str).isin(sel_branches)]
fdf = fdf[fdf["order_type"].astype(str).isin(sel_order_types)]
fdf = fdf[fdf["payment_method"].astype(str).isin(sel_payments)]
fdf = fdf[fdf["status"].astype(str).str.title().isin(sel_status)]
fdf = fdf[fdf.get("grand_total", 0).fillna(0) >= min_amount]

if address_query:
    fdf = fdf[fdf.get("customer_address", "").astype(str).str.contains(address_query, case=False, na=False)]

if customer_query:
    cus_cols = (fdf.get("customer_name", ""), fdf.get("customer_phone", ""))
    mask = cus_cols[0].astype(str).str.contains(customer_query, case=False, na=False) | \
           cus_cols[1].astype(str).str.contains(customer_query, case=False, na=False)
    fdf = fdf[mask]

if fdf.empty:
    st.warning("No rows after applying filters.")
    st.stop()

# ------------------------
# KPIs
# ------------------------
total_orders = len(fdf)
total_gmv = float(fdf["grand_total"].sum(skipna=True)) if "grand_total" in fdf.columns else 0
avg_ticket = total_gmv / total_orders if total_orders else 0
total_items = int(fdf["items_count"].sum())
completed_mask = fdf["status"].astype(str).str.lower().isin(["delivered", "completed", "paid", "done", "closed"])
completion_rate = completed_mask.mean() * 100 if total_orders else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Orders", f"{total_orders:,}")
k2.metric("GMV", f"Rs {total_gmv:,.0f}")
k3.metric("Avg Ticket", f"Rs {avg_ticket:,.0f}")
k4.metric("Items Sold", f"{total_items:,}")
k5.metric("Completion", f"{completion_rate:.1f}%")

# ------------------------
# Trends (hourly or daily depending on range)
# ------------------------
st.subheader("Sales Over Time")
range_days = (end_dt - start_dt).days
if range_days >= 2:
    fdf["bucket"] = fdf["created_at"].dt.tz_convert(None).dt.floor("D")
else:
    fdf["bucket"] = fdf["created_at"].dt.tz_convert(None).dt.floor("H")

by_time = fdf.groupby("bucket", as_index=False).agg(
    orders=("id", "count"),
    gmv=("grand_total", "sum")
).sort_values("bucket")
by_time["cum_gmv"] = by_time["gmv"].cumsum()

base = alt.Chart(by_time).encode(x=alt.X("bucket:T", title="Time"))
st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("orders:Q", title="Orders")), use_container_width=True)
st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("gmv:Q", title="GMV (Rs)")), use_container_width=True)
st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("cum_gmv:Q", title="Cumulative GMV (Rs)")), use_container_width=True)

# ------------------------
# Breakdowns
# ------------------------
st.subheader("Breakdowns")
c1, c2 = st.columns(2)

with c1:
    if "branch" in fdf.columns:
        b1 = fdf.groupby("branch", as_index=False)["grand_total"].sum().rename(columns={"grand_total": "gmv"})
        st.altair_chart(
            alt.Chart(b1).mark_bar().encode(
                x=alt.X("gmv:Q", title="GMV (Rs)"),
                y=alt.Y("branch:N", sort="-x"),
                tooltip=["branch", alt.Tooltip("gmv:Q", format=",")]
            ),
            use_container_width=True,
        )
    if "order_type" in fdf.columns:
        b2 = fdf.groupby("order_type", as_index=False).agg(orders=("id", "count"), gmv=("grand_total", "sum"))
        st.altair_chart(
            alt.Chart(b2).mark_bar().encode(
                x="orders:Q", y=alt.Y("order_type:N", sort="-x"),
                tooltip=["order_type", "orders", alt.Tooltip("gmv:Q", format=",")]
            ),
            use_container_width=True,
        )

with c2:
    if "payment_method" in fdf.columns:
        b3 = fdf.groupby("payment_method", as_index=False).agg(orders=("id", "count"), gmv=("grand_total", "sum"))
        st.altair_chart(
            alt.Chart(b3).mark_bar().encode(
                x="orders:Q", y=alt.Y("payment_method:N", sort="-x"),
                tooltip=["payment_method", "orders", alt.Tooltip("gmv:Q", format=",")]
            ),
            use_container_width=True,
        )
    if "status" in fdf.columns:
        b4 = fdf["status"].astype(str).str.title().value_counts().reset_index()
        b4.columns = ["status", "orders"]
        st.altair_chart(
            alt.Chart(b4).mark_bar().encode(
                x="orders:Q", y=alt.Y("status:N", sort="-x"),
                tooltip=["status", "orders"]
            ),
            use_container_width=True,
        )

# ------------------------
# Address Insights
# ------------------------
st.subheader("Address Insights")

# Top full-addresses
addr_series = fdf.get("customer_address", pd.Series(dtype=str)).astype(str).str.strip()
addr_counts = addr_series[addr_series != ""].value_counts().reset_index()
addr_counts.columns = ["address", "orders"]

if not addr_counts.empty:
    st.altair_chart(
        alt.Chart(addr_counts.head(15)).mark_bar().encode(
            x=alt.X("orders:Q"),
            y=alt.Y("address:N", sort="-x"),
            tooltip=["address", "orders"]
        ),
        use_container_width=True,
    )
    st.dataframe(addr_counts.head(50), use_container_width=True, hide_index=True)
else:
    st.info("No non-empty addresses to summarize.")

# ------------------------
# Top Items
# ------------------------
st.subheader("Top Items")
items_rows = []
for _, row in fdf.iterrows():
    for it in (row.get("items_parsed") or []):
        name = it.get("name") or it.get("item") or "Unknown"
        qty = int(it.get("quantity", 0) or 0)
        revenue = float(it.get("totalPrice", 0) or 0)
        items_rows.append({"name": name, "quantity": qty, "revenue": revenue})
items_df = pd.DataFrame(items_rows)

if not items_df.empty:
    agg = items_df.groupby("name", as_index=False).agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"))
    agg = agg.sort_values(["revenue", "quantity"], ascending=[False, False]).head(20)
    st.altair_chart(
        alt.Chart(agg).mark_bar().encode(
            x=alt.X("revenue:Q", title="Revenue (Rs)"),
            y=alt.Y("name:N", sort="-x"),
            tooltip=["name", "quantity", alt.Tooltip("revenue:Q", format=",")]
        ),
        use_container_width=True,
    )
    st.dataframe(agg, use_container_width=True, hide_index=True)
else:
    st.info("No item-level data available.")

# ------------------------
# Recent Orders + Export
# ------------------------
st.subheader("Recent Orders (Filtered)")
cols = [c for c in ["created_at","order_number","branch","order_type","payment_method",
                    "grand_total","status","cashier_name","customer_name","customer_phone","customer_address"]
        if c in fdf.columns]
st.dataframe(fdf.sort_values("created_at", ascending=False).head(100)[cols], use_container_width=True, hide_index=True)

# Export filtered
csv_buf = io.StringIO()
fdf.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download filtered CSV", data=csv_buf.getvalue(), file_name="orders_filtered.csv", mime="text/csv")

st.caption(f"Connected to Supabase REST: {API_URL}")
