import streamlit as st
import pandas as pd
import plotly.express as px

from db.client import get_client, get_all_articles, get_article_by_id
from auth import require_auth

st.set_page_config(page_title="Article Explorer", layout="wide")
require_auth()
st.title("Article Explorer")

client = get_client()
articles = get_all_articles(client)
df = pd.DataFrame(articles)

if df.empty:
    st.warning("No articles found. Run the ingestion pipeline first.")
    st.stop()

df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

# Compute safe date bounds (drop NaT rows for the date picker)
valid_dates = df["post_date"].dropna()
min_date = valid_dates.min() if not valid_dates.empty else pd.Timestamp("2016-01-01")
max_date = valid_dates.max() if not valid_dates.empty else pd.Timestamp.now()

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    search_query = st.text_input("Search title / subtitle", "")

with col2:
    types = ["All"] + sorted(df["type"].dropna().unique().tolist())
    selected_type = st.selectbox("Type", types)

with col3:
    date_range = st.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
    )

# Apply filters
filtered = df.copy()

if search_query:
    mask = (
        filtered["title"].str.contains(search_query, case=False, na=False)
        | filtered["subtitle"].str.contains(search_query, case=False, na=False)
    )
    filtered = filtered[mask]

if selected_type != "All":
    filtered = filtered[filtered["type"] == selected_type]

if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    has_date = filtered["post_date"].notna()
    filtered = filtered[
        ~has_date | ((filtered["post_date"] >= start) & (filtered["post_date"] <= end))
    ]

st.write(f"**{len(filtered)}** articles found")

# Sortable table
sort_col = st.selectbox("Sort by", ["post_date", "title", "word_count"], index=0)
sort_asc = st.checkbox("Ascending", value=False)
filtered = filtered.sort_values(sort_col, ascending=sort_asc)

# Display table
display_cols = ["title", "subtitle", "post_date", "type", "audience", "word_count"]
st.dataframe(
    filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=400,
)

# Expandable full-text view
st.subheader("Read Article")
article_options = filtered[["id", "title"]].to_dict("records")
if article_options:
    selected = st.selectbox(
        "Select an article to read",
        options=article_options,
        format_func=lambda x: x["title"],
    )
    if selected and st.button("Load full text"):
        full = get_article_by_id(client, selected["id"])
        st.markdown(full["full_text_markdown"])

# Word count distribution
st.subheader("Word Count Distribution")
fig = px.histogram(
    filtered, x="word_count", nbins=30,
    title="Article Word Count Distribution",
    labels={"word_count": "Word Count", "count": "Number of Articles"},
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)
