import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from db.client import (
    get_client, get_all_articles, get_all_chunk_embeddings,
    upsert_gap_feedback, get_all_gap_feedback,
)
from analysis.clustering import (
    compute_article_embeddings,
    cluster_articles,
    compute_tsne,
    label_clusters_with_claude,
)
from config import DEFAULT_CLUSTER_COUNT

from auth import require_auth

st.set_page_config(page_title="Content Gap Analysis", layout="wide")
require_auth()
st.title("Content Gap Analysis")

client = get_client()

n_clusters = st.slider("Number of clusters", min_value=5, max_value=30, value=DEFAULT_CLUSTER_COUNT)

if st.button("Run Analysis"):
    with st.spinner("Loading articles and embeddings..."):
        articles = get_all_articles(client)
        articles_map = {a["id"]: a for a in articles}
        raw_chunks = get_all_chunk_embeddings(client)

    if not raw_chunks:
        st.warning("No chunk embeddings found. Run ingestion first.")
        st.stop()

    with st.spinner("Computing article-level embeddings..."):
        article_embeddings = compute_article_embeddings(raw_chunks)

    with st.spinner(f"Clustering into {n_clusters} groups..."):
        labels, centroids, article_ids = cluster_articles(article_embeddings, n_clusters)

    with st.spinner("Computing t-SNE projection..."):
        tsne_coords = compute_tsne(article_embeddings, article_ids)

    # Build cluster → titles mapping
    cluster_titles = {}
    for aid, cluster_id in labels.items():
        title = articles_map.get(aid, {}).get("title", f"Article {aid}")
        cluster_titles.setdefault(cluster_id, []).append(title)

    with st.spinner("Claude is labeling clusters and finding gaps..."):
        feedback = get_all_gap_feedback(client)
        cluster_info = label_clusters_with_claude(cluster_titles, feedback=feedback)

    # Store results in session state so feedback buttons work without re-running
    st.session_state["cluster_info"] = cluster_info
    st.session_state["cluster_titles"] = cluster_titles
    st.session_state["scatter_data"] = []
    for i, aid in enumerate(article_ids):
        cluster_id = labels[aid]
        info = cluster_info.get(cluster_id, {"label": f"Topic {cluster_id}"})
        st.session_state["scatter_data"].append({
            "x": tsne_coords[i, 0],
            "y": tsne_coords[i, 1],
            "cluster": f"{cluster_id}: {info['label']}",
            "title": articles_map.get(aid, {}).get("title", f"Article {aid}"),
        })

# Render results from session state
if "cluster_info" in st.session_state:
    cluster_info = st.session_state["cluster_info"]
    cluster_titles = st.session_state["cluster_titles"]
    scatter_df = pd.DataFrame(st.session_state["scatter_data"])

    # t-SNE scatter plot
    st.subheader("Topic Clusters")
    st.caption(
        "Each dot is one article. Position is determined by t-SNE, which projects "
        "high-dimensional embeddings into 2D \u2014 articles that are semantically similar "
        "appear close together. Colors represent KMeans clusters (groups of related articles). "
        "Tight clusters = well-covered topics. Scattered dots = unique or cross-topic articles. "
        "Hover over any dot to see the article title."
    )
    fig = px.scatter(
        scatter_df, x="x", y="y", color="cluster",
        hover_data=["title"],
        title="Article Clusters (t-SNE projection)",
        height=600,
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

    # Gap suggestions with feedback buttons
    st.subheader("Content Gaps by Cluster")
    st.caption("Rate each suggestion so future analyses learn your preferences.")

    for cid in sorted(cluster_info.keys()):
        info = cluster_info[cid]
        count = len(cluster_titles.get(cid, []))
        label = info["label"]

        if not info.get("gaps"):
            continue

        st.markdown(f"**Cluster {cid}: {label}** ({count} articles)")
        for gap_idx, gap in enumerate(info["gaps"]):
            col_text, col_up, col_down = st.columns([8, 1, 1])
            with col_text:
                st.write(f"\u2022 {gap}")
            with col_up:
                if st.button("\U0001f44d", key=f"up_{cid}_{gap_idx}", help="I like this suggestion"):
                    upsert_gap_feedback(client, f"{cid}: {label}", gap, "up")
                    st.toast(f"Saved: liked \"{gap[:40]}...\"")
            with col_down:
                if st.button("\U0001f44e", key=f"down_{cid}_{gap_idx}", help="Not useful"):
                    upsert_gap_feedback(client, f"{cid}: {label}", gap, "down")
                    st.toast(f"Saved: disliked \"{gap[:40]}...\"")

    # Feedback stats
    all_feedback = get_all_gap_feedback(client)
    if all_feedback:
        up_count = sum(1 for f in all_feedback if f["rating"] == "up")
        down_count = sum(1 for f in all_feedback if f["rating"] == "down")
        st.caption(f"Feedback so far: {up_count} liked, {down_count} disliked. This shapes future suggestions.")

    # Cluster detail expanders
    st.subheader("Cluster Details")
    for cid in sorted(cluster_info.keys()):
        info = cluster_info[cid]
        titles = cluster_titles.get(cid, [])
        with st.expander(f"Cluster {cid}: {info['label']} ({len(titles)} articles)"):
            st.write("**Articles:**")
            for t in titles:
                st.write(f"- {t}")
            if info.get("gaps"):
                st.write("**Suggested gaps:**")
                for g in info["gaps"]:
                    st.write(f"- {g}")
